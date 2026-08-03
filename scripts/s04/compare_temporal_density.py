"""Compare verified sparse and dense S04 temporal-localization coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from spatial_reconstruction.contracts import PerceptionTarget
from spatial_reconstruction.localization import (
    CorrectedPairState,
    CorrectedTrackingRunSummary,
    TemporalPresentationRunSummary,
    TemporalPresentationState,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--baseline-verification", type=Path, required=True)
    parser.add_argument("--dense-summary", type=Path, required=True)
    parser.add_argument("--dense-verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    output = _resolve(root, args.output)
    if output.exists():
        raise FileExistsError(f"comparison output already exists: {output}")
    baseline = _load_run(root, args.baseline_summary, args.baseline_verification)
    dense = _load_run(root, args.dense_summary, args.dense_verification)
    if baseline.summary.policy != dense.summary.policy:
        raise ValueError("temporal density runs use different D034 policies")
    if len(baseline.summary.presentation_records) != len(
        dense.summary.presentation_records
    ):
        raise ValueError("temporal density runs use different timeline coverage")
    if any(
        record.state is TemporalPresentationState.INFERRED
        for run in (baseline, dense)
        for record in run.summary.presentation_records
    ):
        raise ValueError("temporal density comparison cannot contain inferred positions")

    target_comparison = {
        target.value: _compare_target(baseline.summary, dense.summary, target)
        for target in PerceptionTarget
    }
    baseline_pair_count = len(baseline.corrected.d033_pair_observations)
    dense_pair_count = len(dense.corrected.d033_pair_observations)
    if dense_pair_count <= baseline_pair_count:
        raise ValueError("dense run does not add corrected pair observations")
    if not all(
        values["dense_missing_tick_count"] < values["baseline_missing_tick_count"]
        for values in target_comparison.values()
    ):
        raise ValueError("dense run does not reduce missing ticks for both targets")

    payload = {
        "schema_version": 1,
        "stage": "S04",
        "status": "passed",
        "purpose": "verified_sparse_vs_dense_dynamic_localization_comparison",
        "baseline_summary_ref": _relative(baseline.path, root),
        "baseline_summary_sha256": _sha256(baseline.path),
        "dense_summary_ref": _relative(dense.path, root),
        "dense_summary_sha256": _sha256(dense.path),
        "policy_id": dense.summary.policy.policy_id,
        "timeline_tick_count_per_target": len(dense.summary.presentation_records) // 2,
        "baseline_keyframe_pair_count": baseline_pair_count // 2,
        "dense_keyframe_pair_count": dense_pair_count // 2,
        "baseline_usable_measurement_count": _usable_pair_count(baseline.corrected),
        "dense_usable_measurement_count": _usable_pair_count(dense.corrected),
        "dense_rejected_pair_states": dict(
            sorted(
                Counter(
                    record.state.value
                    for record in dense.corrected.d033_pair_observations
                    if record.state
                    not in {CorrectedPairState.FUSED, CorrectedPairState.SINGLE_CAMERA}
                ).items()
            )
        ),
        "target_comparison": target_comparison,
        "baseline_measured_segment_count": len(
            baseline.summary.measured_trajectory_segments
        ),
        "dense_measured_segment_count": len(dense.summary.measured_trajectory_segments),
        "known_backpack_gap_seconds": dense.verification[
            "known_backpack_gap_seconds"
        ],
        "known_backpack_gap_bridged": False,
        "inferred_position_count": 0,
        "accuracy_interpretation": (
            "The comparison measures temporal evidence coverage and policy consistency, "
            "not absolute XYZ error; no dynamic ground-truth trajectory is available."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


class _LoadedRun:
    def __init__(
        self,
        *,
        path: Path,
        summary: TemporalPresentationRunSummary,
        verification: dict[str, Any],
        corrected: CorrectedTrackingRunSummary,
    ) -> None:
        self.path = path
        self.summary = summary
        self.verification = verification
        self.corrected = corrected


def _load_run(
    root: Path, summary_arg: Path, verification_arg: Path
) -> _LoadedRun:
    summary_path = _resolve(root, summary_arg)
    verification_path = _resolve(root, verification_arg)
    summary = TemporalPresentationRunSummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    verification = _read_object(verification_path)
    if (
        verification.get("status") != "passed"
        or verification.get("source_summary_sha256") != _sha256(summary_path)
        or verification.get("known_backpack_gap_bridged") is not False
    ):
        raise ValueError("temporal density input lacks matching passed verification")
    corrected_path = _resolve(root, Path(summary.source_corrected_summary_ref))
    _require_hash(corrected_path, summary.source_corrected_summary_sha256)
    corrected = CorrectedTrackingRunSummary.model_validate_json(
        corrected_path.read_text(encoding="utf-8")
    )
    return _LoadedRun(
        path=summary_path,
        summary=summary,
        verification=verification,
        corrected=corrected,
    )


def _compare_target(
    baseline: TemporalPresentationRunSummary,
    dense: TemporalPresentationRunSummary,
    target: PerceptionTarget,
) -> dict[str, int | float]:
    baseline_counts = _target_counts(baseline, target)
    dense_counts = _target_counts(dense, target)
    tick_count = sum(baseline_counts.values())
    if sum(dense_counts.values()) != tick_count:
        raise ValueError("target timeline counts differ")
    baseline_display = (
        baseline_counts[TemporalPresentationState.MEASURED]
        + baseline_counts[TemporalPresentationState.STALE]
    )
    dense_display = (
        dense_counts[TemporalPresentationState.MEASURED]
        + dense_counts[TemporalPresentationState.STALE]
    )
    return {
        "baseline_measured_tick_count": baseline_counts[
            TemporalPresentationState.MEASURED
        ],
        "dense_measured_tick_count": dense_counts[TemporalPresentationState.MEASURED],
        "baseline_display_coverage_tick_count": baseline_display,
        "dense_display_coverage_tick_count": dense_display,
        "display_coverage_gain_tick_count": dense_display - baseline_display,
        "baseline_display_coverage_fraction": baseline_display / tick_count,
        "dense_display_coverage_fraction": dense_display / tick_count,
        "display_coverage_gain_percentage_points": 100.0
        * (dense_display - baseline_display)
        / tick_count,
        "baseline_missing_tick_count": baseline_counts[
            TemporalPresentationState.MISSING
        ],
        "dense_missing_tick_count": dense_counts[TemporalPresentationState.MISSING],
        "missing_tick_reduction": baseline_counts[TemporalPresentationState.MISSING]
        - dense_counts[TemporalPresentationState.MISSING],
    }


def _target_counts(
    summary: TemporalPresentationRunSummary, target: PerceptionTarget
) -> Counter[TemporalPresentationState]:
    return Counter(
        record.state
        for record in summary.presentation_records
        if record.target is target
    )


def _usable_pair_count(summary: CorrectedTrackingRunSummary) -> int:
    return sum(
        record.state in {CorrectedPairState.FUSED, CorrectedPairState.SINGLE_CAMERA}
        for record in summary.d033_pair_observations
    )


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _require_hash(path: Path, expected: str) -> None:
    if _sha256(path) != expected:
        raise ValueError(f"artifact hash changed: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
