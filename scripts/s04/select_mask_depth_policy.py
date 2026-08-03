"""Select S04 confidence and visible-surface rules from retained diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from spatial_reconstruction.contracts import PerceptionTarget
from spatial_reconstruction.localization import (
    ActionDepthRunSummary,
    MaskAlignmentRunSummary,
    MaskDepthDiagnosticRecord,
    MaskDepthDiagnosticRunSummary,
    MaskDepthPolicySelectionSummary,
    MaskDepthSamplingPolicy,
    MaskDepthStrategy,
    TargetVisibleSurfaceRule,
    build_mask_depth_candidates,
    select_candidate_relative_confidence,
    summarize_distribution,
)

UInt8Array = NDArray[np.uint8]
Float32Array = NDArray[np.float32]
CAMERA_INDEX = {"camera_a": 0, "camera_b": 1}
SELECTED_STRATEGY = {
    PerceptionTarget.PERSON: MaskDepthStrategy.PERSON_LOWER_BODY,
    PerceptionTarget.BACKPACK: MaskDepthStrategy.CONNECTED_DEPTH_CLUSTER,
}
MINIMUM_RETAINED_SAMPLES = {
    PerceptionTarget.PERSON: 256,
    PerceptionTarget.BACKPACK: 128,
}
CANDIDATE_PERCENTILES = (0.0, 20.0, 40.0, 60.0, 80.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--diagnostics-summary",
        type=Path,
        default=Path("artifacts/s04/mask_depth_diagnostics_20260801/summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/s04/mask_depth_diagnostics_20260801/policy_selection.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    diagnostics_path = _resolve(project_root, args.diagnostics_summary)
    output_path = _resolve(project_root, args.output)
    if output_path.exists():
        raise FileExistsError(f"policy selection output already exists: {output_path}")
    diagnostics = MaskDepthDiagnosticRunSummary.model_validate_json(
        diagnostics_path.read_text(encoding="utf-8")
    )
    alignment_path = _resolve(
        project_root, Path(diagnostics.source_mask_alignment_summary_ref)
    )
    action_path = _resolve(
        project_root, Path(diagnostics.source_action_depth_summary_ref)
    )
    _require_hash(
        alignment_path, diagnostics.source_mask_alignment_summary_sha256
    )
    _require_hash(action_path, diagnostics.source_action_depth_summary_sha256)
    alignment = MaskAlignmentRunSummary.model_validate_json(
        alignment_path.read_text(encoding="utf-8")
    )
    action = ActionDepthRunSummary.model_validate_json(
        action_path.read_text(encoding="utf-8")
    )
    predictions = {record.job.job_id: record for record in action.predictions}
    aligned_by_key = {
        (record.action_depth_job_id, record.camera_id, record.target): record
        for record in alignment.aligned_masks
    }

    raw_cache: dict[str, tuple[Float32Array, Float32Array]] = {}
    mask_cache: dict[str, UInt8Array] = {}
    evaluations: dict[PerceptionTarget, list[dict[str, Any]]] = {
        target: [] for target in PerceptionTarget
    }
    selected_records = [
        record
        for record in diagnostics.records
        if record.strategy is SELECTED_STRATEGY[record.target]
    ]
    for record in selected_records:
        aligned = aligned_by_key.get(
            (record.action_depth_job_id, record.camera_id, record.target)
        )
        if aligned is None:
            raise ValueError("selected diagnostic lacks its aligned-mask identity")
        prediction = predictions.get(record.action_depth_job_id)
        if prediction is None:
            raise ValueError("selected diagnostic lacks its action-depth job")
        raw = raw_cache.get(prediction.raw_prediction_ref)
        if raw is None:
            raw_path = _resolve(project_root, Path(prediction.raw_prediction_ref))
            _require_hash(raw_path, prediction.raw_prediction_sha256)
            with np.load(raw_path, allow_pickle=False) as arrays:
                if bool(arrays["s02_corrections_applied"].item()):
                    raise ValueError("policy evidence unexpectedly contains S02 correction")
                raw = (
                    cast(Float32Array, np.asarray(arrays["depth_m"]).copy()),
                    cast(Float32Array, np.asarray(arrays["confidence"]).copy()),
                )
            raw_cache[prediction.raw_prediction_ref] = raw
        masks = mask_cache.get(aligned.aligned_mask_artifact_ref)
        if masks is None:
            mask_path = _resolve(project_root, Path(aligned.aligned_mask_artifact_ref))
            _require_hash(mask_path, aligned.aligned_mask_artifact_sha256)
            with np.load(mask_path, allow_pickle=False) as arrays:
                masks = cast(UInt8Array, np.asarray(arrays["masks"]).copy())
            mask_cache[aligned.aligned_mask_artifact_ref] = masks

        camera_index = CAMERA_INDEX[record.camera_id]
        depth = raw[0][camera_index]
        confidence = raw[1][camera_index]
        source_mask = masks[aligned.aligned_mask_index]
        candidates = build_mask_depth_candidates(
            source_mask,
            depth,
            target=record.target,
            config=diagnostics.configuration,
        )
        candidate = next(
            item for item in candidates if item.strategy is record.strategy
        )
        unfiltered_depth = depth[
            candidate.mask
            & np.isfinite(depth)
            & (depth > 0)
            & np.isfinite(confidence)
        ]
        unfiltered = summarize_distribution(unfiltered_depth)
        percentile_evaluations: list[dict[str, Any]] = []
        for percentile in CANDIDATE_PERCENTILES:
            selection = select_candidate_relative_confidence(
                candidate_mask=candidate.mask,
                depth_m=depth,
                confidence=confidence,
                percentile=percentile,
                minimum_retained_sample_count=1,
            )
            retained = summarize_distribution(depth[selection.mask])
            percentile_evaluations.append(
                {
                    "candidate_percentile": percentile,
                    "threshold": selection.confidence_threshold,
                    "retained_count": selection.retained_count,
                    "retained_fraction": (
                        selection.retained_count / selection.valid_candidate_count
                    ),
                    "depth_median_m": retained.median,
                    "depth_relative_mad": (
                        retained.median_absolute_deviation / retained.median
                    ),
                    "median_shift_fraction_from_unfiltered": (
                        abs(retained.median - unfiltered.median) / unfiltered.median
                    ),
                }
            )
        selected = next(
            item
            for item in percentile_evaluations
            if item["candidate_percentile"] == 20
        )
        if int(selected["retained_count"]) < MINIMUM_RETAINED_SAMPLES[record.target]:
            raise ValueError("selected candidate-relative p20 evidence is undersampled")
        full_frame_p20 = next(
            item
            for item in record.confidence_sweep
            if item.full_frame_percentile == 20
        )
        evaluations[record.target].append(
            {
                "source_frame_index": record.source_frame_index,
                "phase_id": record.phase_id,
                "camera_id": record.camera_id,
                "candidate_pixel_count": record.candidate_pixel_count,
                "unfiltered_depth_relative_mad": (
                    unfiltered.median_absolute_deviation / unfiltered.median
                ),
                "full_frame_p20_retained_fraction": full_frame_p20.retained_fraction,
                "candidate_relative_evaluations": percentile_evaluations,
            }
        )

    evidence_by_target = {
        target.value: _aggregate_evidence(target, items, diagnostics.records)
        for target, items in evaluations.items()
    }
    policy = MaskDepthSamplingPolicy(
        rules=(
            TargetVisibleSurfaceRule(
                target=PerceptionTarget.PERSON,
                candidate_strategy=MaskDepthStrategy.PERSON_LOWER_BODY,
                confidence_threshold_basis="candidate_valid_sample_percentile",
                confidence_percentile=20,
                minimum_retained_sample_count=MINIMUM_RETAINED_SAMPLES[
                    PerceptionTarget.PERSON
                ],
                depth_aggregate="median_ray_depth",
                insufficient_data_state="unavailable",
                coordinate_semantics=(
                    "Median ray depth of confidence-valid visible lower-body mask "
                    "pixels; not a body centre or ground-contact point."
                ),
            ),
            TargetVisibleSurfaceRule(
                target=PerceptionTarget.BACKPACK,
                candidate_strategy=MaskDepthStrategy.CONNECTED_DEPTH_CLUSTER,
                confidence_threshold_basis="candidate_valid_sample_percentile",
                confidence_percentile=20,
                minimum_retained_sample_count=MINIMUM_RETAINED_SAMPLES[
                    PerceptionTarget.BACKPACK
                ],
                depth_aggregate="median_ray_depth",
                insufficient_data_state="unavailable",
                coordinate_semantics=(
                    "Median ray depth of the confidence-valid largest connected "
                    "eroded in-mask depth cluster; a visible surface, not object centre."
                ),
            ),
        )
    )
    result = MaskDepthPolicySelectionSummary(
        schema_version=1,
        status="selected",
        stage="S04",
        created_at_utc=datetime.now(UTC),
        source_diagnostics_summary_ref=_relative(diagnostics_path, project_root),
        source_diagnostics_summary_sha256=_sha256(diagnostics_path),
        policy=policy,
        evidence_by_target=evidence_by_target,
        rejected_alternatives=(
            {
                "alternative": "same-action full-frame confidence percentile",
                "reason": (
                    "At p20 at least one person and one backpack candidate retains "
                    "zero samples; dynamic objects score below much of the background."
                ),
            },
            {
                "alternative": "whole mask depth",
                "reason": (
                    "Whole masks have the largest median relative depth MAD for both "
                    "targets and retain boundary/background leakage."
                ),
            },
            {
                "alternative": "person whole-mask or generic connected cluster anchor",
                "reason": (
                    "The visible lower-body candidate has lower median relative depth "
                    "MAD and provides the explicit surface semantics needed later."
                ),
            },
        ),
        limitations=(
            "The policy is calibrated to the selected one-person/one-backpack proof of concept.",
            "Candidate-relative confidence is a within-object rank, not calibrated probability.",
            (
                "Visible-surface ray depth does not yet define XYZ, body centre, "
                "object centre, or floor contact."
            ),
            (
                "No fallback to whole masks, stale depth, another timestamp, or "
                "S02 confidence is allowed."
            ),
        ),
        localization_performed=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result.status,
                "policy_id": result.policy.policy_id,
                "source_diagnostics_sha256": result.source_diagnostics_summary_sha256,
                "evidence_by_target": result.evidence_by_target,
                "output": _relative(output_path, project_root),
            },
            indent=2,
        )
    )
    return 0


def _aggregate_evidence(
    target: PerceptionTarget,
    items: list[dict[str, Any]],
    all_records: tuple[MaskDepthDiagnosticRecord, ...],
) -> dict[str, Any]:
    if not items:
        raise ValueError(f"no selected policy evidence for {target.value}")
    strategy = SELECTED_STRATEGY[target]
    selected_diagnostics = [
        record
        for record in all_records
        if record.target is target and record.strategy is strategy
    ]
    percentile_summary: dict[str, Any] = {}
    for percentile in CANDIDATE_PERCENTILES:
        rows = [
            next(
                row
                for row in item["candidate_relative_evaluations"]
                if row["candidate_percentile"] == percentile
            )
            for item in items
        ]
        percentile_summary[str(int(percentile))] = {
            "retained_count": _range_summary(
                [float(row["retained_count"]) for row in rows]
            ),
            "retained_fraction": _range_summary(
                [float(row["retained_fraction"]) for row in rows]
            ),
            "depth_relative_mad": _range_summary(
                [float(row["depth_relative_mad"]) for row in rows]
            ),
            "median_shift_fraction_from_unfiltered": _range_summary(
                [float(row["median_shift_fraction_from_unfiltered"]) for row in rows]
            ),
        }
    alternatives = {}
    for candidate_strategy in MaskDepthStrategy:
        matching = [
            record
            for record in all_records
            if record.target is target and record.strategy is candidate_strategy
        ]
        if not matching:
            continue
        values = [
            record.depth_m.median_absolute_deviation / record.depth_m.median
            for record in matching
        ]
        alternatives[candidate_strategy.value] = _range_summary(values)
    return {
        "selected_strategy": strategy.value,
        "observation_count": len(items),
        "candidate_pixel_count": _range_summary(
            [float(record.candidate_pixel_count) for record in selected_diagnostics]
        ),
        "full_frame_p20_retained_fraction": _range_summary(
            [float(item["full_frame_p20_retained_fraction"]) for item in items]
        ),
        "unfiltered_relative_depth_mad_by_strategy": alternatives,
        "candidate_relative_confidence_percentiles": percentile_summary,
        "selected_candidate_percentile": 20,
        "minimum_retained_sample_count": MINIMUM_RETAINED_SAMPLES[target],
    }


def _range_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("range summary needs values")
    return {
        "minimum": min(values),
        "median": median(values),
        "maximum": max(values),
    }


def _require_hash(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"artifact hash changed for {path}: {actual} != {expected}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _relative(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
