"""Verify S04 mask-depth diagnostics and selected non-XYZ sampling policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from spatial_reconstruction.contracts import PerceptionTarget
from spatial_reconstruction.localization import (
    ActionDepthRunSummary,
    MaskAlignmentRunSummary,
    MaskDepthDiagnosticRunSummary,
    MaskDepthPolicySelectionSummary,
    MaskDepthStrategy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/s04/mask_depth_diagnostics_20260801/summary.json"),
    )
    parser.add_argument(
        "--policy-selection",
        type=Path,
        default=Path(
            "artifacts/s04/mask_depth_diagnostics_20260801/policy_selection.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/s04/mask_depth_diagnostics_20260801/verification.json"),
    )
    parser.add_argument("--visual-qa-passed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.visual_qa_passed:
        raise ValueError("mask-depth verification requires explicit visual QA")
    project_root = args.project_root.resolve()
    summary_path = _resolve(project_root, args.summary)
    policy_path = _resolve(project_root, args.policy_selection)
    output_path = _resolve(project_root, args.output)
    if output_path.exists():
        raise FileExistsError(f"verification output already exists: {output_path}")

    summary = MaskDepthDiagnosticRunSummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    policy = MaskDepthPolicySelectionSummary.model_validate_json(
        policy_path.read_text(encoding="utf-8")
    )
    alignment_path = _resolve(
        project_root, Path(summary.source_mask_alignment_summary_ref)
    )
    action_path = _resolve(project_root, Path(summary.source_action_depth_summary_ref))
    _require_hash(alignment_path, summary.source_mask_alignment_summary_sha256)
    _require_hash(action_path, summary.source_action_depth_summary_sha256)
    _require_hash(summary_path, policy.source_diagnostics_summary_sha256)
    if policy.source_diagnostics_summary_ref != _relative(summary_path, project_root):
        raise ValueError("policy selection refers to a different diagnostic summary")
    alignment = MaskAlignmentRunSummary.model_validate_json(
        alignment_path.read_text(encoding="utf-8")
    )
    action = ActionDepthRunSummary.model_validate_json(
        action_path.read_text(encoding="utf-8")
    )
    known_jobs = {prediction.job.job_id: prediction for prediction in action.predictions}

    _require_hash(
        _resolve(project_root, Path(summary.comparison_csv_ref)),
        summary.comparison_csv_sha256,
    )
    _require_hash(
        _resolve(project_root, Path(summary.strategy_comparison_ref)),
        summary.strategy_comparison_sha256,
    )
    _require_hash(
        _resolve(project_root, Path(summary.contact_sheet_ref)),
        summary.contact_sheet_sha256,
    )
    diagnostic_keys: set[tuple[str, str, str]] = set()
    for artifact in summary.per_mask_diagnostics:
        _require_hash(
            _resolve(project_root, Path(str(artifact["diagnostic_ref"]))),
            str(artifact["diagnostic_sha256"]),
        )
        diagnostic_keys.add(
            (
                str(artifact["action_depth_job_id"]),
                str(artifact["camera_id"]),
                str(artifact["target"]),
            )
        )

    expected_keys: set[tuple[str, str, str, str]] = set()
    for aligned in alignment.aligned_masks:
        strategies = {
            MaskDepthStrategy.WHOLE_MASK,
            MaskDepthStrategy.ERODED_INTERIOR,
            MaskDepthStrategy.CONNECTED_DEPTH_CLUSTER,
        }
        if aligned.target is PerceptionTarget.PERSON:
            strategies.add(MaskDepthStrategy.PERSON_LOWER_BODY)
        for strategy in strategies:
            expected_keys.add(
                (
                    aligned.action_depth_job_id,
                    aligned.camera_id,
                    aligned.target.value,
                    strategy.value,
                )
            )
    actual_keys: set[tuple[str, str, str, str]] = set()
    for record in summary.records:
        if record.action_depth_job_id not in known_jobs:
            raise ValueError("diagnostic record refers to unknown action-depth job")
        prediction = known_jobs[record.action_depth_job_id]
        if (
            record.raw_prediction_ref != prediction.raw_prediction_ref
            or record.raw_prediction_sha256 != prediction.raw_prediction_sha256
        ):
            raise ValueError("diagnostic raw prediction provenance differs")
        if record.finite_positive_depth_count != record.candidate_pixel_count:
            raise ValueError("real diagnostic candidate contains invalid depth")
        if record.finite_confidence_count != record.candidate_pixel_count:
            raise ValueError("real diagnostic candidate contains invalid confidence")
        if tuple(
            item.full_frame_percentile for item in record.confidence_sweep
        ) != summary.configuration.confidence_percentiles:
            raise ValueError("diagnostic confidence sweep differs from configuration")
        actual_keys.add(
            (
                record.action_depth_job_id,
                record.camera_id,
                record.target.value,
                record.strategy.value,
            )
        )
    if actual_keys != expected_keys:
        raise ValueError("diagnostic strategy coverage differs from aligned masks")
    expected_diagnostic_keys = {
        (record.action_depth_job_id, record.camera_id, record.target.value)
        for record in alignment.aligned_masks
    }
    if diagnostic_keys != expected_diagnostic_keys:
        raise ValueError("per-mask diagnostic images do not cover aligned masks")

    selected_policy_evidence: dict[str, dict[str, Any]] = {}
    for rule in policy.policy.rules:
        evidence = policy.evidence_by_target[rule.target.value]
        p20 = evidence["candidate_relative_confidence_percentiles"]["20"]
        retained_count_minimum = float(p20["retained_count"]["minimum"])
        retained_fraction_minimum = float(p20["retained_fraction"]["minimum"])
        retained_fraction_maximum = float(p20["retained_fraction"]["maximum"])
        if retained_count_minimum < rule.minimum_retained_sample_count:
            raise ValueError("selected policy does not meet its minimum sample count")
        if not 0.79 <= retained_fraction_minimum <= retained_fraction_maximum <= 0.81:
            raise ValueError("candidate p20 retention is outside expected rank semantics")
        full_frame_minimum = float(
            evidence["full_frame_p20_retained_fraction"]["minimum"]
        )
        if full_frame_minimum != 0:
            raise ValueError("full-frame p20 rejection evidence unexpectedly changed")
        selected_policy_evidence[rule.target.value] = {
            "strategy": rule.candidate_strategy.value,
            "observation_count": int(evidence["observation_count"]),
            "minimum_candidate_p20_retained_count": retained_count_minimum,
            "candidate_p20_retained_fraction_range": [
                retained_fraction_minimum,
                retained_fraction_maximum,
            ],
            "minimum_full_frame_p20_retained_fraction": full_frame_minimum,
        }

    verification = {
        "schema_version": 1,
        "stage": "S04",
        "status": "passed",
        "purpose": "raw_mask_depth_diagnostics_and_policy_selection_verification",
        "source_summary_ref": _relative(summary_path, project_root),
        "source_summary_sha256": _sha256(summary_path),
        "policy_selection_ref": _relative(policy_path, project_root),
        "policy_selection_sha256": _sha256(policy_path),
        "schema_round_trip_passed": (
            MaskDepthDiagnosticRunSummary.model_validate_json(summary.model_dump_json())
            == summary
            and MaskDepthPolicySelectionSummary.model_validate_json(
                policy.model_dump_json()
            )
            == policy
        ),
        "action_depth_job_count": len(action.predictions),
        "aligned_mask_count": len(alignment.aligned_masks),
        "diagnostic_record_count": len(summary.records),
        "per_mask_diagnostic_count": len(summary.per_mask_diagnostics),
        "strategy_coverage_passed": actual_keys == expected_keys,
        "policy_id": policy.policy.policy_id,
        "selected_policy_evidence": selected_policy_evidence,
        "full_frame_confidence_basis_rejected": True,
        "candidate_relative_confidence_percentile": 20,
        "visual_qa": {
            "status": "passed",
            "finding": (
                "Mask, depth, confidence, candidate outlines, distributions, and "
                "retention curves were inspected across representative phases."
            ),
            "contact_sheet_ref": summary.contact_sheet_ref,
            "strategy_comparison_ref": summary.strategy_comparison_ref,
        },
        "localization_performed": False,
        "xyz_generated": False,
        "s02_confidence_policy_applied": False,
    }
    if not all(
        (
            verification["schema_round_trip_passed"],
            verification["strategy_coverage_passed"],
            verification["full_frame_confidence_basis_rejected"],
            not verification["localization_performed"],
            not verification["xyz_generated"],
            not verification["s02_confidence_policy_applied"],
        )
    ):
        raise RuntimeError("mask-depth diagnostics verification did not pass")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))
    return 0


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
