"""Verify S04 source-mask alignment artifacts and recorded visual QA."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from spatial_reconstruction.localization import (
    ActionDepthRunSummary,
    AlignedMaskRecord,
    MaskAlignmentRunSummary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/s04/mask_alignment_20260801/summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/s04/mask_alignment_20260801/verification.json"),
    )
    parser.add_argument(
        "--visual-qa-passed",
        action="store_true",
        help="Record that a human inspected the contact sheet and accepted alignment.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.visual_qa_passed:
        raise ValueError("mask alignment verification requires explicit visual QA")
    project_root = args.project_root.resolve()
    summary_path = _resolve(project_root, args.summary)
    output_path = _resolve(project_root, args.output)
    if output_path.exists():
        raise FileExistsError(f"verification output already exists: {output_path}")

    summary = MaskAlignmentRunSummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    action_summary_path = _resolve(
        project_root, Path(summary.source_action_depth_summary_ref)
    )
    pose_path = _resolve(project_root, Path(summary.pose_calibration_ref))
    _require_hash(action_summary_path, summary.source_action_depth_summary_sha256)
    _require_hash(pose_path, summary.pose_calibration_sha256)
    action_summary = ActionDepthRunSummary.model_validate_json(
        action_summary_path.read_text(encoding="utf-8")
    )
    known_jobs = {record.job.job_id: record for record in action_summary.predictions}

    contact_sheet = _resolve(project_root, Path(summary.contact_sheet_ref))
    _require_hash(contact_sheet, summary.contact_sheet_sha256)
    overlay_job_ids: set[str] = set()
    for overlay in summary.job_overlays:
        job_id = str(overlay["action_depth_job_id"])
        if job_id not in known_jobs:
            raise ValueError("mask-alignment overlay refers to an unknown depth job")
        overlay_path = _resolve(project_root, Path(str(overlay["overlay_ref"])))
        _require_hash(overlay_path, str(overlay["overlay_sha256"]))
        overlay_job_ids.add(job_id)

    records_by_artifact: dict[str, list[AlignedMaskRecord]] = defaultdict(list)
    for record in summary.aligned_masks:
        if record.action_depth_job_id not in known_jobs:
            raise ValueError("aligned mask refers to an unknown action-depth job")
        records_by_artifact[record.aligned_mask_artifact_ref].append(record)

    verified_artifacts: list[dict[str, Any]] = []
    for artifact_ref, records in sorted(records_by_artifact.items()):
        artifact_path = _resolve(project_root, Path(artifact_ref))
        expected_hashes = {record.aligned_mask_artifact_sha256 for record in records}
        if len(expected_hashes) != 1:
            raise ValueError("one aligned-mask artifact has conflicting hashes")
        expected_hash = next(iter(expected_hashes))
        _require_hash(artifact_path, expected_hash)
        verified_artifact = _verify_mask_artifact(artifact_path, records)
        verified_artifact["artifact_ref"] = artifact_ref
        verified_artifacts.append(verified_artifact)

    rgb_checks = tuple(summary.rgb_reproduction_checks)
    if len(rgb_checks) != len(action_summary.predictions) * 2:
        raise ValueError("RGB reproduction checks do not cover both cameras")
    if not all(bool(check["passed"]) for check in rgb_checks):
        raise ValueError("one or more RGB reproduction checks did not pass")
    max_rgb_error = max(
        int(check["maximum_absolute_channel_difference"])
        for check in rgb_checks
    )
    max_intrinsics_error = max(
        float(check["processed_intrinsics_maximum_absolute_error"])
        for check in rgb_checks
    )
    if max_rgb_error > 1 or max_intrinsics_error > 1e-4:
        raise ValueError("DA3 RGB or intrinsic reproduction exceeded tolerance")

    source_order = [
        prediction.job.bundle.frames[0].source_frame_index
        for prediction in action_summary.predictions
    ]
    verified_job_ids = {
        record.action_depth_job_id for record in summary.aligned_masks
    }
    verification = {
        "schema_version": 1,
        "stage": "S04",
        "status": "passed",
        "purpose": "exact_source_mask_to_da3_grid_alignment_verification",
        "source_summary_ref": _relative(summary_path, project_root),
        "source_summary_sha256": _sha256(summary_path),
        "source_action_depth_summary_ref": summary.source_action_depth_summary_ref,
        "source_action_depth_summary_sha256": (
            summary.source_action_depth_summary_sha256
        ),
        "schema_round_trip_passed": (
            MaskAlignmentRunSummary.model_validate_json(summary.model_dump_json())
            == summary
        ),
        "capture_order_passed": source_order == sorted(source_order),
        "action_depth_job_count": len(action_summary.predictions),
        "aligned_mask_count": len(summary.aligned_masks),
        "aligned_mask_artifact_count": len(verified_artifacts),
        "aligned_mask_artifacts": verified_artifacts,
        "all_action_depth_jobs_covered": (
            verified_job_ids == set(known_jobs) == overlay_job_ids
        ),
        "rgb_reproduction_check_count": len(rgb_checks),
        "maximum_absolute_rgb_channel_error": max_rgb_error,
        "maximum_processed_intrinsics_error": max_intrinsics_error,
        "binary_masks_only": True,
        "nearest_neighbor_mask_mapping_only": True,
        "visual_qa": {
            "status": "passed",
            "contact_sheet_ref": summary.contact_sheet_ref,
            "finding": (
                "Observed person and backpack masks remain attached to their "
                "intended image regions; unavailable observations remain missing."
            ),
        },
        "localization_performed": False,
        "s02_depth_corrections_applied": False,
    }
    required_passes = (
        verification["schema_round_trip_passed"],
        verification["capture_order_passed"],
        verification["all_action_depth_jobs_covered"],
        verification["binary_masks_only"],
        verification["nearest_neighbor_mask_mapping_only"],
    )
    if not all(required_passes):
        raise RuntimeError("S04 mask-alignment verification did not pass")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))
    return 0


def _verify_mask_artifact(
    path: Path, records: list[AlignedMaskRecord]
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as arrays:
        masks = np.asarray(arrays["masks"])
        metadata = json.loads(str(arrays["metadata_json"].item()))
        if masks.dtype != np.uint8 or masks.ndim != 3:
            raise ValueError("aligned masks must be uint8 N-by-H-by-W")
        if tuple(masks.shape[1:]) != (280, 504):
            raise ValueError("aligned masks differ from retained DA3 depth grid")
        if set(np.unique(masks).tolist()) - {0, 1}:
            raise ValueError("aligned masks must remain binary")
        if len(metadata) != len(masks):
            raise ValueError("aligned-mask metadata length differs from mask count")
        if arrays["processed_shape"].tolist() != [280, 504]:
            raise ValueError("aligned-mask processed shape metadata changed")
        if str(arrays["mask_interpolation"].item()) != "nearest":
            raise ValueError("aligned-mask artifact records non-nearest interpolation")
        if bool(arrays["localization_performed"].item()):
            raise ValueError("mask-alignment artifact unexpectedly records localization")
        job_id = str(arrays["action_depth_job_id"].item())
        bundle_id = str(arrays["bundle_id"].item())
        for record in records:
            if record.action_depth_job_id != job_id or record.bundle_id != bundle_id:
                raise ValueError("aligned-mask identity metadata differs from summary")
            if record.aligned_mask_index >= len(masks):
                raise ValueError("aligned-mask index is outside its artifact")
            mask = masks[record.aligned_mask_index]
            if int(np.count_nonzero(mask)) != record.processed_mask_area_pixels:
                raise ValueError("aligned-mask area differs from summary")
            item = metadata[record.aligned_mask_index]
            if (
                str(item["camera_id"]) != record.camera_id
                or str(item["target"]) != record.target.value
                or str(item["frame_id"]) != record.frame_id
                or int(item["detection_index"]) != record.detection_index
            ):
                raise ValueError("aligned-mask item metadata differs from summary")
    return {
        "artifact_ref": str(path),
        "artifact_sha256": _sha256(path),
        "job_id": job_id,
        "bundle_id": bundle_id,
        "mask_count": int(len(masks)),
        "nonzero_pixels_by_mask": [
            int(np.count_nonzero(mask)) for mask in masks
        ],
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
