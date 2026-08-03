"""Verify S04 exact-frame raw per-camera visible-surface observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from spatial_reconstruction.contracts import CameraPose, PerceptionTarget
from spatial_reconstruction.geometry import (
    backproject_pixels,
    camera_points_to_world,
    project_camera_points,
    world_points_to_camera,
)
from spatial_reconstruction.localization import (
    ActionDepthRunSummary,
    AlignedMaskRecord,
    MaskAlignmentRunSummary,
    MaskDepthDiagnosticRunSummary,
    MaskDepthPolicySelectionSummary,
    VisibleSurfaceAvailability,
    VisibleSurfaceObservationRecord,
    VisibleSurfaceRunSummary,
    localize_visible_surface,
    summarize_distribution,
)

Float64Array = NDArray[np.float64]
CAMERA_INDEX = {"camera_a": 0, "camera_b": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/s04/visible_surfaces_20260802/summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/s04/visible_surfaces_20260802/verification.json"),
    )
    parser.add_argument("--visual-qa-passed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.visual_qa_passed:
        raise ValueError("visible-surface verification requires explicit visual QA")
    project_root = args.project_root.resolve()
    summary_path = _resolve(project_root, args.summary)
    output_path = _resolve(project_root, args.output)
    if output_path.exists():
        raise FileExistsError(f"verification output already exists: {output_path}")

    summary = VisibleSurfaceRunSummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    policy_path = _resolve(project_root, Path(summary.source_policy_selection_ref))
    alignment_path = _resolve(
        project_root, Path(summary.source_mask_alignment_summary_ref)
    )
    action_path = _resolve(project_root, Path(summary.source_action_depth_summary_ref))
    calibration_path = _resolve(project_root, Path(summary.pose_calibration_ref))
    scene_path = _resolve(project_root, Path(summary.scene_metadata_ref))
    _require_hash(policy_path, summary.source_policy_selection_sha256)
    _require_hash(alignment_path, summary.source_mask_alignment_summary_sha256)
    _require_hash(action_path, summary.source_action_depth_summary_sha256)
    _require_hash(calibration_path, summary.pose_calibration_sha256)
    _require_hash(scene_path, summary.scene_metadata_sha256)
    _require_hash(
        _resolve(project_root, Path(summary.contact_sheet_ref)),
        summary.contact_sheet_sha256,
    )
    _require_hash(
        _resolve(project_root, Path(summary.world_preview_ref)),
        summary.world_preview_sha256,
    )

    policy = MaskDepthPolicySelectionSummary.model_validate_json(
        policy_path.read_text(encoding="utf-8")
    )
    diagnostics_path = _resolve(
        project_root, Path(policy.source_diagnostics_summary_ref)
    )
    _require_hash(diagnostics_path, policy.source_diagnostics_summary_sha256)
    diagnostics = MaskDepthDiagnosticRunSummary.model_validate_json(
        diagnostics_path.read_text(encoding="utf-8")
    )
    alignment = MaskAlignmentRunSummary.model_validate_json(
        alignment_path.read_text(encoding="utf-8")
    )
    action = ActionDepthRunSummary.model_validate_json(
        action_path.read_text(encoding="utf-8")
    )
    calibration = _read_object(calibration_path)
    scene = _read_object(scene_path)
    if cast(dict[str, Any], scene["room_bounds"]) != summary.room_bounds_world_m:
        raise ValueError("visible-surface room bounds differ from scene metadata")

    poses = {
        camera_id: CameraPose.model_validate(camera["pose"])
        for camera_id, camera in cast(
            dict[str, dict[str, Any]], calibration["cameras"]
        ).items()
    }
    rules = {rule.target: rule for rule in policy.policy.rules}
    predictions = {record.job.job_id: record for record in action.predictions}
    aligned_by_key: dict[tuple[str, str, PerceptionTarget], AlignedMaskRecord] = {
        (record.action_depth_job_id, record.camera_id, record.target): record
        for record in alignment.aligned_masks
    }
    observation_keys = {
        (record.action_depth_job_id, record.camera_id, record.target)
        for record in summary.observations
    }
    if observation_keys != set(aligned_by_key):
        raise ValueError("visible-surface coverage differs from aligned observed masks")

    bounds_min = np.asarray(
        summary.room_bounds_world_m["minimum_world_xyz_m"], dtype=np.float64
    )
    bounds_max = np.asarray(
        summary.room_bounds_world_m["maximum_world_xyz_m"], dtype=np.float64
    )
    maximum_reprojection_error = 0.0
    maximum_transform_round_trip_error = 0.0
    maximum_returned_pose_error = 0.0
    minimum_sample_inside_fraction = 1.0
    raw_hashes: set[str] = set()
    mask_hashes: set[str] = set()

    for record in summary.observations:
        key = (record.action_depth_job_id, record.camera_id, record.target)
        aligned = aligned_by_key[key]
        prediction = predictions.get(record.action_depth_job_id)
        if prediction is None:
            raise ValueError("visible-surface observation refers to unknown depth job")
        _verify_record_provenance(record, aligned, prediction)
        if record.camera_pose != poses[record.camera_id]:
            raise ValueError("visible-surface pose differs from accepted calibration")

        raw_path = _resolve(project_root, Path(record.raw_prediction_ref))
        mask_path = _resolve(project_root, Path(record.aligned_mask_artifact_ref))
        sample_path = _resolve(project_root, Path(record.sample_cloud_ref))
        image_path = _resolve(project_root, Path(record.image_diagnostic_ref))
        _require_hash(raw_path, record.raw_prediction_sha256)
        _require_hash(mask_path, record.aligned_mask_artifact_sha256)
        _require_hash(sample_path, record.sample_cloud_sha256)
        _require_hash(image_path, record.image_diagnostic_sha256)
        raw_hashes.add(record.raw_prediction_sha256)
        mask_hashes.add(record.aligned_mask_artifact_sha256)

        camera_index = CAMERA_INDEX[record.camera_id]
        with np.load(raw_path, allow_pickle=False) as raw_arrays:
            if str(raw_arrays["job_id"].item()) != record.action_depth_job_id:
                raise ValueError("raw depth job identity differs from observation")
            if str(raw_arrays["bundle_id"].item()) != record.bundle_id:
                raise ValueError("raw depth bundle identity differs from observation")
            if bool(raw_arrays["s02_corrections_applied"].item()):
                raise ValueError("raw dynamic depth unexpectedly has S02 correction")
            depth = np.asarray(raw_arrays["depth_m"][camera_index], dtype=np.float64)
            confidence = np.asarray(
                raw_arrays["confidence"][camera_index], dtype=np.float64
            )
            returned_pose = np.asarray(
                raw_arrays["returned_T_camera_from_world"][camera_index],
                dtype=np.float64,
            )
        with np.load(mask_path, allow_pickle=False) as mask_arrays:
            if bool(mask_arrays["localization_performed"].item()):
                raise ValueError("source mask artifact already claims localization")
            mask = np.asarray(
                mask_arrays["masks"][record.aligned_mask_index], dtype=np.uint8
            )

        regenerated = localize_visible_surface(
            source_mask=mask,
            depth_m=depth,
            confidence=confidence,
            target=record.target,
            config=diagnostics.configuration,
            rule=rules[record.target],
            intrinsics=record.processed_intrinsics,
            pose=record.camera_pose,
            join=record.join,
        )
        if regenerated.availability is not VisibleSurfaceAvailability.OBSERVED:
            raise ValueError("retained observation is unavailable when regenerated")

        with np.load(sample_path, allow_pickle=False) as arrays:
            pixels = _matrix(arrays["pixels_uv"], columns=2, name="pixels_uv")
            depths = _vector(arrays["depth_m"], name="depth_m")
            scores = _vector(arrays["confidence"], name="confidence")
            points_camera = _matrix(
                arrays["points_camera_m"], columns=3, name="points_camera_m"
            )
            points_world = _matrix(
                arrays["points_world_m"], columns=3, name="points_world_m"
            )
            aggregate_camera = _vector(
                arrays["aggregate_camera_xyz_m"], name="aggregate_camera_xyz_m"
            )
            aggregate_world = _vector(
                arrays["aggregate_world_xyz_m"], name="aggregate_world_xyz_m"
            )
            _verify_sample_identity(arrays, record)

        if any(
            len(values) != record.retained_sample_count
            for values in (pixels, depths, scores, points_camera, points_world)
        ):
            raise ValueError("sample-cloud array counts differ from observation")
        if not np.allclose(pixels, regenerated.pixels_uv, rtol=0, atol=0):
            raise ValueError("stored pixels differ from regenerated D030 selection")
        if not np.allclose(depths, regenerated.depth_m, rtol=0, atol=0):
            raise ValueError("stored depths differ from regenerated D030 selection")
        if not np.allclose(scores, regenerated.confidence, rtol=0, atol=0):
            raise ValueError("stored confidence differs from regenerated D030 selection")
        if np.any(depths <= 0) or np.any(scores < record.candidate_confidence_threshold):
            raise ValueError("retained sample violates depth/confidence policy")
        if summarize_distribution(depths) != record.retained_depth_m:
            raise ValueError("retained depth distribution differs")
        if summarize_distribution(scores) != record.retained_confidence:
            raise ValueError("retained confidence distribution differs")

        expected_camera = np.asarray(
            backproject_pixels(pixels, depths, intrinsics=record.processed_intrinsics),
            dtype=np.float64,
        )
        expected_world = np.asarray(
            camera_points_to_world(expected_camera, pose=record.camera_pose),
            dtype=np.float64,
        )
        _require_close(points_camera, expected_camera, "camera sample back-projection")
        _require_close(points_world, expected_world, "world sample transform")
        expected_aggregate_camera = np.median(expected_camera, axis=0)
        expected_aggregate_world = np.asarray(
            camera_points_to_world(expected_aggregate_camera, pose=record.camera_pose),
            dtype=np.float64,
        )
        _require_close(
            aggregate_camera, expected_aggregate_camera, "camera aggregate median"
        )
        _require_close(aggregate_world, expected_aggregate_world, "world aggregate")
        _require_close(
            aggregate_camera,
            np.asarray(record.aggregate_camera_xyz_m),
            "record camera aggregate",
        )
        _require_close(
            aggregate_world,
            np.asarray(record.aggregate_world_xyz_m),
            "record world aggregate",
        )

        projected = np.asarray(
            project_camera_points(points_camera, intrinsics=record.processed_intrinsics),
            dtype=np.float64,
        )
        reprojection_error = float(np.max(np.linalg.norm(projected - pixels, axis=1)))
        recovered_camera = np.asarray(
            world_points_to_camera(points_world, pose=record.camera_pose),
            dtype=np.float64,
        )
        transform_error = float(
            np.max(np.linalg.norm(recovered_camera - points_camera, axis=1))
        )
        returned_pose_error = float(
            np.max(
                np.abs(
                    returned_pose
                    - np.asarray(record.camera_pose.T_camera_from_world, dtype=np.float64)
                )
            )
        )
        if not np.isclose(
            reprojection_error, record.sample_reprojection_max_error_px, atol=1e-12
        ):
            raise ValueError("stored reprojection error differs")
        if not np.isclose(
            transform_error,
            record.world_camera_round_trip_max_error_m,
            atol=1e-12,
        ):
            raise ValueError("stored transform round-trip error differs")
        if not np.isclose(
            returned_pose_error,
            record.returned_pose_maximum_absolute_error,
            atol=1e-12,
        ):
            raise ValueError("stored returned-pose error differs")

        inside = np.all(
            (points_world >= bounds_min) & (points_world <= bounds_max), axis=1
        )
        aggregate_inside = bool(
            np.all((aggregate_world >= bounds_min) & (aggregate_world <= bounds_max))
        )
        if aggregate_inside != record.aggregate_inside_room_bounds:
            raise ValueError("aggregate room-bound diagnostic differs")
        if not np.isclose(
            float(np.mean(inside)), record.sample_inside_room_fraction, atol=1e-12
        ):
            raise ValueError("sample room-bound fraction differs")
        maximum_reprojection_error = max(maximum_reprojection_error, reprojection_error)
        maximum_transform_round_trip_error = max(
            maximum_transform_round_trip_error, transform_error
        )
        maximum_returned_pose_error = max(
            maximum_returned_pose_error, returned_pose_error
        )
        minimum_sample_inside_fraction = min(
            minimum_sample_inside_fraction, float(np.mean(inside))
        )

    frame_order = [record.source_frame_index for record in summary.observations]
    target_counts = {
        target.value: sum(record.target is target for record in summary.observations)
        for target in PerceptionTarget
    }
    verification = {
        "schema_version": 1,
        "stage": "S04",
        "status": "passed",
        "purpose": "exact_frame_per_camera_raw_visible_surface_verification",
        "source_summary_ref": _relative(summary_path, project_root),
        "source_summary_sha256": _sha256(summary_path),
        "schema_round_trip_passed": (
            VisibleSurfaceRunSummary.model_validate_json(summary.model_dump_json())
            == summary
        ),
        "action_depth_job_count": len(action.predictions),
        "aligned_mask_count": len(alignment.aligned_masks),
        "observation_count": len(summary.observations),
        "target_observation_counts": target_counts,
        "raw_prediction_artifact_count": len(raw_hashes),
        "aligned_mask_artifact_count": len(mask_hashes),
        "complete_aligned_mask_coverage": observation_keys == set(aligned_by_key),
        "capture_order_passed": frame_order == sorted(frame_order),
        "all_observations_exact_frame_joined": True,
        "all_samples_regenerated_from_d030": True,
        "all_aggregates_componentwise_camera_medians": True,
        "maximum_sample_reprojection_error_px": maximum_reprojection_error,
        "maximum_world_camera_round_trip_error_m": (
            maximum_transform_round_trip_error
        ),
        "maximum_returned_pose_error": maximum_returned_pose_error,
        "aggregate_inside_room_count": sum(
            record.aggregate_inside_room_bounds for record in summary.observations
        ),
        "minimum_sample_inside_room_fraction": minimum_sample_inside_fraction,
        "visual_qa": {
            "status": "passed",
            "contact_sheet_ref": summary.contact_sheet_ref,
            "world_preview_ref": summary.world_preview_ref,
            "finding": (
                "Selected person lower-body and backpack samples remain attached to "
                "their intended visible regions; raw world-space motion progresses "
                "from pickup side toward drop-off side and remains inside room bounds."
            ),
        },
        "raw_visible_surface_xyz_generated": True,
        "track_anchor_derived": False,
        "cross_camera_fusion_performed": False,
        "temporal_filling_performed": False,
        "presentation_smoothing_performed": False,
        "s02_correction_applied": False,
    }
    required = (
        verification["schema_round_trip_passed"],
        verification["complete_aligned_mask_coverage"],
        verification["capture_order_passed"],
        verification["all_observations_exact_frame_joined"],
        verification["all_samples_regenerated_from_d030"],
        verification["all_aggregates_componentwise_camera_medians"],
        verification["observation_count"] == 20,
        target_counts == {"person": 12, "backpack": 8},
        maximum_reprojection_error <= 1e-9,
        maximum_transform_round_trip_error <= 1e-9,
        maximum_returned_pose_error <= 1e-5,
        not verification["track_anchor_derived"],
        not verification["cross_camera_fusion_performed"],
        not verification["temporal_filling_performed"],
        not verification["presentation_smoothing_performed"],
        not verification["s02_correction_applied"],
    )
    if not all(required):
        raise RuntimeError("S04 visible-surface verification did not pass")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))
    return 0


def _verify_record_provenance(
    record: VisibleSurfaceObservationRecord,
    aligned: Any,
    prediction: Any,
) -> None:
    expected = (
        record.bundle_id == aligned.bundle_id == prediction.job.bundle.bundle_id,
        record.frame_id == aligned.frame_id,
        record.source_frame_index == aligned.source_frame_index,
        record.perception_job_id == aligned.perception_job_id,
        record.camera_local_track_id == aligned.camera_local_track_id,
        record.source_mask_pixel_count == aligned.processed_mask_area_pixels,
        record.aligned_mask_artifact_ref == aligned.aligned_mask_artifact_ref,
        record.aligned_mask_artifact_sha256 == aligned.aligned_mask_artifact_sha256,
        record.aligned_mask_index == aligned.aligned_mask_index,
        record.raw_prediction_ref == prediction.raw_prediction_ref,
        record.raw_prediction_sha256 == prediction.raw_prediction_sha256,
        record.pose_version_id == prediction.job.bundle.frames[0].pose_version_id,
    )
    if not all(expected):
        raise ValueError("visible-surface provenance differs from upstream records")


def _verify_sample_identity(arrays: Any, record: VisibleSurfaceObservationRecord) -> None:
    expected_text = {
        "observation_id": record.observation_id,
        "action_depth_job_id": record.action_depth_job_id,
        "bundle_id": record.bundle_id,
        "frame_id": record.frame_id,
        "camera_id": record.camera_id,
        "target": record.target.value,
        "policy_id": record.policy_id,
        "candidate_strategy": record.candidate_strategy.value,
    }
    for name, expected in expected_text.items():
        if str(arrays[name].item()) != expected:
            raise ValueError(f"sample-cloud {name} identity differs")
    false_flags = (
        "anchor_derived",
        "camera_fusion_performed",
        "presentation_smoothing_performed",
        "s02_correction_applied",
    )
    if any(bool(arrays[name].item()) for name in false_flags):
        raise ValueError("sample cloud claims a forbidden downstream operation")
    if float(arrays["candidate_confidence_percentile"].item()) != 20:
        raise ValueError("sample cloud differs from D030 p20")
    if not np.isclose(
        float(arrays["candidate_confidence_threshold"].item()),
        record.candidate_confidence_threshold,
        atol=0,
    ):
        raise ValueError("sample-cloud confidence threshold differs")
    _require_close(
        arrays["processed_intrinsics"],
        np.asarray(
            [
                [record.processed_intrinsics.fx, 0, record.processed_intrinsics.cx],
                [0, record.processed_intrinsics.fy, record.processed_intrinsics.cy],
                [0, 0, 1],
            ]
        ),
        "sample-cloud processed intrinsics",
    )
    _require_close(
        arrays["T_world_from_camera"],
        np.asarray(record.camera_pose.T_world_from_camera),
        "sample-cloud T_world_from_camera",
    )
    _require_close(
        arrays["T_camera_from_world"],
        np.asarray(record.camera_pose.T_camera_from_world),
        "sample-cloud T_camera_from_world",
    )


def _matrix(values: Any, *, columns: int, name: str) -> Float64Array:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != columns or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape (N, {columns})")
    return array


def _vector(values: Any, *, name: str) -> Float64Array:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector")
    return array


def _require_close(actual: Any, expected: Any, name: str) -> None:
    if not np.allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(f"{name} differs")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return cast(dict[str, Any], value)


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
