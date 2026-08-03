"""Back-project D030-selected exact-frame masks into raw per-camera XYZ."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from spatial_reconstruction.contracts import (
    CameraIntrinsics,
    CameraPose,
    PerceptionTarget,
)
from spatial_reconstruction.geometry import project_camera_points
from spatial_reconstruction.localization import (
    ActionDepthRunSummary,
    ExactFrameDepthJoin,
    MaskAlignmentRunSummary,
    MaskDepthDiagnosticRunSummary,
    MaskDepthPolicySelectionSummary,
    VisibleSurfaceAvailability,
    VisibleSurfaceObservationRecord,
    VisibleSurfaceRunSummary,
    localize_visible_surface,
    summarize_distribution,
)

UInt8Array = NDArray[np.uint8]
Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
CAMERA_INDEX = {"camera_a": 0, "camera_b": 1}
TARGET_COLOR = {
    PerceptionTarget.PERSON: "#00cfe8",
    PerceptionTarget.BACKPACK: "#ff2f9e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--policy-selection",
        type=Path,
        default=Path(
            "artifacts/s04/mask_depth_diagnostics_20260801/policy_selection.json"
        ),
    )
    parser.add_argument(
        "--pose-calibration",
        type=Path,
        default=Path(
            "artifacts/s01/calibration/action_take_01_pose/camera_calibration.json"
        ),
    )
    parser.add_argument(
        "--scene-metadata",
        type=Path,
        default=Path("artifacts/s01/scene_metadata.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    policy_path = _resolve(project_root, args.policy_selection)
    calibration_path = _resolve(project_root, args.pose_calibration)
    scene_path = _resolve(project_root, args.scene_metadata)
    output_dir = _resolve(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    samples_dir = output_dir / "sample_clouds"
    diagnostics_dir = output_dir / "image_diagnostics"
    samples_dir.mkdir()
    diagnostics_dir.mkdir()

    policy_selection = MaskDepthPolicySelectionSummary.model_validate_json(
        policy_path.read_text(encoding="utf-8")
    )
    diagnostics_path = _resolve(
        project_root, Path(policy_selection.source_diagnostics_summary_ref)
    )
    _require_hash(
        diagnostics_path, policy_selection.source_diagnostics_summary_sha256
    )
    diagnostics = MaskDepthDiagnosticRunSummary.model_validate_json(
        diagnostics_path.read_text(encoding="utf-8")
    )
    alignment_path = _resolve(
        project_root, Path(diagnostics.source_mask_alignment_summary_ref)
    )
    action_path = _resolve(
        project_root, Path(diagnostics.source_action_depth_summary_ref)
    )
    _require_hash(alignment_path, diagnostics.source_mask_alignment_summary_sha256)
    _require_hash(action_path, diagnostics.source_action_depth_summary_sha256)
    alignment = MaskAlignmentRunSummary.model_validate_json(
        alignment_path.read_text(encoding="utf-8")
    )
    action = ActionDepthRunSummary.model_validate_json(
        action_path.read_text(encoding="utf-8")
    )
    calibration = _read_object(calibration_path)
    scene = _read_object(scene_path)
    _verify_upstream_calibration_hashes(
        action=action,
        calibration_path=calibration_path,
        scene_path=scene_path,
    )
    if calibration.get("pose_version_id") != action.pose_version_id:
        raise ValueError("action-depth and calibration pose versions differ")
    poses = {
        camera_id: CameraPose.model_validate(camera["pose"])
        for camera_id, camera in cast(dict[str, dict[str, Any]], calibration["cameras"]).items()
    }
    rules = {rule.target: rule for rule in policy_selection.policy.rules}
    predictions = {record.job.job_id: record for record in action.predictions}
    room_bounds = cast(dict[str, Any], scene["room_bounds"])
    bounds_min = np.asarray(room_bounds["minimum_world_xyz_m"], dtype=np.float64)
    bounds_max = np.asarray(room_bounds["maximum_world_xyz_m"], dtype=np.float64)

    raw_cache: dict[
        str,
        tuple[
            Float32Array,
            Float32Array,
            UInt8Array,
            Float32Array,
            Float32Array,
        ],
    ] = {}
    mask_cache: dict[str, UInt8Array] = {}
    records: list[VisibleSurfaceObservationRecord] = []
    preview_items: list[
        tuple[VisibleSurfaceObservationRecord, UInt8Array, Float64Array]
    ] = []

    for index, aligned in enumerate(alignment.aligned_masks):
        prediction = predictions.get(aligned.action_depth_job_id)
        if prediction is None:
            raise ValueError("aligned mask refers to an unknown action-depth job")
        frame = next(
            item
            for item in prediction.job.bundle.frames
            if item.camera_id == aligned.camera_id
        )
        evidence = next(
            item
            for item in prediction.job.mask_evidence
            if item.frame_identity.frame_id == aligned.frame_id
            and item.target is aligned.target
        )
        join = ExactFrameDepthJoin(
            action_depth_job_id_from_mask=aligned.action_depth_job_id,
            action_depth_job_id_from_depth=prediction.job.job_id,
            bundle_id_from_mask=aligned.bundle_id,
            bundle_id_from_depth=prediction.job.bundle.bundle_id,
            frame_id_from_mask=evidence.frame_identity.frame_id,
            frame_id_from_depth=frame.frame_id,
            camera_id_from_mask=evidence.frame_identity.camera_id,
            camera_id_from_depth=frame.camera_id,
            capture_timestamp_seconds_from_mask=(
                evidence.frame_identity.capture_timestamp_seconds
            ),
            capture_timestamp_seconds_from_depth=frame.capture_timestamp_seconds,
            timestamp_difference_seconds=abs(
                evidence.frame_identity.capture_timestamp_seconds
                - frame.capture_timestamp_seconds
            ),
        )
        raw = raw_cache.get(prediction.raw_prediction_ref)
        if raw is None:
            raw_path = _resolve(project_root, Path(prediction.raw_prediction_ref))
            _require_hash(raw_path, prediction.raw_prediction_sha256)
            with np.load(raw_path, allow_pickle=False) as arrays:
                _verify_raw_identity(
                    arrays,
                    prediction.job.job_id,
                    prediction.job.bundle.bundle_id,
                )
                if bool(arrays["s02_corrections_applied"].item()):
                    raise ValueError("visible-surface input unexpectedly has S02 correction")
                raw = (
                    cast(Float32Array, np.asarray(arrays["depth_m"]).copy()),
                    cast(Float32Array, np.asarray(arrays["confidence"]).copy()),
                    cast(UInt8Array, np.asarray(arrays["processed_images_rgb"]).copy()),
                    cast(
                        Float32Array,
                        np.asarray(arrays["returned_T_camera_from_world"]).copy(),
                    ),
                    cast(
                        Float32Array,
                        np.asarray(arrays["returned_intrinsics"]).copy(),
                    ),
                )
            raw_cache[prediction.raw_prediction_ref] = raw
        masks = mask_cache.get(aligned.aligned_mask_artifact_ref)
        if masks is None:
            mask_path = _resolve(project_root, Path(aligned.aligned_mask_artifact_ref))
            _require_hash(mask_path, aligned.aligned_mask_artifact_sha256)
            with np.load(mask_path, allow_pickle=False) as arrays:
                if bool(arrays["localization_performed"].item()):
                    raise ValueError("aligned mask already claims localization")
                masks = cast(UInt8Array, np.asarray(arrays["masks"]).copy())
            mask_cache[aligned.aligned_mask_artifact_ref] = masks

        camera_index = CAMERA_INDEX[aligned.camera_id]
        depth, confidence, images, returned_poses, returned_intrinsics = raw
        intrinsics_matrix = returned_intrinsics[camera_index]
        processed_intrinsics = CameraIntrinsics(
            camera_id=aligned.camera_id,
            fx=float(intrinsics_matrix[0, 0]),
            fy=float(intrinsics_matrix[1, 1]),
            cx=float(intrinsics_matrix[0, 2]),
            cy=float(intrinsics_matrix[1, 2]),
            image_width=int(depth.shape[2]),
            image_height=int(depth.shape[1]),
        )
        pose = poses[aligned.camera_id]
        returned_pose_error = float(
            np.max(
                np.abs(
                    returned_poses[camera_index].astype(np.float64)
                    - np.asarray(pose.T_camera_from_world, dtype=np.float64)
                )
            )
        )
        if returned_pose_error > 1e-5:
            raise ValueError("DA3 returned pose differs from accepted action pose")
        surface = localize_visible_surface(
            source_mask=masks[aligned.aligned_mask_index],
            depth_m=depth[camera_index],
            confidence=confidence[camera_index],
            target=aligned.target,
            config=diagnostics.configuration,
            rule=rules[aligned.target],
            intrinsics=processed_intrinsics,
            pose=pose,
            join=join,
        )
        if surface.availability is not VisibleSurfaceAvailability.OBSERVED:
            raise RuntimeError(
                f"real aligned mask is unavailable under D030: {aligned.frame_id}"
            )
        assert surface.aggregate_camera_xyz_m is not None
        assert surface.aggregate_world_xyz_m is not None
        assert surface.confidence_threshold is not None
        assert surface.median_pixel_uv is not None
        assert surface.sample_reprojection_max_error_px is not None
        assert surface.world_camera_round_trip_max_error_m is not None
        inside = np.all(
            (surface.points_world_m >= bounds_min)
            & (surface.points_world_m <= bounds_max),
            axis=1,
        )
        aggregate_world = np.asarray(surface.aggregate_world_xyz_m, dtype=np.float64)
        aggregate_inside = bool(
            np.all((aggregate_world >= bounds_min) & (aggregate_world <= bounds_max))
        )

        sample_path = samples_dir / (
            f"{index:02d}_{aligned.source_frame_index:04d}_{aligned.camera_id}_"
            f"{aligned.target.value}.npz"
        )
        np.savez_compressed(
            sample_path,
            pixels_uv=surface.pixels_uv,
            depth_m=surface.depth_m,
            confidence=surface.confidence,
            points_camera_m=surface.points_camera_m,
            points_world_m=surface.points_world_m,
            aggregate_camera_xyz_m=np.asarray(surface.aggregate_camera_xyz_m),
            aggregate_world_xyz_m=aggregate_world,
            processed_intrinsics=intrinsics_matrix,
            T_world_from_camera=np.asarray(pose.T_world_from_camera),
            T_camera_from_world=np.asarray(pose.T_camera_from_world),
            observation_id=np.asarray(
                VisibleSurfaceObservationRecord.create_observation_id(
                    action_depth_job_id=aligned.action_depth_job_id,
                    bundle_id=aligned.bundle_id,
                    frame_id=aligned.frame_id,
                    camera_id=aligned.camera_id,
                    target=aligned.target,
                    perception_job_id=aligned.perception_job_id,
                    policy_id=policy_selection.policy.policy_id,
                )
            ),
            action_depth_job_id=np.asarray(aligned.action_depth_job_id),
            bundle_id=np.asarray(aligned.bundle_id),
            frame_id=np.asarray(aligned.frame_id),
            camera_id=np.asarray(aligned.camera_id),
            target=np.asarray(aligned.target.value),
            policy_id=np.asarray(policy_selection.policy.policy_id),
            candidate_strategy=np.asarray(surface.strategy.value),
            candidate_confidence_percentile=np.asarray(20.0),
            candidate_confidence_threshold=np.asarray(surface.confidence_threshold),
            anchor_derived=np.asarray(False),
            camera_fusion_performed=np.asarray(False),
            presentation_smoothing_performed=np.asarray(False),
            s02_correction_applied=np.asarray(False),
        )
        sample_hash = _sha256(sample_path)
        image_path = diagnostics_dir / (
            f"{index:02d}_{aligned.source_frame_index:04d}_{aligned.camera_id}_"
            f"{aligned.target.value}.png"
        )
        _save_image_diagnostic(
            image=images[camera_index],
            pixels_uv=surface.pixels_uv,
            aggregate_camera_xyz_m=np.asarray(surface.aggregate_camera_xyz_m),
            aggregate_world_xyz_m=aggregate_world,
            intrinsics=processed_intrinsics,
            phase_id=prediction.job.phase_id,
            camera_id=aligned.camera_id,
            target=aligned.target,
            path=image_path,
        )
        with np.load(sample_path, allow_pickle=False) as sample_arrays:
            observation_id = str(sample_arrays["observation_id"].item())
        record = VisibleSurfaceObservationRecord(
            observation_id=observation_id,
            action_depth_job_id=aligned.action_depth_job_id,
            bundle_id=aligned.bundle_id,
            frame_id=aligned.frame_id,
            capture_timestamp_seconds=frame.capture_timestamp_seconds,
            source_frame_index=aligned.source_frame_index,
            phase_id=prediction.job.phase_id,
            camera_id=aligned.camera_id,
            target=aligned.target,
            perception_job_id=aligned.perception_job_id,
            camera_local_track_id=aligned.camera_local_track_id,
            pose_version_id=frame.pose_version_id,
            join=join,
            policy_id=policy_selection.policy.policy_id,
            candidate_strategy=surface.strategy,
            source_mask_pixel_count=aligned.processed_mask_area_pixels,
            candidate_pixel_count=surface.candidate_pixel_count,
            valid_candidate_count=surface.valid_candidate_count,
            retained_sample_count=surface.retained_sample_count,
            candidate_confidence_percentile=20.0,
            candidate_confidence_threshold=surface.confidence_threshold,
            retained_depth_m=summarize_distribution(surface.depth_m),
            retained_confidence=summarize_distribution(surface.confidence),
            median_pixel_uv=surface.median_pixel_uv,
            aggregate_method="componentwise_median_camera_xyz",
            aggregate_camera_xyz_m=surface.aggregate_camera_xyz_m,
            aggregate_world_xyz_m=surface.aggregate_world_xyz_m,
            processed_intrinsics=processed_intrinsics,
            camera_pose=pose,
            raw_prediction_ref=prediction.raw_prediction_ref,
            raw_prediction_sha256=prediction.raw_prediction_sha256,
            aligned_mask_artifact_ref=aligned.aligned_mask_artifact_ref,
            aligned_mask_artifact_sha256=aligned.aligned_mask_artifact_sha256,
            aligned_mask_index=aligned.aligned_mask_index,
            sample_cloud_ref=_relative(sample_path, project_root),
            sample_cloud_sha256=sample_hash,
            image_diagnostic_ref=_relative(image_path, project_root),
            image_diagnostic_sha256=_sha256(image_path),
            sample_reprojection_max_error_px=surface.sample_reprojection_max_error_px,
            world_camera_round_trip_max_error_m=(
                surface.world_camera_round_trip_max_error_m
            ),
            returned_pose_maximum_absolute_error=returned_pose_error,
            aggregate_inside_room_bounds=aggregate_inside,
            sample_inside_room_fraction=float(np.mean(inside)),
            coordinate_semantics=rules[aligned.target].coordinate_semantics,
        )
        records.append(record)
        preview_items.append((record, images[camera_index], surface.pixels_uv))

    contact_path = output_dir / "visible_surface_contact_sheet.png"
    _save_contact_sheet(preview_items, contact_path)
    world_path = output_dir / "visible_surface_world_preview.png"
    _save_world_preview(
        records=records,
        preview_items=preview_items,
        poses=poses,
        room_bounds=room_bounds,
        zones=cast(dict[str, Any], scene["zones"]),
        project_root=project_root,
        path=world_path,
    )
    summary = VisibleSurfaceRunSummary(
        schema_version=1,
        status="completed_pending_visual_qa",
        stage="S04",
        created_at_utc=datetime.now(UTC),
        source_policy_selection_ref=_relative(policy_path, project_root),
        source_policy_selection_sha256=_sha256(policy_path),
        source_mask_alignment_summary_ref=_relative(alignment_path, project_root),
        source_mask_alignment_summary_sha256=_sha256(alignment_path),
        source_action_depth_summary_ref=_relative(action_path, project_root),
        source_action_depth_summary_sha256=_sha256(action_path),
        pose_calibration_ref=_relative(calibration_path, project_root),
        pose_calibration_sha256=_sha256(calibration_path),
        scene_metadata_ref=_relative(scene_path, project_root),
        scene_metadata_sha256=_sha256(scene_path),
        room_bounds_world_m=room_bounds,
        observations=tuple(records),
        contact_sheet_ref=_relative(contact_path, project_root),
        contact_sheet_sha256=_sha256(contact_path),
        world_preview_ref=_relative(world_path, project_root),
        world_preview_sha256=_sha256(world_path),
        limitations=(
            "Raw XYZ is a per-camera visible-surface aggregate, not a track anchor.",
            "No cross-camera fusion, temporal interpolation, or presentation smoothing occurs.",
            "Room bounds are approximate diagnostics and do not alter retained raw samples.",
        ),
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary.status,
                "observation_count": len(records),
                "aggregate_inside_room_count": sum(
                    record.aggregate_inside_room_bounds for record in records
                ),
                "sample_inside_room_fraction_range": [
                    min(record.sample_inside_room_fraction for record in records),
                    max(record.sample_inside_room_fraction for record in records),
                ],
                "world_xyz_range_m": {
                    "minimum": np.min(
                        np.asarray([record.aggregate_world_xyz_m for record in records]),
                        axis=0,
                    ).tolist(),
                    "maximum": np.max(
                        np.asarray([record.aggregate_world_xyz_m for record in records]),
                        axis=0,
                    ).tolist(),
                },
                "summary": _relative(summary_path, project_root),
            },
            indent=2,
        )
    )
    return 0


def _save_image_diagnostic(
    *,
    image: UInt8Array,
    pixels_uv: Float64Array,
    aggregate_camera_xyz_m: Float64Array,
    aggregate_world_xyz_m: Float64Array,
    intrinsics: CameraIntrinsics,
    phase_id: str,
    camera_id: str,
    target: PerceptionTarget,
    path: Path,
) -> None:
    aggregate_uv = np.asarray(
        project_camera_points(aggregate_camera_xyz_m, intrinsics=intrinsics)
    )
    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    axis.imshow(image)
    step = max(1, len(pixels_uv) // 350)
    shown = pixels_uv[::step]
    axis.scatter(
        shown[:, 0],
        shown[:, 1],
        s=3,
        c=TARGET_COLOR[target],
        alpha=0.5,
        linewidths=0,
    )
    axis.scatter(
        [aggregate_uv[0]],
        [aggregate_uv[1]],
        s=70,
        c="#ffe600",
        edgecolors="black",
        marker="X",
        linewidths=0.8,
    )
    xyz = aggregate_world_xyz_m
    axis.set_title(
        f"{phase_id} · {camera_id} · {target.value}\n"
        f"raw visible surface world XYZ=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) m"
    )
    axis.axis("off")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_contact_sheet(
    items: list[tuple[VisibleSurfaceObservationRecord, UInt8Array, Float64Array]],
    path: Path,
) -> None:
    columns = 4
    rows = (len(items) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(16, rows * 2.5), constrained_layout=True)
    flat_axes = np.asarray(axes).reshape(-1)
    for axis, (record, image, pixels) in zip(flat_axes, items, strict=False):
        axis.imshow(image)
        step = max(1, len(pixels) // 180)
        shown = pixels[::step]
        axis.scatter(
            shown[:, 0],
            shown[:, 1],
            s=2,
            c=TARGET_COLOR[record.target],
            alpha=0.45,
            linewidths=0,
        )
        aggregate_uv = project_camera_points(
            np.asarray(record.aggregate_camera_xyz_m),
            intrinsics=record.processed_intrinsics,
        )
        axis.scatter(
            [aggregate_uv[0]],
            [aggregate_uv[1]],
            s=35,
            c="#ffe600",
            edgecolors="black",
            marker="X",
            linewidths=0.5,
        )
        xyz = record.aggregate_world_xyz_m
        axis.set_title(
            f"f{record.source_frame_index} · {record.camera_id} · {record.target.value}\n"
            f"({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}) m",
            fontsize=8,
        )
        axis.axis("off")
    for axis in flat_axes[len(items) :]:
        axis.axis("off")
    figure.suptitle("S04 raw per-camera visible-surface samples and aggregates")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _save_world_preview(
    *,
    records: list[VisibleSurfaceObservationRecord],
    preview_items: list[
        tuple[VisibleSurfaceObservationRecord, UInt8Array, Float64Array]
    ],
    poses: dict[str, CameraPose],
    room_bounds: dict[str, Any],
    zones: dict[str, Any],
    project_root: Path,
    path: Path,
) -> None:
    figure = plt.figure(figsize=(15, 7), constrained_layout=True)
    axis_3d = figure.add_subplot(1, 2, 1, projection="3d")
    axis_xy = figure.add_subplot(1, 2, 2)
    for record, _, _ in preview_items:
        sample_path = _resolve(project_root, Path(record.sample_cloud_ref))
        with np.load(sample_path, allow_pickle=False) as arrays:
            points = np.asarray(arrays["points_world_m"])
        step = max(1, len(points) // 80)
        shown = points[::step]
        axis_3d.scatter(
            shown[:, 0],
            shown[:, 1],
            shown[:, 2],
            s=2,
            c=TARGET_COLOR[record.target],
            alpha=0.08,
        )
    for target in PerceptionTarget:
        matching = [record for record in records if record.target is target]
        xyz = np.asarray([record.aggregate_world_xyz_m for record in matching])
        axis_3d.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            s=38, c=TARGET_COLOR[target], edgecolors="black", linewidths=0.4,
            label=f"{target.value} raw aggregates",
        )
        for record, point in zip(matching, xyz, strict=True):
            axis_xy.scatter(
                point[0], point[1], s=45, c=TARGET_COLOR[target],
                marker="o" if record.camera_id == "camera_a" else "^",
                edgecolors="black", linewidths=0.4,
            )
            axis_xy.annotate(
                str(record.source_frame_index),
                (point[0], point[1]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
    for camera_id, pose in poses.items():
        centre = np.asarray(pose.T_world_from_camera)[:3, 3]
        axis_3d.scatter(*centre, s=80, marker="^", label=camera_id)
        axis_xy.scatter(centre[0], centre[1], s=80, marker="^", label=camera_id)
    zone_entries = (
        ("pickup", cast(dict[str, Any], zones["pickup_blue_bed"]), "#1677ff"),
        ("drop-off", cast(dict[str, Any], zones["dropoff_white_floor"]), "#777777"),
    )
    for label, zone, color in zone_entries:
        centre = np.asarray(zone["center_world_m"], dtype=np.float64)
        axis_3d.scatter(*centre, s=90, marker="s", c=color, label=label)
        circle = Circle(
            (centre[0], centre[1]),
            float(zone["radius_m"]),
            fill=False,
            color=color,
            linewidth=1.5,
        )
        axis_xy.add_patch(circle)
        axis_xy.scatter(centre[0], centre[1], marker="s", c=color, s=55, label=label)
    minimum = np.asarray(room_bounds["minimum_world_xyz_m"], dtype=np.float64)
    maximum = np.asarray(room_bounds["maximum_world_xyz_m"], dtype=np.float64)
    axis_3d.set_xlim(minimum[0], maximum[0])
    axis_3d.set_ylim(minimum[1], maximum[1])
    axis_3d.set_zlim(minimum[2], maximum[2])
    axis_3d.set_xlabel("world X (m)")
    axis_3d.set_ylabel("world Y (m)")
    axis_3d.set_zlabel("world Z (m)")
    axis_3d.set_title("raw visible-surface sample clouds and aggregates")
    axis_3d.view_init(elev=30, azim=-55)
    axis_3d.legend(fontsize=8, loc="upper left")
    axis_xy.set_xlim(minimum[0], maximum[0])
    axis_xy.set_ylim(minimum[1], maximum[1])
    axis_xy.set_aspect("equal")
    axis_xy.set_xlabel("world X (m)")
    axis_xy.set_ylabel("world Y (m)")
    axis_xy.set_title("top-down raw aggregates (labels are source frame indices)")
    axis_xy.grid(alpha=0.25)
    axis_xy.legend(fontsize=8, loc="upper right")
    figure.suptitle("S04 per-camera visible-surface localization — no fusion or anchors")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _verify_upstream_calibration_hashes(
    *,
    action: ActionDepthRunSummary,
    calibration_path: Path,
    scene_path: Path,
) -> None:
    provenance = action.input_provenance
    _require_hash(calibration_path, str(provenance["pose_calibration_sha256"]))
    _require_hash(scene_path, str(provenance["scene_metadata_sha256"]))


def _verify_raw_identity(
    arrays: Any,
    job_id: str,
    bundle_id: str,
) -> None:
    if str(arrays["job_id"].item()) != job_id:
        raise ValueError("raw prediction job identity differs")
    if str(arrays["bundle_id"].item()) != bundle_id:
        raise ValueError("raw prediction bundle identity differs")


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
