"""Run calibrated DA3 static-room reconstruction for accepted S01 keyframes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import cv2
import matplotlib
import numpy as np
import torch
import trimesh
from numpy.typing import NDArray
from PIL import Image

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from spatial_reconstruction.config import load_project_config
from spatial_reconstruction.contracts import (
    CameraIntrinsics,
    CameraPose,
    SynchronizedFrameBundle,
)
from spatial_reconstruction.geometry import (
    AxisAlignedBounds,
    StaticSceneRunSummary,
    backproject_static_depth,
    confidence_percentile_threshold,
    estimate_marker_depth_scale,
    select_bundles_for_target_times,
    voxel_downsample,
    write_colored_ply,
)
from spatial_reconstruction.ingestion import (
    FileFrameSource,
    TimestampTransform,
    build_synchronized_bundles,
)
from spatial_reconstruction.models import (
    EXPECTED_DA3_VENDOR_FINGERPRINT,
    DA3Adapter,
    compute_vendor_fingerprint,
)
from spatial_reconstruction.models.da3_mps import DA3Precision
from spatial_reconstruction.runtime import (
    PhaseTimer,
    SystemMemorySource,
    sample_memory,
    select_device,
)

FloatArray = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]
MODEL_REVISION = "b2359bdf726fb44ef62acca04d629dcf158053e7"
CAMERA_IDS = ("camera_a", "camera_b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--synchronization-manifest",
        type=Path,
        default=Path(
            "artifacts/s01/empty_room/synchronized/synchronization_manifest.json"
        ),
    )
    parser.add_argument(
        "--pose-calibration",
        type=Path,
        default=Path(
            "artifacts/s01/calibration/empty_room_pose/camera_calibration.json"
        ),
    )
    parser.add_argument(
        "--scene-metadata",
        type=Path,
        default=Path("artifacts/s01/scene_metadata.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--target-time-seconds",
        type=float,
        nargs="+",
        default=[30.0],
        help="Strictly increasing synchronized times inside the accepted empty window.",
    )
    parser.add_argument("--process-resolution", type=int, default=504)
    parser.add_argument("--confidence-percentile", type=float, default=40.0)
    parser.add_argument("--voxel-size-m", type=float, default=0.02)
    parser.add_argument(
        "--marker-scale-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the D025 shared scalar estimated from accepted M40-M42 centres.",
    )
    parser.add_argument(
        "--maximum-marker-scale-relative-deviation",
        type=float,
        default=0.05,
    )
    parser.add_argument("--maximum-target-error-seconds", type=float, default=1.0 / 30.0)
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    config = load_project_config(project_root=project_root)
    output_dir = (
        (project_root / args.output_dir).resolve()
        if args.output_dir is not None
        else _timestamped_output_dir(config.paths.artifacts_dir / "s02")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = (project_root / args.synchronization_manifest).resolve()
    calibration_path = (project_root / args.pose_calibration).resolve()
    scene_metadata_path = (project_root / args.scene_metadata).resolve()

    manifest = _load_json(manifest_path)
    calibration = _load_json(calibration_path)
    scene_metadata = _load_json(scene_metadata_path)
    _validate_stage_inputs(
        manifest=manifest,
        calibration=calibration,
        scene_metadata=scene_metadata,
        target_times=tuple(float(value) for value in args.target_time_seconds),
        process_resolution=int(args.process_resolution),
    )
    manifest_sha256 = _sha256(manifest_path)
    sources = _make_sources(
        project_root=project_root,
        synchronization_manifest_path=manifest_path,
        synchronization_manifest=manifest,
        synchronization_manifest_sha256=manifest_sha256,
        calibration=calibration,
    )
    bundles = tuple(
        build_synchronized_bundles(
            {
                camera_id: sources[camera_id].iter_identities()
                for camera_id in CAMERA_IDS
            },
            expected_camera_ids=CAMERA_IDS,
            reference_camera_id=str(manifest["reference_camera_id"]),
            pairing_tolerance_seconds=1.0 / 60.0,
        )
    )
    content_qa = cast(dict[str, Any], manifest["content_qa"])
    stable_window = cast(
        dict[str, float],
        content_qa["declared_stable_empty_window_sync_seconds"],
    )
    selected_bundles = select_bundles_for_target_times(
        bundles,
        target_times_seconds=tuple(float(value) for value in args.target_time_seconds),
        accepted_start_seconds=float(stable_window["start"]),
        accepted_end_seconds=float(stable_window["end"]),
        maximum_target_error_seconds=float(args.maximum_target_error_seconds),
    )

    camera_intrinsics, camera_poses = _camera_contracts(calibration)
    room_bounds = _room_bounds(scene_metadata)
    marker_ids, marker_centres_world_m = _marker_anchors(calibration)
    keyframe_paths = _extract_undistorted_keyframes(
        output_dir=output_dir,
        sources=sources,
        selected_bundles=selected_bundles,
        camera_intrinsics=camera_intrinsics,
    )

    vendor_fingerprint = compute_vendor_fingerprint(config.paths.da3_vendor_dir)
    if vendor_fingerprint != EXPECTED_DA3_VENDOR_FINGERPRINT:
        raise RuntimeError(
            "DA3 vendor fingerprint changed: "
            f"expected {EXPECTED_DA3_VENDOR_FINGERPRINT}, got {vendor_fingerprint}"
        )
    selection = select_device(
        config.runtime.preferred_device,
        allow_cpu_fallback=False,
    )
    if selection.actual != "mps":
        raise RuntimeError("S02 baseline reconstruction requires native Apple MPS")

    memory_source = SystemMemorySource()
    timings: list[dict[str, object]] = []
    memory: list[dict[str, object]] = [
        sample_memory(
            "before_model_load",
            device="mps",
            source=memory_source,
        ).model_dump(mode="json")
    ]
    load_timer = PhaseTimer(phase="model_load", device="mps")
    with load_timer:
        adapter = DA3Adapter.from_pretrained(
            vendor_dir=config.paths.da3_vendor_dir,
            model_id=config.models.da3,
            model_revision=MODEL_REVISION,
            device=torch.device("mps"),
            precision=cast(DA3Precision, str(args.precision)),
        )
    timings.append(_timer_payload(load_timer))
    memory.append(
        sample_memory(
            "after_model_load",
            device="mps",
            source=memory_source,
        ).model_dump(mode="json")
    )

    points_by_camera: dict[str, list[FloatArray]] = {
        camera_id: [] for camera_id in CAMERA_IDS
    }
    colors_by_camera: dict[str, list[UInt8Array]] = {
        camera_id: [] for camera_id in CAMERA_IDS
    }
    prediction_records: list[dict[str, Any]] = []
    prediction_dir = output_dir / "predictions"
    preview_dir = output_dir / "previews"
    prediction_dir.mkdir(parents=True)
    preview_dir.mkdir(parents=True)

    intrinsics_by_id = {camera.camera_id: camera for camera in camera_intrinsics}
    poses_by_id = {camera.camera_id: camera for camera in camera_poses}
    for selected_index, bundle in enumerate(selected_bundles):
        image_paths = tuple(
            keyframe_paths[(bundle.bundle_id, camera_id)]
            for camera_id in CAMERA_IDS
        )
        inference_timer = PhaseTimer(
            phase=f"inference_bundle_{selected_index:02d}",
            device="mps",
        )
        with inference_timer:
            output = adapter.infer_pose_conditioned(
                image_paths=image_paths,
                camera_intrinsics=tuple(
                    intrinsics_by_id[camera_id] for camera_id in CAMERA_IDS
                ),
                camera_poses=tuple(poses_by_id[camera_id] for camera_id in CAMERA_IDS),
                process_resolution=int(args.process_resolution),
            )
        timings.append(_timer_payload(inference_timer))
        memory.append(
            sample_memory(
                f"after_inference_bundle_{selected_index:02d}",
                device="mps",
                source=memory_source,
            ).model_dump(mode="json")
        )
        if output.processed_images is None:
            raise RuntimeError("DA3 did not return processed images for point coloring")
        supplied_poses = np.asarray(
            [poses_by_id[camera_id].T_camera_from_world for camera_id in CAMERA_IDS],
            dtype=np.float32,
        )
        if not np.allclose(output.T_camera_from_world, supplied_poses, atol=1e-5):
            raise RuntimeError("DA3 returned camera poses differ from calibrated inputs")

        confidence_threshold = confidence_percentile_threshold(
            output.confidence,
            percentile=float(args.confidence_percentile),
        )
        processed_intrinsics = tuple(
            CameraIntrinsics(
                camera_id=camera_id,
                fx=float(output.intrinsics[camera_index, 0, 0]),
                fy=float(output.intrinsics[camera_index, 1, 1]),
                cx=float(output.intrinsics[camera_index, 0, 2]),
                cy=float(output.intrinsics[camera_index, 1, 2]),
                image_width=int(output.depth_m.shape[2]),
                image_height=int(output.depth_m.shape[1]),
            )
            for camera_index, camera_id in enumerate(CAMERA_IDS)
        )
        marker_scale_record: dict[str, Any]
        if bool(args.marker_scale_correction):
            marker_scale = estimate_marker_depth_scale(
                depth_m=output.depth_m,
                camera_intrinsics=processed_intrinsics,
                camera_poses=tuple(
                    poses_by_id[camera_id] for camera_id in CAMERA_IDS
                ),
                marker_ids=marker_ids,
                marker_centres_world_m=marker_centres_world_m,
                patch_radius_pixels=2,
                maximum_relative_deviation=float(
                    args.maximum_marker_scale_relative_deviation
                ),
            )
            geometry_depth_scale = marker_scale.scale
            marker_scale_record = {
                "enabled": True,
                "method": (
                    "shared median expected/predicted camera-Z ratio at "
                    "accepted M40-M42 centres"
                ),
                "scale": marker_scale.scale,
                "maximum_relative_deviation": (
                    marker_scale.maximum_relative_deviation
                ),
                "acceptance_limit": float(
                    args.maximum_marker_scale_relative_deviation
                ),
                "observations": [
                    {
                        "camera_id": observation.camera_id,
                        "marker_id": observation.marker_id,
                        "pixel_uv": list(observation.pixel_uv),
                        "expected_camera_depth_m": (
                            observation.expected_camera_depth_m
                        ),
                        "predicted_depth_m": observation.predicted_depth_m,
                        "expected_over_predicted_ratio": (
                            observation.expected_over_predicted_ratio
                        ),
                    }
                    for observation in marker_scale.observations
                ],
            }
        else:
            geometry_depth_scale = 1.0
            marker_scale_record = {
                "enabled": False,
                "method": "disabled; raw DA3 metric depth used directly",
                "scale": 1.0,
                "observations": [],
            }
        geometry_depth_m = np.asarray(
            output.depth_m * geometry_depth_scale,
            dtype=np.float32,
        )
        raw_path = prediction_dir / f"{selected_index:02d}_{bundle.bundle_id[:12]}.npz"
        np.savez_compressed(
            raw_path,
            depth_m=output.depth_m,
            geometry_depth_m=geometry_depth_m,
            geometry_depth_scale=np.asarray(
                geometry_depth_scale,
                dtype=np.float32,
            ),
            confidence=output.confidence,
            processed_images_rgb=output.processed_images,
            returned_T_camera_from_world=output.T_camera_from_world,
            returned_intrinsics=output.intrinsics,
            bundle_id=np.asarray(bundle.bundle_id),
            capture_timestamp_seconds=np.asarray(bundle.capture_timestamp_seconds),
            frame_ids=np.asarray([frame.frame_id for frame in bundle.frames]),
            camera_ids=np.asarray(CAMERA_IDS),
            model_id=np.asarray(config.models.da3),
            model_revision=np.asarray(MODEL_REVISION),
            process_resolution=np.asarray(int(args.process_resolution), dtype=np.int32),
            confidence_percentile=np.asarray(
                float(args.confidence_percentile),
                dtype=np.float32,
            ),
            confidence_threshold=np.asarray(confidence_threshold, dtype=np.float32),
            is_metric=np.asarray(output.is_metric),
        )
        depth_preview_path = (
            preview_dir / f"{selected_index:02d}_{bundle.bundle_id[:12]}_depth_confidence.png"
        )
        _save_depth_confidence_preview(
            depth_m=output.depth_m,
            confidence=output.confidence,
            processed_images_rgb=output.processed_images,
            camera_ids=CAMERA_IDS,
            path=depth_preview_path,
        )

        camera_records: dict[str, dict[str, Any]] = {}
        for camera_index, camera_id in enumerate(CAMERA_IDS):
            camera_processed_intrinsics = processed_intrinsics[camera_index]
            cloud = backproject_static_depth(
                depth_m=geometry_depth_m[camera_index],
                confidence=output.confidence[camera_index],
                colors_rgb=output.processed_images[camera_index],
                intrinsics=camera_processed_intrinsics,
                pose=poses_by_id[camera_id],
                confidence_threshold=confidence_threshold,
                room_bounds=room_bounds,
            )
            points_by_camera[camera_id].append(cloud.points_world_m)
            colors_by_camera[camera_id].append(cloud.colors_rgb)
            camera_records[camera_id] = {
                "frame_id": bundle.frames[camera_index].frame_id,
                "source_frame_index": bundle.frames[camera_index].source_frame_index,
                "capture_timestamp_seconds": (
                    bundle.frames[camera_index].capture_timestamp_seconds
                ),
                "undistorted_keyframe_ref": _relative(
                    image_paths[camera_index],
                    project_root,
                ),
                "undistorted_keyframe_sha256": _sha256(
                    image_paths[camera_index]
                ),
                "processed_intrinsics": camera_processed_intrinsics.model_dump(
                    mode="json"
                ),
                "filtering": {
                    "total_pixel_count": cloud.stats.total_pixel_count,
                    "valid_depth_count": cloud.stats.valid_depth_count,
                    "finite_confidence_count": cloud.stats.finite_confidence_count,
                    "confidence_retained_count": cloud.stats.confidence_retained_count,
                    "room_bounds_retained_count": cloud.stats.room_bounds_retained_count,
                },
                "depth_range_m": _finite_range(output.depth_m[camera_index]),
                "geometry_depth_range_m": _finite_range(
                    geometry_depth_m[camera_index]
                ),
                "confidence_range": _finite_range(output.confidence[camera_index]),
            }
        prediction_records.append(
            {
                "bundle_id": bundle.bundle_id,
                "bundle_index": bundle.bundle_index,
                "capture_timestamp_seconds": bundle.capture_timestamp_seconds,
                "maximum_frame_time_difference_seconds": (
                    bundle.max_frame_time_difference_seconds
                ),
                "raw_prediction_ref": _relative(raw_path, project_root),
                "raw_prediction_sha256": _sha256(raw_path),
                "depth_confidence_preview_ref": _relative(
                    depth_preview_path,
                    project_root,
                ),
                "depth_confidence_preview_sha256": _sha256(depth_preview_path),
                "confidence_percentile": float(args.confidence_percentile),
                "confidence_threshold": confidence_threshold,
                "marker_depth_scale_correction": marker_scale_record,
                "cameras": camera_records,
            }
        )

    point_cloud_records: dict[str, dict[str, Any]] = {}
    fused_points_parts: list[FloatArray] = []
    fused_colors_parts: list[UInt8Array] = []
    for camera_id in CAMERA_IDS:
        camera_points = _concatenate_points(points_by_camera[camera_id])
        camera_colors = _concatenate_colors(colors_by_camera[camera_id])
        downsampled_points, downsampled_colors = voxel_downsample(
            camera_points,
            camera_colors,
            voxel_size_m=float(args.voxel_size_m),
        )
        camera_ply = output_dir / f"{camera_id}_static_scene.ply"
        write_colored_ply(
            camera_ply,
            points_world_m=downsampled_points,
            colors_rgb=downsampled_colors,
        )
        point_cloud_records[camera_id] = {
            "pre_voxel_point_count": int(camera_points.shape[0]),
            "point_count": int(downsampled_points.shape[0]),
            "ply_ref": _relative(camera_ply, project_root),
            "ply_sha256": _sha256(camera_ply),
            "world_xyz_range_m": _xyz_range(downsampled_points),
        }
        fused_points_parts.append(downsampled_points)
        fused_colors_parts.append(downsampled_colors)

    fused_input_points = _concatenate_points(fused_points_parts)
    fused_input_colors = _concatenate_colors(fused_colors_parts)
    fused_points, fused_colors = voxel_downsample(
        fused_input_points,
        fused_input_colors,
        voxel_size_m=float(args.voxel_size_m),
    )
    if fused_points.shape[0] == 0:
        raise RuntimeError("S02 filtering produced an empty fused point cloud")
    fused_ply = output_dir / "static_scene.ply"
    write_colored_ply(
        fused_ply,
        points_world_m=fused_points,
        colors_rgb=fused_colors,
    )
    geometry_preview_path = preview_dir / "static_scene_geometry.png"
    _save_geometry_preview(
        points_world_m=fused_points,
        colors_rgb=fused_colors,
        camera_poses=camera_poses,
        room_bounds=room_bounds,
        path=geometry_preview_path,
    )
    glb_path = preview_dir / "static_scene_with_cameras.glb"
    _save_glb_preview(
        points_world_m=fused_points,
        colors_rgb=fused_colors,
        camera_intrinsics=camera_intrinsics,
        camera_poses=camera_poses,
        path=glb_path,
    )

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed_pending_visual_qa",
        "stage": "S02",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "capture_session_id": manifest["capture_session_id"],
        "pose_version_id": calibration["pose_version_id"],
        "world_frame": scene_metadata["world_frame"],
        "input_provenance": {
            "synchronization_manifest_ref": _relative(manifest_path, project_root),
            "synchronization_manifest_sha256": manifest_sha256,
            "pose_calibration_ref": _relative(calibration_path, project_root),
            "pose_calibration_sha256": _sha256(calibration_path),
            "scene_metadata_ref": _relative(scene_metadata_path, project_root),
            "scene_metadata_sha256": _sha256(scene_metadata_path),
            "vendor_fingerprint": vendor_fingerprint,
        },
        "model": {
            "model_id": config.models.da3,
            "model_revision": MODEL_REVISION,
            "device": selection.actual,
            "precision": adapter.autocast_policy.reported_precision,
            "is_metric": True,
            "two_view_alignment_policy": (
                "preserve_nested_metric_depth_and_return_supplied_poses"
            ),
        },
        "selection": {
            "accepted_window_seconds": stable_window,
            "target_times_seconds": [
                float(value) for value in args.target_time_seconds
            ],
            "selected_bundle_count": len(selected_bundles),
            "maximum_target_error_seconds": float(
                args.maximum_target_error_seconds
            ),
        },
        "processing": {
            "process_resolution": int(args.process_resolution),
            "confidence_filter": {
                "method": "finite confidence at or above per-pair percentile",
                "percentile": float(args.confidence_percentile),
                "vendor_default_percentile": 40.0,
            },
            "room_bounds": {
                "minimum_world_xyz_m": list(room_bounds.minimum_world_xyz_m),
                "maximum_world_xyz_m": list(room_bounds.maximum_world_xyz_m),
                "use": "gross outlier filtering only; not surveyed surfaces",
            },
            "voxel_size_m": float(args.voxel_size_m),
            "raw_model_outputs_preserved": True,
            "source_frames_undistorted_before_inference": True,
            "marker_depth_scale_correction": {
                "enabled": bool(args.marker_scale_correction),
                "decision": "D025",
                "raw_depth_preserved": True,
                "allowed_influence": "derived S02 static point clouds only",
                "maximum_relative_deviation": float(
                    args.maximum_marker_scale_relative_deviation
                ),
            },
        },
        "predictions": prediction_records,
        "point_clouds": {
            **point_cloud_records,
            "fused": {
                "pre_voxel_point_count": int(fused_input_points.shape[0]),
                "point_count": int(fused_points.shape[0]),
                "ply_ref": _relative(fused_ply, project_root),
                "ply_sha256": _sha256(fused_ply),
                "world_xyz_range_m": _xyz_range(fused_points),
            },
        },
        "previews": {
            "geometry_png_ref": _relative(geometry_preview_path, project_root),
            "geometry_png_sha256": _sha256(geometry_preview_path),
            "glb_ref": _relative(glb_path, project_root),
            "glb_sha256": _sha256(glb_path),
        },
        "runtime": {
            "platform_machine": platform.machine(),
            "torch_version": torch.__version__,
            "timings": timings,
            "memory": memory,
        },
        "limitations": [
            "This run is pending visual and cross-camera geometry QA.",
            "Room bounds are a conservative crop, not surveyed wall geometry.",
            "Confidence uses DA3's adaptive percentile semantics and is not normalized.",
            "The two physical cameras share one bounded intrinsic estimate under D021.",
        ],
    }
    validated_result = StaticSceneRunSummary.model_validate(result).model_dump(
        mode="json"
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(validated_result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validated_result, indent=2))
    return 0


def _validate_stage_inputs(
    *,
    manifest: Mapping[str, Any],
    calibration: Mapping[str, Any],
    scene_metadata: Mapping[str, Any],
    target_times: tuple[float, ...],
    process_resolution: int,
) -> None:
    if manifest.get("capture_session_id") != "s01_capture_20260729":
        raise ValueError("unexpected S02 capture session")
    if manifest.get("purpose") != "empty_room_static_reconstruction_input":
        raise ValueError("synchronization manifest is not the static-room input")
    if not cast(dict[str, Any], manifest["synchronization_validation"]).get(
        "accepted"
    ):
        raise ValueError("empty-room synchronization is not accepted")
    content_qa = cast(dict[str, Any], manifest["content_qa"])
    if not content_qa.get("static_reconstruction_must_not_use_setup_frames"):
        raise ValueError("empty-room content restriction is missing")
    expected_pose = "s01_capture_20260729:empty_room:v1"
    if calibration.get("pose_version_id") != expected_pose:
        raise ValueError("S02 must use the accepted empty-room pose version")
    if not str(calibration.get("calibration_status", "")).startswith("accepted_"):
        raise ValueError("empty-room pose calibration is not accepted")
    accepted_versions = cast(dict[str, str], scene_metadata["accepted_pose_versions"])
    if accepted_versions.get("empty_room") != expected_pose:
        raise ValueError("scene metadata does not accept the empty-room pose version")
    if not target_times:
        raise ValueError("at least one S02 target time is required")
    if process_resolution <= 0 or process_resolution % 14 != 0:
        raise ValueError("DA3 process resolution must be a positive multiple of 14")


def _make_sources(
    *,
    project_root: Path,
    synchronization_manifest_path: Path,
    synchronization_manifest: Mapping[str, Any],
    synchronization_manifest_sha256: str,
    calibration: Mapping[str, Any],
) -> dict[str, FileFrameSource]:
    derived = cast(dict[str, dict[str, Any]], synchronization_manifest["derived_outputs"])
    source_records = cast(dict[str, dict[str, Any]], synchronization_manifest["sources"])
    manifest_ref = _relative(synchronization_manifest_path, project_root)
    return {
        camera_id: FileFrameSource(
            path=project_root / str(derived[camera_id]["path"]),
            capture_session_id=str(synchronization_manifest["capture_session_id"]),
            camera_id=camera_id,
            source_ref=str(derived[camera_id]["path"]),
            expected_sha256=str(derived[camera_id]["sha256"]),
            synchronization_manifest_ref=manifest_ref,
            synchronization_manifest_sha256=synchronization_manifest_sha256,
            pose_version_id=str(calibration["pose_version_id"]),
            timestamp_transform=TimestampTransform(),
            expected_width=int(source_records[camera_id]["image_width"]),
            expected_height=int(source_records[camera_id]["image_height"]),
        )
        for camera_id in CAMERA_IDS
    }


def _camera_contracts(
    calibration: Mapping[str, Any],
) -> tuple[tuple[CameraIntrinsics, ...], tuple[CameraPose, ...]]:
    records = cast(dict[str, dict[str, Any]], calibration["cameras"])
    intrinsics: list[CameraIntrinsics] = []
    poses: list[CameraPose] = []
    for camera_id in CAMERA_IDS:
        intrinsic = cast(dict[str, Any], records[camera_id]["intrinsics"])
        pose = cast(dict[str, Any], records[camera_id]["pose"])
        intrinsics.append(
            CameraIntrinsics(
                camera_id=camera_id,
                fx=float(intrinsic["fx"]),
                fy=float(intrinsic["fy"]),
                cx=float(intrinsic["cx"]),
                cy=float(intrinsic["cy"]),
                image_width=int(intrinsic["image_width"]),
                image_height=int(intrinsic["image_height"]),
                distortion_coefficients=tuple(
                    float(value)
                    for value in cast(list[float], intrinsic["distortion_coefficients"])
                ),
            )
        )
        poses.append(CameraPose.model_validate(pose))
    return tuple(intrinsics), tuple(poses)


def _room_bounds(scene_metadata: Mapping[str, Any]) -> AxisAlignedBounds:
    record = cast(dict[str, Any], scene_metadata["room_bounds"])
    return AxisAlignedBounds(
        minimum_world_xyz_m=cast(
            tuple[float, float, float],
            tuple(float(value) for value in record["minimum_world_xyz_m"]),
        ),
        maximum_world_xyz_m=cast(
            tuple[float, float, float],
            tuple(float(value) for value in record["maximum_world_xyz_m"]),
        ),
    )


def _marker_anchors(
    calibration: Mapping[str, Any],
) -> tuple[tuple[int, ...], FloatArray]:
    marker_model = cast(dict[str, Any], calibration["marker_model"])
    accepted_ids = tuple(
        int(value) for value in cast(list[int], marker_model["pose_anchor_marker_ids"])
    )
    placements = cast(list[dict[str, Any]], marker_model["placements"])
    centres_by_id = {
        int(record["marker_id"]): tuple(
            float(value) for value in record["center_world_m"]
        )
        for record in placements
    }
    if accepted_ids != (40, 41, 42):
        raise ValueError("S02 marker scale correction requires accepted M40-M42")
    if any(marker_id not in centres_by_id for marker_id in accepted_ids):
        raise ValueError("accepted marker centre is missing from calibration")
    centres = np.asarray(
        [centres_by_id[marker_id] for marker_id in accepted_ids],
        dtype=np.float64,
    )
    if centres.shape != (3, 3) or not np.isfinite(centres).all():
        raise ValueError("accepted marker centres must be finite world XYZ")
    return accepted_ids, centres


def _extract_undistorted_keyframes(
    *,
    output_dir: Path,
    sources: Mapping[str, FileFrameSource],
    selected_bundles: tuple[SynchronizedFrameBundle, ...],
    camera_intrinsics: tuple[CameraIntrinsics, ...],
) -> dict[tuple[str, str], Path]:
    keyframe_dir = output_dir / "keyframes"
    keyframe_dir.mkdir(parents=True)
    intrinsics_by_id = {intrinsic.camera_id: intrinsic for intrinsic in camera_intrinsics}
    selected_lookup: dict[str, dict[int, tuple[str, str]]] = {
        camera_id: {} for camera_id in CAMERA_IDS
    }
    for bundle in selected_bundles:
        for frame in bundle.frames:
            selected_lookup[frame.camera_id][frame.source_frame_index] = (
                bundle.bundle_id,
                frame.frame_id,
            )

    result: dict[tuple[str, str], Path] = {}
    for camera_id in CAMERA_IDS:
        intrinsic = intrinsics_by_id[camera_id]
        camera_matrix = np.array(
            [
                [intrinsic.fx, 0.0, intrinsic.cx],
                [0.0, intrinsic.fy, intrinsic.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.asarray(intrinsic.distortion_coefficients, dtype=np.float64)
        pending = selected_lookup[camera_id]
        for decoded in sources[camera_id].iter_frames():
            selected = pending.get(decoded.identity.source_frame_index)
            if selected is None:
                continue
            bundle_id, expected_frame_id = selected
            if decoded.identity.frame_id != expected_frame_id:
                raise RuntimeError("decoded selected frame identity changed")
            undistorted_bgr = cv2.undistort(
                decoded.image_bgr,
                camera_matrix,
                distortion,
                None,
                camera_matrix,
            )
            rgb = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2RGB)
            path = keyframe_dir / (
                f"{bundle_id[:12]}_{camera_id}_frame_"
                f"{decoded.identity.source_frame_index:04d}.png"
            )
            Image.fromarray(rgb).save(path)
            result[(bundle_id, camera_id)] = path
            if len(result) == len(selected_bundles) * len(CAMERA_IDS):
                break
    expected_count = len(selected_bundles) * len(CAMERA_IDS)
    if len(result) != expected_count:
        raise RuntimeError(
            f"decoded {len(result)} selected frames, expected {expected_count}"
        )
    return result


def _save_depth_confidence_preview(
    *,
    depth_m: NDArray[np.float32],
    confidence: NDArray[np.float32],
    processed_images_rgb: UInt8Array,
    camera_ids: tuple[str, str],
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    for index, camera_id in enumerate(camera_ids):
        axes[index, 0].imshow(processed_images_rgb[index])
        axes[index, 0].set_title(f"{camera_id} processed RGB")
        depth_image = axes[index, 1].imshow(
            depth_m[index],
            cmap="turbo",
            vmin=float(np.nanpercentile(depth_m[index], 2)),
            vmax=float(np.nanpercentile(depth_m[index], 98)),
        )
        axes[index, 1].set_title(f"{camera_id} metric depth")
        figure.colorbar(depth_image, ax=axes[index, 1], fraction=0.046)
        confidence_image = axes[index, 2].imshow(confidence[index], cmap="viridis")
        axes[index, 2].set_title(f"{camera_id} confidence")
        figure.colorbar(confidence_image, ax=axes[index, 2], fraction=0.046)
        for axis in axes[index]:
            axis.axis("off")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_geometry_preview(
    *,
    points_world_m: FloatArray,
    colors_rgb: UInt8Array,
    camera_poses: tuple[CameraPose, ...],
    room_bounds: AxisAlignedBounds,
    path: Path,
) -> None:
    sample_indices = np.linspace(
        0,
        points_world_m.shape[0] - 1,
        num=min(60_000, points_world_m.shape[0]),
        dtype=np.int64,
    )
    points = points_world_m[sample_indices]
    colors = colors_rgb[sample_indices].astype(np.float64) / 255.0
    camera_centres = np.asarray(
        [
            [
                pose.T_world_from_camera[0][3],
                pose.T_world_from_camera[1][3],
                pose.T_world_from_camera[2][3],
            ]
            for pose in camera_poses
        ],
        dtype=np.float64,
    )
    figure = plt.figure(figsize=(16, 7), constrained_layout=True)
    axis_3d = figure.add_subplot(1, 2, 1, projection="3d")
    axis_3d.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        s=0.3,
        linewidths=0,
    )
    axis_3d.scatter(
        camera_centres[:, 0],
        camera_centres[:, 1],
        camera_centres[:, 2],
        c=("red", "blue"),
        s=60,
        marker="^",
    )
    axis_3d.set_title("S02 fused world point cloud with calibrated cameras")
    axis_3d.set_xlabel("world X (m)")
    axis_3d.set_ylabel("world Y (m)")
    axis_3d.set_zlabel("world Z (m)")
    axis_3d.view_init(elev=28, azim=-55)

    axis_top = figure.add_subplot(1, 2, 2)
    axis_top.scatter(
        points[:, 0],
        points[:, 1],
        c=colors,
        s=0.3,
        linewidths=0,
    )
    axis_top.scatter(
        camera_centres[:, 0],
        camera_centres[:, 1],
        c=("red", "blue"),
        s=70,
        marker="^",
    )
    for pose, color in zip(camera_poses, ("red", "blue"), strict=True):
        centre = np.asarray(
            [
                pose.T_world_from_camera[0][3],
                pose.T_world_from_camera[1][3],
                pose.T_world_from_camera[2][3],
            ]
        )
        optical_axis = np.asarray(
            [
                pose.T_world_from_camera[0][2],
                pose.T_world_from_camera[1][2],
                pose.T_world_from_camera[2][2],
            ]
        )
        axis_top.arrow(
            centre[0],
            centre[1],
            optical_axis[0] * 0.6,
            optical_axis[1] * 0.6,
            color=color,
            width=0.01,
        )
    axis_top.set_title("Top view (Z up)")
    axis_top.set_xlabel("world X (m)")
    axis_top.set_ylabel("world Y (m)")
    axis_top.set_aspect("equal")
    axis_top.set_xlim(
        room_bounds.minimum_world_xyz_m[0],
        room_bounds.maximum_world_xyz_m[0],
    )
    axis_top.set_ylim(
        room_bounds.minimum_world_xyz_m[1],
        room_bounds.maximum_world_xyz_m[1],
    )
    axis_top.grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_glb_preview(
    *,
    points_world_m: FloatArray,
    colors_rgb: UInt8Array,
    camera_intrinsics: tuple[CameraIntrinsics, ...],
    camera_poses: tuple[CameraPose, ...],
    path: Path,
) -> None:
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.points.PointCloud(  # type: ignore[no-untyped-call]
            vertices=points_world_m,
            colors=colors_rgb,
        ),
        geom_name="static_scene",
    )
    for intrinsic, pose, color in zip(
        camera_intrinsics,
        camera_poses,
        ((255, 60, 60, 255), (60, 100, 255, 255)),
        strict=True,
    ):
        segments = _camera_frustum_segments(
            intrinsic=intrinsic,
            pose=pose,
            distance_m=0.35,
        )
        path_geometry = trimesh.load_path(segments)
        path_geometry.colors = np.tile(
            np.asarray(color, dtype=np.uint8),
            (len(path_geometry.entities), 1),
        )
        scene.add_geometry(path_geometry, geom_name=f"{intrinsic.camera_id}_frustum")
    scene.export(path)  # type: ignore[no-untyped-call]


def _camera_frustum_segments(
    *,
    intrinsic: CameraIntrinsics,
    pose: CameraPose,
    distance_m: float,
) -> FloatArray:
    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [intrinsic.image_width - 1.0, 0.0, 1.0],
            [intrinsic.image_width - 1.0, intrinsic.image_height - 1.0, 1.0],
            [0.0, intrinsic.image_height - 1.0, 1.0],
        ]
    )
    camera_matrix = np.array(
        [
            [intrinsic.fx, 0.0, intrinsic.cx],
            [0.0, intrinsic.fy, intrinsic.cy],
            [0.0, 0.0, 1.0],
        ]
    )
    rays = (np.linalg.inv(camera_matrix) @ corners.T).T
    plane_camera = rays * (distance_m / rays[:, 2:3])
    T_world_from_camera = np.asarray(pose.T_world_from_camera, dtype=np.float64)
    plane_world = (
        plane_camera @ T_world_from_camera[:3, :3].T
        + T_world_from_camera[:3, 3]
    )
    centre = T_world_from_camera[:3, 3]
    segments: list[FloatArray] = [
        np.stack((centre, corner), axis=0) for corner in plane_world
    ]
    for first, second in ((0, 1), (1, 2), (2, 3), (3, 0)):
        segments.append(np.stack((plane_world[first], plane_world[second]), axis=0))
    return np.asarray(segments, dtype=np.float64)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root).as_posix()


def _timer_payload(timer: PhaseTimer) -> dict[str, object]:
    if timer.observation is None:
        raise RuntimeError(f"timer {timer.phase} did not produce an observation")
    return timer.observation.model_dump(mode="json")


def _finite_range(values: NDArray[np.float32]) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise RuntimeError("array contains no finite values")
    return {
        "minimum": float(np.min(finite)),
        "p02": float(np.percentile(finite, 2)),
        "median": float(np.median(finite)),
        "p98": float(np.percentile(finite, 98)),
        "maximum": float(np.max(finite)),
    }


def _xyz_range(points: FloatArray) -> dict[str, list[float]]:
    if points.shape[0] == 0:
        return {"minimum": [], "maximum": []}
    return {
        "minimum": [float(value) for value in np.min(points, axis=0)],
        "maximum": [float(value) for value in np.max(points, axis=0)],
    }


def _concatenate_points(parts: list[FloatArray]) -> FloatArray:
    non_empty = [part for part in parts if part.shape[0] > 0]
    if not non_empty:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(np.concatenate(non_empty, axis=0), dtype=np.float64)


def _concatenate_colors(parts: list[UInt8Array]) -> UInt8Array:
    non_empty = [part for part in parts if part.shape[0] > 0]
    if not non_empty:
        return np.empty((0, 3), dtype=np.uint8)
    return np.asarray(np.concatenate(non_empty, axis=0), dtype=np.uint8)


def _timestamped_output_dir(parent: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return parent / f"static_room_{timestamp}"


if __name__ == "__main__":
    raise SystemExit(main())
