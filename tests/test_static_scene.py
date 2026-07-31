from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from plyfile import PlyData
from pydantic import ValidationError

from spatial_reconstruction.contracts import (
    CameraIntrinsics,
    CameraPose,
    FrameIdentity,
    FrameSourceKind,
    SourceFingerprintKind,
    SynchronizedFrameBundle,
)
from spatial_reconstruction.geometry import (
    AxisAlignedBounds,
    StaticSceneRerunExportSummary,
    StaticSceneRunSummary,
    backproject_static_depth,
    confidence_percentile_threshold,
    estimate_marker_depth_scale,
    radius_overlap_fraction,
    select_keyframe_bundles,
    voxel_downsample,
    write_colored_ply,
)

SOURCE_HASH = "a" * 64
MANIFEST_HASH = "b" * 64
MANIFEST_REF = "artifacts/s01/empty_room/synchronization_manifest.json"


def make_frame(camera_id: str, frame_index: int, timestamp: float) -> FrameIdentity:
    return FrameIdentity.create(
        capture_session_id="session",
        camera_id=camera_id,
        source_kind=FrameSourceKind.FILE,
        source_frame_index=frame_index,
        source_timestamp_seconds=timestamp,
        capture_timestamp_seconds=timestamp,
        source_ref=f"{camera_id}.mp4",
        source_fingerprint=SOURCE_HASH,
        source_fingerprint_kind=SourceFingerprintKind.CONTENT_SHA256,
        synchronization_manifest_ref=MANIFEST_REF,
        synchronization_manifest_sha256=MANIFEST_HASH,
        pose_version_id="session:empty_room:v1",
        image_width=2,
        image_height=2,
    )


def make_bundle(
    index: int,
    timestamp: float,
    *,
    missing_b: bool = False,
) -> SynchronizedFrameBundle:
    frames = (make_frame("camera_a", index, timestamp),)
    if not missing_b:
        frames += (make_frame("camera_b", index, timestamp + 0.001),)
    return SynchronizedFrameBundle.create(
        bundle_index=index,
        capture_session_id="session",
        capture_timestamp_seconds=timestamp,
        reference_camera_id="camera_a",
        expected_camera_ids=("camera_a", "camera_b"),
        frames=frames,
        pairing_tolerance_seconds=0.01,
        synchronization_manifest_ref=MANIFEST_REF,
        synchronization_manifest_sha256=MANIFEST_HASH,
    )


def identity_camera() -> tuple[CameraIntrinsics, CameraPose]:
    matrix = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    return (
        CameraIntrinsics(
            camera_id="camera_a",
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
            image_width=2,
            image_height=2,
        ),
        CameraPose(
            camera_id="camera_a",
            T_world_from_camera=matrix,
            T_camera_from_world=matrix,
        ),
    )


def make_static_summary_payload() -> dict[str, Any]:
    prediction = {
        "bundle_id": "c" * 64,
        "bundle_index": 1,
        "capture_timestamp_seconds": 22.0,
        "maximum_frame_time_difference_seconds": 0.003,
        "raw_prediction_ref": "artifacts/s02/prediction.npz",
        "raw_prediction_sha256": "d" * 64,
        "depth_confidence_preview_ref": "artifacts/s02/preview.png",
        "depth_confidence_preview_sha256": "e" * 64,
        "confidence_percentile": 40.0,
        "confidence_threshold": 2.0,
        "marker_depth_scale_correction": {"enabled": True},
        "cameras": {"camera_a": {}, "camera_b": {}},
    }
    point_cloud = {
        "pre_voxel_point_count": 20,
        "point_count": 10,
        "ply_ref": "artifacts/s02/static_scene.ply",
        "ply_sha256": "f" * 64,
        "world_xyz_range_m": {
            "minimum": [0.0, 0.0, 0.0],
            "maximum": [1.0, 1.0, 1.0],
        },
    }
    return {
        "schema_version": 1,
        "status": "completed_pending_visual_qa",
        "stage": "S02",
        "created_at_utc": "2026-07-31T00:00:00+00:00",
        "capture_session_id": "session",
        "pose_version_id": "session:empty_room:v1",
        "world_frame": {"units": "metres"},
        "input_provenance": {
            "synchronization_manifest_ref": "artifacts/s01/sync.json",
            "synchronization_manifest_sha256": "1" * 64,
            "pose_calibration_ref": "artifacts/s01/calibration.json",
            "pose_calibration_sha256": "2" * 64,
            "scene_metadata_ref": "artifacts/s01/scene.json",
            "scene_metadata_sha256": "3" * 64,
            "vendor_fingerprint": "4" * 64,
        },
        "model": {
            "model_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
            "model_revision": "5" * 40,
            "device": "mps",
            "precision": "float16",
            "is_metric": True,
            "two_view_alignment_policy": (
                "preserve_nested_metric_depth_and_return_supplied_poses"
            ),
        },
        "selection": {
            "accepted_window_seconds": {"start": 22.0, "end": 38.0},
            "target_times_seconds": [22.0],
            "selected_bundle_count": 1,
            "maximum_target_error_seconds": 1.0 / 30.0,
        },
        "processing": {"process_resolution": 504},
        "predictions": [prediction],
        "point_clouds": {
            "camera_a": point_cloud,
            "camera_b": point_cloud,
            "fused": point_cloud,
        },
        "previews": {
            "geometry_png_ref": "artifacts/s02/geometry.png",
            "geometry_png_sha256": "6" * 64,
            "glb_ref": "artifacts/s02/geometry.glb",
            "glb_sha256": "7" * 64,
        },
        "runtime": {"torch_version": "test"},
        "limitations": ["test fixture"],
    }


def test_static_scene_summary_schema_round_trip_and_pair_validation() -> None:
    payload = make_static_summary_payload()
    summary = StaticSceneRunSummary.model_validate(payload)

    assert StaticSceneRunSummary.model_validate_json(
        summary.model_dump_json()
    ) == summary

    del payload["point_clouds"]["camera_b"]
    with pytest.raises(ValidationError, match="both cameras and fused"):
        StaticSceneRunSummary.model_validate(payload)


def test_rerun_summary_schema_rejects_samples_larger_than_sources() -> None:
    payload = {
        "schema_version": 1,
        "status": "passed",
        "stage": "S02",
        "rerun_sdk_version": "0.22.1",
        "source_summary_ref": "artifacts/s02/summary.json",
        "source_summary_sha256": "8" * 64,
        "recording_ref": "artifacts/s02/static_scene.rrd",
        "recording_sha256": "9" * 64,
        "recording_bytes": 100,
        "source_fused_point_count": 10,
        "source_camera_point_counts": {"camera_a": 10, "camera_b": 10},
        "logged_fused_point_count": 11,
        "logged_camera_point_counts": {"camera_a": 5, "camera_b": 5},
        "logged_camera_ids": ["camera_a", "camera_b"],
        "maximum_points_per_rerun_entity": 4_000,
        "camera_image_maximum_dimension": 960,
        "world_coordinates": "right-handed, metres, Z up",
        "camera_coordinates": "OpenCV X right, Y down, Z forward",
        "includes": ["fused static point cloud"],
    }

    with pytest.raises(ValidationError, match="cannot exceed"):
        StaticSceneRerunExportSummary.model_validate(payload)


def test_select_keyframes_is_deterministic_complete_and_interval_bounded() -> None:
    bundles = (
        make_bundle(0, 21.99),
        make_bundle(1, 22.01),
        make_bundle(2, 24.0, missing_b=True),
        make_bundle(3, 24.01),
        make_bundle(4, 25.99),
        make_bundle(5, 26.02),
    )

    selected = select_keyframe_bundles(
        bundles,
        start_seconds=22.0,
        end_seconds=26.0,
        interval_seconds=2.0,
        maximum_target_error_seconds=0.02,
    )

    assert [bundle.bundle_index for bundle in selected] == [1, 3, 4]
    assert all(not bundle.missing_camera_ids for bundle in selected)
    assert select_keyframe_bundles(
        bundles,
        start_seconds=22.0,
        end_seconds=26.0,
        interval_seconds=2.0,
        maximum_target_error_seconds=0.02,
    ) == selected


def test_select_keyframes_rejects_missing_or_distant_candidates() -> None:
    with pytest.raises(ValueError, match="no complete"):
        select_keyframe_bundles(
            (make_bundle(0, 22.0, missing_b=True),),
            start_seconds=22.0,
            end_seconds=24.0,
            interval_seconds=2.0,
            maximum_target_error_seconds=0.02,
        )

    with pytest.raises(ValueError, match="within"):
        select_keyframe_bundles(
            (make_bundle(0, 22.1), make_bundle(1, 24.1)),
            start_seconds=22.0,
            end_seconds=24.2,
            interval_seconds=2.0,
            maximum_target_error_seconds=0.02,
        )


def test_confidence_threshold_uses_finite_vendor_style_percentile() -> None:
    confidence = np.array([1.0, 2.0, 3.0, np.nan])

    assert confidence_percentile_threshold(confidence, percentile=50.0) == 2.0
    with pytest.raises(ValueError, match="between"):
        confidence_percentile_threshold(confidence, percentile=101.0)
    with pytest.raises(ValueError, match="no finite"):
        confidence_percentile_threshold([np.nan], percentile=40.0)


def test_radius_overlap_fraction_is_symmetric_only_when_coverage_is_symmetric() -> None:
    larger = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [1.0, 1.0, 1.0]])
    smaller = np.array([[0.01, 0.0, 0.0]])

    assert radius_overlap_fraction(larger, smaller, radius_m=0.1) == pytest.approx(
        2.0 / 3.0
    )
    assert radius_overlap_fraction(smaller, larger, radius_m=0.1) == pytest.approx(
        1.0
    )

    with pytest.raises(ValueError, match="non-empty"):
        radius_overlap_fraction(np.empty((0, 3)), smaller, radius_m=0.1)


def test_marker_depth_scale_recovers_shared_known_world_ratio() -> None:
    matrix = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    intrinsics = tuple(
        CameraIntrinsics(
            camera_id=camera_id,
            fx=2.0,
            fy=2.0,
            cx=2.0,
            cy=2.0,
            image_width=5,
            image_height=5,
        )
        for camera_id in ("camera_a", "camera_b")
    )
    poses = tuple(
        CameraPose(
            camera_id=camera_id,
            T_world_from_camera=matrix,
            T_camera_from_world=matrix,
        )
        for camera_id in ("camera_a", "camera_b")
    )
    estimate = estimate_marker_depth_scale(
        depth_m=np.ones((2, 5, 5)),
        camera_intrinsics=intrinsics,
        camera_poses=poses,
        marker_ids=(40, 41),
        marker_centres_world_m=np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]]),
        patch_radius_pixels=0,
    )

    assert estimate.scale == pytest.approx(2.0)
    assert estimate.maximum_relative_deviation == pytest.approx(0.0)
    assert len(estimate.observations) == 4
    assert {item.camera_id for item in estimate.observations} == {
        "camera_a",
        "camera_b",
    }


def test_marker_depth_scale_rejects_materially_disagreeing_observations() -> None:
    matrix = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    intrinsics = (
        CameraIntrinsics(
            camera_id="camera_a",
            fx=2.0,
            fy=2.0,
            cx=2.0,
            cy=2.0,
            image_width=5,
            image_height=5,
        ),
    )
    poses = (
        CameraPose(
            camera_id="camera_a",
            T_world_from_camera=matrix,
            T_camera_from_world=matrix,
        ),
    )
    depth = np.ones((1, 5, 5))
    depth[0, 2, 3] = 0.5

    with pytest.raises(ValueError, match="disagree"):
        estimate_marker_depth_scale(
            depth_m=depth,
            camera_intrinsics=intrinsics,
            camera_poses=poses,
            marker_ids=(40, 41),
            marker_centres_world_m=np.array(
                [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]]
            ),
            patch_radius_pixels=0,
            maximum_relative_deviation=0.05,
        )


def test_backprojection_filters_invalid_low_confidence_and_out_of_room_points() -> None:
    intrinsics, pose = identity_camera()
    depth = np.array([[1.0, 1.0], [np.nan, 2.0]])
    confidence = np.array([[3.0, 3.0], [4.0, 1.0]])
    colors = np.array(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )
    bounds = AxisAlignedBounds(
        minimum_world_xyz_m=(-0.1, -0.1, 0.0),
        maximum_world_xyz_m=(0.5, 2.5, 2.5),
    )

    cloud = backproject_static_depth(
        depth_m=depth,
        confidence=confidence,
        colors_rgb=colors,
        intrinsics=intrinsics,
        pose=pose,
        confidence_threshold=2.0,
        room_bounds=bounds,
    )

    assert cloud.points_world_m == pytest.approx(np.array([[0.0, 0.0, 1.0]]))
    assert cloud.colors_rgb.tolist() == [[10, 20, 30]]
    assert cloud.stats.total_pixel_count == 4
    assert cloud.stats.valid_depth_count == 3
    assert cloud.stats.finite_confidence_count == 4
    assert cloud.stats.confidence_retained_count == 2
    assert cloud.stats.room_bounds_retained_count == 1


def test_backprojection_returns_empty_cloud_without_placeholder_xyz() -> None:
    intrinsics, pose = identity_camera()
    cloud = backproject_static_depth(
        depth_m=np.ones((2, 2)),
        confidence=np.zeros((2, 2)),
        colors_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        intrinsics=intrinsics,
        pose=pose,
        confidence_threshold=1.0,
        room_bounds=AxisAlignedBounds(
            minimum_world_xyz_m=(-1.0, -1.0, 0.0),
            maximum_world_xyz_m=(1.0, 1.0, 2.0),
        ),
    )

    assert cloud.points_world_m.shape == (0, 3)
    assert cloud.colors_rgb.shape == (0, 3)
    assert cloud.stats.room_bounds_retained_count == 0


def test_voxel_downsample_is_deterministic_and_averages_colors() -> None:
    points = np.array(
        [
            [0.001, 0.001, 0.001],
            [0.009, 0.009, 0.009],
            [0.021, 0.0, 0.0],
        ]
    )
    colors = np.array([[10, 20, 30], [30, 40, 50], [100, 110, 120]], dtype=np.uint8)

    first_points, first_colors = voxel_downsample(
        points,
        colors,
        voxel_size_m=0.02,
    )
    second_points, second_colors = voxel_downsample(
        points[::-1],
        colors[::-1],
        voxel_size_m=0.02,
    )

    assert first_points == pytest.approx(np.array([[0.005, 0.005, 0.005], [0.021, 0.0, 0.0]]))
    assert first_colors.tolist() == [[20, 30, 40], [100, 110, 120]]
    assert second_points == pytest.approx(first_points)
    assert second_colors.tolist() == first_colors.tolist()


def test_write_ply_round_trips_vertex_count_and_color(tmp_path: Path) -> None:
    path = tmp_path / "cloud.ply"
    points = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]])
    colors = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)

    write_colored_ply(path, points_world_m=points, colors_rgb=colors)
    loaded = PlyData.read(path)

    assert loaded["vertex"].count == 2
    assert loaded["vertex"]["x"].tolist() == pytest.approx([1.0, 0.0])
    assert loaded["vertex"]["red"].tolist() == [10, 40]


def test_room_bounds_reject_invalid_or_reversed_values() -> None:
    with pytest.raises(ValueError, match="greater"):
        AxisAlignedBounds(
            minimum_world_xyz_m=(0.0, 0.0, 0.0),
            maximum_world_xyz_m=(0.0, 1.0, 1.0),
        )
    with pytest.raises(ValueError, match="finite"):
        AxisAlignedBounds(
            minimum_world_xyz_m=(0.0, 0.0, 0.0),
            maximum_world_xyz_m=(1.0, 1.0, np.inf),
        )
