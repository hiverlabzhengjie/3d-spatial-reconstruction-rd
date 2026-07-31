"""Deterministic static-scene point-cloud construction from metric depth."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise, product
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from plyfile import PlyData, PlyElement  # type: ignore[import-untyped]
from pydantic import field_validator, model_validator

from spatial_reconstruction.contracts import (
    CameraIntrinsics,
    CameraPose,
    ContractModel,
    FrameBundleStatus,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    Sha256Digest,
    SynchronizedFrameBundle,
    Vector3,
)
from spatial_reconstruction.geometry.projection import (
    backproject_pixels,
    project_camera_points,
)
from spatial_reconstruction.geometry.transforms import (
    camera_points_to_world,
    world_points_to_camera,
)

FloatArray = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]
IntArray = NDArray[np.int64]


class StaticSceneWindow(ContractModel):
    """Accepted synchronized time interval for static reconstruction."""

    start: NonNegativeFloat
    end: PositiveFloat

    @model_validator(mode="after")
    def require_positive_extent(self) -> Self:
        if self.end <= self.start:
            raise ValueError("static-scene window end must exceed its start")
        return self


class StaticSceneSelectionRecord(ContractModel):
    """Persistent deterministic keyframe-selection parameters."""

    accepted_window_seconds: StaticSceneWindow
    target_times_seconds: tuple[NonNegativeFloat, ...]
    selected_bundle_count: PositiveInt
    maximum_target_error_seconds: NonNegativeFloat

    @model_validator(mode="after")
    def require_matching_ordered_targets(self) -> Self:
        if len(self.target_times_seconds) != self.selected_bundle_count:
            raise ValueError("selected bundle count must match target-time count")
        if tuple(sorted(set(self.target_times_seconds))) != self.target_times_seconds:
            raise ValueError("static-scene target times must be unique and increasing")
        if any(
            value < self.accepted_window_seconds.start
            or value > self.accepted_window_seconds.end
            for value in self.target_times_seconds
        ):
            raise ValueError("static-scene target times must remain in the window")
        return self


class StaticSceneInputProvenance(ContractModel):
    """Immutable accepted-input references and hashes."""

    synchronization_manifest_ref: str
    synchronization_manifest_sha256: Sha256Digest
    pose_calibration_ref: str
    pose_calibration_sha256: Sha256Digest
    scene_metadata_ref: str
    scene_metadata_sha256: Sha256Digest
    vendor_fingerprint: Sha256Digest

    @field_validator(
        "synchronization_manifest_ref",
        "pose_calibration_ref",
        "scene_metadata_ref",
    )
    @classmethod
    def require_non_empty_refs(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("static-scene artifact references must be non-empty")
        return value


class StaticSceneModelRecord(ContractModel):
    """Exact DA3 identity and execution mode used for a static run."""

    model_id: str
    model_revision: str
    device: Literal["mps"]
    precision: Literal["float16", "float32", "bfloat16"]
    is_metric: Literal[True]
    two_view_alignment_policy: Literal[
        "preserve_nested_metric_depth_and_return_supplied_poses"
    ]

    @field_validator("model_revision")
    @classmethod
    def require_git_revision(cls, value: str) -> str:
        if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("DA3 model revision must be a 40-character lowercase hex ID")
        return value


class StaticScenePredictionRecord(ContractModel):
    """One retained synchronized DA3 prediction and its essential provenance."""

    bundle_id: Sha256Digest
    bundle_index: NonNegativeInt
    capture_timestamp_seconds: NonNegativeFloat
    maximum_frame_time_difference_seconds: NonNegativeFloat
    raw_prediction_ref: str
    raw_prediction_sha256: Sha256Digest
    depth_confidence_preview_ref: str
    depth_confidence_preview_sha256: Sha256Digest
    confidence_percentile: NonNegativeFloat
    confidence_threshold: float
    marker_depth_scale_correction: dict[str, Any]
    cameras: dict[str, dict[str, Any]]

    @model_validator(mode="after")
    def require_camera_pair(self) -> Self:
        if set(self.cameras) != {"camera_a", "camera_b"}:
            raise ValueError("static-scene prediction must contain camera A and B")
        return self


class StaticSceneXYZRange(ContractModel):
    """Finite world-space extent of one retained point cloud."""

    minimum: Vector3
    maximum: Vector3

    @model_validator(mode="after")
    def require_ordered_extent(self) -> Self:
        if any(high < low for low, high in zip(self.minimum, self.maximum, strict=True)):
            raise ValueError("static-scene XYZ maximum must not be below minimum")
        return self


class StaticScenePointCloudRecord(ContractModel):
    """Persistent PLY identity, size, and metric extent."""

    pre_voxel_point_count: PositiveInt
    point_count: PositiveInt
    ply_ref: str
    ply_sha256: Sha256Digest
    world_xyz_range_m: StaticSceneXYZRange

    @model_validator(mode="after")
    def require_nonexpanding_voxelization(self) -> Self:
        if self.point_count > self.pre_voxel_point_count:
            raise ValueError("voxelization cannot increase point count")
        return self


class StaticScenePreviewRecord(ContractModel):
    """Inspectable static-scene preview artifacts."""

    geometry_png_ref: str
    geometry_png_sha256: Sha256Digest
    glb_ref: str
    glb_sha256: Sha256Digest


class StaticSceneRunSummary(ContractModel):
    """Strict persistent S02 reconstruction summary contract."""

    schema_version: Literal[1]
    status: Literal["completed_pending_visual_qa"]
    stage: Literal["S02"]
    created_at_utc: datetime
    capture_session_id: str
    pose_version_id: str
    world_frame: dict[str, str]
    input_provenance: StaticSceneInputProvenance
    model: StaticSceneModelRecord
    selection: StaticSceneSelectionRecord
    processing: dict[str, Any]
    predictions: tuple[StaticScenePredictionRecord, ...]
    point_clouds: dict[str, StaticScenePointCloudRecord]
    previews: StaticScenePreviewRecord
    runtime: dict[str, Any]
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def require_complete_pair_artifacts(self) -> Self:
        if len(self.predictions) != self.selection.selected_bundle_count:
            raise ValueError("prediction count must match selected bundle count")
        if set(self.point_clouds) != {"camera_a", "camera_b", "fused"}:
            raise ValueError("point-cloud summary must contain both cameras and fused")
        if len({item.bundle_id for item in self.predictions}) != len(self.predictions):
            raise ValueError("static-scene prediction bundle IDs must be unique")
        return self


class StaticSceneRerunExportSummary(ContractModel):
    """Strict persistent summary for one S02 Rerun recording."""

    schema_version: Literal[1]
    status: Literal["passed"]
    stage: Literal["S02"]
    rerun_sdk_version: str
    source_summary_ref: str
    source_summary_sha256: Sha256Digest
    recording_ref: str
    recording_sha256: Sha256Digest
    recording_bytes: PositiveInt
    source_fused_point_count: PositiveInt
    source_camera_point_counts: dict[str, PositiveInt]
    logged_fused_point_count: PositiveInt
    logged_camera_point_counts: dict[str, PositiveInt]
    logged_camera_ids: tuple[str, ...]
    maximum_points_per_rerun_entity: PositiveInt
    camera_image_maximum_dimension: PositiveInt | None
    world_coordinates: Literal["right-handed, metres, Z up"]
    camera_coordinates: Literal["OpenCV X right, Y down, Z forward"]
    includes: tuple[str, ...]

    @model_validator(mode="after")
    def require_bounded_camera_pair(self) -> Self:
        expected = {"camera_a", "camera_b"}
        if (
            set(self.source_camera_point_counts) != expected
            or set(self.logged_camera_point_counts) != expected
            or set(self.logged_camera_ids) != expected
        ):
            raise ValueError("Rerun summary must contain camera A and B")
        if self.logged_fused_point_count > self.source_fused_point_count:
            raise ValueError("Rerun fused sample cannot exceed its source")
        if any(
            self.logged_camera_point_counts[camera_id]
            > self.source_camera_point_counts[camera_id]
            for camera_id in expected
        ):
            raise ValueError("Rerun camera sample cannot exceed its source")
        return self


@dataclass(frozen=True, slots=True)
class AxisAlignedBounds:
    """Finite world-space axis-aligned filtering bounds in metres."""

    minimum_world_xyz_m: tuple[float, float, float]
    maximum_world_xyz_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum_world_xyz_m, dtype=np.float64)
        maximum = np.asarray(self.maximum_world_xyz_m, dtype=np.float64)
        if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
            raise ValueError("room bounds must contain only finite values")
        if np.any(maximum <= minimum):
            raise ValueError("room-bound maxima must be greater than minima")

    def contains(self, points_world: ArrayLike) -> NDArray[np.bool_]:
        """Return a row mask selecting finite points inside the closed bounds."""

        points = _point_matrix(points_world)
        minimum = np.asarray(self.minimum_world_xyz_m, dtype=np.float64)
        maximum = np.asarray(self.maximum_world_xyz_m, dtype=np.float64)
        return np.asarray(
            np.isfinite(points).all(axis=1)
            & np.all(points >= minimum, axis=1)
            & np.all(points <= maximum, axis=1),
            dtype=np.bool_,
        )


@dataclass(frozen=True, slots=True)
class PointFilteringStats:
    """Counts that make invalid/confidence/bounds filtering inspectable."""

    total_pixel_count: int
    valid_depth_count: int
    finite_confidence_count: int
    confidence_retained_count: int
    room_bounds_retained_count: int


@dataclass(frozen=True, slots=True)
class ColoredPointCloud:
    """One finite world-space colored point cloud and filtering evidence."""

    points_world_m: FloatArray
    colors_rgb: UInt8Array
    stats: PointFilteringStats

    def __post_init__(self) -> None:
        points = _point_matrix(self.points_world_m)
        colors = np.asarray(self.colors_rgb)
        if colors.dtype != np.uint8 or colors.shape != (points.shape[0], 3):
            raise ValueError("point colors must be uint8 RGB with shape (N, 3)")
        if not np.isfinite(points).all():
            raise ValueError("point cloud must contain only finite world points")
        object.__setattr__(self, "points_world_m", points)
        object.__setattr__(self, "colors_rgb", colors.copy())


@dataclass(frozen=True, slots=True)
class DepthScaleObservation:
    """One known-world marker depth compared with one DA3 depth patch."""

    camera_id: str
    marker_id: int
    pixel_uv: tuple[float, float]
    expected_camera_depth_m: float
    predicted_depth_m: float
    expected_over_predicted_ratio: float


@dataclass(frozen=True, slots=True)
class MarkerDepthScaleEstimate:
    """Robust shared scalar correction with inspectable marker evidence."""

    scale: float
    maximum_relative_deviation: float
    observations: tuple[DepthScaleObservation, ...]


def select_keyframe_bundles(
    bundles: tuple[SynchronizedFrameBundle, ...],
    *,
    start_seconds: float,
    end_seconds: float,
    interval_seconds: float,
    maximum_target_error_seconds: float,
) -> tuple[SynchronizedFrameBundle, ...]:
    """Select nearest complete bundles at deterministic interval targets."""

    values = (start_seconds, end_seconds, interval_seconds)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("keyframe selection values must be finite")
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise ValueError("keyframe interval must have a non-negative positive extent")
    if interval_seconds <= 0:
        raise ValueError("selection interval must be positive")

    target_count = int(math.floor((end_seconds - start_seconds) / interval_seconds + 1e-9))
    targets = tuple(start_seconds + index * interval_seconds for index in range(target_count + 1))
    return select_bundles_for_target_times(
        bundles,
        target_times_seconds=targets,
        accepted_start_seconds=start_seconds,
        accepted_end_seconds=end_seconds,
        maximum_target_error_seconds=maximum_target_error_seconds,
    )


def select_bundles_for_target_times(
    bundles: tuple[SynchronizedFrameBundle, ...],
    *,
    target_times_seconds: tuple[float, ...],
    accepted_start_seconds: float,
    accepted_end_seconds: float,
    maximum_target_error_seconds: float,
) -> tuple[SynchronizedFrameBundle, ...]:
    """Select unique complete bundles nearest explicit accepted-window targets."""

    values = (
        *target_times_seconds,
        accepted_start_seconds,
        accepted_end_seconds,
        maximum_target_error_seconds,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("keyframe selection values must be finite")
    if accepted_start_seconds < 0 or accepted_end_seconds <= accepted_start_seconds:
        raise ValueError("accepted interval must have a non-negative positive extent")
    if maximum_target_error_seconds < 0:
        raise ValueError("maximum target error must be non-negative")
    if not target_times_seconds:
        raise ValueError("at least one keyframe target is required")
    if tuple(sorted(set(target_times_seconds))) != target_times_seconds:
        raise ValueError("keyframe targets must be unique and strictly increasing")
    if any(
        target < accepted_start_seconds or target > accepted_end_seconds
        for target in target_times_seconds
    ):
        raise ValueError("keyframe targets must remain inside the accepted interval")
    if not bundles:
        raise ValueError("keyframe selection requires frame bundles")

    bundle_times = [bundle.capture_timestamp_seconds for bundle in bundles]
    if any(current <= previous for previous, current in pairwise(bundle_times)):
        raise ValueError("frame bundles must have strictly increasing capture timestamps")

    candidates = tuple(
        bundle
        for bundle in bundles
        if bundle.status is FrameBundleStatus.COMPLETE
        and accepted_start_seconds
        <= bundle.capture_timestamp_seconds
        <= accepted_end_seconds
    )
    if not candidates:
        raise ValueError("accepted interval contains no complete frame bundles")

    selected: list[SynchronizedFrameBundle] = []
    used_ids: set[str] = set()
    for target in target_times_seconds:
        available = tuple(bundle for bundle in candidates if bundle.bundle_id not in used_ids)
        if not available:
            raise ValueError("not enough unique complete bundles for keyframe targets")
        nearest = min(
            available,
            key=lambda bundle: (
                abs(bundle.capture_timestamp_seconds - target),
                bundle.capture_timestamp_seconds,
                bundle.bundle_index,
            ),
        )
        error = abs(nearest.capture_timestamp_seconds - target)
        if error > maximum_target_error_seconds:
            raise ValueError(
                f"no complete bundle within {maximum_target_error_seconds:.6f}s "
                f"of target {target:.6f}s"
            )
        selected.append(nearest)
        used_ids.add(nearest.bundle_id)
    return tuple(selected)


def confidence_percentile_threshold(
    confidence: ArrayLike,
    *,
    percentile: float,
) -> float:
    """Compute the vendor-style adaptive threshold from finite confidence."""

    if not math.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError("confidence percentile must be between 0 and 100")
    values = np.asarray(confidence, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("confidence contains no finite samples")
    return float(np.percentile(finite, percentile))


def radius_overlap_fraction(
    query_points_world_m: ArrayLike,
    reference_points_world_m: ArrayLike,
    *,
    radius_m: float,
) -> float:
    """Return the exact fraction of query points within a radius of a reference."""

    query = _point_matrix(query_points_world_m)
    reference = _point_matrix(reference_points_world_m)
    if query.shape[0] == 0 or reference.shape[0] == 0:
        raise ValueError("overlap verification requires two non-empty point clouds")
    if not np.isfinite(query).all() or not np.isfinite(reference).all():
        raise ValueError("overlap verification requires finite point clouds")
    if not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("overlap radius must be finite and positive")

    reference_cells = np.floor(reference / radius_m).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[FloatArray]] = {}
    for cell, point in zip(reference_cells, reference, strict=True):
        key = (int(cell[0]), int(cell[1]), int(cell[2]))
        buckets.setdefault(key, []).append(point)

    radius_squared = radius_m * radius_m
    matched = 0
    for point in query:
        cell = np.floor(point / radius_m).astype(np.int64)
        candidates: list[FloatArray] = []
        for offset in product((-1, 0, 1), repeat=3):
            key = (
                int(cell[0]) + offset[0],
                int(cell[1]) + offset[1],
                int(cell[2]) + offset[2],
            )
            candidates.extend(buckets.get(key, ()))
        if candidates:
            candidate_matrix = np.asarray(candidates, dtype=np.float64)
            squared_distances = np.sum(
                np.square(candidate_matrix - point),
                axis=1,
            )
            if bool(np.any(squared_distances <= radius_squared)):
                matched += 1
    return matched / query.shape[0]


def estimate_marker_depth_scale(
    *,
    depth_m: ArrayLike,
    camera_intrinsics: tuple[CameraIntrinsics, ...],
    camera_poses: tuple[CameraPose, ...],
    marker_ids: tuple[int, ...],
    marker_centres_world_m: ArrayLike,
    patch_radius_pixels: int = 2,
    maximum_relative_deviation: float = 0.05,
) -> MarkerDepthScaleEstimate:
    """Estimate one shared depth scale from known floor-marker centres."""

    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 3:
        raise ValueError("marker scaling depth must have shape (N, H, W)")
    if len(camera_intrinsics) != depth.shape[0] or len(camera_poses) != depth.shape[0]:
        raise ValueError("marker scaling cameras must match the depth view count")
    if patch_radius_pixels < 0:
        raise ValueError("marker depth patch radius must be non-negative")
    if (
        not math.isfinite(maximum_relative_deviation)
        or maximum_relative_deviation < 0
    ):
        raise ValueError("maximum marker scale deviation must be finite and non-negative")
    centres = _point_matrix(marker_centres_world_m)
    if len(marker_ids) != centres.shape[0] or not marker_ids:
        raise ValueError("marker IDs and centres must have the same non-zero count")
    if len(set(marker_ids)) != len(marker_ids):
        raise ValueError("marker IDs must be unique")

    pose_by_id = {pose.camera_id: pose for pose in camera_poses}
    if len(pose_by_id) != len(camera_poses):
        raise ValueError("marker scaling camera pose IDs must be unique")
    observations: list[DepthScaleObservation] = []
    for view_index, intrinsics in enumerate(camera_intrinsics):
        pose = pose_by_id.get(intrinsics.camera_id)
        if pose is None:
            raise ValueError(f"missing marker scaling pose for {intrinsics.camera_id}")
        if (
            intrinsics.image_width != depth.shape[2]
            or intrinsics.image_height != depth.shape[1]
        ):
            raise ValueError("marker scaling intrinsics must match depth dimensions")
        points_camera = world_points_to_camera(centres, pose=pose)
        pixels_uv = project_camera_points(points_camera, intrinsics=intrinsics)
        for marker_id, pixel_uv, expected_depth in zip(
            marker_ids,
            pixels_uv,
            points_camera[:, 2],
            strict=True,
        ):
            u, v = (float(pixel_uv[0]), float(pixel_uv[1]))
            x = int(round(u))
            y = int(round(v))
            if (
                x - patch_radius_pixels < 0
                or y - patch_radius_pixels < 0
                or x + patch_radius_pixels >= depth.shape[2]
                or y + patch_radius_pixels >= depth.shape[1]
            ):
                raise ValueError(
                    f"marker {marker_id} projects outside the usable depth image"
                )
            patch = depth[
                view_index,
                y - patch_radius_pixels : y + patch_radius_pixels + 1,
                x - patch_radius_pixels : x + patch_radius_pixels + 1,
            ]
            valid_patch = patch[np.isfinite(patch) & (patch > 0)]
            if valid_patch.size == 0:
                raise ValueError(f"marker {marker_id} depth patch has no valid samples")
            predicted_depth = float(np.median(valid_patch))
            ratio = float(expected_depth / predicted_depth)
            if not math.isfinite(ratio) or ratio <= 0:
                raise ValueError("marker depth scale ratio must be finite and positive")
            observations.append(
                DepthScaleObservation(
                    camera_id=intrinsics.camera_id,
                    marker_id=marker_id,
                    pixel_uv=(u, v),
                    expected_camera_depth_m=float(expected_depth),
                    predicted_depth_m=predicted_depth,
                    expected_over_predicted_ratio=ratio,
                )
            )

    ratios = np.asarray(
        [observation.expected_over_predicted_ratio for observation in observations],
        dtype=np.float64,
    )
    scale = float(np.median(ratios))
    relative_deviations = np.abs(ratios - scale) / scale
    observed_maximum = float(np.max(relative_deviations))
    if observed_maximum > maximum_relative_deviation:
        raise ValueError(
            "marker depth scale observations disagree: "
            f"maximum relative deviation {observed_maximum:.6f} exceeds "
            f"{maximum_relative_deviation:.6f}"
        )
    return MarkerDepthScaleEstimate(
        scale=scale,
        maximum_relative_deviation=observed_maximum,
        observations=tuple(observations),
    )


def backproject_static_depth(
    *,
    depth_m: ArrayLike,
    confidence: ArrayLike,
    colors_rgb: ArrayLike,
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    confidence_threshold: float,
    room_bounds: AxisAlignedBounds,
) -> ColoredPointCloud:
    """Back-project valid confident depth into the calibrated world frame."""

    depth = np.asarray(depth_m, dtype=np.float64)
    scores = np.asarray(confidence, dtype=np.float64)
    colors = np.asarray(colors_rgb)
    if depth.ndim != 2:
        raise ValueError(f"static depth must have shape (H, W), got {depth.shape}")
    if scores.shape != depth.shape:
        raise ValueError("static confidence shape must match depth")
    if colors.dtype != np.uint8 or colors.shape != (*depth.shape, 3):
        raise ValueError("processed colors must be uint8 RGB matching depth dimensions")
    if not math.isfinite(confidence_threshold):
        raise ValueError("confidence threshold must be finite")
    if intrinsics.image_width != depth.shape[1] or intrinsics.image_height != depth.shape[0]:
        raise ValueError("processed intrinsics dimensions must match depth")
    if intrinsics.camera_id != pose.camera_id:
        raise ValueError("intrinsics and pose camera IDs must match")

    valid_depth = np.isfinite(depth) & (depth > 0)
    finite_confidence = np.isfinite(scores)
    confident = valid_depth & finite_confidence & (scores >= confidence_threshold)
    y_pixels, x_pixels = np.nonzero(confident)
    if x_pixels.size == 0:
        return ColoredPointCloud(
            points_world_m=np.empty((0, 3), dtype=np.float64),
            colors_rgb=np.empty((0, 3), dtype=np.uint8),
            stats=PointFilteringStats(
                total_pixel_count=int(depth.size),
                valid_depth_count=int(np.count_nonzero(valid_depth)),
                finite_confidence_count=int(np.count_nonzero(finite_confidence)),
                confidence_retained_count=0,
                room_bounds_retained_count=0,
            ),
        )

    pixels_uv = np.column_stack((x_pixels, y_pixels)).astype(np.float64)
    points_camera = backproject_pixels(
        pixels_uv,
        depth[confident],
        intrinsics=intrinsics,
    )
    points_world = camera_points_to_world(points_camera, pose=pose)
    in_bounds = room_bounds.contains(points_world)
    return ColoredPointCloud(
        points_world_m=np.asarray(points_world[in_bounds], dtype=np.float64),
        colors_rgb=np.asarray(colors[confident][in_bounds], dtype=np.uint8),
        stats=PointFilteringStats(
            total_pixel_count=int(depth.size),
            valid_depth_count=int(np.count_nonzero(valid_depth)),
            finite_confidence_count=int(np.count_nonzero(finite_confidence)),
            confidence_retained_count=int(np.count_nonzero(confident)),
            room_bounds_retained_count=int(np.count_nonzero(in_bounds)),
        ),
    )


def voxel_downsample(
    points_world_m: ArrayLike,
    colors_rgb: ArrayLike,
    *,
    voxel_size_m: float,
) -> tuple[FloatArray, UInt8Array]:
    """Deterministically average points and colors inside metric voxels."""

    if not math.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ValueError("voxel size must be finite and positive")
    points = _point_matrix(points_world_m)
    colors = np.asarray(colors_rgb)
    if colors.dtype != np.uint8 or colors.shape != (points.shape[0], 3):
        raise ValueError("voxel colors must be uint8 RGB with shape (N, 3)")
    if points.shape[0] == 0:
        return points, colors.copy()

    voxel_keys: IntArray = np.floor(points / voxel_size_m).astype(np.int64)
    order = np.lexsort((voxel_keys[:, 2], voxel_keys[:, 1], voxel_keys[:, 0]))
    sorted_keys = voxel_keys[order]
    sorted_points = points[order]
    sorted_colors = colors[order].astype(np.float64)
    starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(np.any(np.diff(sorted_keys, axis=0) != 0, axis=1)) + 1,
        )
    )
    ends = np.concatenate((starts[1:], np.array([points.shape[0]], dtype=np.int64)))
    counts = (ends - starts).astype(np.float64)
    point_sums = np.add.reduceat(sorted_points, starts, axis=0)
    color_sums = np.add.reduceat(sorted_colors, starts, axis=0)
    downsampled_points = np.asarray(point_sums / counts[:, None], dtype=np.float64)
    downsampled_colors = np.asarray(
        np.clip(np.rint(color_sums / counts[:, None]), 0, 255),
        dtype=np.uint8,
    )
    return downsampled_points, downsampled_colors


def write_colored_ply(
    path: Path,
    *,
    points_world_m: ArrayLike,
    colors_rgb: ArrayLike,
) -> None:
    """Write one binary PLY with metric XYZ and RGB vertex properties."""

    points = _point_matrix(points_world_m)
    colors = np.asarray(colors_rgb)
    if colors.dtype != np.uint8 or colors.shape != (points.shape[0], 3):
        raise ValueError("PLY colors must be uint8 RGB with shape (N, 3)")
    vertices = np.empty(
        points.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def _point_matrix(points_world_m: ArrayLike) -> FloatArray:
    points = np.asarray(points_world_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"world points must have shape (N, 3), got {points.shape}")
    return points.copy()
