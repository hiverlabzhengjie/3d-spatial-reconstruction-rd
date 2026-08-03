"""Marker-anchored scaling for derived S04 action-pair depth."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Literal, Self

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import field_validator, model_validator

from spatial_reconstruction.contracts import ContractModel, PositiveInt

Float64Array = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]


class ActionDepthScaleUnavailableError(ValueError):
    """Raised when a pair cannot satisfy the marker-scaling acceptance gate."""


class ActionPairScalePolicy(ContractModel):
    """Fixed D025 acceptance policy for one synchronized S04 action pair."""

    policy_id: Literal["d025_action_pair_marker_scale_v1"] = (
        "d025_action_pair_marker_scale_v1"
    )
    marker_ids: tuple[int, ...] = (40, 41, 42)
    marker_length_m: float = 0.18
    protected_inner_fraction: float = 0.60
    maximum_marker_reprojection_error_px: float = 5.0
    maximum_marker_scale_relative_deviation: float = 0.05
    minimum_markers_per_camera: PositiveInt = 2
    minimum_total_marker_observations: PositiveInt = 5
    minimum_valid_samples_per_marker: PositiveInt = 16
    shared_scale_per_pair: Literal[True] = True
    raw_da3_depth_preserved: Literal[True] = True
    camera_specific_fallback_allowed: Literal[False] = False

    @field_validator(
        "marker_length_m",
        "protected_inner_fraction",
        "maximum_marker_reprojection_error_px",
        "maximum_marker_scale_relative_deviation",
    )
    @classmethod
    def validate_positive_finite(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("action-pair scale thresholds must be finite and positive")
        return value

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if not self.marker_ids or len(set(self.marker_ids)) != len(self.marker_ids):
            raise ValueError("marker IDs must be non-empty and unique")
        if self.protected_inner_fraction > 1:
            raise ValueError("protected marker fraction cannot exceed one")
        maximum_observations = len(self.marker_ids) * 2
        if self.minimum_total_marker_observations > maximum_observations:
            raise ValueError("minimum marker observations exceeds two-camera capacity")
        if self.minimum_markers_per_camera > len(self.marker_ids):
            raise ValueError("minimum markers per camera exceeds configured markers")
        return self


class ActionMarkerScaleObservation(ContractModel):
    """One detected floor marker's robust DA3 scale evidence."""

    camera_id: Literal["camera_a", "camera_b"]
    marker_id: int
    detected_center_uv: tuple[float, float]
    projected_center_uv: tuple[float, float]
    reprojection_error_px: float
    valid_sample_count: PositiveInt
    expected_camera_depth_median: float
    raw_da3_depth_median: float
    expected_over_raw_ratio: float
    ratio_mad: float

    @field_validator(
        "reprojection_error_px",
        "expected_camera_depth_median",
        "raw_da3_depth_median",
        "expected_over_raw_ratio",
        "ratio_mad",
    )
    @classmethod
    def validate_finite_nonnegative(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("marker observation values must be finite and non-negative")
        return value

    @model_validator(mode="after")
    def validate_positive_depth_and_ratio(self) -> Self:
        if min(
            self.expected_camera_depth_median,
            self.raw_da3_depth_median,
            self.expected_over_raw_ratio,
        ) <= 0:
            raise ValueError("marker depth and scale ratio must be positive")
        return self


class ActionPairScaleEstimate(ContractModel):
    """Accepted single scale shared by both cameras in one synchronized pair."""

    policy_id: Literal["d025_action_pair_marker_scale_v1"]
    scale: float
    maximum_relative_deviation: float
    marker_count_by_camera: dict[str, PositiveInt]
    observations: tuple[ActionMarkerScaleObservation, ...]


def estimate_action_pair_scale(
    observations: Sequence[ActionMarkerScaleObservation],
    *,
    policy: ActionPairScalePolicy,
) -> ActionPairScaleEstimate:
    """Apply the all-or-nothing D025 action-pair marker acceptance gate."""

    if len({(item.camera_id, item.marker_id) for item in observations}) != len(
        observations
    ):
        raise ActionDepthScaleUnavailableError("marker evidence contains duplicates")
    allowed = set(policy.marker_ids)
    if any(item.marker_id not in allowed for item in observations):
        raise ActionDepthScaleUnavailableError("marker evidence contains an excluded ID")
    if any(
        item.reprojection_error_px > policy.maximum_marker_reprojection_error_px
        for item in observations
    ):
        raise ActionDepthScaleUnavailableError("marker reprojection error exceeds D025 gate")
    if any(
        item.valid_sample_count < policy.minimum_valid_samples_per_marker
        for item in observations
    ):
        raise ActionDepthScaleUnavailableError("marker has insufficient valid depth samples")

    counts = Counter(item.camera_id for item in observations)
    for camera_id in ("camera_a", "camera_b"):
        if counts[camera_id] < policy.minimum_markers_per_camera:
            raise ActionDepthScaleUnavailableError(
                f"{camera_id} has insufficient accepted marker evidence"
            )
    if len(observations) < policy.minimum_total_marker_observations:
        raise ActionDepthScaleUnavailableError(
            "pair has insufficient total accepted marker evidence"
        )

    ratios = np.asarray(
        [item.expected_over_raw_ratio for item in observations], dtype=np.float64
    )
    if not np.isfinite(ratios).all() or np.any(ratios <= 0):
        raise ActionDepthScaleUnavailableError("marker scale ratios are not finite-positive")
    scale = float(np.median(ratios))
    relative = np.abs(ratios - scale) / scale
    maximum_deviation = float(np.max(relative))
    if maximum_deviation > policy.maximum_marker_scale_relative_deviation:
        raise ActionDepthScaleUnavailableError("marker scale ratios disagree beyond D025 gate")
    return ActionPairScaleEstimate(
        policy_id=policy.policy_id,
        scale=scale,
        maximum_relative_deviation=maximum_deviation,
        marker_count_by_camera={camera_id: counts[camera_id] for camera_id in counts},
        observations=tuple(observations),
    )


def sample_floor_marker_scale(
    *,
    depth_m: ArrayLike,
    processed_intrinsics: ArrayLike,
    T_world_from_camera: ArrayLike,
    marker_center_world_m: tuple[float, float, float],
    marker_length_m: float,
    protected_inner_fraction: float,
) -> tuple[int, float, float, float, float, UInt8Array]:
    """Sample expected/raw camera-Z ratios inside a known floor-marker square."""

    depth = np.asarray(depth_m, dtype=np.float64)
    intrinsics = np.asarray(processed_intrinsics, dtype=np.float64)
    world_from_camera = np.asarray(T_world_from_camera, dtype=np.float64)
    if depth.ndim != 2 or intrinsics.shape != (3, 3) or world_from_camera.shape != (4, 4):
        raise ValueError("marker sampling inputs have invalid shapes")
    if not 0 < protected_inner_fraction <= 1 or marker_length_m <= 0:
        raise ValueError("marker sampling geometry must be positive")

    half = marker_length_m * protected_inner_fraction / 2
    center = np.asarray(marker_center_world_m, dtype=np.float64)
    corners_world = center + np.asarray(
        [[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]],
        dtype=np.float64,
    )
    camera_from_world = np.linalg.inv(world_from_camera)
    corners_camera = (
        camera_from_world[:3, :3] @ corners_world.T
        + camera_from_world[:3, 3:4]
    ).T
    if np.any(corners_camera[:, 2] <= 0):
        raise ValueError("marker square is not in front of the camera")
    projected = (intrinsics @ corners_camera.T).T
    polygon = projected[:, :2] / projected[:, 2:3]
    mask = np.zeros(depth.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), (1,))

    ys, xs = np.nonzero(mask)
    rays_camera = np.linalg.inv(intrinsics) @ np.vstack(
        (xs.astype(np.float64), ys.astype(np.float64), np.ones(xs.size))
    )
    directions_world = world_from_camera[:3, :3] @ rays_camera
    camera_center_world = world_from_camera[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_camera_z = (
            -camera_center_world[2] / directions_world[2]
        )
    raw = depth[ys, xs]
    valid = (
        np.isfinite(raw)
        & (raw > 0)
        & np.isfinite(expected_camera_z)
        & (expected_camera_z > 0)
    )
    ratios = expected_camera_z[valid] / raw[valid]
    if ratios.size == 0:
        raise ActionDepthScaleUnavailableError("marker interior has no valid depth samples")
    median_ratio = float(np.median(ratios))
    ratio_mad = float(np.median(np.abs(ratios - median_ratio)))
    return (
        int(ratios.size),
        float(np.median(expected_camera_z[valid])),
        float(np.median(raw[valid])),
        median_ratio,
        ratio_mad,
        mask,
    )
