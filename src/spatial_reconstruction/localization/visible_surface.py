"""Exact-frame per-camera visible-surface localization for S04."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import field_validator, model_validator

from spatial_reconstruction.contracts import (
    CameraIntrinsics,
    CameraPose,
    ContractModel,
    FiniteFloat,
    NonNegativeFloat,
    NonNegativeInt,
    PerceptionTarget,
    PositiveInt,
    Sha256Digest,
    Vector3,
)
from spatial_reconstruction.geometry import (
    backproject_pixels,
    camera_points_to_world,
    project_camera_points,
    world_points_to_camera,
)
from spatial_reconstruction.localization.mask_depth_diagnostics import (
    InsufficientCandidateSamplesError,
    MaskDepthDiagnosticConfig,
    MaskDepthStrategy,
    SampleDistribution,
    TargetVisibleSurfaceRule,
    build_mask_depth_candidates,
    select_candidate_relative_confidence,
)

Float64Array = NDArray[np.float64]


class VisibleSurfaceAvailability(StrEnum):
    """Availability of one raw per-camera visible-surface measurement."""

    OBSERVED = "observed"
    UNAVAILABLE = "unavailable"


class VisibleSurfaceUnavailableReason(StrEnum):
    """Why an exact-frame mask did not yield an XYZ surface measurement."""

    INVALID_OR_INSUFFICIENT_SAMPLES = "invalid_or_insufficient_samples"


class ExactFrameDepthJoin(ContractModel):
    """Strict identity join between one aligned mask and one DA3 depth plane."""

    action_depth_job_id_from_mask: Sha256Digest
    action_depth_job_id_from_depth: Sha256Digest
    bundle_id_from_mask: Sha256Digest
    bundle_id_from_depth: Sha256Digest
    frame_id_from_mask: Sha256Digest
    frame_id_from_depth: Sha256Digest
    camera_id_from_mask: str
    camera_id_from_depth: str
    capture_timestamp_seconds_from_mask: NonNegativeFloat
    capture_timestamp_seconds_from_depth: NonNegativeFloat
    timestamp_difference_seconds: NonNegativeFloat
    maximum_timestamp_difference_seconds: NonNegativeFloat = 0.0
    match_kind: Literal["exact_frame_identity"] = "exact_frame_identity"
    worker_completion_order_used: Literal[False] = False

    @field_validator("camera_id_from_mask", "camera_id_from_depth")
    @classmethod
    def validate_camera_text(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("join camera IDs must be non-empty and trimmed")
        return value

    @model_validator(mode="after")
    def validate_exact_identity(self) -> Self:
        pairs = (
            (
                self.action_depth_job_id_from_mask,
                self.action_depth_job_id_from_depth,
                "action-depth job",
            ),
            (self.bundle_id_from_mask, self.bundle_id_from_depth, "bundle"),
            (self.frame_id_from_mask, self.frame_id_from_depth, "frame"),
            (self.camera_id_from_mask, self.camera_id_from_depth, "camera"),
        )
        for mask_value, depth_value, name in pairs:
            if mask_value != depth_value:
                raise ValueError(f"exact mask/depth join has mismatched {name} identity")
        actual_difference = abs(
            self.capture_timestamp_seconds_from_mask
            - self.capture_timestamp_seconds_from_depth
        )
        if not math.isclose(
            self.timestamp_difference_seconds,
            actual_difference,
            abs_tol=1e-12,
        ):
            raise ValueError("join timestamp difference is inconsistent")
        if actual_difference != 0:
            raise ValueError("exact frame identity join requires identical timestamps")
        if self.maximum_timestamp_difference_seconds != 0:
            raise ValueError("exact frame identity join must use zero timestamp tolerance")
        return self

    @property
    def camera_id(self) -> str:
        return self.camera_id_from_mask


@dataclass(frozen=True, slots=True)
class VisibleSurfaceLocalization:
    """Runtime result before persistent artifact references are assigned."""

    availability: VisibleSurfaceAvailability
    unavailable_reason: VisibleSurfaceUnavailableReason | None
    strategy: MaskDepthStrategy
    candidate_pixel_count: int
    valid_candidate_count: int
    retained_sample_count: int
    confidence_threshold: float | None
    pixels_uv: Float64Array
    depth_m: Float64Array
    confidence: Float64Array
    points_camera_m: Float64Array
    points_world_m: Float64Array
    median_pixel_uv: tuple[float, float] | None
    aggregate_camera_xyz_m: tuple[float, float, float] | None
    aggregate_world_xyz_m: tuple[float, float, float] | None
    sample_reprojection_max_error_px: float | None
    world_camera_round_trip_max_error_m: float | None

    def __post_init__(self) -> None:
        arrays = {
            "pixels_uv": (self.pixels_uv, 2),
            "depth_m": (self.depth_m, 1),
            "confidence": (self.confidence, 1),
            "points_camera_m": (self.points_camera_m, 3),
            "points_world_m": (self.points_world_m, 3),
        }
        normalized: dict[str, Float64Array] = {}
        for name, (value, columns) in arrays.items():
            array = np.asarray(value, dtype=np.float64)
            if columns == 1:
                if array.ndim != 1:
                    raise ValueError(f"{name} must be one-dimensional")
            elif array.ndim != 2 or array.shape[1] != columns:
                raise ValueError(f"{name} must have shape (N, {columns})")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values")
            array.setflags(write=False)
            normalized[name] = array
            object.__setattr__(self, name, array)
        counts = {array.shape[0] for array in normalized.values()}
        if counts != {self.retained_sample_count}:
            raise ValueError("visible-surface arrays must match retained sample count")

        if self.availability is VisibleSurfaceAvailability.OBSERVED:
            if self.unavailable_reason is not None or self.retained_sample_count <= 0:
                raise ValueError("observed surface has invalid availability fields")
            required = (
                self.confidence_threshold,
                self.median_pixel_uv,
                self.aggregate_camera_xyz_m,
                self.aggregate_world_xyz_m,
                self.sample_reprojection_max_error_px,
                self.world_camera_round_trip_max_error_m,
            )
            if any(value is None for value in required):
                raise ValueError("observed surface requires aggregate and QA values")
        else:
            if self.unavailable_reason is None or self.retained_sample_count != 0:
                raise ValueError("unavailable surface requires a reason and no samples")
            forbidden = (
                self.confidence_threshold,
                self.median_pixel_uv,
                self.aggregate_camera_xyz_m,
                self.aggregate_world_xyz_m,
                self.sample_reprojection_max_error_px,
                self.world_camera_round_trip_max_error_m,
            )
            if any(value is not None for value in forbidden):
                raise ValueError("unavailable surface cannot carry XYZ or QA values")


class VisibleSurfaceObservationRecord(ContractModel):
    """One persistent raw per-camera visible-surface XYZ observation."""

    schema_version: Literal[1] = 1
    observation_id: Sha256Digest
    state: Literal["observed"] = "observed"
    action_depth_job_id: Sha256Digest
    bundle_id: Sha256Digest
    frame_id: Sha256Digest
    capture_timestamp_seconds: NonNegativeFloat
    source_frame_index: NonNegativeInt
    phase_id: str
    camera_id: str
    target: PerceptionTarget
    perception_job_id: Sha256Digest
    camera_local_track_id: str
    pose_version_id: str
    join: ExactFrameDepthJoin
    policy_id: Literal["s04_dynamic_visible_surface_v1"]
    candidate_strategy: MaskDepthStrategy
    source_mask_pixel_count: PositiveInt
    candidate_pixel_count: PositiveInt
    valid_candidate_count: PositiveInt
    retained_sample_count: PositiveInt
    candidate_confidence_percentile: NonNegativeFloat
    candidate_confidence_threshold: NonNegativeFloat
    retained_depth_m: SampleDistribution
    retained_confidence: SampleDistribution
    median_pixel_uv: tuple[FiniteFloat, FiniteFloat]
    aggregate_method: Literal["componentwise_median_camera_xyz"]
    aggregate_camera_xyz_m: Vector3
    aggregate_world_xyz_m: Vector3
    processed_intrinsics: CameraIntrinsics
    camera_pose: CameraPose
    raw_prediction_ref: str
    raw_prediction_sha256: Sha256Digest
    aligned_mask_artifact_ref: str
    aligned_mask_artifact_sha256: Sha256Digest
    aligned_mask_index: NonNegativeInt
    sample_cloud_ref: str
    sample_cloud_sha256: Sha256Digest
    image_diagnostic_ref: str
    image_diagnostic_sha256: Sha256Digest
    sample_reprojection_max_error_px: NonNegativeFloat
    world_camera_round_trip_max_error_m: NonNegativeFloat
    returned_pose_maximum_absolute_error: NonNegativeFloat
    aggregate_inside_room_bounds: bool
    sample_inside_room_fraction: NonNegativeFloat
    coordinate_semantics: str
    anchor_derived: Literal[False] = False
    camera_fusion_performed: Literal[False] = False
    presentation_smoothing_performed: Literal[False] = False
    s02_correction_applied: Literal[False] = False

    @field_validator(
        "phase_id",
        "camera_id",
        "camera_local_track_id",
        "pose_version_id",
        "raw_prediction_ref",
        "aligned_mask_artifact_ref",
        "sample_cloud_ref",
        "image_diagnostic_ref",
        "coordinate_semantics",
    )
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("visible-surface text must be non-empty and trimmed")
        return value

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        if self.join.camera_id != self.camera_id:
            raise ValueError("visible-surface join camera differs from observation")
        if self.processed_intrinsics.camera_id != self.camera_id:
            raise ValueError("processed intrinsics camera differs from observation")
        if self.camera_pose.camera_id != self.camera_id:
            raise ValueError("camera pose differs from observation")
        if self.retained_depth_m.count != self.retained_sample_count:
            raise ValueError("retained depth count differs from sample count")
        if self.retained_confidence.count != self.retained_sample_count:
            raise ValueError("retained confidence count differs from sample count")
        if self.candidate_confidence_percentile != 20:
            raise ValueError("visible-surface observation must apply D030 p20")
        if not 0 <= self.sample_inside_room_fraction <= 1:
            raise ValueError("room-bound sample fraction must be within zero and one")
        identity = {
            "schema_version": self.schema_version,
            "action_depth_job_id": self.action_depth_job_id,
            "bundle_id": self.bundle_id,
            "frame_id": self.frame_id,
            "camera_id": self.camera_id,
            "target": self.target.value,
            "perception_job_id": self.perception_job_id,
            "policy_id": self.policy_id,
        }
        if self.observation_id != _stable_digest(identity):
            raise ValueError("visible-surface observation ID differs from identity")
        return self

    @classmethod
    def create_observation_id(
        cls,
        *,
        action_depth_job_id: str,
        bundle_id: str,
        frame_id: str,
        camera_id: str,
        target: PerceptionTarget,
        perception_job_id: str,
        policy_id: str,
    ) -> str:
        return _stable_digest(
            {
                "schema_version": 1,
                "action_depth_job_id": action_depth_job_id,
                "bundle_id": bundle_id,
                "frame_id": frame_id,
                "camera_id": camera_id,
                "target": target.value,
                "perception_job_id": perception_job_id,
                "policy_id": policy_id,
            }
        )


class VisibleSurfaceRunSummary(ContractModel):
    """Persistent result of exact-frame per-camera visible-surface localization."""

    schema_version: Literal[1]
    status: Literal["completed_pending_visual_qa"]
    stage: Literal["S04"]
    created_at_utc: datetime
    source_policy_selection_ref: str
    source_policy_selection_sha256: Sha256Digest
    source_mask_alignment_summary_ref: str
    source_mask_alignment_summary_sha256: Sha256Digest
    source_action_depth_summary_ref: str
    source_action_depth_summary_sha256: Sha256Digest
    pose_calibration_ref: str
    pose_calibration_sha256: Sha256Digest
    scene_metadata_ref: str
    scene_metadata_sha256: Sha256Digest
    room_bounds_world_m: dict[str, Any]
    observations: tuple[VisibleSurfaceObservationRecord, ...]
    contact_sheet_ref: str
    contact_sheet_sha256: Sha256Digest
    world_preview_ref: str
    world_preview_sha256: Sha256Digest
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        if not self.observations:
            raise ValueError("visible-surface run must contain observations")
        keys = {
            (item.action_depth_job_id, item.camera_id, item.target)
            for item in self.observations
        }
        if len(keys) != len(self.observations):
            raise ValueError("visible-surface observations cannot duplicate identity")
        return self


def localize_visible_surface(
    *,
    source_mask: ArrayLike,
    depth_m: ArrayLike,
    confidence: ArrayLike,
    target: PerceptionTarget,
    config: MaskDepthDiagnosticConfig,
    rule: TargetVisibleSurfaceRule,
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    join: ExactFrameDepthJoin,
) -> VisibleSurfaceLocalization:
    """Apply D030 and back-project one exact-frame per-camera surface."""

    if rule.target is not target:
        raise ValueError("visible-surface target differs from D030 rule")
    if intrinsics.camera_id != join.camera_id or pose.camera_id != join.camera_id:
        raise ValueError("join, intrinsics, and pose camera IDs must match")
    depth = np.asarray(depth_m, dtype=np.float64)
    scores = np.asarray(confidence, dtype=np.float64)
    if depth.ndim != 2 or scores.shape != depth.shape:
        raise ValueError("visible-surface depth/confidence must be matching 2D arrays")
    if (intrinsics.image_height, intrinsics.image_width) != depth.shape:
        raise ValueError("processed intrinsics dimensions must match depth grid")
    try:
        candidates = build_mask_depth_candidates(
            source_mask,
            depth,
            target=target,
            config=config,
        )
    except InsufficientCandidateSamplesError:
        return _unavailable_surface(
            strategy=rule.candidate_strategy,
            candidate_pixel_count=int(np.count_nonzero(source_mask)),
        )
    candidate = next(
        item for item in candidates if item.strategy is rule.candidate_strategy
    )
    try:
        selection = select_candidate_relative_confidence(
            candidate_mask=candidate.mask,
            depth_m=depth,
            confidence=scores,
            percentile=rule.confidence_percentile,
            minimum_retained_sample_count=rule.minimum_retained_sample_count,
        )
    except InsufficientCandidateSamplesError:
        return _unavailable_surface(
            strategy=rule.candidate_strategy,
            candidate_pixel_count=int(np.count_nonzero(candidate.mask)),
        )

    y_pixels, x_pixels = np.nonzero(selection.mask)
    pixels_uv = np.column_stack((x_pixels, y_pixels)).astype(np.float64)
    retained_depth = depth[selection.mask]
    retained_confidence = scores[selection.mask]
    points_camera = np.asarray(
        backproject_pixels(pixels_uv, retained_depth, intrinsics=intrinsics),
        dtype=np.float64,
    )
    points_world = np.asarray(
        camera_points_to_world(points_camera, pose=pose),
        dtype=np.float64,
    )
    aggregate_camera = np.median(points_camera, axis=0)
    aggregate_world = np.asarray(
        camera_points_to_world(aggregate_camera, pose=pose),
        dtype=np.float64,
    )
    reprojected = np.asarray(
        project_camera_points(points_camera, intrinsics=intrinsics),
        dtype=np.float64,
    )
    reprojection_error = np.linalg.norm(reprojected - pixels_uv, axis=1)
    recovered_camera = np.asarray(
        world_points_to_camera(points_world, pose=pose),
        dtype=np.float64,
    )
    round_trip_error = np.linalg.norm(recovered_camera - points_camera, axis=1)
    return VisibleSurfaceLocalization(
        availability=VisibleSurfaceAvailability.OBSERVED,
        unavailable_reason=None,
        strategy=rule.candidate_strategy,
        candidate_pixel_count=int(np.count_nonzero(candidate.mask)),
        valid_candidate_count=selection.valid_candidate_count,
        retained_sample_count=selection.retained_count,
        confidence_threshold=selection.confidence_threshold,
        pixels_uv=pixels_uv,
        depth_m=np.asarray(retained_depth, dtype=np.float64),
        confidence=np.asarray(retained_confidence, dtype=np.float64),
        points_camera_m=points_camera,
        points_world_m=points_world,
        median_pixel_uv=(
            float(np.median(pixels_uv[:, 0])),
            float(np.median(pixels_uv[:, 1])),
        ),
        aggregate_camera_xyz_m=(
            float(aggregate_camera[0]),
            float(aggregate_camera[1]),
            float(aggregate_camera[2]),
        ),
        aggregate_world_xyz_m=(
            float(aggregate_world[0]),
            float(aggregate_world[1]),
            float(aggregate_world[2]),
        ),
        sample_reprojection_max_error_px=float(np.max(reprojection_error)),
        world_camera_round_trip_max_error_m=float(np.max(round_trip_error)),
    )


def _unavailable_surface(
    *,
    strategy: MaskDepthStrategy,
    candidate_pixel_count: int,
) -> VisibleSurfaceLocalization:
    return VisibleSurfaceLocalization(
        availability=VisibleSurfaceAvailability.UNAVAILABLE,
        unavailable_reason=(
            VisibleSurfaceUnavailableReason.INVALID_OR_INSUFFICIENT_SAMPLES
        ),
        strategy=strategy,
        candidate_pixel_count=candidate_pixel_count,
        valid_candidate_count=0,
        retained_sample_count=0,
        confidence_threshold=None,
        pixels_uv=np.empty((0, 2), dtype=np.float64),
        depth_m=np.empty((0,), dtype=np.float64),
        confidence=np.empty((0,), dtype=np.float64),
        points_camera_m=np.empty((0, 3), dtype=np.float64),
        points_world_m=np.empty((0, 3), dtype=np.float64),
        median_pixel_uv=None,
        aggregate_camera_xyz_m=None,
        aggregate_world_xyz_m=None,
        sample_reprojection_max_error_px=None,
        world_camera_round_trip_max_error_m=None,
    )


def _stable_digest(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
