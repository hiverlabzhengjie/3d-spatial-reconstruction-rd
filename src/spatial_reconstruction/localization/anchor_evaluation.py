"""Derived target-anchor candidates and disagreement contracts for S04."""

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
from pydantic import Field, field_validator, model_validator

from spatial_reconstruction.contracts import (
    CameraIntrinsics,
    CameraPose,
    ContractModel,
    FiniteFloat,
    NonNegativeFloat,
    NonNegativeInt,
    PerceptionTarget,
    PositiveFloat,
    PositiveInt,
    Sha256Digest,
    Vector3,
)

Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class AnchorCandidateMethod(StrEnum):
    """Anchor meanings compared before S04 camera fusion."""

    PERSON_RAW_VISIBLE_SURFACE_REFERENCE = "person_raw_visible_surface_reference"
    PERSON_LOWEST_WORLD_Z_DECILE = "person_lowest_world_z_decile"
    PERSON_LOWEST_WORLD_Z_QUINTILE = "person_lowest_world_z_quintile"
    PERSON_BOTTOM_IMAGE_QUINTILE = "person_bottom_image_quintile"
    PERSON_BOTTOM_IMAGE_QUINTILE_FLOOR_INTERSECTION = (
        "person_bottom_image_quintile_floor_intersection"
    )
    PERSON_VALIDATED_GROUND_CONTACT = "person_validated_ground_contact"
    BACKPACK_RAW_VISIBLE_SURFACE_REFERENCE = "backpack_raw_visible_surface_reference"
    BACKPACK_WORLD_COMPONENT_MEDIAN = "backpack_world_component_median"
    BACKPACK_TRIMMED_BOUNDS_CENTER = "backpack_trimmed_bounds_center"
    BACKPACK_TRIMMED_MEAN = "backpack_trimmed_mean"


class AnchorAvailability(StrEnum):
    """Availability of one derived candidate or selected anchor state."""

    OBSERVED = "observed"
    UNAVAILABLE = "unavailable"


class AnchorUnavailableReason(StrEnum):
    """Why an anchor is unavailable without a placeholder coordinate."""

    SOURCE_OBSERVATION_UNAVAILABLE = "source_observation_unavailable"
    INSUFFICIENT_SUPPORT = "insufficient_support"
    INVALID_FLOOR_INTERSECTION = "invalid_floor_intersection"
    INSUFFICIENT_FLOOR_PROXIMITY = "insufficient_floor_proximity"


class CrossCameraAnchorState(StrEnum):
    """Two-camera eligibility before any fusion is performed."""

    PAIRED_ELIGIBLE = "paired_eligible"
    PAIRED_DISAGREEMENT = "paired_disagreement"
    SINGLE_CAMERA = "single_camera"
    UNAVAILABLE = "unavailable"


class AnchorEvaluationConfig(ContractModel):
    """Explicit prototype parameters for the bounded anchor comparison."""

    policy_id: Literal["s04_target_anchor_v1"] = "s04_target_anchor_v1"
    person_low_world_z_decile_percentile: FiniteFloat = 10.0
    person_low_world_z_quintile_percentile: FiniteFloat = 20.0
    person_bottom_image_fraction: float = Field(default=0.20, gt=0.0, le=1.0)
    minimum_candidate_support_count: PositiveInt = 32
    maximum_ground_support_height_m: PositiveFloat = 0.35
    world_floor_z_m: FiniteFloat = 0.0
    backpack_trim_lower_percentile: FiniteFloat = 10.0
    backpack_trim_upper_percentile: FiniteFloat = 90.0
    maximum_cross_camera_disagreement_m: PositiveFloat = 0.35
    camera_fusion_performed: Literal[False] = False

    @model_validator(mode="after")
    def validate_percentiles(self) -> Self:
        if (
            self.person_low_world_z_decile_percentile != 10.0
            or self.person_low_world_z_quintile_percentile != 20.0
            or self.backpack_trim_lower_percentile != 10.0
            or self.backpack_trim_upper_percentile != 90.0
        ):
            raise ValueError("S04 anchor evaluation percentiles are fixed at 10/20/10/90")
        if self.backpack_trim_lower_percentile >= self.backpack_trim_upper_percentile:
            raise ValueError("backpack trim percentiles must be increasing")
        return self


@dataclass(frozen=True, slots=True)
class AnchorCandidate:
    """Runtime candidate result before persistent provenance is assigned."""

    method: AnchorCandidateMethod
    availability: AnchorAvailability
    unavailable_reason: AnchorUnavailableReason | None
    source_sample_count: int
    support_sample_count: int
    world_xyz_m: tuple[float, float, float] | None
    measured_support_world_z_m: float | None
    coordinate_semantics: str

    def __post_init__(self) -> None:
        if self.source_sample_count <= 0 or self.support_sample_count < 0:
            raise ValueError("anchor sample counts are invalid")
        if self.support_sample_count > self.source_sample_count:
            raise ValueError("anchor support cannot exceed source samples")
        if not self.coordinate_semantics.strip():
            raise ValueError("anchor coordinate semantics must not be empty")
        if self.availability is AnchorAvailability.OBSERVED:
            if self.unavailable_reason is not None or self.world_xyz_m is None:
                raise ValueError("observed anchor requires XYZ and no unavailable reason")
            if self.support_sample_count <= 0:
                raise ValueError("observed anchor requires sample support")
        elif self.unavailable_reason is None or self.world_xyz_m is not None:
            raise ValueError("unavailable anchor requires a reason and no XYZ")
        if self.world_xyz_m is not None and not np.isfinite(self.world_xyz_m).all():
            raise ValueError("anchor XYZ must be finite")
        if self.measured_support_world_z_m is not None and not math.isfinite(
            self.measured_support_world_z_m
        ):
            raise ValueError("anchor support height must be finite")


class AnchorCandidateRecord(ContractModel):
    """Persistent derived candidate linked to one raw visible-surface record."""

    schema_version: Literal[1] = 1
    candidate_id: Sha256Digest
    source_observation_id: Sha256Digest
    action_depth_job_id: Sha256Digest
    bundle_id: Sha256Digest
    frame_id: Sha256Digest
    source_frame_index: NonNegativeInt
    capture_timestamp_seconds: NonNegativeFloat
    phase_id: str
    camera_id: Literal["camera_a", "camera_b"]
    target: PerceptionTarget
    method: AnchorCandidateMethod
    availability: AnchorAvailability
    unavailable_reason: AnchorUnavailableReason | None
    source_sample_count: PositiveInt
    support_sample_count: NonNegativeInt
    support_fraction: NonNegativeFloat
    world_xyz_m: Vector3 | None
    measured_support_world_z_m: FiniteFloat | None
    inside_room_bounds: bool | None
    coordinate_semantics: str
    selected_for_tracking: bool
    selected_for_ground_contact: bool
    source_sample_cloud_ref: str
    source_sample_cloud_sha256: Sha256Digest
    source_raw_aggregate_world_xyz_m: Vector3
    camera_fusion_performed: Literal[False] = False
    presentation_smoothing_performed: Literal[False] = False

    @field_validator(
        "phase_id", "source_sample_cloud_ref", "coordinate_semantics"
    )
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("anchor record text must be non-empty and trimmed")
        return value

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        expected_fraction = self.support_sample_count / self.source_sample_count
        if not np.isclose(self.support_fraction, expected_fraction, atol=1e-12):
            raise ValueError("anchor support fraction differs from counts")
        observed = self.availability is AnchorAvailability.OBSERVED
        if observed != (self.world_xyz_m is not None):
            raise ValueError("anchor availability and XYZ differ")
        if observed != (self.inside_room_bounds is not None):
            raise ValueError("anchor availability and room-bounds state differ")
        if observed == (self.unavailable_reason is not None):
            raise ValueError("anchor availability and unavailable reason differ")
        expected_id = self.create_candidate_id(
            source_observation_id=self.source_observation_id,
            method=self.method,
        )
        if self.candidate_id != expected_id:
            raise ValueError("anchor candidate ID differs from identity")
        return self

    @classmethod
    def create_candidate_id(
        cls,
        *,
        source_observation_id: str,
        method: AnchorCandidateMethod,
    ) -> str:
        return _stable_digest(
            {
                "schema_version": 1,
                "source_observation_id": source_observation_id,
                "method": method.value,
            }
        )


class SelectedAnchorStateRecord(ContractModel):
    """Selected per-camera anchor, including missing-source states."""

    action_depth_job_id: Sha256Digest
    bundle_id: Sha256Digest
    frame_id: Sha256Digest
    source_frame_index: NonNegativeInt
    capture_timestamp_seconds: NonNegativeFloat
    phase_id: str
    camera_id: Literal["camera_a", "camera_b"]
    target: PerceptionTarget
    selected_method: AnchorCandidateMethod
    availability: AnchorAvailability
    unavailable_reason: AnchorUnavailableReason | None
    source_observation_id: Sha256Digest | None
    source_candidate_id: Sha256Digest | None
    anchor_world_xyz_m: Vector3 | None
    coordinate_semantics: str
    camera_fusion_performed: Literal[False] = False
    presentation_smoothing_performed: Literal[False] = False

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        observed = self.availability is AnchorAvailability.OBSERVED
        present = (
            self.source_observation_id,
            self.source_candidate_id,
            self.anchor_world_xyz_m,
        )
        if observed and (self.unavailable_reason is not None or any(x is None for x in present)):
            raise ValueError("observed selected anchor lacks source or XYZ")
        if not observed and (
            self.unavailable_reason is None
            or self.source_candidate_id is not None
            or self.anchor_world_xyz_m is not None
        ):
            raise ValueError("unavailable selected anchor carries invalid fields")
        return self


class CrossCameraAnchorComparison(ContractModel):
    """Pairing outcome for selected anchors before camera fusion."""

    action_depth_job_id: Sha256Digest
    bundle_id: Sha256Digest
    source_frame_index: NonNegativeInt
    phase_id: str
    target: PerceptionTarget
    selected_method: AnchorCandidateMethod
    camera_a_availability: AnchorAvailability
    camera_b_availability: AnchorAvailability
    state: CrossCameraAnchorState
    disagreement_distance_m: NonNegativeFloat | None
    maximum_eligible_disagreement_m: PositiveFloat
    eligible_for_fusion: bool
    camera_fusion_performed: Literal[False] = False

    @model_validator(mode="after")
    def validate_comparison(self) -> Self:
        observed_count = sum(
            state is AnchorAvailability.OBSERVED
            for state in (self.camera_a_availability, self.camera_b_availability)
        )
        if observed_count == 2:
            if self.disagreement_distance_m is None:
                raise ValueError("paired anchors require a disagreement distance")
            expected = (
                CrossCameraAnchorState.PAIRED_ELIGIBLE
                if self.disagreement_distance_m
                <= self.maximum_eligible_disagreement_m
                else CrossCameraAnchorState.PAIRED_DISAGREEMENT
            )
            if self.state is not expected:
                raise ValueError("paired anchor state differs from threshold")
            if self.eligible_for_fusion != (
                expected is CrossCameraAnchorState.PAIRED_ELIGIBLE
            ):
                raise ValueError("paired fusion eligibility differs from state")
        elif observed_count == 1:
            if (
                self.state is not CrossCameraAnchorState.SINGLE_CAMERA
                or self.disagreement_distance_m is not None
                or self.eligible_for_fusion
            ):
                raise ValueError("single-camera comparison fields are invalid")
        elif (
            self.state is not CrossCameraAnchorState.UNAVAILABLE
            or self.disagreement_distance_m is not None
            or self.eligible_for_fusion
        ):
            raise ValueError("unavailable comparison fields are invalid")
        return self


class AnchorEvaluationRunSummary(ContractModel):
    """Persistent S04 anchor comparison and selected pre-fusion policy."""

    schema_version: Literal[1]
    status: Literal["completed_pending_visual_qa"]
    stage: Literal["S04"]
    created_at_utc: datetime
    source_visible_surface_summary_ref: str
    source_visible_surface_summary_sha256: Sha256Digest
    source_visible_surface_verification_ref: str
    source_visible_surface_verification_sha256: Sha256Digest
    source_action_depth_summary_ref: str
    source_action_depth_summary_sha256: Sha256Digest
    configuration: AnchorEvaluationConfig
    selected_policy: dict[str, Any]
    candidate_metrics: dict[str, Any]
    candidate_records: tuple[AnchorCandidateRecord, ...]
    selected_anchor_states: tuple[SelectedAnchorStateRecord, ...]
    cross_camera_comparisons: tuple[CrossCameraAnchorComparison, ...]
    comparison_csv_ref: str
    comparison_csv_sha256: Sha256Digest
    candidate_comparison_ref: str
    candidate_comparison_sha256: Sha256Digest
    selected_anchor_world_preview_ref: str
    selected_anchor_world_preview_sha256: Sha256Digest
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        if len({item.candidate_id for item in self.candidate_records}) != len(
            self.candidate_records
        ):
            raise ValueError("anchor candidate IDs must be unique")
        state_keys = {
            (item.action_depth_job_id, item.camera_id, item.target)
            for item in self.selected_anchor_states
        }
        if len(state_keys) != len(self.selected_anchor_states):
            raise ValueError("selected anchor states cannot duplicate identity")
        comparison_keys = {
            (item.action_depth_job_id, item.target)
            for item in self.cross_camera_comparisons
        }
        if len(comparison_keys) != len(self.cross_camera_comparisons):
            raise ValueError("cross-camera comparisons cannot duplicate identity")
        return self


def evaluate_anchor_candidates(
    *,
    target: PerceptionTarget,
    pixels_uv: ArrayLike,
    points_world_m: ArrayLike,
    confidence: ArrayLike,
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    raw_visible_surface_world_xyz_m: ArrayLike,
    config: AnchorEvaluationConfig,
) -> tuple[AnchorCandidate, ...]:
    """Derive bounded person or backpack candidates without camera fusion."""

    pixels = _matrix(pixels_uv, columns=2, name="pixels_uv")
    points = _matrix(points_world_m, columns=3, name="points_world_m")
    scores = np.asarray(confidence, dtype=np.float64)
    raw = np.asarray(raw_visible_surface_world_xyz_m, dtype=np.float64)
    if pixels.shape[0] != points.shape[0] or scores.shape != (len(points),):
        raise ValueError("anchor source arrays must have matching sample counts")
    if not np.isfinite(scores).all() or raw.shape != (3,) or not np.isfinite(raw).all():
        raise ValueError("anchor confidence and raw aggregate must be finite")
    if intrinsics.camera_id != pose.camera_id:
        raise ValueError("anchor intrinsics and pose camera IDs must match")

    if target is PerceptionTarget.PERSON:
        return _person_candidates(
            pixels=pixels,
            points=points,
            intrinsics=intrinsics,
            pose=pose,
            raw=raw,
            config=config,
        )
    return _backpack_candidates(points=points, raw=raw, config=config)


def intersect_pixels_with_world_floor(
    pixels_uv: ArrayLike,
    *,
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    world_floor_z_m: float,
) -> tuple[Float64Array, BoolArray]:
    """Intersect OpenCV pixel rays with a declared horizontal world floor."""

    pixels = _matrix(pixels_uv, columns=2, name="pixels_uv")
    rays_camera = np.column_stack(
        (
            (pixels[:, 0] - intrinsics.cx) / intrinsics.fx,
            (pixels[:, 1] - intrinsics.cy) / intrinsics.fy,
            np.ones(len(pixels), dtype=np.float64),
        )
    )
    transform = np.asarray(pose.T_world_from_camera, dtype=np.float64)
    directions_world = rays_camera @ transform[:3, :3].T
    origin_world = transform[:3, 3]
    denominators = directions_world[:, 2]
    valid = np.isfinite(denominators) & (denominators < -1e-12)
    distances = np.full(len(pixels), np.nan, dtype=np.float64)
    distances[valid] = (world_floor_z_m - origin_world[2]) / denominators[valid]
    valid &= np.isfinite(distances) & (distances > 0)
    intersections = np.full((len(pixels), 3), np.nan, dtype=np.float64)
    intersections[valid] = (
        origin_world + distances[valid, np.newaxis] * directions_world[valid]
    )
    return intersections, np.asarray(valid, dtype=np.bool_)


def _person_candidates(
    *,
    pixels: Float64Array,
    points: Float64Array,
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    raw: Float64Array,
    config: AnchorEvaluationConfig,
) -> tuple[AnchorCandidate, ...]:
    source_count = len(points)
    low_decile = _selected_median_candidate(
        method=AnchorCandidateMethod.PERSON_LOWEST_WORLD_Z_DECILE,
        points=points,
        selection=points[:, 2]
        <= np.percentile(points[:, 2], config.person_low_world_z_decile_percentile),
        minimum_support=config.minimum_candidate_support_count,
        semantics="Measured median of the lowest world-Z decile of visible lower-body samples.",
    )
    low_quintile = _selected_median_candidate(
        method=AnchorCandidateMethod.PERSON_LOWEST_WORLD_Z_QUINTILE,
        points=points,
        selection=points[:, 2]
        <= np.percentile(points[:, 2], config.person_low_world_z_quintile_percentile),
        minimum_support=config.minimum_candidate_support_count,
        semantics="Measured median of the lowest world-Z quintile of visible lower-body samples.",
    )
    bottom_selection = pixels[:, 1] >= np.percentile(
        pixels[:, 1], 100.0 * (1.0 - config.person_bottom_image_fraction)
    )
    bottom_image = _selected_median_candidate(
        method=AnchorCandidateMethod.PERSON_BOTTOM_IMAGE_QUINTILE,
        points=points,
        selection=bottom_selection,
        minimum_support=config.minimum_candidate_support_count,
        semantics="Measured median of samples in the bottom image-space quintile.",
    )
    floor_points, floor_valid = intersect_pixels_with_world_floor(
        pixels[bottom_selection],
        intrinsics=intrinsics,
        pose=pose,
        world_floor_z_m=config.world_floor_z_m,
    )
    floor_support = floor_points[floor_valid]
    if len(floor_support) < config.minimum_candidate_support_count:
        floor_intersection = _unavailable(
            method=(
                AnchorCandidateMethod.PERSON_BOTTOM_IMAGE_QUINTILE_FLOOR_INTERSECTION
            ),
            source_count=source_count,
            support_count=len(floor_support),
            reason=AnchorUnavailableReason.INVALID_FLOOR_INTERSECTION,
            measured_z=None,
            semantics="Median floor intersection of bottom-quintile validated pixel rays.",
        )
    else:
        floor_xyz = np.median(floor_support, axis=0)
        floor_intersection = _observed(
            method=(
                AnchorCandidateMethod.PERSON_BOTTOM_IMAGE_QUINTILE_FLOOR_INTERSECTION
            ),
            source_count=source_count,
            support_count=len(floor_support),
            xyz=floor_xyz,
            measured_z=None,
            semantics="Median floor intersection of bottom-quintile validated pixel rays.",
        )
    if (
        low_quintile.availability is AnchorAvailability.OBSERVED
        and low_quintile.world_xyz_m is not None
        and low_quintile.measured_support_world_z_m is not None
        and low_quintile.measured_support_world_z_m
        <= config.maximum_ground_support_height_m
    ):
        ground_xyz = np.asarray(low_quintile.world_xyz_m, dtype=np.float64).copy()
        ground_xyz[2] = config.world_floor_z_m
        validated_ground = _observed(
            method=AnchorCandidateMethod.PERSON_VALIDATED_GROUND_CONTACT,
            source_count=source_count,
            support_count=low_quintile.support_sample_count,
            xyz=ground_xyz,
            measured_z=low_quintile.measured_support_world_z_m,
            semantics=(
                "Floor-projected XY from a near-floor measured lower-quintile support; "
                "derived ground contact, not raw XYZ."
            ),
        )
    else:
        validated_ground = _unavailable(
            method=AnchorCandidateMethod.PERSON_VALIDATED_GROUND_CONTACT,
            source_count=source_count,
            support_count=low_quintile.support_sample_count,
            reason=AnchorUnavailableReason.INSUFFICIENT_FLOOR_PROXIMITY,
            measured_z=low_quintile.measured_support_world_z_m,
            semantics=(
                "Floor-projected XY requires near-floor lower-quintile support; no "
                "coordinate is emitted when that evidence is absent."
            ),
        )
    return (
        _observed(
            method=AnchorCandidateMethod.PERSON_RAW_VISIBLE_SURFACE_REFERENCE,
            source_count=source_count,
            support_count=source_count,
            xyz=raw,
            measured_z=float(raw[2]),
            semantics="D031 raw per-camera visible-surface aggregate reference.",
        ),
        low_decile,
        low_quintile,
        bottom_image,
        floor_intersection,
        validated_ground,
    )


def _backpack_candidates(
    *,
    points: Float64Array,
    raw: Float64Array,
    config: AnchorEvaluationConfig,
) -> tuple[AnchorCandidate, ...]:
    source_count = len(points)
    lower = np.percentile(points, config.backpack_trim_lower_percentile, axis=0)
    upper = np.percentile(points, config.backpack_trim_upper_percentile, axis=0)
    trimmed = np.all((points >= lower) & (points <= upper), axis=1)
    return (
        _observed(
            method=AnchorCandidateMethod.BACKPACK_RAW_VISIBLE_SURFACE_REFERENCE,
            source_count=source_count,
            support_count=source_count,
            xyz=raw,
            measured_z=float(raw[2]),
            semantics="D031 raw per-camera backpack visible-surface reference.",
        ),
        _observed(
            method=AnchorCandidateMethod.BACKPACK_WORLD_COMPONENT_MEDIAN,
            source_count=source_count,
            support_count=source_count,
            xyz=np.median(points, axis=0),
            measured_z=float(np.median(points[:, 2])),
            semantics="World-frame component-wise median of the visible backpack cluster.",
        ),
        _observed(
            method=AnchorCandidateMethod.BACKPACK_TRIMMED_BOUNDS_CENTER,
            source_count=source_count,
            support_count=source_count,
            xyz=(lower + upper) / 2.0,
            measured_z=float((lower[2] + upper[2]) / 2.0),
            semantics="Centre of the world-frame 10th-to-90th percentile cluster bounds.",
        ),
        _selected_mean_candidate(
            method=AnchorCandidateMethod.BACKPACK_TRIMMED_MEAN,
            points=points,
            selection=trimmed,
            minimum_support=config.minimum_candidate_support_count,
            semantics="Mean of samples inside all world-frame 10th-to-90th percentile bounds.",
        ),
    )


def _selected_median_candidate(
    *,
    method: AnchorCandidateMethod,
    points: Float64Array,
    selection: BoolArray,
    minimum_support: int,
    semantics: str,
) -> AnchorCandidate:
    selected = points[selection]
    if len(selected) < minimum_support:
        return _unavailable(
            method=method,
            source_count=len(points),
            support_count=len(selected),
            reason=AnchorUnavailableReason.INSUFFICIENT_SUPPORT,
            measured_z=None,
            semantics=semantics,
        )
    xyz = np.median(selected, axis=0)
    return _observed(
        method=method,
        source_count=len(points),
        support_count=len(selected),
        xyz=xyz,
        measured_z=float(xyz[2]),
        semantics=semantics,
    )


def _selected_mean_candidate(
    *,
    method: AnchorCandidateMethod,
    points: Float64Array,
    selection: BoolArray,
    minimum_support: int,
    semantics: str,
) -> AnchorCandidate:
    selected = points[selection]
    if len(selected) < minimum_support:
        return _unavailable(
            method=method,
            source_count=len(points),
            support_count=len(selected),
            reason=AnchorUnavailableReason.INSUFFICIENT_SUPPORT,
            measured_z=None,
            semantics=semantics,
        )
    xyz = np.mean(selected, axis=0)
    return _observed(
        method=method,
        source_count=len(points),
        support_count=len(selected),
        xyz=xyz,
        measured_z=float(xyz[2]),
        semantics=semantics,
    )


def _observed(
    *,
    method: AnchorCandidateMethod,
    source_count: int,
    support_count: int,
    xyz: ArrayLike,
    measured_z: float | None,
    semantics: str,
) -> AnchorCandidate:
    point = np.asarray(xyz, dtype=np.float64)
    return AnchorCandidate(
        method=method,
        availability=AnchorAvailability.OBSERVED,
        unavailable_reason=None,
        source_sample_count=source_count,
        support_sample_count=support_count,
        world_xyz_m=(float(point[0]), float(point[1]), float(point[2])),
        measured_support_world_z_m=measured_z,
        coordinate_semantics=semantics,
    )


def _unavailable(
    *,
    method: AnchorCandidateMethod,
    source_count: int,
    support_count: int,
    reason: AnchorUnavailableReason,
    measured_z: float | None,
    semantics: str,
) -> AnchorCandidate:
    return AnchorCandidate(
        method=method,
        availability=AnchorAvailability.UNAVAILABLE,
        unavailable_reason=reason,
        source_sample_count=source_count,
        support_sample_count=support_count,
        world_xyz_m=None,
        measured_support_world_z_m=measured_z,
        coordinate_semantics=semantics,
    )


def _matrix(values: ArrayLike, *, columns: int, name: str) -> Float64Array:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != columns or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape (N, {columns})")
    if len(array) == 0:
        raise ValueError(f"{name} must contain at least one sample")
    return array


def _stable_digest(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
