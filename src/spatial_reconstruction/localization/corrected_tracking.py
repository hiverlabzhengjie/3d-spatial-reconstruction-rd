"""Corrected-depth, margin-aware S04 surface, anchor, and pair tracking."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self, cast

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
from spatial_reconstruction.geometry import (
    backproject_pixels,
    camera_points_to_world,
    project_camera_points,
    world_points_to_camera,
)
from spatial_reconstruction.localization.cross_camera_fusion import reliability_score
from spatial_reconstruction.localization.mask_depth_diagnostics import (
    InsufficientCandidateSamplesError,
    MaskDepthDiagnosticConfig,
    MaskDepthStrategy,
    SampleDistribution,
    build_mask_depth_candidates,
    select_candidate_relative_confidence,
    summarize_distribution,
)

Float64Array = NDArray[np.float64]


class PersonViewValidity(StrEnum):
    """Whether one person mask can support floor-contact reasoning."""

    LOWER_BODY_CANDIDATE = "lower_body_candidate"
    UPPER_BODY_ONLY_BOTTOM_TRUNCATED = "upper_body_only_bottom_truncated"


class CorrectedSurfaceRole(StrEnum):
    """Meaning of one corrected-depth sample cloud."""

    PERSON_LOWER_BODY = "person_lower_body"
    PERSON_UPPER_BODY = "person_upper_body"
    BACKPACK_VISIBLE_CLUSTER = "backpack_visible_cluster"


class CorrectedAnchorKind(StrEnum):
    """Comparable spatial meaning selected from one camera view."""

    PERSON_FOOTPOINT = "person_footpoint"
    PERSON_LOWER_BODY_SURFACE = "person_lower_body_surface"
    PERSON_UPPER_BODY_SURFACE = "person_upper_body_surface"
    BACKPACK_VISIBLE_CLUSTER = "backpack_visible_cluster"


class CorrectedPairState(StrEnum):
    """Same-frame result after semantic compatibility and distance gating."""

    FUSED = "fused"
    SINGLE_CAMERA = "single_camera"
    DISAGREEMENT = "disagreement"
    UNAVAILABLE = "unavailable"


class CorrectedTrackingPolicy(ContractModel):
    """D030-D033 corrected-depth and margin-aware tracking policy."""

    policy_id: Literal["s04_corrected_margin_aware_tracking_v1"] = (
        "s04_corrected_margin_aware_tracking_v1"
    )
    source_depth_policy_id: Literal["d025_action_pair_marker_scale_v1"] = (
        "d025_action_pair_marker_scale_v1"
    )
    confidence_percentile: FiniteFloat = 20.0
    erosion_radius_pixels: PositiveInt = 2
    person_lower_body_fraction: float = Field(default=0.35, gt=0.0, le=1.0)
    person_image_margin_pixels: PositiveInt = 2
    person_minimum_surface_samples: PositiveInt = 256
    backpack_minimum_surface_samples: PositiveInt = 128
    person_lowest_world_z_fraction: float = Field(default=0.20, gt=0.0, le=1.0)
    person_minimum_anchor_support: PositiveInt = 32
    minimum_ground_support_height_m: FiniteFloat = -0.10
    maximum_ground_support_height_m: PositiveFloat = 0.35
    world_floor_z_m: FiniteFloat = 0.0
    maximum_cross_camera_disagreement_m: PositiveFloat = 0.35
    upper_body_floor_projection_allowed: Literal[False] = False
    mixed_anchor_semantics_fusion_allowed: Literal[False] = False
    camera_specific_depth_scale_allowed: Literal[False] = False
    temporal_filling_performed: Literal[False] = False
    presentation_smoothing_performed: Literal[False] = False

    @model_validator(mode="after")
    def validate_ground_band(self) -> Self:
        if self.confidence_percentile != 20.0:
            raise ValueError("corrected candidate confidence percentile is fixed at 20")
        if self.minimum_ground_support_height_m >= self.maximum_ground_support_height_m:
            raise ValueError("ground-support height bounds must be increasing")
        return self


class PersonMaskMarginAssessment(ContractModel):
    """Explicit processed-grid margin evidence for one person mask."""

    margin_pixels: PositiveInt
    image_width: PositiveInt
    image_height: PositiveInt
    x_min: NonNegativeInt
    x_max: NonNegativeInt
    y_min: NonNegativeInt
    y_max: NonNegativeInt
    distance_left_pixels: NonNegativeInt
    distance_right_pixels: NonNegativeInt
    distance_top_pixels: NonNegativeInt
    distance_bottom_pixels: NonNegativeInt
    touches_left_margin: bool
    touches_right_margin: bool
    touches_top_margin: bool
    touches_bottom_margin: bool
    validity: PersonViewValidity
    footpoint_candidate_allowed: bool

    @model_validator(mode="after")
    def validate_assessment(self) -> Self:
        if self.x_max >= self.image_width or self.y_max >= self.image_height:
            raise ValueError("person mask bounds exceed the processed image")
        expected_bottom = self.distance_bottom_pixels <= self.margin_pixels
        if self.touches_bottom_margin != expected_bottom:
            raise ValueError("bottom-margin state differs from distance")
        expected_validity = (
            PersonViewValidity.UPPER_BODY_ONLY_BOTTOM_TRUNCATED
            if expected_bottom
            else PersonViewValidity.LOWER_BODY_CANDIDATE
        )
        if self.validity is not expected_validity:
            raise ValueError("person validity differs from bottom-margin evidence")
        if self.footpoint_candidate_allowed == expected_bottom:
            raise ValueError("footpoint eligibility differs from bottom-margin evidence")
        return self


@dataclass(frozen=True, slots=True)
class CorrectedSurfaceLocalization:
    """Runtime corrected-depth surface before artifact references are assigned."""

    role: CorrectedSurfaceRole
    strategy: MaskDepthStrategy
    margin_assessment: PersonMaskMarginAssessment | None
    candidate_pixel_count: int
    valid_candidate_count: int
    retained_sample_count: int
    confidence_threshold: float
    pixels_uv: Float64Array
    depth_m: Float64Array
    confidence: Float64Array
    points_camera_m: Float64Array
    points_world_m: Float64Array
    aggregate_camera_xyz_m: tuple[float, float, float]
    aggregate_world_xyz_m: tuple[float, float, float]
    reprojection_max_error_px: float
    round_trip_max_error_m: float


@dataclass(frozen=True, slots=True)
class CorrectedAnchor:
    """Runtime primary comparable anchor from one corrected surface."""

    kind: CorrectedAnchorKind
    world_xyz_m: tuple[float, float, float]
    support_sample_count: int
    measured_support_world_z_m: float | None
    footpoint_available: bool
    selection_reason: str


@dataclass(frozen=True, slots=True)
class PairAnchorInput:
    """One optional per-camera input to semantic pair resolution."""

    camera_id: Literal["camera_a", "camera_b"]
    anchor: CorrectedAnchor | None
    reliability_score: float | None


@dataclass(frozen=True, slots=True)
class CorrectedPairResolution:
    """Runtime semantically compatible pair selection and combination."""

    state: CorrectedPairState
    selected_kind: CorrectedAnchorKind | None
    world_xyz_m: tuple[float, float, float] | None
    selected_camera_ids: tuple[str, ...]
    contribution_weights: tuple[float | None, float | None]
    disagreement_distance_m: float | None
    fallback_surface_used: bool
    selection_reason: str


class CorrectedSurfaceRecord(ContractModel):
    """Persistent D030/D031 corrected per-camera surface evidence."""

    schema_version: Literal[1] = 1
    observation_id: Sha256Digest
    policy_id: Literal["s04_corrected_margin_aware_tracking_v1"]
    action_depth_job_id: Sha256Digest
    bundle_id: Sha256Digest
    frame_id: Sha256Digest
    source_frame_index: NonNegativeInt
    capture_timestamp_seconds: NonNegativeFloat
    phase_id: str
    camera_id: Literal["camera_a", "camera_b"]
    target: PerceptionTarget
    perception_job_id: Sha256Digest
    camera_local_track_id: str
    depth_scale: PositiveFloat
    surface_role: CorrectedSurfaceRole
    candidate_strategy: MaskDepthStrategy
    person_margin_assessment: PersonMaskMarginAssessment | None
    source_mask_pixel_count: PositiveInt
    candidate_pixel_count: PositiveInt
    valid_candidate_count: PositiveInt
    retained_sample_count: PositiveInt
    confidence_percentile: FiniteFloat
    confidence_threshold: FiniteFloat
    retained_depth_m: SampleDistribution
    retained_confidence: SampleDistribution
    aggregate_camera_xyz_m: Vector3
    aggregate_world_xyz_m: Vector3
    processed_intrinsics: CameraIntrinsics
    camera_pose: CameraPose
    raw_prediction_ref: str
    raw_prediction_sha256: Sha256Digest
    corrected_prediction_ref: str
    corrected_prediction_sha256: Sha256Digest
    aligned_mask_artifact_ref: str
    aligned_mask_artifact_sha256: Sha256Digest
    aligned_mask_index: NonNegativeInt
    sample_cloud_ref: str
    sample_cloud_sha256: Sha256Digest
    image_diagnostic_ref: str
    image_diagnostic_sha256: Sha256Digest
    reprojection_max_error_px: NonNegativeFloat
    round_trip_max_error_m: NonNegativeFloat
    coordinate_semantics: str

    @field_validator(
        "phase_id",
        "camera_local_track_id",
        "raw_prediction_ref",
        "corrected_prediction_ref",
        "aligned_mask_artifact_ref",
        "sample_cloud_ref",
        "image_diagnostic_ref",
        "coordinate_semantics",
    )
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("corrected surface text must be non-empty and trimmed")
        return value

    @model_validator(mode="after")
    def validate_surface(self) -> Self:
        person = self.target is PerceptionTarget.PERSON
        if person != (self.person_margin_assessment is not None):
            raise ValueError("person margin evidence must exist only for person surfaces")
        if self.surface_role is CorrectedSurfaceRole.PERSON_UPPER_BODY:
            assert self.person_margin_assessment is not None
            if self.person_margin_assessment.footpoint_candidate_allowed:
                raise ValueError("upper-body role requires bottom-truncated evidence")
        if self.retained_depth_m.count != self.retained_sample_count:
            raise ValueError("corrected surface depth count differs")
        if self.retained_confidence.count != self.retained_sample_count:
            raise ValueError("corrected surface confidence count differs")
        if self.confidence_percentile != 20.0:
            raise ValueError("corrected surface must apply candidate-relative p20")
        expected = self.create_observation_id(
            action_depth_job_id=self.action_depth_job_id,
            frame_id=self.frame_id,
            target=self.target,
            policy_id=self.policy_id,
        )
        if self.observation_id != expected:
            raise ValueError("corrected surface observation ID differs")
        return self

    @classmethod
    def create_observation_id(
        cls,
        *,
        action_depth_job_id: str,
        frame_id: str,
        target: PerceptionTarget,
        policy_id: str,
    ) -> str:
        return _stable_digest(
            {
                "schema_version": 1,
                "action_depth_job_id": action_depth_job_id,
                "frame_id": frame_id,
                "target": target.value,
                "policy_id": policy_id,
            }
        )


class CorrectedAnchorRecord(ContractModel):
    """Persistent D032 per-camera primary or fallback anchor."""

    schema_version: Literal[1] = 1
    anchor_id: Sha256Digest
    source_observation_id: Sha256Digest
    action_depth_job_id: Sha256Digest
    bundle_id: Sha256Digest
    frame_id: Sha256Digest
    source_frame_index: NonNegativeInt
    phase_id: str
    camera_id: Literal["camera_a", "camera_b"]
    target: PerceptionTarget
    kind: CorrectedAnchorKind
    world_xyz_m: Vector3
    support_sample_count: PositiveInt
    measured_support_world_z_m: FiniteFloat | None
    footpoint_available: bool
    selection_reason: str
    retained_confidence_median: PositiveFloat
    retained_depth_median_m: PositiveFloat
    retained_depth_mad_m: NonNegativeFloat
    reliability_score: PositiveFloat
    source_sample_cloud_ref: str
    source_sample_cloud_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_anchor(self) -> Self:
        if self.footpoint_available != (
            self.kind is CorrectedAnchorKind.PERSON_FOOTPOINT
        ):
            raise ValueError("footpoint availability differs from anchor kind")
        expected = self.create_anchor_id(
            source_observation_id=self.source_observation_id,
            kind=self.kind,
        )
        if self.anchor_id != expected:
            raise ValueError("corrected anchor ID differs")
        return self

    @classmethod
    def create_anchor_id(
        cls, *, source_observation_id: str, kind: CorrectedAnchorKind
    ) -> str:
        return _stable_digest(
            {
                "schema_version": 1,
                "source_observation_id": source_observation_id,
                "kind": kind.value,
            }
        )


class CorrectedPairSource(ContractModel):
    """One D033 source, including non-contributing alternate semantics."""

    camera_id: Literal["camera_a", "camera_b"]
    source_anchor_id: Sha256Digest | None
    kind: CorrectedAnchorKind | None
    world_xyz_m: Vector3 | None
    reliability_score: PositiveFloat | None
    contribution_weight: NonNegativeFloat | None
    selected_for_output: bool

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        values = (
            self.source_anchor_id,
            self.kind,
            self.world_xyz_m,
            self.reliability_score,
        )
        if all(value is None for value in values):
            if self.contribution_weight is not None or self.selected_for_output:
                raise ValueError("missing pair source cannot contribute")
        elif any(value is None for value in values):
            raise ValueError("observed pair source has incomplete evidence")
        if self.contribution_weight is not None and not 0 <= self.contribution_weight <= 1:
            raise ValueError("pair contribution weight must be within zero and one")
        return self


class CorrectedPairObservationRecord(ContractModel):
    """Persistent D033 same-pair output with explicit anchor semantics."""

    schema_version: Literal[1] = 1
    observation_id: Sha256Digest
    policy_id: Literal["s04_corrected_margin_aware_tracking_v1"]
    action_depth_job_id: Sha256Digest
    bundle_id: Sha256Digest
    source_frame_index: NonNegativeInt
    capture_timestamp_seconds: NonNegativeFloat
    phase_id: str
    target: PerceptionTarget
    state: CorrectedPairState
    selected_kind: CorrectedAnchorKind | None
    world_xyz_m: Vector3 | None
    sources: tuple[CorrectedPairSource, CorrectedPairSource]
    selected_camera_ids: tuple[str, ...]
    disagreement_distance_m: NonNegativeFloat | None
    maximum_cross_camera_disagreement_m: PositiveFloat
    fallback_surface_used: bool
    selection_reason: str
    temporal_filling_performed: Literal[False] = False
    presentation_smoothing_performed: Literal[False] = False

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        if tuple(source.camera_id for source in self.sources) != (
            "camera_a",
            "camera_b",
        ):
            raise ValueError("corrected pair sources must be ordered A then B")
        if self.state in (CorrectedPairState.FUSED, CorrectedPairState.SINGLE_CAMERA):
            if self.selected_kind is None or self.world_xyz_m is None:
                raise ValueError("available pair output requires kind and XYZ")
        elif self.selected_kind is not None or self.world_xyz_m is not None:
            raise ValueError("unavailable/disagreement pair cannot emit XYZ")
        if self.state is CorrectedPairState.DISAGREEMENT:
            if (
                self.disagreement_distance_m is None
                or self.disagreement_distance_m
                <= self.maximum_cross_camera_disagreement_m
            ):
                raise ValueError("pair disagreement distance is invalid")
        elif (
            self.disagreement_distance_m is not None
            and self.state is not CorrectedPairState.FUSED
        ):
            raise ValueError("pair distance belongs only to comparable two-view results")
        expected = self.create_observation_id(
            action_depth_job_id=self.action_depth_job_id,
            target=self.target,
            policy_id=self.policy_id,
        )
        if self.observation_id != expected:
            raise ValueError("corrected pair observation ID differs")
        return self

    @classmethod
    def create_observation_id(
        cls, *, action_depth_job_id: str, target: PerceptionTarget, policy_id: str
    ) -> str:
        return _stable_digest(
            {
                "schema_version": 1,
                "action_depth_job_id": action_depth_job_id,
                "target": target.value,
                "policy_id": policy_id,
            }
        )


class CorrectedTrackingRunSummary(ContractModel):
    """Persistent corrected D030-D033 rebuild."""

    schema_version: Literal[1]
    status: Literal["completed_pending_visual_qa"]
    stage: Literal["S04"]
    created_at_utc: datetime
    policy: CorrectedTrackingPolicy
    source_action_depth_summary_ref: str
    source_action_depth_summary_sha256: Sha256Digest
    source_depth_scale_summary_ref: str
    source_depth_scale_summary_sha256: Sha256Digest
    source_depth_scale_verification_ref: str
    source_depth_scale_verification_sha256: Sha256Digest
    source_mask_alignment_summary_ref: str
    source_mask_alignment_summary_sha256: Sha256Digest
    pose_calibration_ref: str
    pose_calibration_sha256: Sha256Digest
    scene_metadata_ref: str
    scene_metadata_sha256: Sha256Digest
    d030_sampling_summary_ref: str
    d030_sampling_summary_sha256: Sha256Digest
    d031_visible_surface_summary_ref: str
    d031_visible_surface_summary_sha256: Sha256Digest
    d032_anchor_summary_ref: str
    d032_anchor_summary_sha256: Sha256Digest
    d033_observation_summary_ref: str
    d033_observation_summary_sha256: Sha256Digest
    d030_surface_records: tuple[CorrectedSurfaceRecord, ...]
    d032_anchor_records: tuple[CorrectedAnchorRecord, ...]
    d033_pair_observations: tuple[CorrectedPairObservationRecord, ...]
    observation_csv_ref: str
    observation_csv_sha256: Sha256Digest
    margin_contact_sheet_ref: str
    margin_contact_sheet_sha256: Sha256Digest
    world_preview_ref: str
    world_preview_sha256: Sha256Digest
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        if not self.d030_surface_records:
            raise ValueError("corrected rebuild requires retained surfaces")
        surface_keys = {
            (record.action_depth_job_id, record.camera_id, record.target)
            for record in self.d030_surface_records
        }
        if len(surface_keys) != len(self.d030_surface_records):
            raise ValueError("corrected surfaces cannot duplicate job/camera/target")
        anchor_keys = {
            (record.action_depth_job_id, record.camera_id, record.target)
            for record in self.d032_anchor_records
        }
        if anchor_keys != surface_keys or len(anchor_keys) != len(
            self.d032_anchor_records
        ):
            raise ValueError("corrected rebuild requires one anchor per retained surface")
        job_ids = {record.action_depth_job_id for record in self.d030_surface_records}
        expected_pair_keys = {
            (job_id, target) for job_id in job_ids for target in PerceptionTarget
        }
        pair_keys = {
            (record.action_depth_job_id, record.target)
            for record in self.d033_pair_observations
        }
        if pair_keys != expected_pair_keys or len(pair_keys) != len(
            self.d033_pair_observations
        ):
            raise ValueError(
                "corrected rebuild requires one pair record per job and target"
            )
        return self


def assess_person_mask_margins(
    mask: ArrayLike, *, margin_pixels: int
) -> PersonMaskMarginAssessment:
    """Classify bottom truncation without treating top/side contact as hidden feet."""

    value = np.asarray(mask)
    if value.ndim != 2 or value.dtype not in (np.bool_, np.uint8):
        raise ValueError("person mask must be a 2D bool or uint8 array")
    foreground = value.astype(bool)
    if not np.any(foreground) or margin_pixels <= 0:
        raise ValueError("person margin assessment requires mask pixels and positive margin")
    ys, xs = np.nonzero(foreground)
    height, width = foreground.shape
    x_min, x_max = int(np.min(xs)), int(np.max(xs))
    y_min, y_max = int(np.min(ys)), int(np.max(ys))
    distances = {
        "left": x_min,
        "right": width - 1 - x_max,
        "top": y_min,
        "bottom": height - 1 - y_max,
    }
    bottom = distances["bottom"] <= margin_pixels
    return PersonMaskMarginAssessment(
        margin_pixels=margin_pixels,
        image_width=width,
        image_height=height,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        distance_left_pixels=distances["left"],
        distance_right_pixels=distances["right"],
        distance_top_pixels=distances["top"],
        distance_bottom_pixels=distances["bottom"],
        touches_left_margin=distances["left"] <= margin_pixels,
        touches_right_margin=distances["right"] <= margin_pixels,
        touches_top_margin=distances["top"] <= margin_pixels,
        touches_bottom_margin=bottom,
        validity=(
            PersonViewValidity.UPPER_BODY_ONLY_BOTTOM_TRUNCATED
            if bottom
            else PersonViewValidity.LOWER_BODY_CANDIDATE
        ),
        footpoint_candidate_allowed=not bottom,
    )


def localize_corrected_surface(
    *,
    source_mask: ArrayLike,
    corrected_depth_m: ArrayLike,
    confidence: ArrayLike,
    target: PerceptionTarget,
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    policy: CorrectedTrackingPolicy,
) -> CorrectedSurfaceLocalization:
    """Apply corrected D030 sampling and D031 back-projection for one view."""

    depth = np.asarray(corrected_depth_m, dtype=np.float64)
    scores = np.asarray(confidence, dtype=np.float64)
    mask = np.asarray(source_mask)
    if depth.ndim != 2 or scores.shape != depth.shape or mask.shape != depth.shape:
        raise ValueError("corrected surface mask/depth/confidence shapes must match")
    if intrinsics.camera_id != pose.camera_id:
        raise ValueError("corrected surface camera contracts must match")
    config = MaskDepthDiagnosticConfig(
        erosion_radius_pixels=policy.erosion_radius_pixels,
        person_lower_body_fraction=policy.person_lower_body_fraction,
    )
    margins: PersonMaskMarginAssessment | None = None
    if target is PerceptionTarget.PERSON:
        margins = assess_person_mask_margins(
            mask, margin_pixels=policy.person_image_margin_pixels
        )
        if margins.footpoint_candidate_allowed:
            strategy = MaskDepthStrategy.PERSON_LOWER_BODY
            role = CorrectedSurfaceRole.PERSON_LOWER_BODY
        else:
            strategy = MaskDepthStrategy.CONNECTED_DEPTH_CLUSTER
            role = CorrectedSurfaceRole.PERSON_UPPER_BODY
        minimum_samples = policy.person_minimum_surface_samples
    else:
        strategy = MaskDepthStrategy.CONNECTED_DEPTH_CLUSTER
        role = CorrectedSurfaceRole.BACKPACK_VISIBLE_CLUSTER
        minimum_samples = policy.backpack_minimum_surface_samples
    candidate = next(
        item
        for item in build_mask_depth_candidates(
            mask, depth, target=target, config=config
        )
        if item.strategy is strategy
    )
    selection = select_candidate_relative_confidence(
        candidate_mask=candidate.mask,
        depth_m=depth,
        confidence=scores,
        percentile=policy.confidence_percentile,
        minimum_retained_sample_count=minimum_samples,
    )
    ys, xs = np.nonzero(selection.mask)
    pixels = np.column_stack((xs, ys)).astype(np.float64)
    selected_depth = depth[selection.mask]
    selected_scores = scores[selection.mask]
    points_camera = np.asarray(
        backproject_pixels(pixels, selected_depth, intrinsics=intrinsics),
        dtype=np.float64,
    )
    points_world = np.asarray(
        camera_points_to_world(points_camera, pose=pose), dtype=np.float64
    )
    aggregate_camera = np.median(points_camera, axis=0)
    aggregate_world = np.asarray(
        camera_points_to_world(aggregate_camera, pose=pose), dtype=np.float64
    )
    reprojection = np.linalg.norm(
        np.asarray(
            project_camera_points(points_camera, intrinsics=intrinsics),
            dtype=np.float64,
        )
        - pixels,
        axis=1,
    )
    round_trip = np.linalg.norm(
        np.asarray(world_points_to_camera(points_world, pose=pose), dtype=np.float64)
        - points_camera,
        axis=1,
    )
    return CorrectedSurfaceLocalization(
        role=role,
        strategy=strategy,
        margin_assessment=margins,
        candidate_pixel_count=int(np.count_nonzero(candidate.mask)),
        valid_candidate_count=selection.valid_candidate_count,
        retained_sample_count=selection.retained_count,
        confidence_threshold=selection.confidence_threshold,
        pixels_uv=pixels,
        depth_m=selected_depth,
        confidence=selected_scores,
        points_camera_m=points_camera,
        points_world_m=points_world,
        aggregate_camera_xyz_m=cast(
            tuple[float, float, float], tuple(float(item) for item in aggregate_camera)
        ),
        aggregate_world_xyz_m=cast(
            tuple[float, float, float], tuple(float(item) for item in aggregate_world)
        ),
        reprojection_max_error_px=float(np.max(reprojection)),
        round_trip_max_error_m=float(np.max(round_trip)),
    )


def derive_corrected_anchor(
    surface: CorrectedSurfaceLocalization,
    *,
    target: PerceptionTarget,
    policy: CorrectedTrackingPolicy,
) -> CorrectedAnchor:
    """Derive one footpoint, fallback body surface, or backpack anchor."""

    points = surface.points_world_m
    if target is PerceptionTarget.BACKPACK:
        xyz = np.median(points, axis=0)
        return _anchor(
            CorrectedAnchorKind.BACKPACK_VISIBLE_CLUSTER,
            xyz,
            len(points),
            float(xyz[2]),
            False,
            "World component median of the corrected visible backpack cluster.",
        )
    if surface.role is CorrectedSurfaceRole.PERSON_UPPER_BODY:
        xyz = np.median(points, axis=0)
        return _anchor(
            CorrectedAnchorKind.PERSON_UPPER_BODY_SURFACE,
            xyz,
            len(points),
            float(xyz[2]),
            False,
            (
                "Bottom-truncated mask: retain measured upper-body surface only; "
                "no floor projection."
            ),
        )
    cutoff = float(
        np.percentile(points[:, 2], 100.0 * policy.person_lowest_world_z_fraction)
    )
    low = points[points[:, 2] <= cutoff]
    if len(low) < policy.person_minimum_anchor_support:
        raise InsufficientCandidateSamplesError("person low-Z anchor has insufficient support")
    low_xyz = np.median(low, axis=0)
    measured_z = float(low_xyz[2])
    assert surface.margin_assessment is not None
    ground_valid = (
        surface.margin_assessment.footpoint_candidate_allowed
        and policy.minimum_ground_support_height_m
        <= measured_z
        <= policy.maximum_ground_support_height_m
    )
    if ground_valid:
        footpoint = low_xyz.copy()
        footpoint[2] = policy.world_floor_z_m
        return _anchor(
            CorrectedAnchorKind.PERSON_FOOTPOINT,
            footpoint,
            len(low),
            measured_z,
            True,
            "Margin-valid near-floor low-Z support projected vertically to world Z=0.",
        )
    xyz = np.median(points, axis=0)
    return _anchor(
        CorrectedAnchorKind.PERSON_LOWER_BODY_SURFACE,
        xyz,
        len(points),
        measured_z,
        False,
        (
            "Lower-body surface retained, but floor proximity is insufficient "
            "for a footpoint."
        ),
    )


def resolve_corrected_pair(
    *,
    target: PerceptionTarget,
    camera_a: PairAnchorInput,
    camera_b: PairAnchorInput,
    maximum_disagreement_m: float,
) -> CorrectedPairResolution:
    """Prefer semantically consistent footpoints; use body surfaces only as fallback."""

    inputs = (camera_a, camera_b)
    available = [item for item in inputs if item.anchor is not None]
    if target is PerceptionTarget.PERSON:
        footpoints = [
            item
            for item in available
            if item.anchor is not None
            and item.anchor.kind is CorrectedAnchorKind.PERSON_FOOTPOINT
        ]
        if footpoints:
            selected = footpoints
            fallback = False
            reason = (
                "Prefer synchronized margin-valid footpoint evidence over "
                "body-surface anchors."
            )
        else:
            priority = {
                CorrectedAnchorKind.PERSON_LOWER_BODY_SURFACE: 0,
                CorrectedAnchorKind.PERSON_UPPER_BODY_SURFACE: 1,
            }
            if not available:
                selected = []
            else:
                best = min(
                    priority[item.anchor.kind]
                    for item in available
                    if item.anchor is not None
                )
                selected = [
                    item
                    for item in available
                    if item.anchor is not None and priority[item.anchor.kind] == best
                ]
            fallback = bool(selected)
            reason = (
                "No valid footpoint: retain the best measured body-surface semantic "
                "without floor projection or mixed-semantic fusion."
            )
    else:
        selected = available
        fallback = False
        reason = "Use corrected visible-backpack cluster anchors."
    return _combine_selected(
        inputs=inputs,
        selected=selected,
        maximum_disagreement_m=maximum_disagreement_m,
        fallback=fallback,
        reason=reason,
    )


def surface_statistics(
    surface: CorrectedSurfaceLocalization,
) -> tuple[SampleDistribution, SampleDistribution]:
    """Return persistent corrected depth and confidence summaries."""

    return summarize_distribution(surface.depth_m), summarize_distribution(
        surface.confidence
    )


def anchor_reliability(
    anchor: CorrectedAnchor, surface: CorrectedSurfaceLocalization
) -> float:
    """Apply D033's transparent reliability formula to a corrected anchor."""

    depth_median = float(np.median(surface.depth_m))
    depth_mad = float(np.median(np.abs(surface.depth_m - depth_median)))
    return reliability_score(
        support_sample_count=anchor.support_sample_count,
        retained_confidence_median=float(np.median(surface.confidence)),
        retained_depth_median_m=depth_median,
        retained_depth_mad_m=depth_mad,
    )


def _combine_selected(
    *,
    inputs: tuple[PairAnchorInput, PairAnchorInput],
    selected: list[PairAnchorInput],
    maximum_disagreement_m: float,
    fallback: bool,
    reason: str,
) -> CorrectedPairResolution:
    weights: list[float | None] = [None, None]
    if not selected:
        return CorrectedPairResolution(
            state=CorrectedPairState.UNAVAILABLE,
            selected_kind=None,
            world_xyz_m=None,
            selected_camera_ids=(),
            contribution_weights=(None, None),
            disagreement_distance_m=None,
            fallback_surface_used=False,
            selection_reason="No current compatible camera anchor is available.",
        )
    assert selected[0].anchor is not None
    kind = selected[0].anchor.kind
    if len(selected) == 1:
        index = inputs.index(selected[0])
        weights[index] = 1.0
        return CorrectedPairResolution(
            state=CorrectedPairState.SINGLE_CAMERA,
            selected_kind=kind,
            world_xyz_m=selected[0].anchor.world_xyz_m,
            selected_camera_ids=(selected[0].camera_id,),
            contribution_weights=(weights[0], weights[1]),
            disagreement_distance_m=None,
            fallback_surface_used=fallback,
            selection_reason=reason,
        )
    left, right = selected
    assert left.anchor is not None and right.anchor is not None
    if left.anchor.kind is not right.anchor.kind:
        raise ValueError("mixed anchor semantics cannot enter pair combination")
    distance = float(
        np.linalg.norm(
            np.asarray(left.anchor.world_xyz_m)
            - np.asarray(right.anchor.world_xyz_m)
        )
    )
    if distance > maximum_disagreement_m:
        return CorrectedPairResolution(
            state=CorrectedPairState.DISAGREEMENT,
            selected_kind=None,
            world_xyz_m=None,
            selected_camera_ids=(left.camera_id, right.camera_id),
            contribution_weights=(None, None),
            disagreement_distance_m=distance,
            fallback_surface_used=fallback,
            selection_reason=reason + " Comparable sources exceed the distance gate.",
        )
    if left.reliability_score is None or right.reliability_score is None:
        raise ValueError("paired fusion requires two reliability scores")
    score_sum = left.reliability_score + right.reliability_score
    left_weight = left.reliability_score / score_sum
    right_weight = right.reliability_score / score_sum
    point = (
        left_weight * np.asarray(left.anchor.world_xyz_m)
        + right_weight * np.asarray(right.anchor.world_xyz_m)
    )
    return CorrectedPairResolution(
        state=CorrectedPairState.FUSED,
        selected_kind=kind,
        world_xyz_m=(float(point[0]), float(point[1]), float(point[2])),
        selected_camera_ids=(left.camera_id, right.camera_id),
        contribution_weights=(left_weight, right_weight),
        disagreement_distance_m=distance,
        fallback_surface_used=fallback,
        selection_reason=reason,
    )


def _anchor(
    kind: CorrectedAnchorKind,
    xyz: ArrayLike,
    support: int,
    measured_z: float | None,
    footpoint: bool,
    reason: str,
) -> CorrectedAnchor:
    point = np.asarray(xyz, dtype=np.float64)
    return CorrectedAnchor(
        kind=kind,
        world_xyz_m=(float(point[0]), float(point[1]), float(point[2])),
        support_sample_count=support,
        measured_support_world_z_m=measured_z,
        footpoint_available=footpoint,
        selection_reason=reason,
    )


def _stable_digest(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
