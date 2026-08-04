"""Honest temporal presentation over corrected S04 spatial observations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self, cast

import numpy as np
from pydantic import field_validator, model_validator

from spatial_reconstruction.contracts import (
    ContractModel,
    NonNegativeFloat,
    NonNegativeInt,
    PerceptionTarget,
    PositiveFloat,
    PositiveInt,
    Sha256Digest,
    Vector3,
)
from spatial_reconstruction.localization.corrected_tracking import (
    CorrectedAnchorKind,
    CorrectedPairObservationRecord,
    CorrectedPairState,
)
from spatial_reconstruction.perception import PerceptionPresenceState


class TemporalPresentationState(StrEnum):
    """Whether one timeline tick has current, held, or unavailable XYZ."""

    MEASURED = "measured"
    STALE = "stale"
    MISSING = "missing"
    OCCLUDED = "occluded"
    INFERRED = "inferred"


class TemporalCoordinateProvenance(StrEnum):
    """Origin and authority of a presentation coordinate."""

    CURRENT_MEASUREMENT = "current_measurement"
    STALE_HOLD = "stale_hold"
    INFERRED_MODEL = "inferred_model"
    NONE = "none"


class TemporalPresentationReason(StrEnum):
    """Inspectable reason for one presentation state."""

    CURRENT_CORRECTED_OBSERVATION = "current_corrected_observation"
    RECENT_MEASUREMENT_HELD_STALE = "recent_measurement_held_stale"
    NO_PRIOR_MEASUREMENT = "no_prior_measurement"
    STALE_HORIZON_EXPIRED = "stale_horizon_expired"
    NO_CURRENT_DEPTH = "no_current_depth"
    SOURCE_TARGET_UNAVAILABLE = "source_target_unavailable"
    SOURCE_FAILURE = "source_failure"
    CONFIRMED_OCCLUSION = "confirmed_occlusion"
    INFERRED_POSITION = "inferred_position"


class TemporalPresentationPolicy(ContractModel):
    """D034 presentation-only freshness and trajectory policy."""

    policy_id: Literal["s04_temporal_presentation_v1"] = (
        "s04_temporal_presentation_v1"
    )
    source_observation_policy_id: Literal[
        "s04_corrected_margin_aware_tracking_v1"
    ] = "s04_corrected_margin_aware_tracking_v1"
    timeline_frame_stride: PositiveInt = 6
    nominal_timeline_rate_fps: PositiveFloat = 5.0
    maximum_stale_age_seconds: PositiveFloat = 1.0
    maximum_trajectory_segment_gap_seconds: PositiveFloat = 3.0
    interpolation_allowed: Literal[False] = False
    motion_extrapolation_allowed: Literal[False] = False
    inferred_positions_allowed: Literal[False] = False
    stale_positions_update_zone_membership: Literal[False] = False
    stale_positions_extend_trajectory: Literal[False] = False
    anchor_kind_conversion_allowed: Literal[False] = False
    occlusion_requires_explicit_evidence: Literal[True] = True

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if self.maximum_stale_age_seconds >= self.maximum_trajectory_segment_gap_seconds:
            raise ValueError("stale horizon must be shorter than the segment gap")
        return self


@dataclass(frozen=True, slots=True)
class TemporalPresentationResolution:
    """Runtime result before persistent identity is assigned."""

    state: TemporalPresentationState
    coordinate_provenance: TemporalCoordinateProvenance
    reason: TemporalPresentationReason
    raw_world_xyz_m: tuple[float, float, float] | None
    presentation_world_xyz_m: tuple[float, float, float] | None
    anchor_kind: CorrectedAnchorKind | None
    source_observation_id: str | None
    source_measurement_frame_index: int | None
    source_measurement_timestamp_seconds: float | None
    source_measurement_camera_ids: tuple[str, ...]
    measurement_age_seconds: float | None
    may_update_zone_membership: bool
    may_extend_trajectory: bool


class TemporalPresentationRecord(ContractModel):
    """Persistent presentation state at one authoritative capture-time tick."""

    schema_version: Literal[1] = 1
    record_id: Sha256Digest
    policy_id: Literal["s04_temporal_presentation_v1"]
    target: PerceptionTarget
    source_frame_index: NonNegativeInt
    capture_timestamp_seconds: NonNegativeFloat
    camera_a_perception_state: PerceptionPresenceState
    camera_b_perception_state: PerceptionPresenceState
    state: TemporalPresentationState
    coordinate_provenance: TemporalCoordinateProvenance
    reason: TemporalPresentationReason
    raw_world_xyz_m: Vector3 | None
    presentation_world_xyz_m: Vector3 | None
    anchor_kind: CorrectedAnchorKind | None
    source_observation_id: Sha256Digest | None
    source_measurement_frame_index: NonNegativeInt | None
    source_measurement_timestamp_seconds: NonNegativeFloat | None
    source_measurement_camera_ids: tuple[str, ...] = ()
    measurement_age_seconds: NonNegativeFloat | None
    may_update_zone_membership: bool
    may_extend_trajectory: bool
    visual_style_id: str

    @field_validator("source_measurement_camera_ids")
    @classmethod
    def validate_camera_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)) or any(not value.strip() for value in values):
            raise ValueError("source measurement camera IDs must be unique and non-empty")
        return values

    @field_validator("visual_style_id")
    @classmethod
    def validate_style(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("visual style ID must be non-empty and trimmed")
        return value

    @model_validator(mode="after")
    def validate_state_semantics(self) -> Self:
        measurement_fields = (
            self.anchor_kind,
            self.source_observation_id,
            self.source_measurement_frame_index,
            self.source_measurement_timestamp_seconds,
            self.measurement_age_seconds,
        )
        if self.state is TemporalPresentationState.MEASURED:
            if any(value is None for value in measurement_fields):
                raise ValueError("measured presentation requires complete source evidence")
            if self.raw_world_xyz_m is None or self.presentation_world_xyz_m is None:
                raise ValueError("measured presentation requires raw and presented XYZ")
            if self.raw_world_xyz_m != self.presentation_world_xyz_m:
                raise ValueError("measured raw and presentation XYZ must match")
            if self.coordinate_provenance is not TemporalCoordinateProvenance.CURRENT_MEASUREMENT:
                raise ValueError("measured presentation requires current provenance")
            if self.measurement_age_seconds != 0.0:
                raise ValueError("measured presentation age must be zero")
            if not self.may_update_zone_membership or not self.may_extend_trajectory:
                raise ValueError("current measurement must retain spatial authority")
        elif self.state is TemporalPresentationState.STALE:
            if any(value is None for value in measurement_fields):
                raise ValueError("stale presentation requires last-measurement evidence")
            if self.raw_world_xyz_m is not None or self.presentation_world_xyz_m is None:
                raise ValueError("stale state may hold only presentation XYZ")
            if self.coordinate_provenance is not TemporalCoordinateProvenance.STALE_HOLD:
                raise ValueError("stale state requires stale-hold provenance")
            if self.measurement_age_seconds is None or self.measurement_age_seconds <= 0:
                raise ValueError("stale state requires positive measurement age")
            if self.may_update_zone_membership or self.may_extend_trajectory:
                raise ValueError("stale state cannot update spatial facts or trajectory")
        elif self.state is TemporalPresentationState.INFERRED:
            if self.raw_world_xyz_m is not None or self.presentation_world_xyz_m is None:
                raise ValueError("inferred state may have presentation XYZ only")
            if self.coordinate_provenance is not TemporalCoordinateProvenance.INFERRED_MODEL:
                raise ValueError("inferred state requires inferred provenance")
            if self.may_update_zone_membership or self.may_extend_trajectory:
                raise ValueError("inferred state cannot update measured spatial facts")
        else:
            if self.raw_world_xyz_m is not None or self.presentation_world_xyz_m is not None:
                raise ValueError("missing/occluded states cannot carry XYZ")
            if self.coordinate_provenance is not TemporalCoordinateProvenance.NONE:
                raise ValueError("missing/occluded states require no-coordinate provenance")
            if self.anchor_kind is not None:
                raise ValueError("missing/occluded states cannot claim an anchor kind")
            if self.may_update_zone_membership or self.may_extend_trajectory:
                raise ValueError("missing/occluded states cannot update spatial facts")
        expected = self.create_record_id(
            policy_id=self.policy_id,
            target=self.target,
            source_frame_index=self.source_frame_index,
            capture_timestamp_seconds=self.capture_timestamp_seconds,
        )
        if self.record_id != expected:
            raise ValueError("temporal presentation record ID differs")
        return self

    @classmethod
    def create_record_id(
        cls,
        *,
        policy_id: str,
        target: PerceptionTarget,
        source_frame_index: int,
        capture_timestamp_seconds: float,
    ) -> str:
        return _stable_digest(
            {
                "schema_version": 1,
                "policy_id": policy_id,
                "target": target.value,
                "source_frame_index": source_frame_index,
                "capture_timestamp_seconds": capture_timestamp_seconds,
            }
        )


class MeasuredTrajectorySegment(ContractModel):
    """A line between adjacent, semantically matching measured endpoints."""

    schema_version: Literal[1] = 1
    segment_id: Sha256Digest
    policy_id: Literal["s04_temporal_presentation_v1"]
    target: PerceptionTarget
    anchor_kind: CorrectedAnchorKind
    start_observation_id: Sha256Digest
    end_observation_id: Sha256Digest
    start_source_frame_index: NonNegativeInt
    end_source_frame_index: NonNegativeInt
    start_timestamp_seconds: NonNegativeFloat
    end_timestamp_seconds: NonNegativeFloat
    start_world_xyz_m: Vector3
    end_world_xyz_m: Vector3
    elapsed_seconds: PositiveFloat
    distance_m: NonNegativeFloat
    maximum_allowed_gap_seconds: PositiveFloat
    interpolation_performed: Literal[False] = False
    stale_points_used: Literal[False] = False

    @model_validator(mode="after")
    def validate_segment(self) -> Self:
        if self.end_source_frame_index <= self.start_source_frame_index:
            raise ValueError("trajectory segment frames must increase")
        if self.end_timestamp_seconds <= self.start_timestamp_seconds:
            raise ValueError("trajectory segment timestamps must increase")
        if self.elapsed_seconds > self.maximum_allowed_gap_seconds:
            raise ValueError("trajectory segment exceeds the allowed time gap")
        expected_elapsed = self.end_timestamp_seconds - self.start_timestamp_seconds
        if abs(self.elapsed_seconds - expected_elapsed) > 1e-9:
            raise ValueError("trajectory segment elapsed time differs")
        expected_distance = float(
            np.linalg.norm(
                np.asarray(self.end_world_xyz_m) - np.asarray(self.start_world_xyz_m)
            )
        )
        if abs(self.distance_m - expected_distance) > 1e-9:
            raise ValueError("trajectory segment distance differs")
        expected = self.create_segment_id(
            policy_id=self.policy_id,
            target=self.target,
            anchor_kind=self.anchor_kind,
            start_observation_id=self.start_observation_id,
            end_observation_id=self.end_observation_id,
        )
        if self.segment_id != expected:
            raise ValueError("trajectory segment ID differs")
        return self

    @classmethod
    def create_segment_id(
        cls,
        *,
        policy_id: str,
        target: PerceptionTarget,
        anchor_kind: CorrectedAnchorKind,
        start_observation_id: str,
        end_observation_id: str,
    ) -> str:
        return _stable_digest(
            {
                "schema_version": 1,
                "policy_id": policy_id,
                "target": target.value,
                "anchor_kind": anchor_kind.value,
                "start_observation_id": start_observation_id,
                "end_observation_id": end_observation_id,
            }
        )


class TemporalPresentationRunSummary(ContractModel):
    """Persistent D034 timeline and measured-segment artifact contract."""

    schema_version: Literal[1] = 1
    stage: Literal["S04"] = "S04"
    status: Literal["completed_pending_visual_qa"]
    created_at_utc: datetime
    policy: TemporalPresentationPolicy
    source_corrected_summary_ref: str
    source_corrected_summary_sha256: Sha256Digest
    source_corrected_verification_ref: str
    source_corrected_verification_sha256: Sha256Digest
    source_perception_summary_ref: str
    source_perception_summary_sha256: Sha256Digest
    source_camera_a_timeline_ref: str
    source_camera_a_timeline_sha256: Sha256Digest
    source_camera_b_timeline_ref: str
    source_camera_b_timeline_sha256: Sha256Digest
    source_visibility_summary_ref: str | None = None
    source_visibility_summary_sha256: Sha256Digest | None = None
    presentation_records: tuple[TemporalPresentationRecord, ...]
    measured_trajectory_segments: tuple[MeasuredTrajectorySegment, ...]
    state_counts: dict[str, NonNegativeInt]
    anchor_kind_counts: dict[str, NonNegativeInt]
    timeline_records_ref: str
    timeline_records_sha256: Sha256Digest
    trajectory_segments_ref: str
    trajectory_segments_sha256: Sha256Digest
    review_csv_ref: str
    review_csv_sha256: Sha256Digest
    timeline_diagnostic_ref: str
    timeline_diagnostic_sha256: Sha256Digest
    world_preview_ref: str
    world_preview_sha256: Sha256Digest
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        if (self.source_visibility_summary_ref is None) != (
            self.source_visibility_summary_sha256 is None
        ):
            raise ValueError("visibility summary reference and hash must appear together")
        if len(self.presentation_records) != 320:
            raise ValueError("D034 run requires two targets across 160 timeline ticks")
        if any(
            record.state is TemporalPresentationState.INFERRED
            for record in self.presentation_records
        ):
            raise ValueError("D034 policy forbids inferred positions")
        if any(
            record.state is TemporalPresentationState.STALE
            and record.measurement_age_seconds is not None
            and record.measurement_age_seconds > self.policy.maximum_stale_age_seconds
            for record in self.presentation_records
        ):
            raise ValueError("stale record exceeds the policy horizon")
        return self


def resolve_temporal_presentation(
    *,
    source_frame_index: int,
    capture_timestamp_seconds: float,
    target: PerceptionTarget,
    camera_a_perception_state: PerceptionPresenceState,
    camera_b_perception_state: PerceptionPresenceState,
    current_observation: CorrectedPairObservationRecord | None,
    last_measurement: CorrectedPairObservationRecord | None,
    confirmed_occluded: bool,
    policy: TemporalPresentationPolicy,
) -> TemporalPresentationResolution:
    """Resolve one tick without interpolation, extrapolation, or hidden XYZ."""

    if current_observation is not None:
        _validate_current_observation(
            current_observation,
            source_frame_index=source_frame_index,
            capture_timestamp_seconds=capture_timestamp_seconds,
            target=target,
        )
        if current_observation.state in {
            CorrectedPairState.FUSED,
            CorrectedPairState.SINGLE_CAMERA,
        }:
            assert current_observation.world_xyz_m is not None
            assert current_observation.selected_kind is not None
            xyz = cast(
                tuple[float, float, float],
                tuple(float(value) for value in current_observation.world_xyz_m),
            )
            return TemporalPresentationResolution(
                state=TemporalPresentationState.MEASURED,
                coordinate_provenance=TemporalCoordinateProvenance.CURRENT_MEASUREMENT,
                reason=TemporalPresentationReason.CURRENT_CORRECTED_OBSERVATION,
                raw_world_xyz_m=xyz,
                presentation_world_xyz_m=xyz,
                anchor_kind=current_observation.selected_kind,
                source_observation_id=current_observation.observation_id,
                source_measurement_frame_index=current_observation.source_frame_index,
                source_measurement_timestamp_seconds=(
                    current_observation.capture_timestamp_seconds
                ),
                source_measurement_camera_ids=current_observation.selected_camera_ids,
                measurement_age_seconds=0.0,
                may_update_zone_membership=True,
                may_extend_trajectory=True,
            )

    if confirmed_occluded:
        return _unavailable_resolution(
            state=TemporalPresentationState.OCCLUDED,
            reason=TemporalPresentationReason.CONFIRMED_OCCLUSION,
        )

    if last_measurement is not None:
        if last_measurement.world_xyz_m is None or last_measurement.selected_kind is None:
            raise ValueError("last measurement must contain selected kind and XYZ")
        age = capture_timestamp_seconds - last_measurement.capture_timestamp_seconds
        if age < 0:
            raise ValueError("last measurement cannot be in the future")
        if 0 < age <= policy.maximum_stale_age_seconds + 1e-9:
            xyz = cast(
                tuple[float, float, float],
                tuple(float(value) for value in last_measurement.world_xyz_m),
            )
            return TemporalPresentationResolution(
                state=TemporalPresentationState.STALE,
                coordinate_provenance=TemporalCoordinateProvenance.STALE_HOLD,
                reason=TemporalPresentationReason.RECENT_MEASUREMENT_HELD_STALE,
                raw_world_xyz_m=None,
                presentation_world_xyz_m=xyz,
                anchor_kind=last_measurement.selected_kind,
                source_observation_id=last_measurement.observation_id,
                source_measurement_frame_index=last_measurement.source_frame_index,
                source_measurement_timestamp_seconds=(
                    last_measurement.capture_timestamp_seconds
                ),
                source_measurement_camera_ids=last_measurement.selected_camera_ids,
                measurement_age_seconds=age,
                may_update_zone_membership=False,
                may_extend_trajectory=False,
            )

    reason = _missing_reason(
        last_measurement=last_measurement,
        camera_a_perception_state=camera_a_perception_state,
        camera_b_perception_state=camera_b_perception_state,
    )
    return _unavailable_resolution(
        state=TemporalPresentationState.MISSING,
        reason=reason,
    )


def make_temporal_record(
    *,
    source_frame_index: int,
    capture_timestamp_seconds: float,
    target: PerceptionTarget,
    camera_a_perception_state: PerceptionPresenceState,
    camera_b_perception_state: PerceptionPresenceState,
    resolution: TemporalPresentationResolution,
    policy: TemporalPresentationPolicy,
) -> TemporalPresentationRecord:
    """Assign deterministic identity and visual semantics to a resolution."""

    record_id = TemporalPresentationRecord.create_record_id(
        policy_id=policy.policy_id,
        target=target,
        source_frame_index=source_frame_index,
        capture_timestamp_seconds=capture_timestamp_seconds,
    )
    return TemporalPresentationRecord(
        record_id=record_id,
        policy_id=policy.policy_id,
        target=target,
        source_frame_index=source_frame_index,
        capture_timestamp_seconds=capture_timestamp_seconds,
        camera_a_perception_state=camera_a_perception_state,
        camera_b_perception_state=camera_b_perception_state,
        state=resolution.state,
        coordinate_provenance=resolution.coordinate_provenance,
        reason=resolution.reason,
        raw_world_xyz_m=resolution.raw_world_xyz_m,
        presentation_world_xyz_m=resolution.presentation_world_xyz_m,
        anchor_kind=resolution.anchor_kind,
        source_observation_id=resolution.source_observation_id,
        source_measurement_frame_index=resolution.source_measurement_frame_index,
        source_measurement_timestamp_seconds=(
            resolution.source_measurement_timestamp_seconds
        ),
        source_measurement_camera_ids=resolution.source_measurement_camera_ids,
        measurement_age_seconds=resolution.measurement_age_seconds,
        may_update_zone_membership=resolution.may_update_zone_membership,
        may_extend_trajectory=resolution.may_extend_trajectory,
        visual_style_id=_visual_style(resolution.state, resolution.anchor_kind),
    )


def build_measured_trajectory_segments(
    observations: tuple[CorrectedPairObservationRecord, ...],
    *,
    policy: TemporalPresentationPolicy,
) -> tuple[MeasuredTrajectorySegment, ...]:
    """Connect only adjacent matching measurements inside the time-gap gate."""

    segments: list[MeasuredTrajectorySegment] = []
    for target in PerceptionTarget:
        ordered = sorted(
            (item for item in observations if item.target is target),
            key=lambda item: item.capture_timestamp_seconds,
        )
        for start, end in zip(ordered, ordered[1:], strict=False):
            if (
                start.world_xyz_m is None
                or end.world_xyz_m is None
                or start.selected_kind is None
                or end.selected_kind is None
                or start.selected_kind is not end.selected_kind
            ):
                continue
            elapsed = end.capture_timestamp_seconds - start.capture_timestamp_seconds
            if elapsed <= 0 or elapsed > policy.maximum_trajectory_segment_gap_seconds:
                continue
            distance = float(
                np.linalg.norm(
                    np.asarray(end.world_xyz_m) - np.asarray(start.world_xyz_m)
                )
            )
            segment_id = MeasuredTrajectorySegment.create_segment_id(
                policy_id=policy.policy_id,
                target=target,
                anchor_kind=start.selected_kind,
                start_observation_id=start.observation_id,
                end_observation_id=end.observation_id,
            )
            segments.append(
                MeasuredTrajectorySegment(
                    segment_id=segment_id,
                    policy_id=policy.policy_id,
                    target=target,
                    anchor_kind=start.selected_kind,
                    start_observation_id=start.observation_id,
                    end_observation_id=end.observation_id,
                    start_source_frame_index=start.source_frame_index,
                    end_source_frame_index=end.source_frame_index,
                    start_timestamp_seconds=start.capture_timestamp_seconds,
                    end_timestamp_seconds=end.capture_timestamp_seconds,
                    start_world_xyz_m=start.world_xyz_m,
                    end_world_xyz_m=end.world_xyz_m,
                    elapsed_seconds=elapsed,
                    distance_m=distance,
                    maximum_allowed_gap_seconds=(
                        policy.maximum_trajectory_segment_gap_seconds
                    ),
                )
            )
    return tuple(segments)


def _validate_current_observation(
    observation: CorrectedPairObservationRecord,
    *,
    source_frame_index: int,
    capture_timestamp_seconds: float,
    target: PerceptionTarget,
) -> None:
    if observation.source_frame_index != source_frame_index:
        raise ValueError("current observation frame differs from timeline tick")
    if abs(observation.capture_timestamp_seconds - capture_timestamp_seconds) > 1e-9:
        raise ValueError("current observation timestamp differs from timeline tick")
    if observation.target is not target:
        raise ValueError("current observation target differs from timeline tick")


def _missing_reason(
    *,
    last_measurement: CorrectedPairObservationRecord | None,
    camera_a_perception_state: PerceptionPresenceState,
    camera_b_perception_state: PerceptionPresenceState,
) -> TemporalPresentationReason:
    states = (camera_a_perception_state, camera_b_perception_state)
    if PerceptionPresenceState.FAILED in states:
        return TemporalPresentationReason.SOURCE_FAILURE
    if PerceptionPresenceState.OBSERVED in states:
        return TemporalPresentationReason.NO_CURRENT_DEPTH
    if last_measurement is None:
        return TemporalPresentationReason.NO_PRIOR_MEASUREMENT
    if all(state is PerceptionPresenceState.MISSING for state in states):
        return TemporalPresentationReason.SOURCE_TARGET_UNAVAILABLE
    return TemporalPresentationReason.STALE_HORIZON_EXPIRED


def _unavailable_resolution(
    *, state: TemporalPresentationState, reason: TemporalPresentationReason
) -> TemporalPresentationResolution:
    return TemporalPresentationResolution(
        state=state,
        coordinate_provenance=TemporalCoordinateProvenance.NONE,
        reason=reason,
        raw_world_xyz_m=None,
        presentation_world_xyz_m=None,
        anchor_kind=None,
        source_observation_id=None,
        source_measurement_frame_index=None,
        source_measurement_timestamp_seconds=None,
        source_measurement_camera_ids=(),
        measurement_age_seconds=None,
        may_update_zone_membership=False,
        may_extend_trajectory=False,
    )


def _visual_style(
    state: TemporalPresentationState, anchor_kind: CorrectedAnchorKind | None
) -> str:
    if state in {TemporalPresentationState.MISSING, TemporalPresentationState.OCCLUDED}:
        return state.value
    if anchor_kind is None:
        return state.value
    return f"{state.value}:{anchor_kind.value}"


def _stable_digest(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
