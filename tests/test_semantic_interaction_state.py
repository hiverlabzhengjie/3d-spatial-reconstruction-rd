from __future__ import annotations

import pytest
from pydantic import ValidationError

from spatial_reconstruction.contracts import PerceptionTarget
from spatial_reconstruction.interaction import (
    InteractionPhase,
    InteractionZone,
    LocalizationAvailability,
    PhaseAuthority,
    SemanticInteractionPolicy,
    SemanticInteractionRecord,
    resolve_semantic_interaction_tick,
)
from spatial_reconstruction.localization import (
    CorrectedAnchorKind,
    TemporalCoordinateProvenance,
    TemporalPresentationReason,
    TemporalPresentationRecord,
    TemporalPresentationState,
)
from spatial_reconstruction.perception import (
    BackpackVisibilityRecord,
    BackpackVisibilityState,
    PerceptionPresenceState,
    VisibilityEvidenceSource,
)

SHA_A = "a" * 64

PICKUP_ZONE = InteractionZone(
    zone_id="pickup",
    role="pickup",
    center_world_m=(0.0, 0.0, 0.0),
    radius_m=0.3,
    coordinate_source="test",
)
DROPOFF_ZONE = InteractionZone(
    zone_id="dropoff",
    role="dropoff",
    center_world_m=(2.0, 0.0, 0.0),
    radius_m=0.3,
    coordinate_source="test",
)


def _temporal(
    *,
    target: PerceptionTarget,
    frame: int,
    timestamp: float,
    state: TemporalPresentationState,
    xyz: tuple[float, float, float] | None,
) -> TemporalPresentationRecord:
    record_id = TemporalPresentationRecord.create_record_id(
        policy_id="s04_temporal_presentation_v1",
        target=target,
        source_frame_index=frame,
        capture_timestamp_seconds=timestamp,
    )
    if state is TemporalPresentationState.MEASURED:
        assert xyz is not None
        anchor = (
            CorrectedAnchorKind.PERSON_FOOTPOINT
            if target is PerceptionTarget.PERSON
            else CorrectedAnchorKind.BACKPACK_VISIBLE_CLUSTER
        )
        return TemporalPresentationRecord(
            record_id=record_id,
            policy_id="s04_temporal_presentation_v1",
            target=target,
            source_frame_index=frame,
            capture_timestamp_seconds=timestamp,
            camera_a_perception_state=PerceptionPresenceState.OBSERVED,
            camera_b_perception_state=PerceptionPresenceState.MISSING,
            state=state,
            coordinate_provenance=TemporalCoordinateProvenance.CURRENT_MEASUREMENT,
            reason=TemporalPresentationReason.CURRENT_CORRECTED_OBSERVATION,
            raw_world_xyz_m=xyz,
            presentation_world_xyz_m=xyz,
            anchor_kind=anchor,
            source_observation_id=SHA_A,
            source_measurement_frame_index=frame,
            source_measurement_timestamp_seconds=timestamp,
            source_measurement_camera_ids=("camera_a",),
            measurement_age_seconds=0.0,
            may_update_zone_membership=True,
            may_extend_trajectory=True,
            visual_style_id="measured",
        )
    return TemporalPresentationRecord(
        record_id=record_id,
        policy_id="s04_temporal_presentation_v1",
        target=target,
        source_frame_index=frame,
        capture_timestamp_seconds=timestamp,
        camera_a_perception_state=PerceptionPresenceState.MISSING,
        camera_b_perception_state=PerceptionPresenceState.MISSING,
        state=state,
        coordinate_provenance=TemporalCoordinateProvenance.NONE,
        reason=(
            TemporalPresentationReason.CONFIRMED_OCCLUSION
            if state is TemporalPresentationState.OCCLUDED
            else TemporalPresentationReason.NO_CURRENT_DEPTH
        ),
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
        visual_style_id=state.value,
    )


def _visibility(
    *, frame: int, timestamp: float, partial: bool
) -> BackpackVisibilityRecord:
    state = (
        BackpackVisibilityState.PARTIALLY_OCCLUDED
        if partial
        else BackpackVisibilityState.VISIBLE
    )
    source = (
        VisibilityEvidenceSource.SYNCHRONIZED_VIDEO_REVIEW
        if partial
        else VisibilityEvidenceSource.DETECTOR_OBSERVATION
    )
    return BackpackVisibilityRecord(
        record_id=BackpackVisibilityRecord.create_record_id(
            policy_id="s05_backpack_visibility_overlay_v1",
            source_frame_index=frame,
            capture_timestamp_seconds=timestamp,
        ),
        policy_id="s05_backpack_visibility_overlay_v1",
        source_frame_index=frame,
        capture_timestamp_seconds=timestamp,
        camera_a_detection_state=(
            PerceptionPresenceState.MISSING
            if partial
            else PerceptionPresenceState.OBSERVED
        ),
        camera_b_detection_state=PerceptionPresenceState.MISSING,
        visibility_state=state,
        evidence_source=source,
        confirmed_occluded_for_localization=partial,
        rationale="test visibility evidence",
        evidence_refs=("test-frame.jpg",) if partial else (),
    )


def _tick(
    *,
    frame: int,
    timestamp: float,
    backpack_state: TemporalPresentationState,
    backpack_xyz: tuple[float, float, float] | None,
    person_xyz: tuple[float, float, float],
    partial: bool,
    previous: SemanticInteractionRecord | None,
    policy: SemanticInteractionPolicy,
) -> SemanticInteractionRecord:
    return resolve_semantic_interaction_tick(
        person=_temporal(
            target=PerceptionTarget.PERSON,
            frame=frame,
            timestamp=timestamp,
            state=TemporalPresentationState.MEASURED,
            xyz=person_xyz,
        ),
        backpack=_temporal(
            target=PerceptionTarget.BACKPACK,
            frame=frame,
            timestamp=timestamp,
            state=backpack_state,
            xyz=backpack_xyz,
        ),
        visibility=_visibility(frame=frame, timestamp=timestamp, partial=partial),
        previous=previous,
        pickup_zone=PICKUP_ZONE,
        dropoff_zone=DROPOFF_ZONE,
        policy=policy,
    )


def test_carry_phase_coexists_with_partial_occlusion_and_unavailable_xyz() -> None:
    policy = SemanticInteractionPolicy()
    at_pickup = _tick(
        frame=0,
        timestamp=0.0,
        backpack_state=TemporalPresentationState.MEASURED,
        backpack_xyz=(0.1, 0.0, 0.0),
        person_xyz=(0.5, 0.0, 0.0),
        partial=False,
        previous=None,
        policy=policy,
    )
    pickup = _tick(
        frame=6,
        timestamp=0.2,
        backpack_state=TemporalPresentationState.MEASURED,
        backpack_xyz=(0.4, 0.0, 0.0),
        person_xyz=(0.5, 0.0, 0.0),
        partial=False,
        previous=at_pickup,
        policy=policy,
    )
    carry = _tick(
        frame=12,
        timestamp=0.4,
        backpack_state=TemporalPresentationState.OCCLUDED,
        backpack_xyz=None,
        person_xyz=(0.8, 0.0, 0.0),
        partial=True,
        previous=pickup,
        policy=policy,
    )

    assert carry.phase is InteractionPhase.CARRY
    assert carry.phase_authority is PhaseAuthority.SEQUENCE_CONTINUITY
    assert carry.visibility_state is BackpackVisibilityState.PARTIALLY_OCCLUDED
    assert carry.localization_availability is LocalizationAvailability.UNAVAILABLE
    assert carry.backpack_world_xyz_m is None
    assert not carry.phase_has_current_spatial_authority


def test_unlocalized_carry_expires_without_a_place_measurement() -> None:
    policy = SemanticInteractionPolicy(maximum_unlocalized_carry_seconds=1.0)
    at_pickup = _tick(
        frame=0,
        timestamp=0.0,
        backpack_state=TemporalPresentationState.MEASURED,
        backpack_xyz=(0.1, 0.0, 0.0),
        person_xyz=(0.5, 0.0, 0.0),
        partial=False,
        previous=None,
        policy=policy,
    )
    pickup = _tick(
        frame=6,
        timestamp=0.2,
        backpack_state=TemporalPresentationState.MEASURED,
        backpack_xyz=(0.4, 0.0, 0.0),
        person_xyz=(0.5, 0.0, 0.0),
        partial=False,
        previous=at_pickup,
        policy=policy,
    )
    expired = _tick(
        frame=42,
        timestamp=1.4,
        backpack_state=TemporalPresentationState.OCCLUDED,
        backpack_xyz=None,
        person_xyz=(1.0, 0.0, 0.0),
        partial=True,
        previous=pickup,
        policy=policy,
    )
    assert expired.phase is InteractionPhase.UNKNOWN
    assert expired.phase_authority is PhaseAuthority.NONE


def test_partial_occlusion_requires_explicit_review_not_detector_missing() -> None:
    with pytest.raises(ValidationError, match="occlusion requires synchronized-video review"):
        BackpackVisibilityRecord(
            record_id=BackpackVisibilityRecord.create_record_id(
                policy_id="s05_backpack_visibility_overlay_v1",
                source_frame_index=12,
                capture_timestamp_seconds=0.4,
            ),
            policy_id="s05_backpack_visibility_overlay_v1",
            source_frame_index=12,
            capture_timestamp_seconds=0.4,
            camera_a_detection_state=PerceptionPresenceState.MISSING,
            camera_b_detection_state=PerceptionPresenceState.MISSING,
            visibility_state=BackpackVisibilityState.PARTIALLY_OCCLUDED,
            evidence_source=VisibilityEvidenceSource.NONE,
            confirmed_occluded_for_localization=True,
            rationale="detector missing only",
        )
