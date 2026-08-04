from __future__ import annotations

import pytest

from spatial_reconstruction.contracts import PerceptionTarget
from spatial_reconstruction.interaction import (
    BackpackInteractionState,
    InteractionEventKind,
    InteractionEvidence,
    InteractionPolicy,
    InteractionTransitionReason,
    InteractionZone,
    InteractionZoneMembership,
    build_event_candidates,
    build_interaction_evidence,
    resolve_interaction_state,
)
from spatial_reconstruction.localization import (
    CorrectedAnchorKind,
    TemporalCoordinateProvenance,
    TemporalPresentationReason,
    TemporalPresentationRecord,
    TemporalPresentationState,
)
from spatial_reconstruction.perception import PerceptionPresenceState

SHA_C = "c" * 64


def _zone(
    *,
    zone_id: str,
    role: str,
    center: tuple[float, float, float],
    radius: float = 0.3,
) -> InteractionZone:
    return InteractionZone(
        zone_id=zone_id,
        role=role,  # type: ignore[arg-type]
        center_world_m=center,
        radius_m=radius,
        coordinate_source="video_estimated_and_user_validated",
    )


PICKUP_ZONE = _zone(zone_id="pickup", role="pickup", center=(0.0, 0.0, 0.6))
DROPOFF_ZONE = _zone(zone_id="dropoff", role="dropoff", center=(2.0, 0.0, 0.0))


def _record(
    *,
    target: PerceptionTarget,
    frame: int,
    timestamp: float,
    state: TemporalPresentationState = TemporalPresentationState.MEASURED,
    xyz: tuple[float, float, float] | None = None,
    anchor_kind: CorrectedAnchorKind | None = None,
) -> TemporalPresentationRecord:
    record_id = TemporalPresentationRecord.create_record_id(
        policy_id="s04_temporal_presentation_v1",
        target=target,
        source_frame_index=frame,
        capture_timestamp_seconds=timestamp,
    )
    if state is TemporalPresentationState.MEASURED:
        if xyz is None:
            xyz = (0.0, 0.0, 0.0)
        if anchor_kind is None:
            anchor_kind = (
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
            anchor_kind=anchor_kind,
            source_observation_id=SHA_C,
            source_measurement_frame_index=frame,
            source_measurement_timestamp_seconds=timestamp,
            source_measurement_camera_ids=("camera_a",),
            measurement_age_seconds=0.0,
            may_update_zone_membership=True,
            may_extend_trajectory=True,
            visual_style_id="measured",
        )

    if state is TemporalPresentationState.STALE:
        if xyz is None:
            xyz = (0.0, 0.0, 0.0)
        if anchor_kind is None:
            anchor_kind = (
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
            camera_a_perception_state=PerceptionPresenceState.MISSING,
            camera_b_perception_state=PerceptionPresenceState.MISSING,
            state=state,
            coordinate_provenance=TemporalCoordinateProvenance.STALE_HOLD,
            reason=TemporalPresentationReason.RECENT_MEASUREMENT_HELD_STALE,
            raw_world_xyz_m=None,
            presentation_world_xyz_m=xyz,
            anchor_kind=anchor_kind,
            source_observation_id=SHA_C,
            source_measurement_frame_index=frame - 1,
            source_measurement_timestamp_seconds=timestamp - 0.2,
            source_measurement_camera_ids=("camera_a",),
            measurement_age_seconds=0.2,
            may_update_zone_membership=False,
            may_extend_trajectory=False,
            visual_style_id="stale",
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


def _evidence(
    *,
    frame: int,
    timestamp: float,
    backpack_xyz: tuple[float, float, float] | None,
    person_xyz: tuple[float, float, float] | None,
    backpack_state: TemporalPresentationState = TemporalPresentationState.MEASURED,
    person_state: TemporalPresentationState = TemporalPresentationState.MEASURED,
    person_kind: CorrectedAnchorKind = CorrectedAnchorKind.PERSON_FOOTPOINT,
) -> tuple[InteractionPolicy, InteractionEvidence]:
    policy = InteractionPolicy()
    person = _record(
        target=PerceptionTarget.PERSON,
        frame=frame,
        timestamp=timestamp,
        state=person_state,
        xyz=person_xyz,
        anchor_kind=person_kind if person_state is TemporalPresentationState.MEASURED else None,
    )
    backpack = _record(
        target=PerceptionTarget.BACKPACK,
        frame=frame,
        timestamp=timestamp,
        state=backpack_state,
        xyz=backpack_xyz,
    )
    return policy, build_interaction_evidence(
        person_record=person,
        backpack_record=backpack,
        pickup_zone=PICKUP_ZONE,
        dropoff_zone=DROPOFF_ZONE,
        policy=policy,
    )


def test_pickup_carry_unknown_gap_and_place_sequence() -> None:
    policy, at_pickup_evidence = _evidence(
        frame=10,
        timestamp=1.0,
        backpack_xyz=(0.1, 0.0, 0.8),
        person_xyz=(0.7, 0.0, 0.0),
    )
    at_pickup = resolve_interaction_state(
        previous=None, evidence=at_pickup_evidence, policy=policy
    )
    assert at_pickup.state is BackpackInteractionState.AT_PICKUP

    _, pickup_evidence = _evidence(
        frame=20,
        timestamp=2.0,
        backpack_xyz=(0.4, 0.0, 0.8),
        person_xyz=(0.6, 0.0, 0.0),
    )
    pickup = resolve_interaction_state(
        previous=at_pickup, evidence=pickup_evidence, policy=policy
    )
    assert pickup.state is BackpackInteractionState.PICKUP
    assert pickup.pickup_confirmed

    _, carry_evidence = _evidence(
        frame=30,
        timestamp=3.0,
        backpack_xyz=(1.0, 0.0, 0.8),
        person_xyz=(1.2, 0.0, 0.0),
    )
    carry = resolve_interaction_state(
        previous=pickup, evidence=carry_evidence, policy=policy
    )
    assert carry.state is BackpackInteractionState.CARRY

    _, gap_evidence = _evidence(
        frame=40,
        timestamp=4.0,
        backpack_xyz=None,
        person_xyz=(1.5, 0.0, 0.0),
        backpack_state=TemporalPresentationState.MISSING,
    )
    gap = resolve_interaction_state(previous=carry, evidence=gap_evidence, policy=policy)
    assert gap.state is BackpackInteractionState.UNKNOWN
    assert gap.last_authoritative_state is BackpackInteractionState.CARRY
    assert not gap.spatial_transition_authority

    _, place_evidence = _evidence(
        frame=50,
        timestamp=5.0,
        backpack_xyz=(2.0, 0.1, 0.2),
        person_xyz=(2.2, 0.1, 0.0),
    )
    place = resolve_interaction_state(previous=gap, evidence=place_evidence, policy=policy)
    assert place.state is BackpackInteractionState.PLACE
    assert place.reason is InteractionTransitionReason.PLACE_SPATIAL_EVIDENCE

    _, later_gap_evidence = _evidence(
        frame=60,
        timestamp=6.0,
        backpack_xyz=None,
        person_xyz=(2.3, 0.1, 0.0),
        backpack_state=TemporalPresentationState.MISSING,
    )
    later_gap = resolve_interaction_state(
        previous=place, evidence=later_gap_evidence, policy=policy
    )
    _, repeated_place_evidence = _evidence(
        frame=70,
        timestamp=7.0,
        backpack_xyz=(2.0, 0.1, 0.2),
        person_xyz=(2.4, 0.1, 0.0),
    )
    repeated_place = resolve_interaction_state(
        previous=later_gap, evidence=repeated_place_evidence, policy=policy
    )
    candidates = build_event_candidates(
        state_records=(
            at_pickup,
            pickup,
            carry,
            gap,
            place,
            later_gap,
            repeated_place,
        ),
        video_duration_seconds=10.0,
        policy=policy,
    )
    assert [candidate.event_kind for candidate in candidates] == [
        InteractionEventKind.PICKUP,
        InteractionEventKind.CARRY,
        InteractionEventKind.PLACE,
    ]


def test_stale_backpack_coordinate_cannot_create_zone_or_transition_fact() -> None:
    policy, evidence = _evidence(
        frame=20,
        timestamp=2.0,
        backpack_xyz=(0.0, 0.0, 0.8),
        person_xyz=(0.1, 0.0, 0.0),
        backpack_state=TemporalPresentationState.STALE,
    )

    assert evidence.backpack_zone_membership is InteractionZoneMembership.UNKNOWN
    assert evidence.backpack_pickup_center_distance_xy_m is None
    assert not evidence.backpack_spatial_authority
    result = resolve_interaction_state(previous=None, evidence=evidence, policy=policy)
    assert result.state is BackpackInteractionState.UNKNOWN


def test_explicit_occlusion_has_no_spatial_authority() -> None:
    policy, evidence = _evidence(
        frame=20,
        timestamp=2.0,
        backpack_xyz=None,
        person_xyz=(0.1, 0.0, 0.0),
        backpack_state=TemporalPresentationState.OCCLUDED,
    )

    result = resolve_interaction_state(previous=None, evidence=evidence, policy=policy)
    assert result.state is BackpackInteractionState.OCCLUDED
    assert result.reason is InteractionTransitionReason.CONFIRMED_OCCLUSION
    assert result.backpack_zone_membership is InteractionZoneMembership.UNKNOWN
    assert not result.spatial_transition_authority


def test_initial_dropoff_measurement_does_not_invent_prior_pickup() -> None:
    policy, evidence = _evidence(
        frame=10,
        timestamp=1.0,
        backpack_xyz=(2.0, 0.0, 0.2),
        person_xyz=(2.1, 0.0, 0.0),
    )

    result = resolve_interaction_state(previous=None, evidence=evidence, policy=policy)
    assert result.state is BackpackInteractionState.UNKNOWN
    assert result.reason is InteractionTransitionReason.MEASURED_SEQUENCE_UNPROVEN
    assert not result.pickup_confirmed


def test_person_anchor_semantics_are_retained_without_footpoint_conversion() -> None:
    policy, evidence = _evidence(
        frame=10,
        timestamp=1.0,
        backpack_xyz=(0.1, 0.0, 0.8),
        person_xyz=(0.2, 0.0, 1.3),
        person_kind=CorrectedAnchorKind.PERSON_UPPER_BODY_SURFACE,
    )

    assert evidence.person_anchor_kind is CorrectedAnchorKind.PERSON_UPPER_BODY_SURFACE
    assert evidence.person_backpack_distance_xy_m == pytest.approx(0.1)


def test_state_machine_rejects_out_of_order_capture_time() -> None:
    policy, first_evidence = _evidence(
        frame=20,
        timestamp=2.0,
        backpack_xyz=(0.0, 0.0, 0.8),
        person_xyz=(0.5, 0.0, 0.0),
    )
    first = resolve_interaction_state(previous=None, evidence=first_evidence, policy=policy)
    _, older_evidence = _evidence(
        frame=10,
        timestamp=1.0,
        backpack_xyz=(0.0, 0.0, 0.8),
        person_xyz=(0.5, 0.0, 0.0),
    )

    with pytest.raises(ValueError, match="frames must increase"):
        resolve_interaction_state(previous=first, evidence=older_evidence, policy=policy)


def test_qwen_cannot_be_enabled_as_spatial_state_authority() -> None:
    with pytest.raises(ValueError):
        InteractionPolicy(qwen_may_change_spatial_state=True)  # type: ignore[arg-type]


def test_interaction_state_record_round_trips_through_persistent_schema() -> None:
    policy, evidence = _evidence(
        frame=10,
        timestamp=1.0,
        backpack_xyz=(0.1, 0.0, 0.8),
        person_xyz=(0.7, 0.0, 0.0),
    )
    record = resolve_interaction_state(previous=None, evidence=evidence, policy=policy)

    assert type(record).model_validate_json(record.model_dump_json()) == record
