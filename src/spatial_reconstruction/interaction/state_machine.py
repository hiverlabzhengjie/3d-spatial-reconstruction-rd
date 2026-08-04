"""Legacy D036 measured-only interaction state retained for artifact replay.

New S05 work must use :mod:`semantic_state`, which separates interaction phase,
visibility, and localization under D037.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self

import numpy as np
from pydantic import field_validator, model_validator

from spatial_reconstruction.contracts import (
    ContractModel,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    Sha256Digest,
    Vector3,
)
from spatial_reconstruction.localization import (
    CorrectedAnchorKind,
    TemporalPresentationRecord,
    TemporalPresentationState,
)


class BackpackInteractionState(StrEnum):
    """Inspectable interaction phase at one authoritative capture-time tick."""

    UNKNOWN = "unknown"
    AT_PICKUP = "at_pickup"
    PICKUP = "pickup"
    CARRY = "carry"
    PLACE = "place"
    OCCLUDED = "occluded"


class InteractionZoneMembership(StrEnum):
    """Measured backpack membership in the two accepted interaction zones."""

    UNKNOWN = "unknown"
    PICKUP = "pickup"
    DROPOFF = "dropoff"
    OUTSIDE = "outside"


class InteractionTransitionReason(StrEnum):
    """Deterministic reason for an interaction-state result."""

    INITIAL_MEASURED_AT_PICKUP = "initial_measured_at_pickup"
    REMAINS_AT_PICKUP = "remains_at_pickup"
    PICKUP_SPATIAL_EVIDENCE = "pickup_spatial_evidence"
    CARRY_SPATIAL_EVIDENCE = "carry_spatial_evidence"
    REMAINS_CARRY = "remains_carry"
    PLACE_SPATIAL_EVIDENCE = "place_spatial_evidence"
    REMAINS_PLACED = "remains_placed"
    CONFIRMED_OCCLUSION = "confirmed_occlusion"
    SPATIAL_EVIDENCE_UNAVAILABLE = "spatial_evidence_unavailable"
    MEASURED_SEQUENCE_UNPROVEN = "measured_sequence_unproven"


class InteractionEventKind(StrEnum):
    """Candidate semantic event emitted by an authoritative state transition."""

    PICKUP = "pickup"
    CARRY = "carry"
    PLACE = "place"


class InteractionZone(ContractModel):
    """Accepted horizontal circular zone in the metric world frame."""

    zone_id: str
    role: Literal["pickup", "dropoff"]
    center_world_m: Vector3
    radius_m: PositiveFloat
    coordinate_source: str

    @field_validator("zone_id", "coordinate_source")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("zone text must be non-empty and trimmed")
        return value


class InteractionPolicy(ContractModel):
    """Bounded S05 spatial authority and transition policy."""

    policy_id: Literal["s05_interaction_state_v1"] = "s05_interaction_state_v1"
    source_presentation_policy_id: Literal["s04_temporal_presentation_v1"] = (
        "s04_temporal_presentation_v1"
    )
    maximum_person_backpack_xy_distance_m: PositiveFloat = 1.0
    minimum_pickup_center_distance_m: PositiveFloat = 0.30
    candidate_clip_pre_seconds: PositiveFloat = 2.0
    candidate_clip_post_seconds: PositiveFloat = 2.0
    zone_membership_uses_xy_only: Literal[True] = True
    current_measurement_required_for_spatial_facts: Literal[True] = True
    stale_coordinates_allowed_for_transitions: Literal[False] = False
    inferred_coordinates_allowed_for_transitions: Literal[False] = False
    qwen_may_change_spatial_state: Literal[False] = False


class InteractionEvidence(ContractModel):
    """Authority-checked evidence for one person/backpack timeline tick."""

    schema_version: Literal[1] = 1
    policy_id: Literal["s05_interaction_state_v1"]
    source_frame_index: NonNegativeInt
    capture_timestamp_seconds: NonNegativeFloat
    backpack_record_id: Sha256Digest
    person_record_id: Sha256Digest
    backpack_presentation_state: TemporalPresentationState
    person_presentation_state: TemporalPresentationState
    backpack_anchor_kind: CorrectedAnchorKind | None
    person_anchor_kind: CorrectedAnchorKind | None
    backpack_zone_membership: InteractionZoneMembership
    backpack_pickup_center_distance_xy_m: NonNegativeFloat | None
    person_backpack_distance_xy_m: NonNegativeFloat | None
    backpack_spatial_authority: bool
    person_backpack_proximity_authority: bool

    @model_validator(mode="after")
    def validate_authority(self) -> Self:
        backpack_measured = (
            self.backpack_presentation_state is TemporalPresentationState.MEASURED
        )
        person_measured = self.person_presentation_state is TemporalPresentationState.MEASURED

        if backpack_measured:
            if not self.backpack_spatial_authority:
                raise ValueError("measured backpack evidence must retain spatial authority")
            if self.backpack_anchor_kind is not CorrectedAnchorKind.BACKPACK_VISIBLE_CLUSTER:
                raise ValueError("measured backpack evidence requires its visible-cluster kind")
            if self.backpack_zone_membership is InteractionZoneMembership.UNKNOWN:
                raise ValueError("measured backpack evidence requires known zone membership")
            if self.backpack_pickup_center_distance_xy_m is None:
                raise ValueError("measured backpack evidence requires pickup-centre distance")
        else:
            if self.backpack_spatial_authority:
                raise ValueError("non-measured backpack evidence cannot have spatial authority")
            if self.backpack_anchor_kind is not None:
                raise ValueError("non-measured backpack evidence cannot claim an anchor")
            if self.backpack_zone_membership is not InteractionZoneMembership.UNKNOWN:
                raise ValueError("non-measured backpack evidence cannot claim zone membership")
            if self.backpack_pickup_center_distance_xy_m is not None:
                raise ValueError("non-measured backpack evidence cannot carry spatial distance")

        if person_measured:
            if self.person_anchor_kind not in {
                CorrectedAnchorKind.PERSON_FOOTPOINT,
                CorrectedAnchorKind.PERSON_LOWER_BODY_SURFACE,
                CorrectedAnchorKind.PERSON_UPPER_BODY_SURFACE,
            }:
                raise ValueError("measured person evidence requires its original anchor kind")
        elif self.person_anchor_kind is not None:
            raise ValueError("non-measured person evidence cannot claim an anchor")

        expected_proximity_authority = backpack_measured and person_measured
        if self.person_backpack_proximity_authority != expected_proximity_authority:
            raise ValueError("person/backpack proximity authority differs from measured inputs")
        if expected_proximity_authority == (self.person_backpack_distance_xy_m is None):
            raise ValueError("person/backpack distance presence differs from spatial authority")
        return self


class InteractionStateRecord(ContractModel):
    """Persistent state-machine result at one capture-time tick."""

    schema_version: Literal[1] = 1
    record_id: Sha256Digest
    policy_id: Literal["s05_interaction_state_v1"]
    source_frame_index: NonNegativeInt
    capture_timestamp_seconds: NonNegativeFloat
    evidence_backpack_record_id: Sha256Digest
    evidence_person_record_id: Sha256Digest
    previous_state: BackpackInteractionState | None
    state: BackpackInteractionState
    last_authoritative_state: BackpackInteractionState | None
    pickup_confirmed: bool
    reason: InteractionTransitionReason
    backpack_zone_membership: InteractionZoneMembership
    backpack_pickup_center_distance_xy_m: NonNegativeFloat | None
    person_backpack_distance_xy_m: NonNegativeFloat | None
    backpack_anchor_kind: CorrectedAnchorKind | None
    person_anchor_kind: CorrectedAnchorKind | None
    spatial_transition_authority: bool
    invented_xyz: Literal[False] = False
    qwen_influenced_spatial_state: Literal[False] = False

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        authoritative_states = {
            BackpackInteractionState.AT_PICKUP,
            BackpackInteractionState.PICKUP,
            BackpackInteractionState.CARRY,
            BackpackInteractionState.PLACE,
        }
        if self.state in authoritative_states:
            if not self.spatial_transition_authority:
                raise ValueError("authoritative interaction state requires measured evidence")
            if self.last_authoritative_state is not self.state:
                raise ValueError("authoritative state must update phase memory")
        else:
            if self.spatial_transition_authority:
                raise ValueError("unknown/occluded state cannot claim spatial authority")
            if self.state is BackpackInteractionState.OCCLUDED and self.reason is not (
                InteractionTransitionReason.CONFIRMED_OCCLUSION
            ):
                raise ValueError("occluded interaction state requires explicit evidence")
        if self.pickup_confirmed and self.last_authoritative_state is (
            BackpackInteractionState.AT_PICKUP
        ):
            raise ValueError("pickup confirmation cannot precede the pickup transition")
        expected = self.create_record_id(
            policy_id=self.policy_id,
            source_frame_index=self.source_frame_index,
            capture_timestamp_seconds=self.capture_timestamp_seconds,
            backpack_record_id=self.evidence_backpack_record_id,
            person_record_id=self.evidence_person_record_id,
        )
        if self.record_id != expected:
            raise ValueError("interaction state record ID differs")
        return self

    @classmethod
    def create_record_id(
        cls,
        *,
        policy_id: str,
        source_frame_index: int,
        capture_timestamp_seconds: float,
        backpack_record_id: str,
        person_record_id: str,
    ) -> str:
        return _stable_digest(
            {
                "schema_version": 1,
                "policy_id": policy_id,
                "source_frame_index": source_frame_index,
                "capture_timestamp_seconds": capture_timestamp_seconds,
                "backpack_record_id": backpack_record_id,
                "person_record_id": person_record_id,
            }
        )


class InteractionEventCandidate(ContractModel):
    """Bounded video-review window around one authoritative interaction change."""

    schema_version: Literal[1] = 1
    candidate_id: Sha256Digest
    policy_id: Literal["s05_interaction_state_v1"]
    event_kind: InteractionEventKind
    source_state_record_id: Sha256Digest
    source_frame_index: NonNegativeInt
    capture_timestamp_seconds: NonNegativeFloat
    clip_start_timestamp_seconds: NonNegativeFloat
    clip_end_timestamp_seconds: NonNegativeFloat
    camera_ids: tuple[Literal["camera_a", "camera_b"], ...]
    spatial_transition_authority: Literal[True] = True
    qwen_review_pending: Literal[True] = True

    @model_validator(mode="after")
    def validate_candidate(self) -> Self:
        if self.clip_start_timestamp_seconds > self.capture_timestamp_seconds:
            raise ValueError("candidate clip starts after its transition")
        if self.clip_end_timestamp_seconds < self.capture_timestamp_seconds:
            raise ValueError("candidate clip ends before its transition")
        if self.clip_end_timestamp_seconds <= self.clip_start_timestamp_seconds:
            raise ValueError("candidate clip interval must be positive")
        if self.camera_ids != ("camera_a", "camera_b"):
            raise ValueError("S05 candidate requires both synchronized cameras")
        expected = self.create_candidate_id(
            policy_id=self.policy_id,
            event_kind=self.event_kind,
            source_state_record_id=self.source_state_record_id,
        )
        if self.candidate_id != expected:
            raise ValueError("interaction event candidate ID differs")
        return self

    @classmethod
    def create_candidate_id(
        cls,
        *,
        policy_id: str,
        event_kind: InteractionEventKind,
        source_state_record_id: str,
    ) -> str:
        return _stable_digest(
            {
                "schema_version": 1,
                "policy_id": policy_id,
                "event_kind": event_kind.value,
                "source_state_record_id": source_state_record_id,
            }
        )


class InteractionTimelineRunSummary(ContractModel):
    """Persistent S05 interaction-state artifact and source provenance."""

    schema_version: Literal[1] = 1
    stage: Literal["S05"] = "S05"
    status: Literal["completed_pending_visual_qa"]
    created_at_utc: datetime
    policy: InteractionPolicy
    pickup_zone: InteractionZone
    dropoff_zone: InteractionZone
    source_temporal_summary_ref: str
    source_temporal_summary_sha256: Sha256Digest
    source_temporal_verification_ref: str
    source_temporal_verification_sha256: Sha256Digest
    source_zone_metadata_ref: str
    source_zone_metadata_sha256: Sha256Digest
    source_synchronization_manifest_ref: str
    source_synchronization_manifest_sha256: Sha256Digest
    source_camera_a_video_ref: str
    source_camera_a_video_sha256: Sha256Digest
    source_camera_b_video_ref: str
    source_camera_b_video_sha256: Sha256Digest
    state_records: tuple[InteractionStateRecord, ...]
    event_candidates: tuple[InteractionEventCandidate, ...]
    state_counts: dict[str, NonNegativeInt]
    transition_counts: dict[str, NonNegativeInt]
    records_ref: str
    records_sha256: Sha256Digest
    candidates_ref: str
    candidates_sha256: Sha256Digest
    review_csv_ref: str
    review_csv_sha256: Sha256Digest
    timeline_diagnostic_ref: str
    timeline_diagnostic_sha256: Sha256Digest
    candidate_contact_sheet_ref: str
    candidate_contact_sheet_sha256: Sha256Digest
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        if len(self.state_records) != 160:
            raise ValueError("S05 interaction timeline requires 160 capture ticks")
        ordered = sorted(
            self.state_records,
            key=lambda record: (
                record.capture_timestamp_seconds,
                record.source_frame_index,
            ),
        )
        if list(self.state_records) != ordered:
            raise ValueError("S05 interaction records must be capture-time ordered")
        if any(record.qwen_influenced_spatial_state for record in self.state_records):
            raise ValueError("Qwen cannot influence S05 interaction spatial state")
        candidate_sources = {
            candidate.source_state_record_id for candidate in self.event_candidates
        }
        known_sources = {record.record_id for record in self.state_records}
        if not candidate_sources.issubset(known_sources):
            raise ValueError("event candidate references an unknown state record")
        return self


def build_interaction_timeline(
    *,
    presentation_records: tuple[TemporalPresentationRecord, ...],
    pickup_zone: InteractionZone,
    dropoff_zone: InteractionZone,
    policy: InteractionPolicy,
) -> tuple[InteractionStateRecord, ...]:
    """Resolve one interaction record for every paired D034 capture tick."""

    by_frame: dict[int, dict[str, TemporalPresentationRecord]] = {}
    for record in presentation_records:
        target_records = by_frame.setdefault(record.source_frame_index, {})
        target = record.target.value
        if target in target_records:
            raise ValueError("duplicate target record at one interaction frame")
        target_records[target] = record
    if len(by_frame) != 160:
        raise ValueError("S05 interaction input requires 160 paired capture ticks")

    previous: InteractionStateRecord | None = None
    output: list[InteractionStateRecord] = []
    for frame in sorted(by_frame):
        target_records = by_frame[frame]
        if set(target_records) != {"person", "backpack"}:
            raise ValueError("interaction tick requires exactly person and backpack")
        evidence = build_interaction_evidence(
            person_record=target_records["person"],
            backpack_record=target_records["backpack"],
            pickup_zone=pickup_zone,
            dropoff_zone=dropoff_zone,
            policy=policy,
        )
        current = resolve_interaction_state(
            previous=previous,
            evidence=evidence,
            policy=policy,
        )
        output.append(current)
        previous = current
    return tuple(output)


def build_event_candidates(
    *,
    state_records: tuple[InteractionStateRecord, ...],
    video_duration_seconds: float,
    policy: InteractionPolicy,
) -> tuple[InteractionEventCandidate, ...]:
    """Create bounded Qwen-review candidates from authoritative state changes."""

    if video_duration_seconds <= 0:
        raise ValueError("candidate video duration must be positive")
    reason_to_event = {
        InteractionTransitionReason.PICKUP_SPATIAL_EVIDENCE: InteractionEventKind.PICKUP,
        InteractionTransitionReason.CARRY_SPATIAL_EVIDENCE: InteractionEventKind.CARRY,
        InteractionTransitionReason.PLACE_SPATIAL_EVIDENCE: InteractionEventKind.PLACE,
    }
    candidates: list[InteractionEventCandidate] = []
    for record in state_records:
        event_kind = reason_to_event.get(record.reason)
        if event_kind is None or not record.spatial_transition_authority:
            continue
        start = max(0.0, record.capture_timestamp_seconds - policy.candidate_clip_pre_seconds)
        end = min(
            video_duration_seconds,
            record.capture_timestamp_seconds + policy.candidate_clip_post_seconds,
        )
        candidates.append(
            InteractionEventCandidate(
                candidate_id=InteractionEventCandidate.create_candidate_id(
                    policy_id=policy.policy_id,
                    event_kind=event_kind,
                    source_state_record_id=record.record_id,
                ),
                policy_id=policy.policy_id,
                event_kind=event_kind,
                source_state_record_id=record.record_id,
                source_frame_index=record.source_frame_index,
                capture_timestamp_seconds=record.capture_timestamp_seconds,
                clip_start_timestamp_seconds=start,
                clip_end_timestamp_seconds=end,
                camera_ids=("camera_a", "camera_b"),
            )
        )
    return tuple(candidates)


def build_interaction_evidence(
    *,
    person_record: TemporalPresentationRecord,
    backpack_record: TemporalPresentationRecord,
    pickup_zone: InteractionZone,
    dropoff_zone: InteractionZone,
    policy: InteractionPolicy,
) -> InteractionEvidence:
    """Build facts only from exact, paired, current S04 measurements."""

    _validate_paired_records(person_record, backpack_record, policy=policy)
    _validate_zone_pair(pickup_zone, dropoff_zone)

    backpack_measured = (
        backpack_record.state is TemporalPresentationState.MEASURED
        and backpack_record.may_update_zone_membership
        and backpack_record.raw_world_xyz_m is not None
    )
    person_measured = (
        person_record.state is TemporalPresentationState.MEASURED
        and person_record.may_update_zone_membership
        and person_record.raw_world_xyz_m is not None
    )

    membership = InteractionZoneMembership.UNKNOWN
    pickup_distance: float | None = None
    person_distance: float | None = None
    backpack_kind: CorrectedAnchorKind | None = None
    person_kind: CorrectedAnchorKind | None = None

    if backpack_measured:
        backpack_xyz = backpack_record.raw_world_xyz_m
        assert backpack_xyz is not None
        backpack_kind = backpack_record.anchor_kind
        pickup_distance = _xy_distance(backpack_xyz, pickup_zone.center_world_m)
        dropoff_distance = _xy_distance(backpack_xyz, dropoff_zone.center_world_m)
        in_pickup = pickup_distance <= pickup_zone.radius_m
        in_dropoff = dropoff_distance <= dropoff_zone.radius_m
        if in_pickup and in_dropoff:
            raise ValueError("accepted interaction zones overlap at the backpack measurement")
        if in_pickup:
            membership = InteractionZoneMembership.PICKUP
        elif in_dropoff:
            membership = InteractionZoneMembership.DROPOFF
        else:
            membership = InteractionZoneMembership.OUTSIDE

        if person_measured:
            person_xyz = person_record.raw_world_xyz_m
            assert person_xyz is not None
            person_distance = _xy_distance(person_xyz, backpack_xyz)

    if person_measured:
        person_kind = person_record.anchor_kind

    return InteractionEvidence(
        policy_id=policy.policy_id,
        source_frame_index=backpack_record.source_frame_index,
        capture_timestamp_seconds=backpack_record.capture_timestamp_seconds,
        backpack_record_id=backpack_record.record_id,
        person_record_id=person_record.record_id,
        backpack_presentation_state=backpack_record.state,
        person_presentation_state=person_record.state,
        backpack_anchor_kind=backpack_kind,
        person_anchor_kind=person_kind,
        backpack_zone_membership=membership,
        backpack_pickup_center_distance_xy_m=pickup_distance,
        person_backpack_distance_xy_m=person_distance,
        backpack_spatial_authority=backpack_measured,
        person_backpack_proximity_authority=backpack_measured and person_measured,
    )


def resolve_interaction_state(
    *,
    previous: InteractionStateRecord | None,
    evidence: InteractionEvidence,
    policy: InteractionPolicy,
) -> InteractionStateRecord:
    """Advance one deterministic transition without inferred spatial facts."""

    if evidence.policy_id != policy.policy_id:
        raise ValueError("interaction evidence policy differs from state policy")
    if previous is not None:
        if previous.policy_id != policy.policy_id:
            raise ValueError("previous interaction policy differs")
        if evidence.source_frame_index <= previous.source_frame_index:
            raise ValueError("interaction frames must increase")
        if evidence.capture_timestamp_seconds <= previous.capture_timestamp_seconds:
            raise ValueError("interaction timestamps must increase")

    previous_state = None if previous is None else previous.state
    last_authoritative = None if previous is None else previous.last_authoritative_state
    pickup_confirmed = False if previous is None else previous.pickup_confirmed

    if evidence.backpack_presentation_state is TemporalPresentationState.OCCLUDED:
        state = BackpackInteractionState.OCCLUDED
        reason = InteractionTransitionReason.CONFIRMED_OCCLUSION
        spatial_authority = False
    elif not evidence.backpack_spatial_authority:
        state = BackpackInteractionState.UNKNOWN
        reason = InteractionTransitionReason.SPATIAL_EVIDENCE_UNAVAILABLE
        spatial_authority = False
    else:
        state, reason, pickup_confirmed = _resolve_measured_state(
            previous=previous,
            evidence=evidence,
            policy=policy,
        )
        spatial_authority = state is not BackpackInteractionState.UNKNOWN
        if spatial_authority:
            last_authoritative = state

    return InteractionStateRecord(
        record_id=InteractionStateRecord.create_record_id(
            policy_id=policy.policy_id,
            source_frame_index=evidence.source_frame_index,
            capture_timestamp_seconds=evidence.capture_timestamp_seconds,
            backpack_record_id=evidence.backpack_record_id,
            person_record_id=evidence.person_record_id,
        ),
        policy_id=policy.policy_id,
        source_frame_index=evidence.source_frame_index,
        capture_timestamp_seconds=evidence.capture_timestamp_seconds,
        evidence_backpack_record_id=evidence.backpack_record_id,
        evidence_person_record_id=evidence.person_record_id,
        previous_state=previous_state,
        state=state,
        last_authoritative_state=last_authoritative,
        pickup_confirmed=pickup_confirmed,
        reason=reason,
        backpack_zone_membership=evidence.backpack_zone_membership,
        backpack_pickup_center_distance_xy_m=(
            evidence.backpack_pickup_center_distance_xy_m
        ),
        person_backpack_distance_xy_m=evidence.person_backpack_distance_xy_m,
        backpack_anchor_kind=evidence.backpack_anchor_kind,
        person_anchor_kind=evidence.person_anchor_kind,
        spatial_transition_authority=spatial_authority,
    )


def _resolve_measured_state(
    *,
    previous: InteractionStateRecord | None,
    evidence: InteractionEvidence,
    policy: InteractionPolicy,
) -> tuple[BackpackInteractionState, InteractionTransitionReason, bool]:
    membership = evidence.backpack_zone_membership
    prior_authoritative = None if previous is None else previous.last_authoritative_state
    pickup_confirmed = False if previous is None else previous.pickup_confirmed

    if membership is InteractionZoneMembership.PICKUP:
        reason = (
            InteractionTransitionReason.INITIAL_MEASURED_AT_PICKUP
            if prior_authoritative is None
            else InteractionTransitionReason.REMAINS_AT_PICKUP
        )
        return BackpackInteractionState.AT_PICKUP, reason, False

    if membership is InteractionZoneMembership.DROPOFF:
        if pickup_confirmed and prior_authoritative in {
            BackpackInteractionState.PICKUP,
            BackpackInteractionState.CARRY,
            BackpackInteractionState.PLACE,
        }:
            reason = (
                InteractionTransitionReason.REMAINS_PLACED
                if prior_authoritative is BackpackInteractionState.PLACE
                else InteractionTransitionReason.PLACE_SPATIAL_EVIDENCE
            )
            return BackpackInteractionState.PLACE, reason, True
        return (
            BackpackInteractionState.UNKNOWN,
            InteractionTransitionReason.MEASURED_SEQUENCE_UNPROVEN,
            pickup_confirmed,
        )

    near_person = (
        evidence.person_backpack_distance_xy_m is not None
        and evidence.person_backpack_distance_xy_m
        <= policy.maximum_person_backpack_xy_distance_m
    )
    sufficiently_departed = (
        evidence.backpack_pickup_center_distance_xy_m is not None
        and evidence.backpack_pickup_center_distance_xy_m
        >= policy.minimum_pickup_center_distance_m
    )
    if (
        prior_authoritative is BackpackInteractionState.AT_PICKUP
        and near_person
        and sufficiently_departed
    ):
        return (
            BackpackInteractionState.PICKUP,
            InteractionTransitionReason.PICKUP_SPATIAL_EVIDENCE,
            True,
        )
    if prior_authoritative is BackpackInteractionState.PICKUP and near_person:
        return (
            BackpackInteractionState.CARRY,
            InteractionTransitionReason.CARRY_SPATIAL_EVIDENCE,
            True,
        )
    if prior_authoritative is BackpackInteractionState.CARRY and near_person:
        return (
            BackpackInteractionState.CARRY,
            InteractionTransitionReason.REMAINS_CARRY,
            True,
        )
    return (
        BackpackInteractionState.UNKNOWN,
        InteractionTransitionReason.MEASURED_SEQUENCE_UNPROVEN,
        pickup_confirmed,
    )


def _validate_paired_records(
    person_record: TemporalPresentationRecord,
    backpack_record: TemporalPresentationRecord,
    *,
    policy: InteractionPolicy,
) -> None:
    if person_record.policy_id != policy.source_presentation_policy_id:
        raise ValueError("person record has the wrong S04 presentation policy")
    if backpack_record.policy_id != policy.source_presentation_policy_id:
        raise ValueError("backpack record has the wrong S04 presentation policy")
    if person_record.target.value != "person":
        raise ValueError("person input has the wrong target")
    if backpack_record.target.value != "backpack":
        raise ValueError("backpack input has the wrong target")
    if person_record.source_frame_index != backpack_record.source_frame_index:
        raise ValueError("interaction records must share one source frame")
    if abs(
        person_record.capture_timestamp_seconds
        - backpack_record.capture_timestamp_seconds
    ) > 1e-9:
        raise ValueError("interaction records must share one capture timestamp")


def _validate_zone_pair(pickup_zone: InteractionZone, dropoff_zone: InteractionZone) -> None:
    if pickup_zone.role != "pickup" or dropoff_zone.role != "dropoff":
        raise ValueError("interaction zones must be supplied in pickup/dropoff order")
    centre_distance = _xy_distance(
        pickup_zone.center_world_m, dropoff_zone.center_world_m
    )
    if centre_distance <= pickup_zone.radius_m + dropoff_zone.radius_m:
        raise ValueError("accepted pickup and dropoff zones must not overlap")


def _xy_distance(first: Vector3, second: Vector3) -> float:
    return float(np.linalg.norm(np.asarray(first[:2]) - np.asarray(second[:2])))


def _stable_digest(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
