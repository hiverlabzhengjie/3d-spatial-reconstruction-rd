"""Orthogonal S05 interaction phase, visibility, and localization contracts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self

import numpy as np
from pydantic import model_validator

from spatial_reconstruction.contracts import (
    ContractModel,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    Sha256Digest,
    Vector3,
)
from spatial_reconstruction.interaction.state_machine import (
    InteractionEventKind,
    InteractionZone,
    InteractionZoneMembership,
)
from spatial_reconstruction.localization import (
    TemporalPresentationRecord,
    TemporalPresentationState,
)
from spatial_reconstruction.perception import (
    BackpackVisibilityRecord,
    BackpackVisibilityState,
    VisibilityEvidenceSource,
)


class InteractionPhase(StrEnum):
    """Semantic interaction sequence, independent of visibility and XYZ."""

    UNKNOWN = "unknown"
    AT_PICKUP = "at_pickup"
    PICKUP = "pickup"
    CARRY = "carry"
    PLACE = "place"


class LocalizationAvailability(StrEnum):
    """Whether current backpack XYZ exists, is held stale, or is unavailable."""

    MEASURED = "measured"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


class PhaseAuthority(StrEnum):
    """Evidence authority for one semantic phase label."""

    MEASURED_SPATIAL = "measured_spatial"
    SEQUENCE_CONTINUITY = "sequence_continuity"
    NONE = "none"


class SemanticTransitionReason(StrEnum):
    """Inspectable reason for one v2 phase result."""

    INITIAL_MEASURED_AT_PICKUP = "initial_measured_at_pickup"
    REMAINS_AT_PICKUP_MEASURED = "remains_at_pickup_measured"
    AT_PICKUP_SEQUENCE_CONTINUITY = "at_pickup_sequence_continuity"
    PICKUP_MEASURED_SPATIAL_EVIDENCE = "pickup_measured_spatial_evidence"
    CARRY_MEASURED_SPATIAL_EVIDENCE = "carry_measured_spatial_evidence"
    CARRY_SEQUENCE_CONTINUITY = "carry_sequence_continuity"
    PLACE_MEASURED_SPATIAL_EVIDENCE = "place_measured_spatial_evidence"
    PLACE_SEQUENCE_CONTINUITY = "place_sequence_continuity"
    SPATIAL_SEQUENCE_UNPROVEN = "spatial_sequence_unproven"
    UNLOCALIZED_CARRY_HORIZON_EXPIRED = "unlocalized_carry_horizon_expired"


class SemanticInteractionPolicy(ContractModel):
    """S05 v2 policy separating semantic and spatial authority."""

    policy_id: Literal["s05_semantic_interaction_v2"] = "s05_semantic_interaction_v2"
    source_presentation_policy_id: Literal["s04_temporal_presentation_v1"] = (
        "s04_temporal_presentation_v1"
    )
    source_visibility_policy_id: Literal["s05_backpack_visibility_overlay_v1"] = (
        "s05_backpack_visibility_overlay_v1"
    )
    maximum_person_backpack_xy_distance_m: PositiveFloat = 1.0
    minimum_pickup_center_distance_m: PositiveFloat = 0.30
    maximum_unlocalized_carry_seconds: PositiveFloat = 10.0
    candidate_clip_pre_seconds: PositiveFloat = 2.0
    candidate_clip_post_seconds: PositiveFloat = 2.0
    current_measurement_required_for_spatial_facts: Literal[True] = True
    sequence_continuity_may_establish_semantic_phase: Literal[True] = True
    sequence_continuity_may_supply_xyz: Literal[False] = False
    visibility_may_supply_xyz: Literal[False] = False
    qwen_may_change_spatial_facts: Literal[False] = False


class SemanticInteractionRecord(ContractModel):
    """One tick with independent phase, visibility, and localization axes."""

    schema_version: Literal[2] = 2
    record_id: Sha256Digest
    policy_id: Literal["s05_semantic_interaction_v2"]
    source_frame_index: NonNegativeInt
    capture_timestamp_seconds: NonNegativeFloat
    source_backpack_record_id: Sha256Digest
    source_person_record_id: Sha256Digest
    source_visibility_record_id: Sha256Digest
    previous_phase: InteractionPhase | None
    phase: InteractionPhase
    phase_authority: PhaseAuthority
    reason: SemanticTransitionReason
    visibility_state: BackpackVisibilityState
    visibility_evidence_source: VisibilityEvidenceSource
    localization_availability: LocalizationAvailability
    backpack_presentation_state: TemporalPresentationState
    backpack_world_xyz_m: Vector3 | None
    person_world_xyz_m: Vector3 | None
    backpack_zone_membership: InteractionZoneMembership
    backpack_pickup_center_distance_xy_m: NonNegativeFloat | None
    person_backpack_distance_xy_m: NonNegativeFloat | None
    pickup_confirmed: bool
    pickup_timestamp_seconds: NonNegativeFloat | None
    phase_has_current_spatial_authority: bool
    invented_xyz: Literal[False] = False
    qwen_influenced_spatial_facts: Literal[False] = False

    @model_validator(mode="after")
    def validate_axes(self) -> Self:
        if self.phase_authority is PhaseAuthority.MEASURED_SPATIAL:
            if not self.phase_has_current_spatial_authority:
                raise ValueError("measured phase authority must retain spatial authority")
        elif self.phase_has_current_spatial_authority:
            raise ValueError("non-measured phase authority cannot claim spatial authority")
        if self.localization_availability is LocalizationAvailability.MEASURED:
            if self.backpack_world_xyz_m is None:
                raise ValueError("measured localization requires backpack XYZ")
            if self.backpack_zone_membership is InteractionZoneMembership.UNKNOWN:
                raise ValueError("measured localization requires known zone membership")
        else:
            if self.backpack_world_xyz_m is not None:
                raise ValueError("stale/unavailable localization cannot expose backpack XYZ")
            if self.backpack_zone_membership is not InteractionZoneMembership.UNKNOWN:
                raise ValueError("stale/unavailable localization cannot claim a zone")
            if self.backpack_pickup_center_distance_xy_m is not None:
                raise ValueError("stale/unavailable localization cannot claim distance")
            if self.person_backpack_distance_xy_m is not None:
                raise ValueError("stale/unavailable localization cannot claim proximity")
        if self.phase is InteractionPhase.CARRY and self.phase_authority is (
            PhaseAuthority.SEQUENCE_CONTINUITY
        ):
            if not self.pickup_confirmed or self.pickup_timestamp_seconds is None:
                raise ValueError("continuity carry requires an earlier confirmed pickup")
        if self.pickup_confirmed != (self.pickup_timestamp_seconds is not None):
            raise ValueError("pickup timestamp presence differs from pickup confirmation")
        expected = self.create_record_id(
            policy_id=self.policy_id,
            source_frame_index=self.source_frame_index,
            capture_timestamp_seconds=self.capture_timestamp_seconds,
            backpack_record_id=self.source_backpack_record_id,
            visibility_record_id=self.source_visibility_record_id,
        )
        if self.record_id != expected:
            raise ValueError("semantic interaction record ID differs")
        return self

    @classmethod
    def create_record_id(
        cls,
        *,
        policy_id: str,
        source_frame_index: int,
        capture_timestamp_seconds: float,
        backpack_record_id: str,
        visibility_record_id: str,
    ) -> str:
        return _stable_digest(
            {
                "schema_version": 2,
                "policy_id": policy_id,
                "source_frame_index": source_frame_index,
                "capture_timestamp_seconds": capture_timestamp_seconds,
                "backpack_record_id": backpack_record_id,
                "visibility_record_id": visibility_record_id,
            }
        )


class SemanticEventCandidate(ContractModel):
    """Qwen-review candidate whose semantic and spatial authorities remain explicit."""

    schema_version: Literal[2] = 2
    candidate_id: Sha256Digest
    policy_id: Literal["s05_semantic_interaction_v2"]
    event_kind: InteractionEventKind
    source_state_record_id: Sha256Digest
    source_frame_index: NonNegativeInt
    capture_timestamp_seconds: NonNegativeFloat
    clip_start_timestamp_seconds: NonNegativeFloat
    clip_end_timestamp_seconds: NonNegativeFloat
    phase_authority: PhaseAuthority
    spatial_transition_authority: bool
    qwen_review_pending: Literal[True] = True
    qwen_may_change_spatial_facts: Literal[False] = False

    @model_validator(mode="after")
    def validate_candidate(self) -> Self:
        if not (
            self.clip_start_timestamp_seconds
            <= self.capture_timestamp_seconds
            <= self.clip_end_timestamp_seconds
        ):
            raise ValueError("candidate transition is outside its clip")
        if self.clip_end_timestamp_seconds <= self.clip_start_timestamp_seconds:
            raise ValueError("candidate clip interval must be positive")
        if self.spatial_transition_authority != (
            self.phase_authority is PhaseAuthority.MEASURED_SPATIAL
        ):
            raise ValueError("candidate spatial authority differs from phase authority")
        expected = _stable_digest(
            {
                "schema_version": 2,
                "policy_id": self.policy_id,
                "event_kind": self.event_kind.value,
                "source_state_record_id": self.source_state_record_id,
            }
        )
        if self.candidate_id != expected:
            raise ValueError("semantic candidate ID differs")
        return self


class SemanticInteractionRunSummary(ContractModel):
    """Persistent S05 v2 artifact with all three axes and source provenance."""

    schema_version: Literal[2] = 2
    stage: Literal["S05"] = "S05"
    status: Literal["completed_pending_visual_qa"]
    created_at_utc: datetime
    policy: SemanticInteractionPolicy
    pickup_zone: InteractionZone
    dropoff_zone: InteractionZone
    source_temporal_summary_ref: str
    source_temporal_summary_sha256: Sha256Digest
    source_temporal_verification_ref: str
    source_temporal_verification_sha256: Sha256Digest
    source_visibility_summary_ref: str
    source_visibility_summary_sha256: Sha256Digest
    source_visibility_verification_ref: str
    source_visibility_verification_sha256: Sha256Digest
    source_zone_metadata_ref: str
    source_zone_metadata_sha256: Sha256Digest
    source_synchronization_manifest_ref: str
    source_synchronization_manifest_sha256: Sha256Digest
    records: tuple[SemanticInteractionRecord, ...]
    event_candidates: tuple[SemanticEventCandidate, ...]
    phase_counts: dict[str, NonNegativeInt]
    visibility_counts: dict[str, NonNegativeInt]
    localization_counts: dict[str, NonNegativeInt]
    records_ref: str
    records_sha256: Sha256Digest
    candidates_ref: str
    candidates_sha256: Sha256Digest
    review_csv_ref: str
    review_csv_sha256: Sha256Digest
    timeline_diagnostic_ref: str
    timeline_diagnostic_sha256: Sha256Digest
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        if len(self.records) != 160:
            raise ValueError("S05 v2 timeline requires all 160 ticks")
        if any(record.invented_xyz for record in self.records):
            raise ValueError("S05 v2 cannot invent XYZ")
        if any(record.qwen_influenced_spatial_facts for record in self.records):
            raise ValueError("Qwen cannot influence spatial facts")
        return self


def build_semantic_interaction_timeline(
    *,
    presentation_records: tuple[TemporalPresentationRecord, ...],
    visibility_records: tuple[BackpackVisibilityRecord, ...],
    pickup_zone: InteractionZone,
    dropoff_zone: InteractionZone,
    policy: SemanticInteractionPolicy,
) -> tuple[SemanticInteractionRecord, ...]:
    """Resolve phase continuity without fabricating spatial evidence."""

    paired: dict[int, dict[str, TemporalPresentationRecord]] = {}
    for record in presentation_records:
        paired.setdefault(record.source_frame_index, {})[record.target.value] = record
    visibility = {record.source_frame_index: record for record in visibility_records}
    if len(paired) != 160 or len(visibility) != 160:
        raise ValueError("S05 v2 inputs must cover 160 paired ticks")
    previous: SemanticInteractionRecord | None = None
    output: list[SemanticInteractionRecord] = []
    for frame in sorted(paired):
        targets = paired[frame]
        if set(targets) != {"person", "backpack"} or frame not in visibility:
            raise ValueError("S05 v2 tick lacks person, backpack, or visibility evidence")
        current = resolve_semantic_interaction_tick(
            person=targets["person"],
            backpack=targets["backpack"],
            visibility=visibility[frame],
            previous=previous,
            pickup_zone=pickup_zone,
            dropoff_zone=dropoff_zone,
            policy=policy,
        )
        output.append(current)
        previous = current
    return tuple(output)


def build_semantic_event_candidates(
    *,
    records: tuple[SemanticInteractionRecord, ...],
    video_duration_seconds: float,
    policy: SemanticInteractionPolicy,
) -> tuple[SemanticEventCandidate, ...]:
    """Emit the first pickup, carry, and place transition for semantic review."""

    phase_to_kind = {
        InteractionPhase.PICKUP: InteractionEventKind.PICKUP,
        InteractionPhase.CARRY: InteractionEventKind.CARRY,
        InteractionPhase.PLACE: InteractionEventKind.PLACE,
    }
    candidates: list[SemanticEventCandidate] = []
    seen: set[InteractionEventKind] = set()
    for record in records:
        kind = phase_to_kind.get(record.phase)
        if kind is None or kind in seen or record.previous_phase is record.phase:
            continue
        seen.add(kind)
        start = max(0.0, record.capture_timestamp_seconds - policy.candidate_clip_pre_seconds)
        end = min(
            video_duration_seconds,
            record.capture_timestamp_seconds + policy.candidate_clip_post_seconds,
        )
        candidate_id = _stable_digest(
            {
                "schema_version": 2,
                "policy_id": policy.policy_id,
                "event_kind": kind.value,
                "source_state_record_id": record.record_id,
            }
        )
        candidates.append(
            SemanticEventCandidate(
                candidate_id=candidate_id,
                policy_id=policy.policy_id,
                event_kind=kind,
                source_state_record_id=record.record_id,
                source_frame_index=record.source_frame_index,
                capture_timestamp_seconds=record.capture_timestamp_seconds,
                clip_start_timestamp_seconds=start,
                clip_end_timestamp_seconds=end,
                phase_authority=record.phase_authority,
                spatial_transition_authority=record.phase_has_current_spatial_authority,
            )
        )
    return tuple(candidates)


def resolve_semantic_interaction_tick(
    *,
    person: TemporalPresentationRecord,
    backpack: TemporalPresentationRecord,
    visibility: BackpackVisibilityRecord,
    previous: SemanticInteractionRecord | None,
    pickup_zone: InteractionZone,
    dropoff_zone: InteractionZone,
    policy: SemanticInteractionPolicy,
) -> SemanticInteractionRecord:
    if (
        person.source_frame_index != backpack.source_frame_index
        or person.source_frame_index != visibility.source_frame_index
        or abs(person.capture_timestamp_seconds - backpack.capture_timestamp_seconds) > 0.01
        or abs(person.capture_timestamp_seconds - visibility.capture_timestamp_seconds) > 0.01
    ):
        raise ValueError("S05 v2 sources are not paired at one capture tick")
    if backpack.policy_id != policy.source_presentation_policy_id:
        raise ValueError("S05 v2 source presentation policy differs")
    if visibility.policy_id != policy.source_visibility_policy_id:
        raise ValueError("S05 v2 source visibility policy differs")

    localization = _localization_availability(backpack.state)
    backpack_xyz = (
        backpack.raw_world_xyz_m
        if localization is LocalizationAvailability.MEASURED
        else None
    )
    person_xyz = (
        person.raw_world_xyz_m
        if person.state is TemporalPresentationState.MEASURED
        else None
    )
    membership = InteractionZoneMembership.UNKNOWN
    pickup_distance: float | None = None
    person_distance: float | None = None
    if backpack_xyz is not None:
        pickup_distance = _xy_distance(backpack_xyz, pickup_zone.center_world_m)
        dropoff_distance = _xy_distance(backpack_xyz, dropoff_zone.center_world_m)
        if pickup_distance <= pickup_zone.radius_m:
            membership = InteractionZoneMembership.PICKUP
        elif dropoff_distance <= dropoff_zone.radius_m:
            membership = InteractionZoneMembership.DROPOFF
        else:
            membership = InteractionZoneMembership.OUTSIDE
        if person_xyz is not None:
            person_distance = _xy_distance(backpack_xyz, person_xyz)

    previous_phase = previous.phase if previous is not None else None
    pickup_confirmed = previous.pickup_confirmed if previous is not None else False
    pickup_timestamp = previous.pickup_timestamp_seconds if previous is not None else None
    phase = InteractionPhase.UNKNOWN
    authority = PhaseAuthority.NONE
    reason = SemanticTransitionReason.SPATIAL_SEQUENCE_UNPROVEN

    if localization is LocalizationAvailability.MEASURED:
        near_person = (
            person_distance is not None
            and person_distance <= policy.maximum_person_backpack_xy_distance_m
        )
        departed_pickup = (
            pickup_distance is not None
            and pickup_distance >= policy.minimum_pickup_center_distance_m
        )
        if membership is InteractionZoneMembership.DROPOFF and pickup_confirmed:
            phase = InteractionPhase.PLACE
            authority = PhaseAuthority.MEASURED_SPATIAL
            reason = SemanticTransitionReason.PLACE_MEASURED_SPATIAL_EVIDENCE
        elif (
            previous_phase is InteractionPhase.AT_PICKUP
            and departed_pickup
            and near_person
        ):
            phase = InteractionPhase.PICKUP
            authority = PhaseAuthority.MEASURED_SPATIAL
            reason = SemanticTransitionReason.PICKUP_MEASURED_SPATIAL_EVIDENCE
            pickup_confirmed = True
            pickup_timestamp = backpack.capture_timestamp_seconds
        elif pickup_confirmed and membership is InteractionZoneMembership.OUTSIDE and near_person:
            phase = InteractionPhase.CARRY
            authority = PhaseAuthority.MEASURED_SPATIAL
            reason = SemanticTransitionReason.CARRY_MEASURED_SPATIAL_EVIDENCE
        elif membership is InteractionZoneMembership.PICKUP and not pickup_confirmed:
            phase = InteractionPhase.AT_PICKUP
            authority = PhaseAuthority.MEASURED_SPATIAL
            reason = (
                SemanticTransitionReason.INITIAL_MEASURED_AT_PICKUP
                if previous is None
                else SemanticTransitionReason.REMAINS_AT_PICKUP_MEASURED
            )
    elif previous_phase is InteractionPhase.PLACE:
        phase = InteractionPhase.PLACE
        authority = PhaseAuthority.SEQUENCE_CONTINUITY
        reason = SemanticTransitionReason.PLACE_SEQUENCE_CONTINUITY
    elif pickup_confirmed and pickup_timestamp is not None:
        carry_age = backpack.capture_timestamp_seconds - pickup_timestamp
        if carry_age <= policy.maximum_unlocalized_carry_seconds + 1e-9:
            phase = InteractionPhase.CARRY
            authority = PhaseAuthority.SEQUENCE_CONTINUITY
            reason = SemanticTransitionReason.CARRY_SEQUENCE_CONTINUITY
        else:
            reason = SemanticTransitionReason.UNLOCALIZED_CARRY_HORIZON_EXPIRED
    elif previous_phase is InteractionPhase.AT_PICKUP:
        phase = InteractionPhase.AT_PICKUP
        authority = PhaseAuthority.SEQUENCE_CONTINUITY
        reason = SemanticTransitionReason.AT_PICKUP_SEQUENCE_CONTINUITY

    spatial_authority = authority is PhaseAuthority.MEASURED_SPATIAL
    return SemanticInteractionRecord(
        record_id=SemanticInteractionRecord.create_record_id(
            policy_id=policy.policy_id,
            source_frame_index=backpack.source_frame_index,
            capture_timestamp_seconds=backpack.capture_timestamp_seconds,
            backpack_record_id=backpack.record_id,
            visibility_record_id=visibility.record_id,
        ),
        policy_id=policy.policy_id,
        source_frame_index=backpack.source_frame_index,
        capture_timestamp_seconds=backpack.capture_timestamp_seconds,
        source_backpack_record_id=backpack.record_id,
        source_person_record_id=person.record_id,
        source_visibility_record_id=visibility.record_id,
        previous_phase=previous_phase,
        phase=phase,
        phase_authority=authority,
        reason=reason,
        visibility_state=visibility.visibility_state,
        visibility_evidence_source=visibility.evidence_source,
        localization_availability=localization,
        backpack_presentation_state=backpack.state,
        backpack_world_xyz_m=backpack_xyz,
        person_world_xyz_m=person_xyz,
        backpack_zone_membership=membership,
        backpack_pickup_center_distance_xy_m=pickup_distance,
        person_backpack_distance_xy_m=person_distance,
        pickup_confirmed=pickup_confirmed,
        pickup_timestamp_seconds=pickup_timestamp,
        phase_has_current_spatial_authority=spatial_authority,
    )


def _localization_availability(
    state: TemporalPresentationState,
) -> LocalizationAvailability:
    if state is TemporalPresentationState.MEASURED:
        return LocalizationAvailability.MEASURED
    if state is TemporalPresentationState.STALE:
        return LocalizationAvailability.STALE
    return LocalizationAvailability.UNAVAILABLE


def _xy_distance(a: Vector3, b: Vector3) -> float:
    return float(np.linalg.norm(np.asarray(a[:2]) - np.asarray(b[:2])))


def _stable_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
