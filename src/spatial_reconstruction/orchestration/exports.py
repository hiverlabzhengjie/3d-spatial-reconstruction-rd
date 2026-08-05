"""Typed dedicated S06 event export preserving spatial-authority boundaries."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Self

from pydantic import model_validator

from spatial_reconstruction.contracts import (
    ContractModel,
    NonNegativeFloat,
    NonNegativeInt,
    Sha256Digest,
)
from spatial_reconstruction.interaction import PhaseAuthority, SemanticEventCandidate
from spatial_reconstruction.orchestration.rerun_presentation import RerunEventMarker


class Stage06EventExportRecord(ContractModel):
    """One transition plus separate semantic review and authority provenance."""

    schema_version: Literal[1] = 1
    event_id: Sha256Digest
    candidate_id: Sha256Digest
    source_state_record_id: Sha256Digest
    qwen_job_id: Sha256Digest
    event_kind: Literal["pickup", "carry", "place"]
    transition_frame_index: NonNegativeInt
    transition_timestamp_seconds: NonNegativeFloat
    review_frame_index: NonNegativeInt
    review_timestamp_seconds: NonNegativeFloat
    phase_authority: PhaseAuthority
    spatial_transition_authority: bool
    qwen_outcome: Literal["completed"]
    qwen_event_label: Literal["pickup", "carry", "place", "unknown"]
    qwen_summary: str
    qwen_matches_candidate: bool
    qwen_changed_spatial_facts: Literal[False] = False

    @classmethod
    def create(
        cls,
        *,
        candidate: SemanticEventCandidate,
        marker: RerunEventMarker,
        qwen_job_id: str,
        qwen_outcome: str,
    ) -> Self:
        payload = {
            "candidate_id": candidate.candidate_id,
            "source_state_record_id": candidate.source_state_record_id,
            "qwen_job_id": qwen_job_id,
            "event_kind": marker.event_kind,
            "transition_frame_index": marker.transition_frame_index,
            "transition_timestamp_seconds": marker.transition_timestamp_seconds,
            "review_frame_index": marker.review_frame_index,
            "review_timestamp_seconds": marker.review_timestamp_seconds,
            "phase_authority": candidate.phase_authority,
            "spatial_transition_authority": candidate.spatial_transition_authority,
            "qwen_outcome": qwen_outcome,
            "qwen_event_label": marker.qwen_event_label,
            "qwen_summary": marker.qwen_summary,
            "qwen_matches_candidate": marker.qwen_matches_candidate,
            "qwen_changed_spatial_facts": False,
        }
        identity_payload = {"schema_version": 1, **payload}
        return cls.model_validate({"event_id": _digest(identity_payload), **identity_payload})

    @model_validator(mode="after")
    def validate_event(self) -> Self:
        if self.event_kind != self.qwen_event_label or not self.qwen_matches_candidate:
            raise ValueError("accepted S06 event export requires a matching Qwen label")
        if self.spatial_transition_authority != (
            self.phase_authority is PhaseAuthority.MEASURED_SPATIAL
        ):
            raise ValueError("event spatial authority differs from phase authority")
        if self.event_kind == "carry":
            if self.transition_frame_index == self.review_frame_index:
                raise ValueError("carry transition and review identities must remain separate")
        elif (
            self.transition_frame_index != self.review_frame_index
            or self.transition_timestamp_seconds != self.review_timestamp_seconds
        ):
            raise ValueError("pickup/place transition and review identities must match")
        expected = _digest(self.model_dump(mode="json", exclude={"event_id"}))
        if self.event_id != expected:
            raise ValueError("S06 event export ID differs from its content")
        return self


def _digest(payload: object) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=lambda value: value.value,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
