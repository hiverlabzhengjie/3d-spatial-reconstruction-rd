"""Explicit backpack visibility evidence over an immutable S03 detector timeline."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import field_validator, model_validator

from spatial_reconstruction.contracts import (
    ContractModel,
    NonNegativeFloat,
    NonNegativeInt,
    Sha256Digest,
)
from spatial_reconstruction.perception.timeline import PerceptionPresenceState


class BackpackVisibilityState(StrEnum):
    """Reviewed optical condition, independent of detector and XYZ availability."""

    VISIBLE = "visible"
    PARTIALLY_OCCLUDED = "partially_occluded"
    FULLY_OCCLUDED = "fully_occluded"
    OUT_OF_VIEW = "out_of_view"
    UNKNOWN = "unknown"


class VisibilityEvidenceSource(StrEnum):
    """Authority that established one visibility label."""

    DETECTOR_OBSERVATION = "detector_observation"
    SYNCHRONIZED_VIDEO_REVIEW = "synchronized_video_review"
    NONE = "none"


class VisibilityReviewInterval(ContractModel):
    """Inclusive, capture-frame-aligned interval established by video review."""

    start_source_frame_index: NonNegativeInt
    end_source_frame_index: NonNegativeInt
    visibility_state: Literal[
        BackpackVisibilityState.PARTIALLY_OCCLUDED,
        BackpackVisibilityState.FULLY_OCCLUDED,
        BackpackVisibilityState.OUT_OF_VIEW,
    ]
    evidence_source: Literal[VisibilityEvidenceSource.SYNCHRONIZED_VIDEO_REVIEW]
    rationale: str
    evidence_refs: tuple[str, ...]

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("visibility-review rationale must be non-empty and trimmed")
        return value

    @field_validator("evidence_refs")
    @classmethod
    def validate_refs(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values or len(values) != len(set(values)):
            raise ValueError("visibility review requires unique evidence references")
        if any(not value or value.strip() != value for value in values):
            raise ValueError("visibility evidence references must be non-empty and trimmed")
        return values

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.end_source_frame_index < self.start_source_frame_index:
            raise ValueError("visibility review interval ends before it starts")
        return self


class BackpackVisibilityPolicy(ContractModel):
    """Rules for adding explicit visibility evidence without changing S03 facts."""

    policy_id: Literal["s05_backpack_visibility_overlay_v1"] = (
        "s05_backpack_visibility_overlay_v1"
    )
    timeline_frame_stride: Literal[6] = 6
    missing_detection_implies_occlusion: Literal[False] = False
    reviewed_occlusion_may_supply_xyz: Literal[False] = False
    explicit_review_overrides_detector_visibility: Literal[True] = True
    review_intervals: tuple[VisibilityReviewInterval, ...]

    @model_validator(mode="after")
    def validate_intervals(self) -> Self:
        ordered = sorted(
            self.review_intervals,
            key=lambda item: item.start_source_frame_index,
        )
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current.start_source_frame_index <= previous.end_source_frame_index:
                raise ValueError("visibility review intervals overlap")
        for interval in ordered:
            if (
                interval.start_source_frame_index % self.timeline_frame_stride
                or interval.end_source_frame_index % self.timeline_frame_stride
            ):
                raise ValueError("visibility interval must align to the S03 frame grid")
        return self


class BackpackVisibilityRecord(ContractModel):
    """One visibility label with the original per-camera detector facts retained."""

    schema_version: Literal[1] = 1
    record_id: Sha256Digest
    policy_id: Literal["s05_backpack_visibility_overlay_v1"]
    source_frame_index: NonNegativeInt
    capture_timestamp_seconds: NonNegativeFloat
    camera_a_detection_state: PerceptionPresenceState
    camera_b_detection_state: PerceptionPresenceState
    visibility_state: BackpackVisibilityState
    evidence_source: VisibilityEvidenceSource
    confirmed_occluded_for_localization: bool
    rationale: str
    evidence_refs: tuple[str, ...] = ()
    supplies_xyz: Literal[False] = False

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("visibility rationale must be non-empty and trimmed")
        return value

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        explicitly_occluded = self.visibility_state in {
            BackpackVisibilityState.PARTIALLY_OCCLUDED,
            BackpackVisibilityState.FULLY_OCCLUDED,
        }
        if self.confirmed_occluded_for_localization != explicitly_occluded:
            raise ValueError("localization occlusion flag differs from visibility state")
        if self.evidence_source is VisibilityEvidenceSource.SYNCHRONIZED_VIDEO_REVIEW:
            if not self.evidence_refs:
                raise ValueError("reviewed visibility requires evidence references")
        elif self.evidence_refs:
            raise ValueError("only reviewed visibility may carry evidence references")
        if explicitly_occluded and self.evidence_source is not (
            VisibilityEvidenceSource.SYNCHRONIZED_VIDEO_REVIEW
        ):
            raise ValueError("occlusion requires synchronized-video review")
        if self.visibility_state is BackpackVisibilityState.VISIBLE and (
            self.evidence_source is not VisibilityEvidenceSource.DETECTOR_OBSERVATION
        ):
            raise ValueError("visible state requires a detector observation")
        if self.visibility_state is BackpackVisibilityState.UNKNOWN and (
            self.evidence_source is not VisibilityEvidenceSource.NONE
        ):
            raise ValueError("unknown visibility requires no evidence authority")
        expected = self.create_record_id(
            policy_id=self.policy_id,
            source_frame_index=self.source_frame_index,
            capture_timestamp_seconds=self.capture_timestamp_seconds,
        )
        if self.record_id != expected:
            raise ValueError("visibility record ID differs")
        return self

    @classmethod
    def create_record_id(
        cls,
        *,
        policy_id: str,
        source_frame_index: int,
        capture_timestamp_seconds: float,
    ) -> str:
        return _stable_digest(
            {
                "schema_version": 1,
                "policy_id": policy_id,
                "source_frame_index": source_frame_index,
                "capture_timestamp_seconds": capture_timestamp_seconds,
            }
        )


class BackpackVisibilityRunSummary(ContractModel):
    """Versioned visibility overlay and immutable S03 source provenance."""

    schema_version: Literal[1] = 1
    stage: Literal["S05"] = "S05"
    status: Literal["completed_pending_visual_qa"]
    created_at_utc: datetime
    policy: BackpackVisibilityPolicy
    source_perception_summary_ref: str
    source_perception_summary_sha256: Sha256Digest
    source_camera_a_timeline_ref: str
    source_camera_a_timeline_sha256: Sha256Digest
    source_camera_b_timeline_ref: str
    source_camera_b_timeline_sha256: Sha256Digest
    records: tuple[BackpackVisibilityRecord, ...]
    state_counts: dict[str, NonNegativeInt]
    records_ref: str
    records_sha256: Sha256Digest
    review_csv_ref: str
    review_csv_sha256: Sha256Digest
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        if len(self.records) != 160:
            raise ValueError("visibility overlay requires all 160 S03 timeline ticks")
        ordered = sorted(
            self.records,
            key=lambda item: (item.capture_timestamp_seconds, item.source_frame_index),
        )
        if list(self.records) != ordered:
            raise ValueError("visibility records must be capture-time ordered")
        if any(record.supplies_xyz for record in self.records):
            raise ValueError("visibility evidence cannot supply XYZ")
        return self


def _stable_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
