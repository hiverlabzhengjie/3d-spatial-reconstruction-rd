"""Bounded, deduplicated S05 Qwen event-review contracts and queue runtime."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Protocol, Self, cast

from pydantic import field_validator, model_validator

from spatial_reconstruction.contracts import (
    ContractModel,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    Sha256Digest,
)
from spatial_reconstruction.interaction.semantic_state import SemanticEventCandidate
from spatial_reconstruction.interaction.state_machine import InteractionEventKind


class QwenEventLabel(StrEnum):
    """Schema-bounded semantic label; unknown is the safe fallback."""

    PICKUP = "pickup"
    CARRY = "carry"
    PLACE = "place"
    UNKNOWN = "unknown"


class QwenEvidenceStrength(StrEnum):
    """Qualitative visible-evidence strength, not a calibrated probability."""

    UNKNOWN = "unknown"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class QwenSampleRole(StrEnum):
    """Temporal role of one ordered camera frame in an event request."""

    BEFORE = "before"
    TRANSITION = "transition"
    AFTER = "after"


class QwenEventResultOutcome(StrEnum):
    """Terminal result of one accepted Qwen attempt."""

    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    INVALID_OUTPUT = "invalid_output"


class QwenQueueOverflowPolicy(StrEnum):
    """Bounded queue policy shared by offline and future live execution."""

    THROTTLE = "throttle"
    DROP_OLDEST = "drop_oldest"


class QwenQueueSubmissionDisposition(StrEnum):
    """Observable disposition for every event-job submission."""

    ACCEPTED = "accepted"
    DUPLICATE_COALESCED = "duplicate_coalesced"
    THROTTLE_REQUIRED = "throttle_required"
    ACCEPTED_AFTER_DROP_OLDEST = "accepted_after_drop_oldest"


class QwenResponseNormalization(StrEnum):
    """Auditable normalization applied before strict JSON parsing."""

    NONE = "none"
    JSON_CODE_FENCE = "json_code_fence"


class QwenResponseValidationError(ValueError):
    """Raised when generated event JSON cannot satisfy the S05 schema."""


class QwenEventReviewPolicy(ContractModel):
    """Bounded deterministic event-review and retry policy."""

    policy_id: Literal[
        "s05_qwen_event_review_v1",
        "s05_qwen_event_review_v2",
        "s05_qwen_event_review_v3",
        "s05_qwen_event_review_v4",
    ] = "s05_qwen_event_review_v4"
    source_candidate_policy_id: Literal["s05_semantic_interaction_v2"] = (
        "s05_semantic_interaction_v2"
    )
    model_id: Literal["Qwen/Qwen3-VL-2B-Instruct"] = (
        "Qwen/Qwen3-VL-2B-Instruct"
    )
    model_revision: str = "89644892e4d85e24eaac8bacfd4f463576704203"
    queue_capacity: PositiveInt = 3
    overflow_policy: QwenQueueOverflowPolicy = QwenQueueOverflowPolicy.THROTTLE
    frame_sample_offsets_seconds: tuple[float, float, float] = (-2.0, 0.0, 2.0)
    frames_per_event: Literal[6] = 6
    max_new_tokens: PositiveInt = 160
    timeout_seconds: PositiveFloat = 45.0
    maximum_attempts: Literal[2] = 2
    deterministic_decoding: Literal[True] = True
    qwen_may_change_coordinates: Literal[False] = False
    qwen_may_change_track_identity: Literal[False] = False
    qwen_may_change_capture_timestamps: Literal[False] = False
    qwen_may_change_zone_membership: Literal[False] = False
    qwen_may_change_spatial_authority: Literal[False] = False

    @field_validator("model_revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("Qwen model revision must be a 40-character lowercase commit")
        return value

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if self.frame_sample_offsets_seconds != (-2.0, 0.0, 2.0):
            raise ValueError("S05 event sampling requires before/transition/after offsets")
        if self.max_new_tokens > 256:
            raise ValueError("Qwen token bound cannot exceed adapter maximum")
        expected_token_bound = (
            96 if self.policy_id == "s05_qwen_event_review_v1" else 160
        )
        if self.max_new_tokens != expected_token_bound:
            raise ValueError("Qwen token bound differs from the versioned policy")
        return self


class QwenVideoSource(ContractModel):
    """Immutable synchronized source-video identity used by Qwen jobs."""

    camera_id: Literal["camera_a", "camera_b"]
    source_ref: str
    source_sha256: Sha256Digest
    decoded_frame_count: PositiveInt
    duration_seconds: PositiveFloat
    nominal_frame_rate_fps: PositiveFloat

    @field_validator("source_ref")
    @classmethod
    def validate_ref(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("Qwen video reference must be non-empty and trimmed")
        return value


class QwenEventFrameInput(ContractModel):
    """One ordered, exact camera-frame input planned for semantic review."""

    sequence_index: NonNegativeInt
    camera_id: Literal["camera_a", "camera_b"]
    sample_role: QwenSampleRole
    source_frame_index: NonNegativeInt
    capture_timestamp_seconds: NonNegativeFloat
    source_video_ref: str
    source_video_sha256: Sha256Digest

    @field_validator("source_video_ref")
    @classmethod
    def validate_ref(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("Qwen frame source reference must be non-empty and trimmed")
        return value


class QwenEventPromptSpec(ContractModel):
    """Stable prompt identity and text with an explicit spatial-write boundary."""

    prompt_id: Literal[
        "s05_qwen_event_json_v1",
        "s05_qwen_event_json_v2",
        "s05_qwen_event_json_v3",
    ] = "s05_qwen_event_json_v3"
    prompt_sha256: Sha256Digest
    expected_event: InteractionEventKind
    prompt_text: str
    assistant_prefill: str | None = None

    @field_validator("prompt_text")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("Qwen event prompt must be non-empty and trimmed")
        return value

    @model_validator(mode="after")
    def validate_hash(self) -> Self:
        if self.prompt_id == "s05_qwen_event_json_v3":
            if self.assistant_prefill != '{"event_label":"':
                raise ValueError("Qwen v3 prompt requires the exact JSON prefill")
            expected = _stable_digest(
                {
                    "prompt_text": self.prompt_text,
                    "assistant_prefill": self.assistant_prefill,
                }
            )
        else:
            if self.assistant_prefill is not None:
                raise ValueError("Qwen v1/v2 prompts cannot claim an assistant prefill")
            expected = hashlib.sha256(self.prompt_text.encode("utf-8")).hexdigest()
        if self.prompt_sha256 != expected:
            raise ValueError("Qwen event prompt hash differs")
        return self


class QwenEventJob(ContractModel):
    """One immutable semantic-review attempt with no spatial write fields."""

    schema_version: Literal[1] = 1
    job_id: Sha256Digest
    deduplication_key: Sha256Digest
    policy_id: Literal[
        "s05_qwen_event_review_v1",
        "s05_qwen_event_review_v2",
        "s05_qwen_event_review_v3",
        "s05_qwen_event_review_v4",
    ]
    source_candidate_id: Sha256Digest
    source_state_record_id: Sha256Digest
    event_kind: InteractionEventKind
    source_frame_index: NonNegativeInt
    capture_timestamp_seconds: NonNegativeFloat
    clip_start_timestamp_seconds: NonNegativeFloat
    clip_end_timestamp_seconds: NonNegativeFloat
    review_frame_index: NonNegativeInt | None = None
    review_timestamp_seconds: NonNegativeFloat | None = None
    review_clip_start_timestamp_seconds: NonNegativeFloat | None = None
    review_clip_end_timestamp_seconds: NonNegativeFloat | None = None
    capture_session_id: str
    synchronization_manifest_ref: str
    synchronization_manifest_sha256: Sha256Digest
    frame_inputs: tuple[QwenEventFrameInput, ...]
    prompt: QwenEventPromptSpec
    model_id: Literal["Qwen/Qwen3-VL-2B-Instruct"]
    model_revision: str
    max_new_tokens: PositiveInt
    timeout_seconds: PositiveFloat
    attempt: PositiveInt
    priority: NonNegativeInt
    created_processing_seconds: NonNegativeFloat

    @field_validator("capture_session_id", "synchronization_manifest_ref")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("Qwen job identity text must be non-empty and trimmed")
        return value

    @field_validator("model_revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("Qwen job model revision must be a lowercase commit")
        return value

    @model_validator(mode="after")
    def validate_job(self) -> Self:
        if self.prompt.expected_event is not self.event_kind:
            raise ValueError("Qwen prompt expected event differs from candidate")
        expected_prompt_id = (
            "s05_qwen_event_json_v1"
            if self.policy_id == "s05_qwen_event_review_v1"
            else (
                "s05_qwen_event_json_v2"
                if self.policy_id == "s05_qwen_event_review_v2"
                else "s05_qwen_event_json_v3"
            )
        )
        expected_token_bound = (
            96 if self.policy_id == "s05_qwen_event_review_v1" else 160
        )
        if self.prompt.prompt_id != expected_prompt_id:
            raise ValueError("Qwen prompt version differs from job policy")
        if self.max_new_tokens != expected_token_bound:
            raise ValueError("Qwen job token bound differs from its policy")
        if not (
            self.clip_start_timestamp_seconds
            <= self.capture_timestamp_seconds
            <= self.clip_end_timestamp_seconds
        ):
            raise ValueError("Qwen event transition lies outside its clip")
        is_review_centered = self.policy_id == "s05_qwen_event_review_v4"
        review_values = (
            self.review_frame_index,
            self.review_timestamp_seconds,
            self.review_clip_start_timestamp_seconds,
            self.review_clip_end_timestamp_seconds,
        )
        if is_review_centered and any(value is None for value in review_values):
            raise ValueError("Qwen v4 job requires an explicit review centre and clip")
        if not is_review_centered and any(value is not None for value in review_values):
            raise ValueError("Qwen v1-v3 job cannot claim a separate review centre")
        review_frame_index = (
            self.source_frame_index
            if self.review_frame_index is None
            else self.review_frame_index
        )
        review_timestamp_seconds = (
            self.capture_timestamp_seconds
            if self.review_timestamp_seconds is None
            else self.review_timestamp_seconds
        )
        review_clip_start = (
            self.clip_start_timestamp_seconds
            if self.review_clip_start_timestamp_seconds is None
            else self.review_clip_start_timestamp_seconds
        )
        review_clip_end = (
            self.clip_end_timestamp_seconds
            if self.review_clip_end_timestamp_seconds is None
            else self.review_clip_end_timestamp_seconds
        )
        if not review_clip_start <= review_timestamp_seconds <= review_clip_end:
            raise ValueError("Qwen review centre lies outside its review clip")
        if len(self.frame_inputs) != 6:
            raise ValueError("Qwen event job requires six ordered frame inputs")
        if tuple(item.sequence_index for item in self.frame_inputs) != tuple(range(6)):
            raise ValueError("Qwen frame sequence indices must be contiguous")
        expected_roles = tuple(
            role
            for role in QwenSampleRole
            for _camera_id in ("camera_a", "camera_b")
        )
        if tuple(item.sample_role for item in self.frame_inputs) != expected_roles:
            raise ValueError("Qwen frames must be time-major before/transition/after")
        expected_cameras = ("camera_a", "camera_b") * 3
        if tuple(item.camera_id for item in self.frame_inputs) != expected_cameras:
            raise ValueError("Qwen frames must pair Camera A then B at every sample")
        if any(
            item.capture_timestamp_seconds < review_clip_start - 1e-9
            or item.capture_timestamp_seconds > review_clip_end + 1e-9
            for item in self.frame_inputs
        ):
            raise ValueError("Qwen frame input lies outside the candidate clip")
        transition_inputs = tuple(
            item
            for item in self.frame_inputs
            if item.sample_role is QwenSampleRole.TRANSITION
        )
        if any(
            item.source_frame_index != review_frame_index
            or abs(item.capture_timestamp_seconds - review_timestamp_seconds) > 1e-9
            for item in transition_inputs
        ):
            raise ValueError("Qwen transition frames differ from source candidate")
        expected_dedup = self.create_deduplication_key(
            policy_id=self.policy_id,
            candidate_id=self.source_candidate_id,
            prompt_sha256=self.prompt.prompt_sha256,
            model_revision=self.model_revision,
        )
        if self.deduplication_key != expected_dedup:
            raise ValueError("Qwen deduplication key differs")
        expected_job = self.create_job_id(
            deduplication_key=self.deduplication_key,
            attempt=self.attempt,
            priority=self.priority,
        )
        if self.job_id != expected_job:
            raise ValueError("Qwen event job ID differs")
        return self

    @classmethod
    def create_deduplication_key(
        cls,
        *,
        policy_id: str,
        candidate_id: str,
        prompt_sha256: str,
        model_revision: str,
    ) -> str:
        return _stable_digest(
            {
                "schema_version": 1,
                "policy_id": policy_id,
                "candidate_id": candidate_id,
                "prompt_sha256": prompt_sha256,
                "model_revision": model_revision,
            }
        )

    @classmethod
    def create_job_id(
        cls,
        *,
        deduplication_key: str,
        attempt: int,
        priority: int,
    ) -> str:
        return _stable_digest(
            {
                "schema_version": 1,
                "deduplication_key": deduplication_key,
                "attempt": attempt,
                "priority": priority,
            }
        )


class QwenEventInterpretation(ContractModel):
    """Schema-validated semantic output without spatial mutation fields."""

    schema_version: Literal[1] = 1
    expected_event: InteractionEventKind
    event_label: QwenEventLabel
    matches_candidate: bool
    evidence_strength: QwenEvidenceStrength
    summary: str
    visible_evidence: tuple[str, ...]
    uncertainty: str | None
    spatial_claims_present: Literal[False] = False

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        if not value or value.strip() != value or len(value) > 500:
            raise ValueError("Qwen event summary must be trimmed and at most 500 characters")
        return value

    @field_validator("visible_evidence")
    @classmethod
    def validate_evidence(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) > 4 or any(
            not value or value.strip() != value or len(value) > 300 for value in values
        ):
            raise ValueError("Qwen visible evidence must contain up to four concise items")
        return values

    @field_validator("uncertainty")
    @classmethod
    def validate_uncertainty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or value.strip() != value or len(value) > 500:
            raise ValueError("Qwen uncertainty must be trimmed and at most 500 characters")
        return value

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        expected_label = QwenEventLabel(self.expected_event.value)
        if self.matches_candidate != (self.event_label is expected_label):
            raise ValueError("Qwen candidate match flag differs from event label")
        if self.event_label is QwenEventLabel.UNKNOWN:
            if self.evidence_strength is not QwenEvidenceStrength.UNKNOWN:
                raise ValueError("unknown Qwen event requires unknown evidence strength")
            if self.visible_evidence:
                raise ValueError("unknown Qwen event cannot claim visible evidence")
            if self.uncertainty is None:
                raise ValueError("unknown Qwen event requires uncertainty")
        else:
            if self.evidence_strength is QwenEvidenceStrength.UNKNOWN:
                raise ValueError("labelled Qwen event requires evidence strength")
            if not self.visible_evidence:
                raise ValueError("labelled Qwen event requires visible evidence")
        return self

    @classmethod
    def unknown_fallback(
        cls,
        *,
        expected_event: InteractionEventKind,
        reason: str,
    ) -> Self:
        normalized = reason.strip() or "Qwen event interpretation unavailable."
        return cls(
            expected_event=expected_event,
            event_label=QwenEventLabel.UNKNOWN,
            matches_candidate=False,
            evidence_strength=QwenEvidenceStrength.UNKNOWN,
            summary="Event could not be confirmed from the Qwen review.",
            visible_evidence=(),
            uncertainty=normalized[:500],
        )


@dataclass(frozen=True, slots=True)
class QwenEventProcessingOutput:
    """Raw adapter response and diagnostics before schema/result wrapping."""

    raw_response_text: str
    input_token_count: int
    output_token_count: int
    output_token_ids: tuple[int, ...]
    input_shapes: dict[str, tuple[int, ...]]


class QwenEventProcessor(Protocol):
    """Replaceable asynchronous processor boundary for one Qwen event job."""

    def process(self, job: QwenEventJob) -> Awaitable[QwenEventProcessingOutput]: ...


class QwenEventResult(ContractModel):
    """One terminal attempt with a schema-valid interpretation or unknown fallback."""

    schema_version: Literal[1] = 1
    job: QwenEventJob
    outcome: QwenEventResultOutcome
    interpretation: QwenEventInterpretation
    raw_response_text: str | None
    input_token_count: NonNegativeInt | None
    output_token_count: NonNegativeInt | None
    output_token_ids: tuple[NonNegativeInt, ...]
    input_shapes: dict[str, tuple[NonNegativeInt, ...]]
    processing_started_seconds: NonNegativeFloat
    processing_finished_seconds: NonNegativeFloat
    error_type: str | None
    error_message: str | None
    response_normalization: QwenResponseNormalization = QwenResponseNormalization.NONE

    @field_validator("raw_response_text", "error_type", "error_message")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or value.strip() != value:
            raise ValueError("Qwen result text must be non-empty and trimmed")
        return value

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.processing_started_seconds < self.job.created_processing_seconds:
            raise ValueError("Qwen processing cannot start before job creation")
        if self.processing_finished_seconds < self.processing_started_seconds:
            raise ValueError("Qwen processing finish cannot precede start")
        if self.interpretation.expected_event is not self.job.event_kind:
            raise ValueError("Qwen result interpretation differs from its job")
        if (
            self.response_normalization is QwenResponseNormalization.JSON_CODE_FENCE
            and self.outcome is not QwenEventResultOutcome.COMPLETED
        ):
            raise ValueError("only completed Qwen JSON may record fence normalization")
        if self.outcome is QwenEventResultOutcome.COMPLETED:
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("completed Qwen result cannot contain an error")
            if (
                self.raw_response_text is None
                or self.input_token_count is None
                or self.output_token_count is None
                or not self.output_token_ids
                or not self.input_shapes
            ):
                raise ValueError("completed Qwen result requires raw token diagnostics")
            if self.output_token_count != len(self.output_token_ids):
                raise ValueError("Qwen output token count differs from raw IDs")
        elif self.outcome is QwenEventResultOutcome.INVALID_OUTPUT:
            if not self.error_type or not self.error_message:
                raise ValueError("non-completed Qwen result requires an explicit error")
            if self.interpretation.event_label is not QwenEventLabel.UNKNOWN:
                raise ValueError("non-completed Qwen result must fall back to unknown")
            if (
                self.input_token_count is None
                or self.output_token_count is None
                or not self.output_token_ids
                or not self.input_shapes
            ):
                raise ValueError("invalid Qwen output must retain available token diagnostics")
            if self.output_token_count != len(self.output_token_ids):
                raise ValueError("invalid Qwen output token count differs from raw IDs")
        else:
            if not self.error_type or not self.error_message:
                raise ValueError("non-completed Qwen result requires an explicit error")
            if self.interpretation.event_label is not QwenEventLabel.UNKNOWN:
                raise ValueError("non-completed Qwen result must fall back to unknown")
            if self.raw_response_text is not None:
                raise ValueError("failed/timed-out Qwen result cannot claim raw output")
            if self.input_token_count is not None or self.output_token_count is not None:
                raise ValueError("failed/timed-out Qwen result cannot claim token counts")
            if self.output_token_ids or self.input_shapes:
                raise ValueError("failed/timed-out Qwen result cannot claim diagnostics")
        return self


class QwenQueueSubmission(ContractModel):
    """Persistent queue submission or coalescing disposition."""

    job_id: Sha256Digest
    disposition: QwenQueueSubmissionDisposition
    accepted: bool
    existing_job_id: Sha256Digest | None = None
    dropped_job_id: Sha256Digest | None = None
    queue_depth_after: NonNegativeInt

    @model_validator(mode="after")
    def validate_disposition(self) -> Self:
        if self.disposition is QwenQueueSubmissionDisposition.ACCEPTED:
            if not self.accepted or self.existing_job_id or self.dropped_job_id:
                raise ValueError("ordinary Qwen acceptance cannot identify other jobs")
        elif self.disposition is QwenQueueSubmissionDisposition.DUPLICATE_COALESCED:
            if self.accepted or self.existing_job_id is None or self.dropped_job_id:
                raise ValueError("coalesced Qwen submission must identify existing work")
        elif self.disposition is QwenQueueSubmissionDisposition.THROTTLE_REQUIRED:
            if self.accepted or self.existing_job_id or self.dropped_job_id:
                raise ValueError("throttled Qwen submission cannot alter queued work")
        elif not self.accepted or self.dropped_job_id is None or self.existing_job_id:
            raise ValueError("drop-oldest Qwen acceptance must identify dropped work")
        return self


class QwenQueueDiagnostics(ContractModel):
    """Inspectable queue, deduplication, retry, and terminal counters."""

    capacity: PositiveInt
    overflow_policy: QwenQueueOverflowPolicy
    maximum_attempts: PositiveInt
    current_depth: NonNegativeInt
    in_flight_count: NonNegativeInt
    accepted_count: NonNegativeInt
    retry_accepted_count: NonNegativeInt
    duplicate_coalesced_count: NonNegativeInt
    popped_count: NonNegativeInt
    completed_count: NonNegativeInt
    failed_count: NonNegativeInt
    timed_out_count: NonNegativeInt
    invalid_output_count: NonNegativeInt
    throttled_count: NonNegativeInt
    dropped_oldest_count: NonNegativeInt
    cancelled_count: NonNegativeInt

    @model_validator(mode="after")
    def validate_depth(self) -> Self:
        if self.current_depth > self.capacity:
            raise ValueError("Qwen queue depth exceeds capacity")
        return self


class QwenEventJobPlanRunSummary(ContractModel):
    """Persistent, verified S05 Qwen job plan before model execution."""

    schema_version: Literal[1] = 1
    stage: Literal["S05"] = "S05"
    status: Literal["completed_pending_execution"]
    created_at_utc: datetime
    policy: QwenEventReviewPolicy
    source_semantic_summary_ref: str
    source_semantic_summary_sha256: Sha256Digest
    source_semantic_verification_ref: str
    source_semantic_verification_sha256: Sha256Digest
    source_synchronization_manifest_ref: str
    source_synchronization_manifest_sha256: Sha256Digest
    source_qwen_gate_summary_ref: str
    source_qwen_gate_summary_sha256: Sha256Digest
    video_sources: tuple[QwenVideoSource, QwenVideoSource]
    jobs: tuple[QwenEventJob, ...]
    queue_submissions: tuple[QwenQueueSubmission, ...]
    queue_diagnostics: QwenQueueDiagnostics
    jobs_ref: str
    jobs_sha256: Sha256Digest
    prompt_manifest_ref: str
    prompt_manifest_sha256: Sha256Digest
    review_csv_ref: str
    review_csv_sha256: Sha256Digest
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if len(self.jobs) != 3 or tuple(job.event_kind for job in self.jobs) != (
            InteractionEventKind.PICKUP,
            InteractionEventKind.CARRY,
            InteractionEventKind.PLACE,
        ):
            raise ValueError("Qwen job plan requires pickup, carry, and place")
        if len(self.queue_submissions) != 4:
            raise ValueError("Qwen plan requires three acceptances and one dedup check")
        if sum(submission.accepted for submission in self.queue_submissions) != 3:
            raise ValueError("Qwen plan must accept exactly three unique jobs")
        if sum(
            submission.disposition
            is QwenQueueSubmissionDisposition.DUPLICATE_COALESCED
            for submission in self.queue_submissions
        ) != 1:
            raise ValueError("Qwen plan must demonstrate one duplicate coalescing")
        if (
            self.queue_diagnostics.current_depth != 3
            or self.queue_diagnostics.accepted_count != 3
            or self.queue_diagnostics.duplicate_coalesced_count != 1
            or self.queue_diagnostics.completed_count
            or self.queue_diagnostics.failed_count
            or self.queue_diagnostics.timed_out_count
            or self.queue_diagnostics.invalid_output_count
        ):
            raise ValueError("Qwen plan diagnostics differ from pre-execution state")
        return self


class BoundedQwenEventQueue:
    """Capture-ordered queue with logical-event deduplication and bounded retries."""

    def __init__(
        self,
        *,
        capacity: int,
        overflow_policy: QwenQueueOverflowPolicy,
        maximum_attempts: int,
    ) -> None:
        if capacity <= 0 or maximum_attempts <= 0:
            raise ValueError("Qwen queue capacity and maximum attempts must be positive")
        self.capacity = capacity
        self.overflow_policy = overflow_policy
        self.maximum_attempts = maximum_attempts
        self._pending: deque[QwenEventJob] = deque()
        self._active_keys: set[str] = set()
        self._in_flight: dict[str, QwenEventJob] = {}
        self._latest_job_by_key: dict[str, QwenEventJob] = {}
        self._terminal_outcome_by_key: dict[str, QwenEventResultOutcome] = {}
        self._last_initial_capture_timestamp: float | None = None
        self._accepted_count = 0
        self._retry_accepted_count = 0
        self._duplicate_coalesced_count = 0
        self._popped_count = 0
        self._completed_count = 0
        self._failed_count = 0
        self._timed_out_count = 0
        self._invalid_output_count = 0
        self._throttled_count = 0
        self._dropped_oldest_count = 0
        self._cancelled_count = 0

    def submit(self, job: QwenEventJob) -> QwenQueueSubmission:
        """Accept new/retry work, coalesce duplicates, or apply overflow policy."""

        key = job.deduplication_key
        existing = self._latest_job_by_key.get(key)
        if key in self._active_keys or (
            existing is not None
            and self._terminal_outcome_by_key.get(key) is QwenEventResultOutcome.COMPLETED
        ):
            self._duplicate_coalesced_count += 1
            assert existing is not None
            return QwenQueueSubmission(
                job_id=job.job_id,
                disposition=QwenQueueSubmissionDisposition.DUPLICATE_COALESCED,
                accepted=False,
                existing_job_id=existing.job_id,
                queue_depth_after=len(self._pending),
            )

        is_retry = existing is not None
        if is_retry:
            assert existing is not None
            expected_attempt = existing.attempt + 1
            if job.attempt <= existing.attempt:
                self._duplicate_coalesced_count += 1
                return QwenQueueSubmission(
                    job_id=job.job_id,
                    disposition=QwenQueueSubmissionDisposition.DUPLICATE_COALESCED,
                    accepted=False,
                    existing_job_id=existing.job_id,
                    queue_depth_after=len(self._pending),
                )
            if job.attempt != expected_attempt or job.attempt > self.maximum_attempts:
                raise ValueError("Qwen retry attempt is not the next bounded attempt")
            if job.source_candidate_id != existing.source_candidate_id:
                raise ValueError("Qwen retry candidate identity differs")
        elif job.attempt != 1:
            raise ValueError("first Qwen submission must use attempt one")

        if not is_retry and self._last_initial_capture_timestamp is not None and (
            job.capture_timestamp_seconds <= self._last_initial_capture_timestamp
        ):
            raise ValueError("initial Qwen jobs must be submitted in capture-time order")

        dropped_job_id: str | None = None
        if len(self._pending) >= self.capacity:
            if self.overflow_policy is QwenQueueOverflowPolicy.THROTTLE:
                self._throttled_count += 1
                return QwenQueueSubmission(
                    job_id=job.job_id,
                    disposition=QwenQueueSubmissionDisposition.THROTTLE_REQUIRED,
                    accepted=False,
                    queue_depth_after=len(self._pending),
                )
            dropped = self._pending.popleft()
            dropped_job_id = dropped.job_id
            self._active_keys.remove(dropped.deduplication_key)
            self._terminal_outcome_by_key[dropped.deduplication_key] = (
                QwenEventResultOutcome.FAILED
            )
            self._dropped_oldest_count += 1

        self._pending.append(job)
        self._active_keys.add(key)
        self._latest_job_by_key[key] = job
        if not is_retry:
            self._last_initial_capture_timestamp = job.capture_timestamp_seconds
        self._accepted_count += 1
        if is_retry:
            self._retry_accepted_count += 1
        return QwenQueueSubmission(
            job_id=job.job_id,
            disposition=(
                QwenQueueSubmissionDisposition.ACCEPTED_AFTER_DROP_OLDEST
                if dropped_job_id is not None
                else QwenQueueSubmissionDisposition.ACCEPTED
            ),
            accepted=True,
            dropped_job_id=dropped_job_id,
            queue_depth_after=len(self._pending),
        )

    def pop(self) -> QwenEventJob | None:
        """Pop the earliest pending job without using worker completion order."""

        if not self._pending:
            return None
        job = self._pending.popleft()
        self._in_flight[job.job_id] = job
        self._popped_count += 1
        return job

    def mark_result(self, result: QwenEventResult) -> None:
        """Record one terminal result and release its logical event for retry."""

        job = self._in_flight.pop(result.job.job_id, None)
        if job is None or job != result.job:
            raise ValueError("Qwen result does not match an in-flight job")
        key = job.deduplication_key
        self._active_keys.remove(key)
        self._terminal_outcome_by_key[key] = result.outcome
        if result.outcome is QwenEventResultOutcome.COMPLETED:
            self._completed_count += 1
        elif result.outcome is QwenEventResultOutcome.TIMED_OUT:
            self._timed_out_count += 1
        elif result.outcome is QwenEventResultOutcome.INVALID_OUTPUT:
            self._invalid_output_count += 1
        else:
            self._failed_count += 1

    def cancel_pending(self) -> tuple[QwenEventJob, ...]:
        """Cancel only pending work with explicit accounting."""

        cancelled = tuple(self._pending)
        self._pending.clear()
        for job in cancelled:
            self._active_keys.remove(job.deduplication_key)
            self._terminal_outcome_by_key[job.deduplication_key] = (
                QwenEventResultOutcome.FAILED
            )
        self._cancelled_count += len(cancelled)
        return cancelled

    @property
    def diagnostics(self) -> QwenQueueDiagnostics:
        """Return one immutable validated queue snapshot."""

        return QwenQueueDiagnostics(
            capacity=self.capacity,
            overflow_policy=self.overflow_policy,
            maximum_attempts=self.maximum_attempts,
            current_depth=len(self._pending),
            in_flight_count=len(self._in_flight),
            accepted_count=self._accepted_count,
            retry_accepted_count=self._retry_accepted_count,
            duplicate_coalesced_count=self._duplicate_coalesced_count,
            popped_count=self._popped_count,
            completed_count=self._completed_count,
            failed_count=self._failed_count,
            timed_out_count=self._timed_out_count,
            invalid_output_count=self._invalid_output_count,
            throttled_count=self._throttled_count,
            dropped_oldest_count=self._dropped_oldest_count,
            cancelled_count=self._cancelled_count,
        )


async def process_next_qwen_event(
    queue: BoundedQwenEventQueue,
    processor: QwenEventProcessor,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> QwenEventResult | None:
    """Process one job asynchronously with timeout and safe unknown fallback."""

    job = queue.pop()
    if job is None:
        return None
    started = _clock_value(clock)
    raw_response_text: str | None = None
    processing_output: QwenEventProcessingOutput | None = None
    try:
        output = await asyncio.wait_for(
            processor.process(job),
            timeout=job.timeout_seconds,
        )
        processing_output = output
        raw_response_text = output.raw_response_text.strip() or None
        raw_response_text = _validate_processing_output(output, job=job)
        interpretation, response_normalization = _parse_qwen_event_response(
            raw_response_text,
            expected_event=job.event_kind,
            prompt_id=job.prompt.prompt_id,
        )
        finished = _clock_value(clock)
        result = QwenEventResult(
            job=job,
            outcome=QwenEventResultOutcome.COMPLETED,
            interpretation=interpretation,
            raw_response_text=raw_response_text,
            input_token_count=output.input_token_count,
            output_token_count=output.output_token_count,
            output_token_ids=output.output_token_ids,
            input_shapes=output.input_shapes,
            processing_started_seconds=started,
            processing_finished_seconds=finished,
            error_type=None,
            error_message=None,
            response_normalization=response_normalization,
        )
    except TimeoutError as exc:
        finished = _clock_value(clock)
        result = _fallback_result(
            job=job,
            outcome=QwenEventResultOutcome.TIMED_OUT,
            started=started,
            finished=finished,
            error=exc,
        )
    except QwenResponseValidationError as exc:
        finished = _clock_value(clock)
        result = _fallback_result(
            job=job,
            outcome=QwenEventResultOutcome.INVALID_OUTPUT,
            started=started,
            finished=finished,
            error=exc,
            raw_response_text=raw_response_text,
            processing_output=processing_output,
        )
    except Exception as exc:
        finished = _clock_value(clock)
        result = _fallback_result(
            job=job,
            outcome=QwenEventResultOutcome.FAILED,
            started=started,
            finished=finished,
            error=exc,
        )
    queue.mark_result(result)
    return result


def parse_qwen_event_response(
    raw_text: str,
    *,
    expected_event: InteractionEventKind,
    prompt_id: str = "s05_qwen_event_json_v3",
) -> QwenEventInterpretation:
    """Parse versioned event JSON or raise for the worker's unknown fallback path."""

    interpretation, _normalization = _parse_qwen_event_response(
        raw_text,
        expected_event=expected_event,
        prompt_id=prompt_id,
    )
    return interpretation


def _parse_qwen_event_response(
    raw_text: str,
    *,
    expected_event: InteractionEventKind,
    prompt_id: str,
) -> tuple[QwenEventInterpretation, QwenResponseNormalization]:
    """Normalize one bounded wrapper and parse the versioned semantic payload."""

    normalized, normalization = _normalize_qwen_json_text(raw_text)
    if not normalized:
        raise QwenResponseValidationError("Qwen returned empty event response")
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise QwenResponseValidationError("Qwen event response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise QwenResponseValidationError("Qwen event response must be one JSON object")
    if prompt_id == "s05_qwen_event_json_v1":
        required_keys = {
            "event_label",
            "matches_candidate",
            "evidence_strength",
            "summary",
            "visible_evidence",
            "uncertainty",
            "spatial_claims_present",
        }
    elif prompt_id in {"s05_qwen_event_json_v2", "s05_qwen_event_json_v3"}:
        required_keys = {
            "event_label",
            "evidence_strength",
            "summary",
            "visible_evidence",
            "uncertainty",
        }
    else:
        raise QwenResponseValidationError("unsupported Qwen event prompt version")
    if set(payload) != required_keys:
        raise QwenResponseValidationError(
            "Qwen event response must contain exactly the required semantic keys"
        )
    if prompt_id in {"s05_qwen_event_json_v2", "s05_qwen_event_json_v3"}:
        visible_evidence = payload.get("visible_evidence")
        if not isinstance(visible_evidence, str):
            raise QwenResponseValidationError(
                "Qwen v2 visible evidence must be one concise string"
            )
        label = payload.get("event_label")
        payload["matches_candidate"] = label == expected_event.value
        payload["visible_evidence"] = (
            [visible_evidence] if visible_evidence.strip() else []
        )
        payload["spatial_claims_present"] = False
    payload["expected_event"] = expected_event.value
    payload["schema_version"] = 1
    try:
        interpretation = QwenEventInterpretation.model_validate(payload)
    except Exception as exc:
        raise QwenResponseValidationError(
            "Qwen event response does not satisfy the semantic schema"
        ) from exc
    return interpretation, normalization


def _normalize_qwen_json_text(
    raw_text: str,
) -> tuple[str, QwenResponseNormalization]:
    normalized = raw_text.strip()
    if normalized.startswith("```json\n") and normalized.endswith("\n```"):
        inner = normalized[len("```json\n") : -len("\n```")].strip()
        if "```" in inner:
            raise QwenResponseValidationError("Qwen response has nested code fences")
        return inner, QwenResponseNormalization.JSON_CODE_FENCE
    if "```" in normalized:
        raise QwenResponseValidationError("Qwen response has an incomplete code fence")
    return normalized, QwenResponseNormalization.NONE


def build_qwen_event_jobs(
    *,
    candidates: tuple[SemanticEventCandidate, ...],
    video_sources: tuple[QwenVideoSource, QwenVideoSource],
    capture_session_id: str,
    synchronization_manifest_ref: str,
    synchronization_manifest_sha256: str,
    policy: QwenEventReviewPolicy,
    created_processing_seconds: float,
) -> tuple[QwenEventJob, ...]:
    """Build one stable attempt-one job for each capture-ordered candidate."""

    if tuple(source.camera_id for source in video_sources) != ("camera_a", "camera_b"):
        raise ValueError("Qwen video sources must be ordered Camera A then Camera B")
    if len(candidates) != 3 or tuple(item.event_kind for item in candidates) != (
        InteractionEventKind.PICKUP,
        InteractionEventKind.CARRY,
        InteractionEventKind.PLACE,
    ):
        raise ValueError("S05 Qwen plan requires pickup, carry, and place candidates")
    jobs_list: list[QwenEventJob] = []
    for candidate in candidates:
        review_timestamp_seconds = candidate.capture_timestamp_seconds
        if (
            policy.policy_id == "s05_qwen_event_review_v4"
            and candidate.event_kind is InteractionEventKind.CARRY
        ):
            place_candidate = candidates[2]
            if place_candidate.event_kind is not InteractionEventKind.PLACE:
                raise ValueError("Qwen carry review requires the following place candidate")
            midpoint = (
                candidate.capture_timestamp_seconds
                + place_candidate.capture_timestamp_seconds
            ) / 2.0
            review_frame = round(midpoint * video_sources[0].nominal_frame_rate_fps)
            review_timestamp_seconds = (
                review_frame / video_sources[0].nominal_frame_rate_fps
            )
        jobs_list.append(
            _build_qwen_event_job(
                candidate=candidate,
                review_timestamp_seconds=review_timestamp_seconds,
                video_sources=video_sources,
                capture_session_id=capture_session_id,
                synchronization_manifest_ref=synchronization_manifest_ref,
                synchronization_manifest_sha256=synchronization_manifest_sha256,
                policy=policy,
                created_processing_seconds=created_processing_seconds,
            )
        )
    jobs = tuple(jobs_list)
    if tuple(job.capture_timestamp_seconds for job in jobs) != tuple(
        sorted(job.capture_timestamp_seconds for job in jobs)
    ):
        raise ValueError("Qwen jobs are not capture-time ordered")
    return jobs


def build_qwen_event_prompt(
    expected_event: InteractionEventKind,
    *,
    policy_id: str = "s05_qwen_event_review_v4",
) -> QwenEventPromptSpec:
    """Build the stable event-specific JSON prompt."""

    prompt_id: Literal[
        "s05_qwen_event_json_v1",
        "s05_qwen_event_json_v2",
        "s05_qwen_event_json_v3",
    ]
    assistant_prefill: str | None = None
    if policy_id == "s05_qwen_event_review_v1":
        prompt_id = "s05_qwen_event_json_v1"
        prompt_text = (
            "Review the six ordered frames: before, transition, and after from Camera A "
            "and Camera B. Describe only directly visible action involving the person and "
            f"backpack. The candidate event is {expected_event.value}. Return exactly one "
            "JSON object with keys event_label, matches_candidate, evidence_strength, "
            "summary, visible_evidence, uncertainty, and spatial_claims_present. "
            "event_label must be pickup, carry, place, or unknown. evidence_strength must "
            "be weak, moderate, strong, or unknown. Use unknown when the action is not "
            "visually supportable. spatial_claims_present must be false. Do not infer or "
            "return coordinates, track identity, timestamps, zones, distances, or spatial "
            "authority. Return JSON only."
        )
    elif policy_id == "s05_qwen_event_review_v2":
        prompt_id = "s05_qwen_event_json_v2"
        prompt_text = (
            "Inspect six ordered frames: before A/B, transition A/B, after A/B. "
            f"Candidate={expected_event.value}. Judge only visible person-backpack action. "
            "Return one compact JSON object, starting with { and ending with }; no markdown. "
            "Use exactly five keys: event_label (pickup|carry|place|unknown), "
            "evidence_strength (weak|moderate|strong|unknown), summary (one short string), "
            "visible_evidence (one short string; empty only if unknown), and uncertainty "
            "(short string or null). If event_label is unknown, evidence_strength must be "
            "unknown. Do not include coordinates, identities, times, zones, distances, "
            "spatial claims, or extra keys."
        )
    elif policy_id in {"s05_qwen_event_review_v3", "s05_qwen_event_review_v4"}:
        prompt_id = "s05_qwen_event_json_v3"
        assistant_prefill = '{"event_label":"'
        prompt_text = (
            "Inspect six ordered frames: before A/B, transition A/B, after A/B. "
            f"Candidate={expected_event.value}. Judge only visible person-backpack action. "
            "Complete the compact JSON object already started by the assistant. "
            "After event_label, use exactly these remaining keys in order: "
            "evidence_strength, summary, visible_evidence, uncertainty. "
            "event_label is pickup, carry, place, or unknown. evidence_strength is weak, "
            "moderate, strong, or unknown. summary and visible_evidence are each one short "
            "string. uncertainty is one short string or null. Close the object immediately. "
            "No markdown, scene inventory, coordinates, identities, times, zones, distances, "
            "spatial claims, or extra keys."
        )
    else:
        raise ValueError("unsupported Qwen event policy version")
    return QwenEventPromptSpec(
        prompt_id=prompt_id,
        prompt_sha256=(
            _stable_digest(
                {
                    "prompt_text": prompt_text,
                    "assistant_prefill": assistant_prefill,
                }
            )
            if assistant_prefill is not None
            else hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        ),
        expected_event=expected_event,
        prompt_text=prompt_text,
        assistant_prefill=assistant_prefill,
    )


def _build_qwen_event_job(
    *,
    candidate: SemanticEventCandidate,
    review_timestamp_seconds: float,
    video_sources: tuple[QwenVideoSource, QwenVideoSource],
    capture_session_id: str,
    synchronization_manifest_ref: str,
    synchronization_manifest_sha256: str,
    policy: QwenEventReviewPolicy,
    created_processing_seconds: float,
) -> QwenEventJob:
    if candidate.policy_id != policy.source_candidate_policy_id:
        raise ValueError("Qwen candidate policy differs")
    role_by_offset = {
        -2.0: QwenSampleRole.BEFORE,
        0.0: QwenSampleRole.TRANSITION,
        2.0: QwenSampleRole.AFTER,
    }
    frame_inputs: list[QwenEventFrameInput] = []
    review_clip_start = max(0.0, review_timestamp_seconds - 2.0)
    review_clip_end = review_timestamp_seconds + 2.0
    for offset in policy.frame_sample_offsets_seconds:
        timestamp = review_timestamp_seconds + offset
        if timestamp < review_clip_start - 1e-9 or timestamp > review_clip_end + 1e-9:
            raise ValueError("Qwen sample offset lies outside review clip")
        for source in video_sources:
            frame_index = round(timestamp * source.nominal_frame_rate_fps)
            if frame_index >= source.decoded_frame_count:
                raise ValueError("Qwen sample frame lies outside synchronized video")
            frame_inputs.append(
                QwenEventFrameInput(
                    sequence_index=len(frame_inputs),
                    camera_id=source.camera_id,
                    sample_role=role_by_offset[offset],
                    source_frame_index=frame_index,
                    capture_timestamp_seconds=timestamp,
                    source_video_ref=source.source_ref,
                    source_video_sha256=source.source_sha256,
                )
            )
    prompt = build_qwen_event_prompt(candidate.event_kind, policy_id=policy.policy_id)
    deduplication_key = QwenEventJob.create_deduplication_key(
        policy_id=policy.policy_id,
        candidate_id=candidate.candidate_id,
        prompt_sha256=prompt.prompt_sha256,
        model_revision=policy.model_revision,
    )
    priority = {
        InteractionEventKind.PICKUP: 0,
        InteractionEventKind.CARRY: 1,
        InteractionEventKind.PLACE: 0,
    }[candidate.event_kind]
    return QwenEventJob(
        job_id=QwenEventJob.create_job_id(
            deduplication_key=deduplication_key,
            attempt=1,
            priority=priority,
        ),
        deduplication_key=deduplication_key,
        policy_id=policy.policy_id,
        source_candidate_id=candidate.candidate_id,
        source_state_record_id=candidate.source_state_record_id,
        event_kind=candidate.event_kind,
        source_frame_index=candidate.source_frame_index,
        capture_timestamp_seconds=candidate.capture_timestamp_seconds,
        clip_start_timestamp_seconds=candidate.clip_start_timestamp_seconds,
        clip_end_timestamp_seconds=candidate.clip_end_timestamp_seconds,
        review_frame_index=(
            round(review_timestamp_seconds * video_sources[0].nominal_frame_rate_fps)
            if policy.policy_id == "s05_qwen_event_review_v4"
            else None
        ),
        review_timestamp_seconds=(
            review_timestamp_seconds
            if policy.policy_id == "s05_qwen_event_review_v4"
            else None
        ),
        review_clip_start_timestamp_seconds=(
            review_clip_start
            if policy.policy_id == "s05_qwen_event_review_v4"
            else None
        ),
        review_clip_end_timestamp_seconds=(
            review_clip_end
            if policy.policy_id == "s05_qwen_event_review_v4"
            else None
        ),
        capture_session_id=capture_session_id,
        synchronization_manifest_ref=synchronization_manifest_ref,
        synchronization_manifest_sha256=synchronization_manifest_sha256,
        frame_inputs=tuple(frame_inputs),
        prompt=prompt,
        model_id=policy.model_id,
        model_revision=policy.model_revision,
        max_new_tokens=policy.max_new_tokens,
        timeout_seconds=policy.timeout_seconds,
        attempt=1,
        priority=priority,
        created_processing_seconds=created_processing_seconds,
    )


def make_qwen_retry_job(
    job: QwenEventJob,
    *,
    created_processing_seconds: float,
) -> QwenEventJob:
    """Create the one schema-identical repair attempt for a failed logical event."""

    next_attempt = job.attempt + 1
    payload = job.model_dump(mode="json")
    payload.update(
        {
            "job_id": QwenEventJob.create_job_id(
                deduplication_key=job.deduplication_key,
                attempt=next_attempt,
                priority=job.priority,
            ),
            "attempt": next_attempt,
            "created_processing_seconds": created_processing_seconds,
        }
    )
    return QwenEventJob.model_validate(payload)


def _validate_processing_output(
    output: QwenEventProcessingOutput,
    *,
    job: QwenEventJob,
) -> str:
    raw_text = output.raw_response_text.strip()
    if not raw_text:
        raise QwenResponseValidationError("Qwen returned empty event response")
    if output.input_token_count <= 0:
        raise QwenResponseValidationError("Qwen event input token count must be positive")
    if output.output_token_count != len(output.output_token_ids):
        raise QwenResponseValidationError("Qwen event output token count differs")
    if output.output_token_count <= 0 or output.output_token_count > job.max_new_tokens:
        raise QwenResponseValidationError("Qwen event output exceeds token bounds")
    if not output.input_shapes:
        raise QwenResponseValidationError("Qwen event output lacks input-shape diagnostics")
    return raw_text


def _fallback_result(
    *,
    job: QwenEventJob,
    outcome: QwenEventResultOutcome,
    started: float,
    finished: float,
    error: Exception,
    raw_response_text: str | None = None,
    processing_output: QwenEventProcessingOutput | None = None,
) -> QwenEventResult:
    message = str(error).strip() or type(error).__name__
    return QwenEventResult(
        job=job,
        outcome=outcome,
        interpretation=QwenEventInterpretation.unknown_fallback(
            expected_event=job.event_kind,
            reason=message,
        ),
        raw_response_text=raw_response_text,
        input_token_count=(
            processing_output.input_token_count if processing_output is not None else None
        ),
        output_token_count=(
            processing_output.output_token_count if processing_output is not None else None
        ),
        output_token_ids=(
            processing_output.output_token_ids if processing_output is not None else ()
        ),
        input_shapes=(processing_output.input_shapes if processing_output is not None else {}),
        processing_started_seconds=started,
        processing_finished_seconds=finished,
        error_type=type(error).__name__,
        error_message=message,
    )


def _clock_value(clock: Callable[[], float]) -> float:
    value = float(clock())
    if not math.isfinite(value) or value < 0:
        raise ValueError("Qwen worker clock must return finite non-negative seconds")
    return value


def _stable_digest(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=lambda value: value.value if isinstance(value, StrEnum) else cast(str, value),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
