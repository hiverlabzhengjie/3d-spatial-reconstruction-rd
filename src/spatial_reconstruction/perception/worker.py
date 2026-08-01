"""Bounded perception queue and explicit independent worker outcomes."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import field_validator, model_validator

from spatial_reconstruction.contracts import (
    ContractModel,
    FrameIdentity,
    NonNegativeFloat,
    NonNegativeInt,
    PerceptionCandidate,
    PositiveInt,
    Sha256Digest,
)

UInt8Array = NDArray[np.uint8]


class QueueOverflowPolicy(StrEnum):
    """Bounded queue action when capacity is exhausted."""

    THROTTLE = "throttle"
    DROP_OLDEST = "drop_oldest"


class QueueSubmissionDisposition(StrEnum):
    """Observable outcome of one queue submission attempt."""

    ACCEPTED = "accepted"
    THROTTLE_REQUIRED = "throttle_required"
    ACCEPTED_AFTER_DROP_OLDEST = "accepted_after_drop_oldest"


class PerceptionResultOutcome(StrEnum):
    """Independent terminal state for one accepted perception job."""

    COMPLETED = "completed"
    FAILED = "failed"


class PerceptionJob(ContractModel):
    """Stable perception work identity with immutable capture provenance."""

    schema_version: Literal[1] = 1
    job_id: Sha256Digest
    frame_identity: FrameIdentity
    model_id: str
    model_revision: Sha256Digest
    policy_id: str
    attempt: PositiveInt = 1
    priority: NonNegativeInt = 0
    created_processing_seconds: NonNegativeFloat

    @field_validator("model_id", "policy_id")
    @classmethod
    def validate_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or normalized != value:
            raise ValueError("perception job text must be non-empty without whitespace")
        return value

    @classmethod
    def create(
        cls,
        *,
        frame_identity: FrameIdentity,
        model_id: str,
        model_revision: str,
        policy_id: str,
        attempt: int = 1,
        priority: int = 0,
        created_processing_seconds: float,
    ) -> Self:
        payload = {
            "schema_version": 1,
            "frame_identity": frame_identity,
            "model_id": model_id,
            "model_revision": model_revision,
            "policy_id": policy_id,
            "attempt": attempt,
            "priority": priority,
            "created_processing_seconds": created_processing_seconds,
        }
        identity_payload = {
            "schema_version": 1,
            "frame_id": frame_identity.frame_id,
            "model_id": model_id,
            "model_revision": model_revision,
            "policy_id": policy_id,
            "attempt": attempt,
            "priority": priority,
        }
        payload["job_id"] = _stable_digest(identity_payload)
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def validate_job_id(self) -> Self:
        expected = _stable_digest(
            {
                "schema_version": self.schema_version,
                "frame_id": self.frame_identity.frame_id,
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "policy_id": self.policy_id,
                "attempt": self.attempt,
                "priority": self.priority,
            }
        )
        if self.job_id != expected:
            raise ValueError("job_id does not match perception job identity")
        return self


@dataclass(frozen=True, slots=True)
class PerceptionWorkItem:
    """Runtime job plus immutable RGB pixels excluded from persistence."""

    job: PerceptionJob
    image_rgb: UInt8Array

    def __post_init__(self) -> None:
        image = np.asarray(self.image_rgb)
        expected_shape = (
            self.job.frame_identity.image_height,
            self.job.frame_identity.image_width,
            3,
        )
        if image.dtype != np.uint8 or image.shape != expected_shape:
            raise ValueError("perception image must be uint8 RGB matching frame identity")
        immutable = np.ascontiguousarray(image).copy()
        immutable.setflags(write=False)
        object.__setattr__(self, "image_rgb", immutable)


@dataclass(frozen=True, slots=True)
class PerceptionProcessingOutput:
    """Successful processor payload before timing and outcome wrapping."""

    candidates: tuple[PerceptionCandidate, ...] = ()
    raw_artifact_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(not value.strip() or value.strip() != value for value in self.raw_artifact_refs):
            raise ValueError("raw artifact references must be non-empty without whitespace")


class PerceptionProcessor(Protocol):
    """Replaceable processor invoked by the bounded worker runtime."""

    def process(self, item: PerceptionWorkItem) -> PerceptionProcessingOutput: ...


class PerceptionFrameResult(ContractModel):
    """Explicit success or failure tied to exactly one perception job."""

    schema_version: Literal[1] = 1
    job: PerceptionJob
    outcome: PerceptionResultOutcome
    candidates: tuple[PerceptionCandidate, ...] = ()
    raw_artifact_refs: tuple[str, ...] = ()
    processing_started_seconds: NonNegativeFloat
    processing_finished_seconds: NonNegativeFloat
    error_type: str | None = None
    error_message: str | None = None

    @field_validator("raw_artifact_refs")
    @classmethod
    def validate_artifact_refs(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() or value.strip() != value for value in values):
            raise ValueError("artifact references must be non-empty without whitespace")
        return values

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.processing_started_seconds < self.job.created_processing_seconds:
            raise ValueError("perception processing cannot start before job creation")
        if self.processing_finished_seconds < self.processing_started_seconds:
            raise ValueError("perception processing finish cannot precede start")
        if self.outcome is PerceptionResultOutcome.COMPLETED:
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("completed perception result cannot contain an error")
        else:
            if self.candidates or self.raw_artifact_refs:
                raise ValueError("failed perception result cannot contain model outputs")
            if not self.error_type or not self.error_message:
                raise ValueError("failed perception result requires explicit error details")
        return self


class QueueSubmission(ContractModel):
    """Persistent bounded-queue submission disposition."""

    job_id: Sha256Digest
    disposition: QueueSubmissionDisposition
    accepted: bool
    dropped_job_id: Sha256Digest | None = None
    queue_depth_after: NonNegativeInt

    @model_validator(mode="after")
    def validate_disposition(self) -> Self:
        if self.disposition is QueueSubmissionDisposition.THROTTLE_REQUIRED:
            if self.accepted or self.dropped_job_id is not None:
                raise ValueError("throttled submission cannot be accepted or drop work")
        elif self.disposition is QueueSubmissionDisposition.ACCEPTED:
            if not self.accepted or self.dropped_job_id is not None:
                raise ValueError("ordinary accepted submission cannot report a drop")
        elif not self.accepted or self.dropped_job_id is None:
            raise ValueError("drop-oldest submission must identify dropped accepted work")
        return self


class PerceptionQueueDiagnostics(ContractModel):
    """Inspectable queue counters for offline and future-live execution."""

    capacity: PositiveInt
    overflow_policy: QueueOverflowPolicy
    current_depth: NonNegativeInt
    in_flight_count: NonNegativeInt
    accepted_count: NonNegativeInt
    popped_count: NonNegativeInt
    completed_count: NonNegativeInt
    failed_count: NonNegativeInt
    throttled_count: NonNegativeInt
    dropped_oldest_count: NonNegativeInt
    cancelled_count: NonNegativeInt

    @model_validator(mode="after")
    def validate_depth(self) -> Self:
        if self.current_depth > self.capacity:
            raise ValueError("queue depth cannot exceed capacity")
        return self


class BoundedPerceptionQueue:
    """FIFO queue with deterministic offline and explicit live overflow policy."""

    def __init__(self, *, capacity: int, overflow_policy: QueueOverflowPolicy) -> None:
        if capacity <= 0:
            raise ValueError("perception queue capacity must be positive")
        self.capacity = capacity
        self.overflow_policy = overflow_policy
        self._pending: deque[PerceptionWorkItem] = deque()
        self._seen_job_ids: set[str] = set()
        self._terminal_job_ids: set[str] = set()
        self._in_flight_job_ids: set[str] = set()
        self._bound_camera_id: str | None = None
        self._last_accepted_frame_index: int | None = None
        self._last_accepted_capture_timestamp: float | None = None
        self._accepted_count = 0
        self._popped_count = 0
        self._completed_count = 0
        self._failed_count = 0
        self._throttled_count = 0
        self._dropped_oldest_count = 0
        self._cancelled_count = 0

    def submit(self, item: PerceptionWorkItem) -> QueueSubmission:
        """Submit one job without ever exceeding configured capacity."""

        job_id = item.job.job_id
        if job_id in self._seen_job_ids:
            raise ValueError(f"duplicate perception job_id: {job_id}")
        identity = item.job.frame_identity
        if self._bound_camera_id is None:
            self._bound_camera_id = identity.camera_id
        elif identity.camera_id != self._bound_camera_id:
            raise ValueError("one perception queue cannot mix camera streams")
        if (
            self._last_accepted_frame_index is not None
            and identity.source_frame_index <= self._last_accepted_frame_index
        ):
            raise ValueError("perception jobs must have increasing source frame indices")
        if (
            self._last_accepted_capture_timestamp is not None
            and identity.capture_timestamp_seconds
            <= self._last_accepted_capture_timestamp
        ):
            raise ValueError("perception jobs must have increasing capture timestamps")
        dropped_job_id: str | None = None
        if len(self._pending) >= self.capacity:
            if self.overflow_policy is QueueOverflowPolicy.THROTTLE:
                self._throttled_count += 1
                return QueueSubmission(
                    job_id=job_id,
                    disposition=QueueSubmissionDisposition.THROTTLE_REQUIRED,
                    accepted=False,
                    queue_depth_after=len(self._pending),
                )
            dropped = self._pending.popleft()
            dropped_job_id = dropped.job.job_id
            self._terminal_job_ids.add(dropped_job_id)
            self._dropped_oldest_count += 1

        self._pending.append(item)
        self._seen_job_ids.add(job_id)
        self._last_accepted_frame_index = identity.source_frame_index
        self._last_accepted_capture_timestamp = identity.capture_timestamp_seconds
        self._accepted_count += 1
        disposition = (
            QueueSubmissionDisposition.ACCEPTED_AFTER_DROP_OLDEST
            if dropped_job_id is not None
            else QueueSubmissionDisposition.ACCEPTED
        )
        return QueueSubmission(
            job_id=job_id,
            disposition=disposition,
            accepted=True,
            dropped_job_id=dropped_job_id,
            queue_depth_after=len(self._pending),
        )

    def pop(self) -> PerceptionWorkItem | None:
        """Pop the earliest accepted job in FIFO capture-submission order."""

        if not self._pending:
            return None
        self._popped_count += 1
        item = self._pending.popleft()
        self._in_flight_job_ids.add(item.job.job_id)
        return item

    def mark_result(self, result: PerceptionFrameResult) -> None:
        """Record exactly one terminal processor result for an accepted job."""

        job_id = result.job.job_id
        if job_id not in self._in_flight_job_ids:
            raise ValueError("perception result does not reference an in-flight job")
        if job_id in self._terminal_job_ids:
            raise ValueError("perception job already has a terminal disposition")
        self._in_flight_job_ids.remove(job_id)
        self._terminal_job_ids.add(job_id)
        if result.outcome is PerceptionResultOutcome.COMPLETED:
            self._completed_count += 1
        else:
            self._failed_count += 1

    def cancel_pending(self) -> tuple[PerceptionJob, ...]:
        """Drain pending work with explicit cancellation accounting."""

        cancelled = tuple(item.job for item in self._pending)
        self._pending.clear()
        for job in cancelled:
            self._terminal_job_ids.add(job.job_id)
        self._cancelled_count += len(cancelled)
        return cancelled

    @property
    def diagnostics(self) -> PerceptionQueueDiagnostics:
        """Return an immutable validated snapshot of queue state."""

        return PerceptionQueueDiagnostics(
            capacity=self.capacity,
            overflow_policy=self.overflow_policy,
            current_depth=len(self._pending),
            in_flight_count=len(self._in_flight_job_ids),
            accepted_count=self._accepted_count,
            popped_count=self._popped_count,
            completed_count=self._completed_count,
            failed_count=self._failed_count,
            throttled_count=self._throttled_count,
            dropped_oldest_count=self._dropped_oldest_count,
            cancelled_count=self._cancelled_count,
        )


def process_next_perception_item(
    queue: BoundedPerceptionQueue,
    processor: PerceptionProcessor,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> PerceptionFrameResult | None:
    """Process one FIFO item and convert exceptions into explicit failures."""

    item = queue.pop()
    if item is None:
        return None
    started = _clock_value(clock)
    try:
        output = processor.process(item)
        _validate_output_identity(output, item.job)
        finished = _clock_value(clock)
        result = PerceptionFrameResult(
            job=item.job,
            outcome=PerceptionResultOutcome.COMPLETED,
            candidates=output.candidates,
            raw_artifact_refs=output.raw_artifact_refs,
            processing_started_seconds=started,
            processing_finished_seconds=finished,
        )
    except Exception as exc:
        finished = _clock_value(clock)
        result = PerceptionFrameResult(
            job=item.job,
            outcome=PerceptionResultOutcome.FAILED,
            processing_started_seconds=started,
            processing_finished_seconds=finished,
            error_type=type(exc).__name__,
            error_message=str(exc) or repr(exc),
        )
    queue.mark_result(result)
    return result


def _clock_value(clock: Callable[[], float]) -> float:
    value = float(clock())
    if not math.isfinite(value) or value < 0:
        raise ValueError("worker clock must return finite non-negative seconds")
    return value


def _validate_output_identity(
    output: PerceptionProcessingOutput, job: PerceptionJob
) -> None:
    expected_frame = job.frame_identity.as_frame_ref()
    for candidate in output.candidates:
        if candidate.source_detection.frame != expected_frame:
            raise ValueError("processor candidate does not match perception job frame")


def _stable_digest(payload: dict[str, object]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
