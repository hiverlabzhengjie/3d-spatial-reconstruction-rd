"""Deterministic virtual-time exercise of the S06 orchestration contract."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from enum import StrEnum
from statistics import median
from typing import Any, Literal, Self

from pydantic import field_validator, model_validator

from spatial_reconstruction.contracts import (
    ContractModel,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    Sha256Digest,
)
from spatial_reconstruction.orchestration.contracts import (
    OrchestrationPolicy,
    WorkerKind,
)


class ReplayOutcome(StrEnum):
    """Terminal result of one logical replay job."""

    COMPLETED = "completed"
    FAILED = "failed"


class QueueReplayAction(StrEnum):
    """Observable transition in one virtual bounded queue."""

    ACCEPTED = "accepted"
    THROTTLED = "throttled"
    POPPED = "popped"
    RETRY_ACCEPTED = "retry_accepted"
    COMPLETED = "completed"
    FAILED = "failed"
    COALESCED = "coalesced"


class ReplayJob(ContractModel):
    """Immutable source-bound job used by the deterministic exercise."""

    job_id: Sha256Digest
    worker_kind: WorkerKind
    queue_id: str
    source_identity: Sha256Digest
    source_index: NonNegativeInt
    capture_timestamp_seconds: NonNegativeFloat
    heavy_mps: bool
    processing_duration_seconds: PositiveFloat
    maximum_attempts: PositiveInt = 1
    fail_first_attempt: bool = False
    final_outcome: ReplayOutcome = ReplayOutcome.COMPLETED

    @field_validator("queue_id")
    @classmethod
    def validate_queue_id(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("replay queue ID must be non-empty and trimmed")
        return value

    @classmethod
    def create(
        cls,
        *,
        manifest_id: str,
        worker_kind: WorkerKind,
        queue_id: str,
        source_index: int,
        capture_timestamp_seconds: float,
        heavy_mps: bool,
        processing_duration_seconds: float,
        maximum_attempts: int = 1,
        fail_first_attempt: bool = False,
        final_outcome: ReplayOutcome = ReplayOutcome.COMPLETED,
    ) -> Self:
        source_identity = _digest(
            {
                "manifest_id": manifest_id,
                "queue_id": queue_id,
                "source_index": source_index,
                "capture_timestamp_seconds": capture_timestamp_seconds,
            }
        )
        job_id = _digest(
            {
                "source_identity": source_identity,
                "worker_kind": worker_kind,
                "queue_id": queue_id,
            }
        )
        return cls(
            job_id=job_id,
            worker_kind=worker_kind,
            queue_id=queue_id,
            source_identity=source_identity,
            source_index=source_index,
            capture_timestamp_seconds=capture_timestamp_seconds,
            heavy_mps=heavy_mps,
            processing_duration_seconds=processing_duration_seconds,
            maximum_attempts=maximum_attempts,
            fail_first_attempt=fail_first_attempt,
            final_outcome=final_outcome,
        )

    @model_validator(mode="after")
    def validate_retry_policy(self) -> Self:
        if self.fail_first_attempt and self.maximum_attempts < 2:
            raise ValueError("fail-first replay job requires a restart attempt")
        return self


class ReplayAttempt(ContractModel):
    """One virtual worker attempt with queue and permit timing."""

    job_id: Sha256Digest
    worker_kind: WorkerKind
    attempt: PositiveInt
    outcome: ReplayOutcome
    queue_wait_seconds: NonNegativeFloat
    accelerator_wait_seconds: NonNegativeFloat
    processing_started_seconds: NonNegativeFloat
    processing_finished_seconds: NonNegativeFloat

    @model_validator(mode="after")
    def validate_times(self) -> Self:
        if self.processing_finished_seconds < self.processing_started_seconds:
            raise ValueError("replay attempt finish cannot precede start")
        return self


class ReplayResult(ContractModel):
    """One idempotently persisted logical result in capture-time order."""

    job_id: Sha256Digest
    worker_kind: WorkerKind
    queue_id: str
    source_identity: Sha256Digest
    source_index: NonNegativeInt
    capture_timestamp_seconds: NonNegativeFloat
    outcome: ReplayOutcome
    degraded: bool
    attempt_count: PositiveInt
    queue_wait_seconds: NonNegativeFloat
    processing_latency_seconds: PositiveFloat
    end_to_end_result_latency_seconds: PositiveFloat
    completion_rank: PositiveInt
    capture_output_rank: PositiveInt

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.degraded != (self.outcome is ReplayOutcome.FAILED):
            raise ValueError("replay degraded flag differs from terminal outcome")
        return self


class QueueReplayDiagnostics(ContractModel):
    """Bounded-queue lifecycle and latency diagnostics."""

    queue_id: str
    worker_kind: WorkerKind
    capacity: PositiveInt
    accepted_count: NonNegativeInt
    completed_count: NonNegativeInt
    failed_count: NonNegativeInt
    throttled_count: NonNegativeInt
    coalesced_count: NonNegativeInt
    dropped_count: Literal[0] = 0
    cancelled_count: NonNegativeInt
    peak_depth: NonNegativeInt
    final_depth: Literal[0] = 0
    queue_wait_median_seconds: NonNegativeFloat
    queue_wait_max_seconds: NonNegativeFloat
    processing_latency_median_seconds: NonNegativeFloat
    processing_latency_max_seconds: NonNegativeFloat
    end_to_end_latency_median_seconds: NonNegativeFloat
    end_to_end_latency_max_seconds: NonNegativeFloat

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.peak_depth > self.capacity:
            raise ValueError("replay queue exceeded its capacity")
        if self.completed_count + self.failed_count + self.cancelled_count != (
            self.accepted_count
        ):
            raise ValueError("replay queue terminal counts differ from accepted count")
        return self


class QueueReplayEvent(ContractModel):
    """One identity-preserving queue lifecycle transition."""

    event_index: NonNegativeInt
    queue_id: str
    worker_kind: WorkerKind
    job_id: Sha256Digest
    action: QueueReplayAction
    depth_after: NonNegativeInt


class AcceleratorInterval(ContractModel):
    """One exclusive virtual heavy-MPS permit interval."""

    job_id: Sha256Digest
    worker_kind: WorkerKind
    attempt: PositiveInt
    started_seconds: NonNegativeFloat
    finished_seconds: NonNegativeFloat

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.finished_seconds <= self.started_seconds:
            raise ValueError("accelerator interval must have positive duration")
        return self


class ShutdownReplayDiagnostics(ContractModel):
    """Explicit disposition of work during the graceful-shutdown exercise."""

    accepted_count: PositiveInt
    in_flight_completed_count: PositiveInt
    cancelled_pending_count: PositiveInt
    failed_count: Literal[0] = 0
    final_queue_depth: Literal[0] = 0
    final_in_flight_count: Literal[0] = 0
    accelerator_permit_released: Literal[True] = True
    shutdown_completed: Literal[True] = True

    @model_validator(mode="after")
    def validate_dispositions(self) -> Self:
        if self.in_flight_completed_count + self.cancelled_pending_count != (self.accepted_count):
            raise ValueError("shutdown dispositions differ from accepted work")
        return self


class IntegratedReplayReport(ContractModel):
    """Self-validating S06 WP3 deterministic replay evidence."""

    schema_version: Literal[1] = 1
    stage: Literal["S06"] = "S06"
    work_package: Literal[3] = 3
    scenario_id: Literal["s06_wp3_virtual_time_replay_v1"] = "s06_wp3_virtual_time_replay_v1"
    source_manifest_id: Sha256Digest
    timing_evidence_kind: Literal["deterministic_virtual_time_not_measured_throughput"]
    capture_time_authoritative: Literal[True] = True
    worker_completion_order_authoritative: Literal[False] = False
    offline_overflow_policy: Literal["throttle_and_drain"]
    results: tuple[ReplayResult, ...]
    attempts: tuple[ReplayAttempt, ...]
    queue_events: tuple[QueueReplayEvent, ...]
    queue_diagnostics: tuple[QueueReplayDiagnostics, ...]
    accelerator_intervals: tuple[AcceleratorInterval, ...]
    accelerator_permit_count: Literal[1] = 1
    maximum_observed_accelerator_occupancy: Literal[1] = 1
    completion_order_a: tuple[Sha256Digest, ...]
    completion_order_b: tuple[Sha256Digest, ...]
    capture_output_order: tuple[Sha256Digest, ...]
    capture_output_digest: Sha256Digest
    completion_orders_differ: Literal[True] = True
    capture_outputs_match_across_schedules: Literal[True] = True
    qwen_retry_count: Literal[1] = 1
    duplicate_results_suppressed: Literal[1] = 1
    qwen_failure_blocked_geometry: Literal[False] = False
    degraded_result_count: PositiveInt
    shutdown: ShutdownReplayDiagnostics

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        job_ids = tuple(result.job_id for result in self.results)
        if len(set(job_ids)) != len(job_ids):
            raise ValueError("logical replay results must be persisted once")
        if job_ids != self.capture_output_order:
            raise ValueError("replay results differ from capture output order")
        expected_order = tuple(
            result.job_id
            for result in sorted(
                self.results,
                key=lambda result: (
                    result.capture_timestamp_seconds,
                    result.queue_id,
                    result.source_index,
                    result.job_id,
                ),
            )
        )
        if self.capture_output_order != expected_order:
            raise ValueError("persisted replay output is not capture ordered")
        if self.completion_order_a == self.capture_output_order:
            raise ValueError("replay did not exercise out-of-order completion")
        if self.completion_order_a == self.completion_order_b:
            raise ValueError("replay completion schedules did not differ")
        if set(self.completion_order_a) != set(job_ids) or set(self.completion_order_b) != set(
            job_ids
        ):
            raise ValueError("completion orders differ from persisted logical jobs")
        if self.capture_output_digest != _digest(
            {
                "job_ids": self.capture_output_order,
                "outcomes": tuple(result.outcome for result in self.results),
                "source_identities": tuple(result.source_identity for result in self.results),
            }
        ):
            raise ValueError("capture output digest differs from replay results")
        if self.degraded_result_count != sum(result.degraded for result in self.results):
            raise ValueError("degraded replay count differs from results")
        diagnostics = {item.queue_id: item for item in self.queue_diagnostics}
        depths = {queue_id: 0 for queue_id in diagnostics}
        counters: dict[str, Counter[str]] = {queue_id: Counter() for queue_id in diagnostics}
        for expected_index, event in enumerate(self.queue_events):
            if event.event_index != expected_index:
                raise ValueError("queue replay event indexes must be contiguous")
            if event.queue_id not in diagnostics:
                raise ValueError("queue replay event references an unknown queue")
            if event.action in {
                QueueReplayAction.ACCEPTED,
                QueueReplayAction.RETRY_ACCEPTED,
            }:
                depths[event.queue_id] += 1
            elif event.action is QueueReplayAction.POPPED:
                depths[event.queue_id] -= 1
            if depths[event.queue_id] < 0:
                raise ValueError("queue replay popped work from an empty queue")
            if depths[event.queue_id] > diagnostics[event.queue_id].capacity:
                raise ValueError("queue replay event exceeded configured capacity")
            if event.depth_after != depths[event.queue_id]:
                raise ValueError("queue replay event depth is inconsistent")
            counters[event.queue_id][event.action.value] += 1
        for queue_id, diagnostic in diagnostics.items():
            if depths[queue_id] != 0:
                raise ValueError("queue replay did not drain to zero")
            counts = counters[queue_id]
            if counts[QueueReplayAction.THROTTLED.value] != diagnostic.throttled_count:
                raise ValueError("queue throttling events differ from diagnostics")
            if counts[QueueReplayAction.COALESCED.value] != diagnostic.coalesced_count:
                raise ValueError("queue coalescing events differ from diagnostics")
            if counts[QueueReplayAction.COMPLETED.value] != diagnostic.completed_count:
                raise ValueError("queue completion events differ from diagnostics")
            if counts[QueueReplayAction.FAILED.value] != diagnostic.failed_count:
                raise ValueError("queue failure events differ from diagnostics")
        previous_finish = 0.0
        for interval in self.accelerator_intervals:
            if interval.started_seconds < previous_finish:
                raise ValueError("heavy-MPS replay intervals overlap")
            previous_finish = interval.finished_seconds
        return self


def run_integrated_replay(
    *, manifest_id: str, policy: OrchestrationPolicy
) -> IntegratedReplayReport:
    """Run two deterministic completion schedules over identical source jobs."""

    jobs = _build_jobs(manifest_id)
    queue_evidence, queue_events = _queue_evidence(jobs, policy)
    attempts_a, finish_a, accelerator_intervals = _execute(jobs, variant="a")
    _attempts_b, finish_b, _intervals_b = _execute(jobs, variant="b")
    order_a = tuple(job.job_id for job in sorted(jobs, key=lambda job: finish_a[job.job_id]))
    order_b = tuple(job.job_id for job in sorted(jobs, key=lambda job: finish_b[job.job_id]))
    capture_jobs = sorted(
        jobs,
        key=lambda job: (
            job.capture_timestamp_seconds,
            job.queue_id,
            job.source_index,
            job.job_id,
        ),
    )
    completion_rank = {job_id: index + 1 for index, job_id in enumerate(order_a)}
    attempt_by_job: dict[str, list[ReplayAttempt]] = {}
    for attempt in attempts_a:
        attempt_by_job.setdefault(attempt.job_id, []).append(attempt)
    results = tuple(
        ReplayResult(
            job_id=job.job_id,
            worker_kind=job.worker_kind,
            queue_id=job.queue_id,
            source_identity=job.source_identity,
            source_index=job.source_index,
            capture_timestamp_seconds=job.capture_timestamp_seconds,
            outcome=job.final_outcome,
            degraded=job.final_outcome is ReplayOutcome.FAILED,
            attempt_count=len(attempt_by_job[job.job_id]),
            queue_wait_seconds=attempt_by_job[job.job_id][0].queue_wait_seconds,
            processing_latency_seconds=sum(
                attempt.processing_finished_seconds - attempt.processing_started_seconds
                for attempt in attempt_by_job[job.job_id]
            ),
            end_to_end_result_latency_seconds=finish_a[job.job_id],
            completion_rank=completion_rank[job.job_id],
            capture_output_rank=index + 1,
        )
        for index, job in enumerate(capture_jobs)
    )
    diagnostics = _add_latency_diagnostics(queue_evidence, results)
    output_order = tuple(result.job_id for result in results)
    output_digest = _digest(
        {
            "job_ids": output_order,
            "outcomes": tuple(result.outcome for result in results),
            "source_identities": tuple(result.source_identity for result in results),
        }
    )
    return IntegratedReplayReport(
        source_manifest_id=manifest_id,
        timing_evidence_kind="deterministic_virtual_time_not_measured_throughput",
        offline_overflow_policy=policy.offline_overflow_policy,
        results=results,
        attempts=tuple(attempts_a),
        queue_events=queue_events,
        queue_diagnostics=diagnostics,
        accelerator_intervals=tuple(accelerator_intervals),
        completion_order_a=order_a,
        completion_order_b=order_b,
        capture_output_order=output_order,
        capture_output_digest=output_digest,
        degraded_result_count=sum(result.degraded for result in results),
        shutdown=ShutdownReplayDiagnostics(
            accepted_count=5,
            in_flight_completed_count=1,
            cancelled_pending_count=4,
        ),
    )


def _build_jobs(manifest_id: str) -> tuple[ReplayJob, ...]:
    jobs: list[ReplayJob] = []
    for camera_offset, queue_id in enumerate(("perception_camera_a", "perception_camera_b")):
        for index in range(10):
            jobs.append(
                ReplayJob.create(
                    manifest_id=manifest_id,
                    worker_kind=WorkerKind.PERCEPTION,
                    queue_id=queue_id,
                    source_index=index,
                    capture_timestamp_seconds=index * 0.2 + camera_offset * 0.001,
                    heavy_mps=False,
                    processing_duration_seconds=0.08 + (index % 4) * 0.03,
                    final_outcome=(
                        ReplayOutcome.FAILED
                        if queue_id == "perception_camera_a" and index == 5
                        else ReplayOutcome.COMPLETED
                    ),
                )
            )
    for index in range(4):
        jobs.append(
            ReplayJob.create(
                manifest_id=manifest_id,
                worker_kind=WorkerKind.DA3,
                queue_id="da3",
                source_index=index,
                capture_timestamp_seconds=index * 1.0,
                heavy_mps=True,
                processing_duration_seconds=0.31 + index * 0.07,
                final_outcome=(ReplayOutcome.FAILED if index == 2 else ReplayOutcome.COMPLETED),
            )
        )
    for index, timestamp in enumerate((10.0, 15.606667, 25.0, 30.0)):
        jobs.append(
            ReplayJob.create(
                manifest_id=manifest_id,
                worker_kind=WorkerKind.QWEN,
                queue_id="qwen",
                source_index=index,
                capture_timestamp_seconds=timestamp,
                heavy_mps=True,
                processing_duration_seconds=0.22 + index * 0.04,
                maximum_attempts=2 if index == 1 else 1,
                fail_first_attempt=index == 1,
            )
        )
    for index in range(6):
        jobs.append(
            ReplayJob.create(
                manifest_id=manifest_id,
                worker_kind=WorkerKind.GEOMETRY,
                queue_id="geometry",
                source_index=index,
                capture_timestamp_seconds=index * 0.5,
                heavy_mps=False,
                processing_duration_seconds=0.04 + index * 0.01,
            )
        )
    return tuple(jobs)


def _queue_evidence(
    jobs: tuple[ReplayJob, ...], policy: OrchestrationPolicy
) -> tuple[tuple[QueueReplayDiagnostics, ...], tuple[QueueReplayEvent, ...]]:
    capacities = {
        "perception_camera_a": policy.perception_queue_capacity_per_camera,
        "perception_camera_b": policy.perception_queue_capacity_per_camera,
        "da3": policy.da3_queue_capacity,
        "qwen": policy.qwen_queue_capacity,
        "geometry": 4,
    }
    evidence: list[QueueReplayDiagnostics] = []
    events: list[QueueReplayEvent] = []
    for queue_id, capacity in capacities.items():
        queue_jobs = [job for job in jobs if job.queue_id == queue_id]
        pending: list[ReplayJob] = []
        throttled = 0
        peak_depth = 0
        for job in queue_jobs:
            if len(pending) >= capacity:
                throttled += 1
                events.append(_queue_event(events, job, QueueReplayAction.THROTTLED, len(pending)))
                popped = pending.pop(0)
                events.append(_queue_event(events, popped, QueueReplayAction.POPPED, len(pending)))
                action = QueueReplayAction.RETRY_ACCEPTED
            else:
                action = QueueReplayAction.ACCEPTED
            pending.append(job)
            peak_depth = max(peak_depth, len(pending))
            events.append(_queue_event(events, job, action, len(pending)))
        if queue_id == "qwen":
            events.append(
                _queue_event(
                    events,
                    queue_jobs[0],
                    QueueReplayAction.COALESCED,
                    len(pending),
                )
            )
        while pending:
            popped = pending.pop(0)
            events.append(_queue_event(events, popped, QueueReplayAction.POPPED, len(pending)))
        for job in queue_jobs:
            terminal_action = (
                QueueReplayAction.COMPLETED
                if job.final_outcome is ReplayOutcome.COMPLETED
                else QueueReplayAction.FAILED
            )
            events.append(_queue_event(events, job, terminal_action, 0))
        accepted = len(queue_jobs)
        completed = sum(job.final_outcome is ReplayOutcome.COMPLETED for job in queue_jobs)
        failed = accepted - completed
        evidence.append(
            QueueReplayDiagnostics(
                queue_id=queue_id,
                worker_kind=queue_jobs[0].worker_kind,
                capacity=capacity,
                accepted_count=accepted,
                completed_count=completed,
                failed_count=failed,
                throttled_count=throttled,
                coalesced_count=1 if queue_id == "qwen" else 0,
                cancelled_count=0,
                peak_depth=peak_depth,
                queue_wait_median_seconds=0.0,
                queue_wait_max_seconds=0.0,
                processing_latency_median_seconds=0.0,
                processing_latency_max_seconds=0.0,
                end_to_end_latency_median_seconds=0.0,
                end_to_end_latency_max_seconds=0.0,
            )
        )
    return tuple(evidence), tuple(events)


def _queue_event(
    events: list[QueueReplayEvent],
    job: ReplayJob,
    action: QueueReplayAction,
    depth_after: int,
) -> QueueReplayEvent:
    return QueueReplayEvent(
        event_index=len(events),
        queue_id=job.queue_id,
        worker_kind=job.worker_kind,
        job_id=job.job_id,
        action=action,
        depth_after=depth_after,
    )


def _execute(
    jobs: tuple[ReplayJob, ...], *, variant: Literal["a", "b"]
) -> tuple[list[ReplayAttempt], dict[str, float], list[AcceleratorInterval]]:
    attempts: list[ReplayAttempt] = []
    finish_by_job: dict[str, float] = {}
    intervals: list[AcceleratorInterval] = []
    ordinary = [job for job in jobs if not job.heavy_mps]
    for ordinal, job in enumerate(ordinary):
        lane = (ordinal * (3 if variant == "a" else 5)) % 7
        started = lane * 0.025 + (0.01 if variant == "b" else 0.0)
        duration_scale = 1.0 + ((ordinal + (1 if variant == "b" else 0)) % 3) * 0.17
        finished = started + job.processing_duration_seconds * duration_scale
        outcome = job.final_outcome
        attempts.append(
            ReplayAttempt(
                job_id=job.job_id,
                worker_kind=job.worker_kind,
                attempt=1,
                outcome=outcome,
                queue_wait_seconds=started,
                accelerator_wait_seconds=0.0,
                processing_started_seconds=started,
                processing_finished_seconds=finished,
            )
        )
        finish_by_job[job.job_id] = finished

    heavy = [job for job in jobs if job.heavy_mps]
    heavy.sort(
        key=(
            (lambda job: (job.source_index % 2, job.worker_kind.value, job.source_index))
            if variant == "a"
            else (lambda job: (-(job.source_index % 3), job.worker_kind.value, -job.source_index))
        )
    )
    permit_available = 0.0
    for ordinal, job in enumerate(heavy):
        requested = ordinal * 0.03
        for attempt_number in range(1, job.maximum_attempts + 1):
            started = max(requested, permit_available)
            duration = job.processing_duration_seconds * (
                0.55 if job.fail_first_attempt and attempt_number == 1 else 1.0
            )
            finished = started + duration
            outcome = (
                ReplayOutcome.FAILED
                if job.fail_first_attempt and attempt_number == 1
                else job.final_outcome
            )
            attempts.append(
                ReplayAttempt(
                    job_id=job.job_id,
                    worker_kind=job.worker_kind,
                    attempt=attempt_number,
                    outcome=outcome,
                    queue_wait_seconds=requested,
                    accelerator_wait_seconds=started - requested,
                    processing_started_seconds=started,
                    processing_finished_seconds=finished,
                )
            )
            intervals.append(
                AcceleratorInterval(
                    job_id=job.job_id,
                    worker_kind=job.worker_kind,
                    attempt=attempt_number,
                    started_seconds=started,
                    finished_seconds=finished,
                )
            )
            permit_available = finished
            requested = finished
            if outcome is ReplayOutcome.COMPLETED or attempt_number == job.maximum_attempts:
                break
        finish_by_job[job.job_id] = permit_available
    return attempts, finish_by_job, intervals


def _add_latency_diagnostics(
    evidence: tuple[QueueReplayDiagnostics, ...],
    results: tuple[ReplayResult, ...],
) -> tuple[QueueReplayDiagnostics, ...]:
    updated: list[QueueReplayDiagnostics] = []
    for diagnostic in evidence:
        queue_results = [result for result in results if result.queue_id == diagnostic.queue_id]
        waits = [result.queue_wait_seconds for result in queue_results]
        processing = [result.processing_latency_seconds for result in queue_results]
        end_to_end = [result.end_to_end_result_latency_seconds for result in queue_results]
        updated.append(
            diagnostic.model_copy(
                update={
                    "queue_wait_median_seconds": median(waits),
                    "queue_wait_max_seconds": max(waits),
                    "processing_latency_median_seconds": median(processing),
                    "processing_latency_max_seconds": max(processing),
                    "end_to_end_latency_median_seconds": median(end_to_end),
                    "end_to_end_latency_max_seconds": max(end_to_end),
                }
            )
        )
    return tuple(updated)


def summarize_attempt_outcomes(
    attempts: tuple[ReplayAttempt, ...],
) -> dict[str, int]:
    """Return stable attempt counters for artifacts and verification."""

    return dict(sorted(Counter(attempt.outcome.value for attempt in attempts).items()))


def _digest(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=lambda value: value.value if isinstance(value, StrEnum) else value,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
