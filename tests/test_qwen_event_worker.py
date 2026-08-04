from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from spatial_reconstruction.interaction import (
    BoundedQwenEventQueue,
    InteractionEventKind,
    PhaseAuthority,
    QwenEventJob,
    QwenEventLabel,
    QwenEventProcessingOutput,
    QwenEventResultOutcome,
    QwenEventReviewPolicy,
    QwenQueueOverflowPolicy,
    QwenQueueSubmissionDisposition,
    QwenResponseNormalization,
    QwenResponseValidationError,
    QwenSampleRole,
    QwenVideoSource,
    SemanticEventCandidate,
    build_qwen_event_jobs,
    make_qwen_retry_job,
    parse_qwen_event_response,
    process_next_qwen_event,
)

SYNC_HASH = "a" * 64
VIDEO_A_HASH = "b" * 64
VIDEO_B_HASH = "c" * 64
STATE_HASHES = ("d" * 64, "e" * 64, "f" * 64)


def _digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate(
    *,
    kind: InteractionEventKind,
    frame: int,
    timestamp: float,
    state_id: str,
) -> SemanticEventCandidate:
    candidate_id = _digest(
        {
            "schema_version": 2,
            "policy_id": "s05_semantic_interaction_v2",
            "event_kind": kind.value,
            "source_state_record_id": state_id,
        }
    )
    measured = kind is not InteractionEventKind.CARRY
    return SemanticEventCandidate(
        candidate_id=candidate_id,
        policy_id="s05_semantic_interaction_v2",
        event_kind=kind,
        source_state_record_id=state_id,
        source_frame_index=frame,
        capture_timestamp_seconds=timestamp,
        clip_start_timestamp_seconds=timestamp - 2.0,
        clip_end_timestamp_seconds=timestamp + 2.0,
        phase_authority=(
            PhaseAuthority.MEASURED_SPATIAL
            if measured
            else PhaseAuthority.SEQUENCE_CONTINUITY
        ),
        spatial_transition_authority=measured,
    )


def _jobs() -> tuple[QwenEventJob, ...]:
    candidates = (
        _candidate(
            kind=InteractionEventKind.PICKUP,
            frame=462,
            timestamp=15.4,
            state_id=STATE_HASHES[0],
        ),
        _candidate(
            kind=InteractionEventKind.CARRY,
            frame=468,
            timestamp=15.6,
            state_id=STATE_HASHES[1],
        ),
        _candidate(
            kind=InteractionEventKind.PLACE,
            frame=666,
            timestamp=22.2,
            state_id=STATE_HASHES[2],
        ),
    )
    sources = (
        QwenVideoSource(
            camera_id="camera_a",
            source_ref="artifacts/camera_a.mp4",
            source_sha256=VIDEO_A_HASH,
            decoded_frame_count=1047,
            duration_seconds=34.9,
            nominal_frame_rate_fps=30.0,
        ),
        QwenVideoSource(
            camera_id="camera_b",
            source_ref="artifacts/camera_b.mp4",
            source_sha256=VIDEO_B_HASH,
            decoded_frame_count=1047,
            duration_seconds=34.9,
            nominal_frame_rate_fps=30.0,
        ),
    )
    return build_qwen_event_jobs(
        candidates=candidates,
        video_sources=sources,
        capture_session_id="session_01",
        synchronization_manifest_ref="artifacts/sync.json",
        synchronization_manifest_sha256=SYNC_HASH,
        policy=QwenEventReviewPolicy(),
        created_processing_seconds=1.0,
    )


def _valid_response(kind: InteractionEventKind) -> str:
    return json.dumps(
        {
            "event_label": kind.value,
            "evidence_strength": "strong",
            "summary": f"Visible {kind.value} action.",
            "visible_evidence": "The backpack changes relative to the person's hands.",
            "uncertainty": None,
        }
    )


def _output(text: str) -> QwenEventProcessingOutput:
    return QwenEventProcessingOutput(
        raw_response_text=text,
        input_token_count=100,
        output_token_count=3,
        output_token_ids=(1, 2, 3),
        input_shapes={"input_ids": (1, 100), "pixel_values": (6, 3, 8, 8)},
    )


def _clock(*values: float) -> Iterator[float]:
    yield from values


class ValidProcessor:
    async def process(self, job: QwenEventJob) -> QwenEventProcessingOutput:
        return _output(_valid_response(job.event_kind))


class InvalidProcessor:
    async def process(self, job: QwenEventJob) -> QwenEventProcessingOutput:
        assert job.event_kind is InteractionEventKind.PICKUP
        return _output("not-json")


class FailingProcessor:
    async def process(self, job: QwenEventJob) -> QwenEventProcessingOutput:
        raise RuntimeError(f"failed {job.event_kind.value}")


class SlowProcessor:
    async def process(self, job: QwenEventJob) -> QwenEventProcessingOutput:
        await asyncio.sleep(0.02)
        return _output(_valid_response(job.event_kind))


def _queue(*, capacity: int = 3) -> BoundedQwenEventQueue:
    return BoundedQwenEventQueue(
        capacity=capacity,
        overflow_policy=QwenQueueOverflowPolicy.THROTTLE,
        maximum_attempts=2,
    )


def test_jobs_are_stable_ordered_and_have_no_spatial_write_fields() -> None:
    first = _jobs()
    replay = _jobs()

    assert [job.event_kind.value for job in first] == ["pickup", "carry", "place"]
    assert [job.job_id for job in first] == [job.job_id for job in replay]
    assert [job.source_frame_index for job in first] == [462, 468, 666]
    assert [job.review_frame_index for job in first] == [462, 567, 666]
    assert first[1].review_timestamp_seconds == pytest.approx(18.9)
    assert [item.source_frame_index for item in first[1].frame_inputs] == [
        507,
        507,
        567,
        567,
        627,
        627,
    ]
    for job in first:
        assert len(job.frame_inputs) == 6
        assert [item.sample_role for item in job.frame_inputs] == [
            QwenSampleRole.BEFORE,
            QwenSampleRole.BEFORE,
            QwenSampleRole.TRANSITION,
            QwenSampleRole.TRANSITION,
            QwenSampleRole.AFTER,
            QwenSampleRole.AFTER,
        ]
        assert [item.camera_id for item in job.frame_inputs] == [
            "camera_a",
            "camera_b",
        ] * 3
        fields = set(job.model_dump())
        assert not fields.intersection(
            {
                "world_xyz_m",
                "track_identity",
                "zone_membership",
                "spatial_authority",
            }
        )


def test_queue_is_bounded_capture_ordered_and_coalesces_duplicates() -> None:
    jobs = _jobs()
    queue = _queue(capacity=2)

    assert queue.submit(jobs[0]).accepted
    assert queue.submit(jobs[1]).accepted
    duplicate = queue.submit(jobs[0])
    throttled = queue.submit(jobs[2])

    assert duplicate.disposition is QwenQueueSubmissionDisposition.DUPLICATE_COALESCED
    assert duplicate.existing_job_id == jobs[0].job_id
    assert throttled.disposition is QwenQueueSubmissionDisposition.THROTTLE_REQUIRED
    assert [queue.pop(), queue.pop()] == [jobs[0], jobs[1]]
    diagnostics = queue.diagnostics
    assert diagnostics.duplicate_coalesced_count == 1
    assert diagnostics.throttled_count == 1
    assert diagnostics.current_depth == 0
    assert diagnostics.in_flight_count == 2


def test_live_drop_oldest_and_pending_cancellation_are_explicit() -> None:
    jobs = _jobs()
    live_queue = BoundedQwenEventQueue(
        capacity=1,
        overflow_policy=QwenQueueOverflowPolicy.DROP_OLDEST,
        maximum_attempts=2,
    )
    live_queue.submit(jobs[0])
    replacement = live_queue.submit(jobs[1])
    assert replacement.disposition is (
        QwenQueueSubmissionDisposition.ACCEPTED_AFTER_DROP_OLDEST
    )
    assert replacement.dropped_job_id == jobs[0].job_id
    assert live_queue.diagnostics.dropped_oldest_count == 1

    cancelled = live_queue.cancel_pending()
    assert cancelled == (jobs[1],)
    assert live_queue.pop() is None
    assert live_queue.diagnostics.cancelled_count == 1


def test_initial_jobs_must_arrive_in_capture_order() -> None:
    jobs = _jobs()
    queue = _queue()
    queue.submit(jobs[1])
    with pytest.raises(ValueError, match="capture-time order"):
        queue.submit(jobs[0])


def test_valid_async_result_is_schema_bounded_and_terminal() -> None:
    job = _jobs()[0]
    queue = _queue()
    queue.submit(job)
    clock = _clock(2.0, 3.0)

    result = asyncio.run(
        process_next_qwen_event(
            queue,
            ValidProcessor(),
            clock=lambda: next(clock),
        )
    )

    assert result is not None
    assert result.outcome is QwenEventResultOutcome.COMPLETED
    assert result.interpretation.event_label is QwenEventLabel.PICKUP
    assert result.interpretation.matches_candidate
    assert not result.interpretation.spatial_claims_present
    assert result.response_normalization is QwenResponseNormalization.NONE
    assert queue.diagnostics.completed_count == 1
    duplicate = queue.submit(job)
    assert duplicate.disposition is QwenQueueSubmissionDisposition.DUPLICATE_COALESCED


def test_invalid_output_becomes_unknown_and_allows_one_repair_attempt() -> None:
    job = _jobs()[0]
    queue = _queue()
    queue.submit(job)
    first_clock = _clock(2.0, 3.0)
    invalid = asyncio.run(
        process_next_qwen_event(
            queue,
            InvalidProcessor(),
            clock=lambda: next(first_clock),
        )
    )
    assert invalid is not None
    assert invalid.outcome is QwenEventResultOutcome.INVALID_OUTPUT
    assert invalid.interpretation.event_label is QwenEventLabel.UNKNOWN
    assert invalid.raw_response_text == "not-json"
    assert invalid.input_token_count == 100
    assert invalid.output_token_count == 3
    assert invalid.output_token_ids == (1, 2, 3)
    assert invalid.input_shapes == {
        "input_ids": (1, 100),
        "pixel_values": (6, 3, 8, 8),
    }

    retry = make_qwen_retry_job(job, created_processing_seconds=4.0)
    assert retry.attempt == 2
    assert retry.deduplication_key == job.deduplication_key
    assert queue.submit(retry).accepted
    second_clock = _clock(5.0, 6.0)
    completed = asyncio.run(
        process_next_qwen_event(
            queue,
            ValidProcessor(),
            clock=lambda: next(second_clock),
        )
    )
    assert completed is not None
    assert completed.outcome is QwenEventResultOutcome.COMPLETED
    assert queue.diagnostics.retry_accepted_count == 1
    assert queue.diagnostics.invalid_output_count == 1


def test_retry_cannot_skip_or_exceed_the_attempt_bound() -> None:
    job = _jobs()[0]
    queue = _queue()
    queue.submit(job)
    first_clock = _clock(2.0, 3.0)
    first = asyncio.run(
        process_next_qwen_event(
            queue,
            FailingProcessor(),
            clock=lambda: next(first_clock),
        )
    )
    assert first is not None and first.outcome is QwenEventResultOutcome.FAILED

    retry = make_qwen_retry_job(job, created_processing_seconds=4.0)
    queue.submit(retry)
    second_clock = _clock(5.0, 6.0)
    second = asyncio.run(
        process_next_qwen_event(
            queue,
            FailingProcessor(),
            clock=lambda: next(second_clock),
        )
    )
    assert second is not None and second.outcome is QwenEventResultOutcome.FAILED

    third = make_qwen_retry_job(retry, created_processing_seconds=7.0)
    with pytest.raises(ValueError, match="bounded attempt"):
        queue.submit(third)


def test_timeout_and_exception_do_not_escape_or_block_queue() -> None:
    timeout_job_payload = _jobs()[0].model_dump(mode="json")
    timeout_job_payload["timeout_seconds"] = 0.001
    timeout_job = QwenEventJob.model_validate(timeout_job_payload)
    timeout_queue = _queue()
    timeout_queue.submit(timeout_job)
    timeout_clock = _clock(2.0, 3.0)
    timed_out = asyncio.run(
        process_next_qwen_event(
            timeout_queue,
            SlowProcessor(),
            clock=lambda: next(timeout_clock),
        )
    )
    assert timed_out is not None
    assert timed_out.outcome is QwenEventResultOutcome.TIMED_OUT
    assert timed_out.interpretation.event_label is QwenEventLabel.UNKNOWN
    assert timeout_queue.diagnostics.timed_out_count == 1

    failure_job = _jobs()[1]
    failure_queue = _queue()
    failure_queue.submit(failure_job)
    failure_clock = _clock(2.0, 3.0)
    failed = asyncio.run(
        process_next_qwen_event(
            failure_queue,
            FailingProcessor(),
            clock=lambda: next(failure_clock),
        )
    )
    assert failed is not None
    assert failed.outcome is QwenEventResultOutcome.FAILED
    assert failed.interpretation.event_label is QwenEventLabel.UNKNOWN
    assert failure_queue.diagnostics.failed_count == 1


def test_response_rejects_spatial_claims_and_extra_coordinate_fields() -> None:
    payload = json.loads(_valid_response(InteractionEventKind.PICKUP))
    payload["world_xyz_m"] = [1.0, 2.0, 3.0]

    with pytest.raises(QwenResponseValidationError, match="required semantic keys"):
        parse_qwen_event_response(
            json.dumps(payload),
            expected_event=InteractionEventKind.PICKUP,
        )


def test_v2_accepts_one_complete_json_fence_and_records_normalization() -> None:
    job = _jobs()[1]
    queue = _queue()
    queue.submit(job)

    class FencedProcessor:
        async def process(self, queued_job: QwenEventJob) -> QwenEventProcessingOutput:
            return _output(f"```json\n{_valid_response(queued_job.event_kind)}\n```")

    clock = _clock(2.0, 3.0)
    result = asyncio.run(
        process_next_qwen_event(
            queue,
            FencedProcessor(),
            clock=lambda: next(clock),
        )
    )
    assert result is not None
    assert result.outcome is QwenEventResultOutcome.COMPLETED
    assert result.interpretation.event_label is QwenEventLabel.CARRY
    assert result.response_normalization is QwenResponseNormalization.JSON_CODE_FENCE


def test_v1_artifact_contract_and_parser_remain_supported() -> None:
    policy = QwenEventReviewPolicy(
        policy_id="s05_qwen_event_review_v1",
        max_new_tokens=96,
    )
    assert policy.max_new_tokens == 96
    response = json.dumps(
        {
            "event_label": "pickup",
            "matches_candidate": True,
            "evidence_strength": "strong",
            "summary": "Visible pickup action.",
            "visible_evidence": ["The backpack leaves the bed."],
            "uncertainty": None,
            "spatial_claims_present": False,
        }
    )
    interpretation = parse_qwen_event_response(
        response,
        expected_event=InteractionEventKind.PICKUP,
        prompt_id="s05_qwen_event_json_v1",
    )
    assert interpretation.matches_candidate


def test_job_identity_is_tamper_evident() -> None:
    payload = _jobs()[0].model_dump(mode="json")
    payload["priority"] = 9
    with pytest.raises(ValidationError, match="job ID differs"):
        QwenEventJob.model_validate(payload)
