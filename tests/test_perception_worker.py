from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torch import nn

from spatial_reconstruction.contracts import (
    FrameIdentity,
    FrameSourceKind,
    PerceptionCandidate,
    PerceptionTarget,
    PixelBox,
    SegmentationDetection,
    SourceFingerprintKind,
)
from spatial_reconstruction.models import YOLOSegAdapter
from spatial_reconstruction.perception import (
    BoundedPerceptionQueue,
    PerceptionJob,
    PerceptionProcessingOutput,
    PerceptionResultOutcome,
    PerceptionWorkItem,
    QueueOverflowPolicy,
    QueueSubmissionDisposition,
    YOLOByteTrackProcessor,
    process_next_perception_item,
)

MODEL_REVISION = "a" * 64


def make_identity(index: int, *, camera_id: str = "camera_a") -> FrameIdentity:
    timestamp = index / 5.0
    return FrameIdentity.create(
        capture_session_id="session_01",
        camera_id=camera_id,
        source_kind=FrameSourceKind.FILE,
        source_frame_index=index,
        source_timestamp_seconds=timestamp,
        capture_timestamp_seconds=timestamp,
        source_ref=f"captures/{camera_id}.mp4",
        source_fingerprint="b" * 64,
        source_fingerprint_kind=SourceFingerprintKind.CONTENT_SHA256,
        synchronization_manifest_ref="captures/sync.json",
        synchronization_manifest_sha256="c" * 64,
        pose_version_id="session_01:pose:v1",
        image_width=4,
        image_height=3,
    )


def make_item(index: int, *, created: float = 0.0) -> PerceptionWorkItem:
    job = PerceptionJob.create(
        frame_identity=make_identity(index),
        model_id="yolov8n-seg.pt",
        model_revision=MODEL_REVISION,
        policy_id="d028_guarded_backpack_handbag_v1",
        created_processing_seconds=created,
    )
    return PerceptionWorkItem(
        job=job,
        image_rgb=np.zeros((3, 4, 3), dtype=np.uint8),
    )


def test_perception_job_identity_is_stable_and_tamper_evident() -> None:
    first = make_item(2, created=1.0).job
    replay = make_item(2, created=9.0).job

    assert first.job_id == replay.job_id
    assert PerceptionJob.model_validate_json(first.model_dump_json()) == first

    payload = first.model_dump()
    payload["priority"] = 4
    with pytest.raises(ValidationError, match="job_id does not match"):
        PerceptionJob.model_validate(payload)


def test_work_item_copies_pixels_and_makes_them_read_only() -> None:
    source = np.zeros((3, 4, 3), dtype=np.uint8)
    item = PerceptionWorkItem(job=make_item(0).job, image_rgb=source)

    source[0, 0] = 255
    assert np.all(item.image_rgb[0, 0] == 0)
    with pytest.raises(ValueError):
        item.image_rgb[0, 0] = 1


def test_offline_queue_throttles_without_dropping_and_preserves_fifo() -> None:
    queue = BoundedPerceptionQueue(
        capacity=2,
        overflow_policy=QueueOverflowPolicy.THROTTLE,
    )
    first, second, third = (make_item(index) for index in range(3))

    assert queue.submit(first).disposition is QueueSubmissionDisposition.ACCEPTED
    assert queue.submit(second).disposition is QueueSubmissionDisposition.ACCEPTED
    throttled = queue.submit(third)

    assert throttled.disposition is QueueSubmissionDisposition.THROTTLE_REQUIRED
    assert not throttled.accepted
    assert throttled.dropped_job_id is None
    popped_first = queue.pop()
    assert popped_first is not None and popped_first.job == first.job
    assert queue.submit(third).accepted
    popped_second = queue.pop()
    popped_third = queue.pop()
    assert popped_second is not None and popped_second.job == second.job
    assert popped_third is not None and popped_third.job == third.job
    diagnostics = queue.diagnostics
    assert diagnostics.throttled_count == 1
    assert diagnostics.dropped_oldest_count == 0
    assert diagnostics.current_depth == 0
    assert diagnostics.in_flight_count == 3


def test_live_queue_drops_oldest_with_explicit_identity() -> None:
    queue = BoundedPerceptionQueue(
        capacity=2,
        overflow_policy=QueueOverflowPolicy.DROP_OLDEST,
    )
    first, second, third = (make_item(index) for index in range(3))
    queue.submit(first)
    queue.submit(second)

    submission = queue.submit(third)

    assert submission.disposition is QueueSubmissionDisposition.ACCEPTED_AFTER_DROP_OLDEST
    assert submission.dropped_job_id == first.job.job_id
    popped_second = queue.pop()
    popped_third = queue.pop()
    assert popped_second is not None and popped_second.job == second.job
    assert popped_third is not None and popped_third.job == third.job
    assert queue.diagnostics.dropped_oldest_count == 1


def test_queue_rejects_duplicate_mixed_camera_and_non_monotonic_work() -> None:
    queue = BoundedPerceptionQueue(
        capacity=3,
        overflow_policy=QueueOverflowPolicy.THROTTLE,
    )
    first = make_item(1)
    queue.submit(first)

    with pytest.raises(ValueError, match="duplicate"):
        queue.submit(first)
    mixed_job = PerceptionJob.create(
        frame_identity=make_identity(2, camera_id="camera_b"),
        model_id="yolov8n-seg.pt",
        model_revision=MODEL_REVISION,
        policy_id="d028_guarded_backpack_handbag_v1",
        created_processing_seconds=0.0,
    )
    with pytest.raises(ValueError, match="cannot mix camera"):
        queue.submit(
            PerceptionWorkItem(
                job=mixed_job,
                image_rgb=np.zeros((3, 4, 3), dtype=np.uint8),
            )
        )
    with pytest.raises(ValueError, match="increasing source frame"):
        queue.submit(make_item(0))


class EmptyProcessor:
    def process(self, item: PerceptionWorkItem) -> PerceptionProcessingOutput:
        assert item.job.frame_identity.camera_id == "camera_a"
        return PerceptionProcessingOutput(
            candidates=(),
            raw_artifact_refs=("artifacts/s03/frame_0000.npz",),
        )


class FailingProcessor:
    def process(self, item: PerceptionWorkItem) -> PerceptionProcessingOutput:
        raise RuntimeError(f"model failed for {item.job.frame_identity.frame_id}")


class WrongFrameProcessor:
    def process(self, item: PerceptionWorkItem) -> PerceptionProcessingOutput:
        detection = SegmentationDetection(
            frame=make_identity(item.job.frame_identity.source_frame_index + 1).as_frame_ref(),
            class_id=0,
            class_name="person",
            confidence=0.9,
            box=PixelBox(x_min=0.0, y_min=0.0, x_max=2.0, y_max=2.0),
            mask_ref="artifacts/s03/wrong_mask.npz#mask_0000",
            camera_local_track_id="camera_a:1",
        )
        return PerceptionProcessingOutput(
            candidates=(
                PerceptionCandidate(
                    detection_index=0,
                    target=PerceptionTarget.PERSON,
                    source_detection=detection,
                    policy_id="d028_guarded_backpack_handbag_v1",
                ),
            )
        )


def clock_values(*values: float) -> Iterator[float]:
    yield from values


def test_worker_preserves_explicit_empty_success_and_failure_outcomes() -> None:
    success_queue = BoundedPerceptionQueue(
        capacity=1,
        overflow_policy=QueueOverflowPolicy.THROTTLE,
    )
    success_queue.submit(make_item(0, created=1.0))
    success_clock = clock_values(2.0, 2.5)

    success = process_next_perception_item(
        success_queue,
        EmptyProcessor(),
        clock=lambda: next(success_clock),
    )

    assert success is not None
    assert success.outcome is PerceptionResultOutcome.COMPLETED
    assert success.candidates == ()
    assert success.raw_artifact_refs == ("artifacts/s03/frame_0000.npz",)
    assert success.job.frame_identity.frame_id == make_identity(0).frame_id
    assert success_queue.diagnostics.completed_count == 1
    assert success_queue.diagnostics.in_flight_count == 0

    failure_queue = BoundedPerceptionQueue(
        capacity=1,
        overflow_policy=QueueOverflowPolicy.THROTTLE,
    )
    failure_queue.submit(make_item(1, created=3.0))
    failure_clock = clock_values(4.0, 4.2)
    failure = process_next_perception_item(
        failure_queue,
        FailingProcessor(),
        clock=lambda: next(failure_clock),
    )

    assert failure is not None
    assert failure.outcome is PerceptionResultOutcome.FAILED
    assert failure.candidates == ()
    assert failure.raw_artifact_refs == ()
    assert failure.error_type == "RuntimeError"
    assert "model failed" in str(failure.error_message)
    assert failure_queue.diagnostics.failed_count == 1


def test_pending_work_is_cancelled_and_accounted_without_results() -> None:
    queue = BoundedPerceptionQueue(
        capacity=2,
        overflow_policy=QueueOverflowPolicy.THROTTLE,
    )
    queue.submit(make_item(0))
    queue.submit(make_item(1))

    cancelled = queue.cancel_pending()

    assert [job.frame_identity.source_frame_index for job in cancelled] == [0, 1]
    assert queue.pop() is None
    assert queue.diagnostics.cancelled_count == 2
    assert queue.diagnostics.current_depth == 0


def test_worker_rejects_candidate_from_the_wrong_source_frame() -> None:
    queue = BoundedPerceptionQueue(
        capacity=1,
        overflow_policy=QueueOverflowPolicy.THROTTLE,
    )
    queue.submit(make_item(0, created=1.0))
    clock = clock_values(2.0, 2.1)

    result = process_next_perception_item(
        queue,
        WrongFrameProcessor(),
        clock=lambda: next(clock),
    )

    assert result is not None
    assert result.outcome is PerceptionResultOutcome.FAILED
    assert result.candidates == ()
    assert result.error_type == "ValueError"
    assert "does not match perception job frame" in str(result.error_message)


class FakeTrackedResult:
    orig_shape = (3, 4)
    names = {26: "handbag"}
    boxes = SimpleNamespace(
        xyxy=torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
        cls=torch.tensor([26.0]),
        conf=torch.tensor([0.8]),
        id=torch.tensor([3.0]),
    )
    masks = SimpleNamespace(data=torch.ones((1, 3, 4)))
    speed = {"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5}

    def plot(self) -> np.ndarray:
        return np.zeros((3, 4, 3), dtype=np.uint8)


class FakeTrackingModel:
    task = "segment"
    ckpt_path = "yolov8n-seg.pt"

    def __init__(self) -> None:
        self.model = nn.Linear(2, 2)

    def predict(self, **kwargs: object) -> list[FakeTrackedResult]:
        return [FakeTrackedResult()]

    def track(self, **kwargs: object) -> list[FakeTrackedResult]:
        return [FakeTrackedResult()]


def test_real_yolo_processor_persists_raw_outputs_and_d028_candidate(
    tmp_path: Path,
) -> None:
    weight_path = tmp_path / "yolov8n-seg.pt"
    weight_path.write_bytes(b"weights")
    adapter = YOLOSegAdapter(model=FakeTrackingModel(), weight_path=weight_path)
    item = make_item(0, created=1.0)
    item = PerceptionWorkItem(
        job=PerceptionJob.create(
            frame_identity=item.job.frame_identity,
            model_id="yolov8n-seg.pt",
            model_revision=adapter.weight_sha256,
            policy_id="d028_guarded_backpack_handbag_v1",
            created_processing_seconds=1.0,
        ),
        image_rgb=item.image_rgb,
    )
    processor = YOLOByteTrackProcessor(
        adapter=adapter,
        project_root=tmp_path,
        output_dir=tmp_path / "artifacts/s03/raw",
        device="mps",
        image_size=640,
        confidence_threshold=0.25,
        tracked_class_ids=(0, 24, 26),
        bag_class_aliases=("backpack", "handbag"),
        excluded_bag_classes=("suitcase",),
        policy_id="d028_guarded_backpack_handbag_v1",
    )

    output = processor.process(item)

    assert len(output.candidates) == 1
    candidate = output.candidates[0]
    assert candidate.target is PerceptionTarget.BACKPACK
    assert candidate.source_detection.class_name == "handbag"
    assert candidate.source_detection.camera_local_track_id == "camera_a:3"
    assert all((tmp_path / ref).is_file() for ref in output.raw_artifact_refs)
    with np.load(tmp_path / output.raw_artifact_refs[0]) as arrays:
        assert arrays["raw_track_ids"].tolist() == [3]
