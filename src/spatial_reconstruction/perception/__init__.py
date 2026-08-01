"""Bounded S03 perception worker contracts and runtime."""

from spatial_reconstruction.perception.timeline import (
    CandidateMaskMetrics,
    ImagePlaneVisibility,
    PerceptionPresenceState,
    PerceptionTargetFrameState,
    build_target_frame_states,
)
from spatial_reconstruction.perception.worker import (
    BoundedPerceptionQueue,
    PerceptionFrameResult,
    PerceptionJob,
    PerceptionProcessingOutput,
    PerceptionQueueDiagnostics,
    PerceptionResultOutcome,
    PerceptionWorkItem,
    QueueOverflowPolicy,
    QueueSubmission,
    QueueSubmissionDisposition,
    process_next_perception_item,
)
from spatial_reconstruction.perception.yolo_processor import YOLOByteTrackProcessor

__all__ = [
    "BoundedPerceptionQueue",
    "CandidateMaskMetrics",
    "ImagePlaneVisibility",
    "PerceptionFrameResult",
    "PerceptionJob",
    "PerceptionPresenceState",
    "PerceptionProcessingOutput",
    "PerceptionQueueDiagnostics",
    "PerceptionResultOutcome",
    "PerceptionTargetFrameState",
    "PerceptionWorkItem",
    "QueueOverflowPolicy",
    "QueueSubmission",
    "QueueSubmissionDisposition",
    "YOLOByteTrackProcessor",
    "build_target_frame_states",
    "process_next_perception_item",
]
