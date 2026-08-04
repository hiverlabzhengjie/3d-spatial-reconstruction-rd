"""Bounded S03 perception worker contracts and runtime."""

from spatial_reconstruction.perception.timeline import (
    CandidateMaskMetrics,
    ImagePlaneVisibility,
    PerceptionPresenceState,
    PerceptionTargetFrameState,
    build_target_frame_states,
)
from spatial_reconstruction.perception.visibility import (
    BackpackVisibilityPolicy,
    BackpackVisibilityRecord,
    BackpackVisibilityRunSummary,
    BackpackVisibilityState,
    VisibilityEvidenceSource,
    VisibilityReviewInterval,
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
    "BackpackVisibilityPolicy",
    "BackpackVisibilityRecord",
    "BackpackVisibilityRunSummary",
    "BackpackVisibilityState",
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
    "VisibilityEvidenceSource",
    "VisibilityReviewInterval",
    "YOLOByteTrackProcessor",
    "build_target_frame_states",
    "process_next_perception_item",
]
