"""Integrated S06 orchestration contracts and worker supervision."""

from spatial_reconstruction.orchestration.contracts import (
    ArtifactRole,
    OrchestrationArtifact,
    OrchestrationPolicy,
    SourceVideo,
    Stage06OrchestrationManifest,
    WorkerKind,
)
from spatial_reconstruction.orchestration.exports import Stage06EventExportRecord
from spatial_reconstruction.orchestration.replay import (
    AcceleratorInterval,
    IntegratedReplayReport,
    QueueReplayAction,
    QueueReplayDiagnostics,
    QueueReplayEvent,
    ReplayAttempt,
    ReplayJob,
    ReplayOutcome,
    ReplayResult,
    ShutdownReplayDiagnostics,
    run_integrated_replay,
    summarize_attempt_outcomes,
)
from spatial_reconstruction.orchestration.rerun_presentation import (
    RerunEventMarker,
    RerunPointStyle,
    build_event_markers,
    coordinate_log_text,
    coordinate_point_label,
    point_style,
)
from spatial_reconstruction.orchestration.supervisor import (
    ProcessOutcome,
    ProcessWorkerSpec,
    SupervisedAttemptResult,
    SupervisedWorkerRun,
    run_supervised_worker,
)

__all__ = [
    "ArtifactRole",
    "AcceleratorInterval",
    "IntegratedReplayReport",
    "OrchestrationArtifact",
    "OrchestrationPolicy",
    "ProcessOutcome",
    "ProcessWorkerSpec",
    "QueueReplayAction",
    "QueueReplayDiagnostics",
    "QueueReplayEvent",
    "RerunEventMarker",
    "RerunPointStyle",
    "ReplayAttempt",
    "ReplayJob",
    "ReplayOutcome",
    "ReplayResult",
    "SourceVideo",
    "Stage06OrchestrationManifest",
    "Stage06EventExportRecord",
    "ShutdownReplayDiagnostics",
    "SupervisedAttemptResult",
    "SupervisedWorkerRun",
    "WorkerKind",
    "build_event_markers",
    "coordinate_log_text",
    "coordinate_point_label",
    "point_style",
    "run_integrated_replay",
    "run_supervised_worker",
    "summarize_attempt_outcomes",
]
