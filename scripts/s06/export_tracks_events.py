"""Export dedicated S06 track states, trajectory segments, and events."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spatial_reconstruction.interaction import (
    QwenEventJobPlanRunSummary,
    SemanticInteractionRunSummary,
)
from spatial_reconstruction.localization import (
    MeasuredTrajectorySegment,
    TemporalPresentationRunSummary,
)
from spatial_reconstruction.orchestration import (
    ArtifactRole,
    Stage06EventExportRecord,
    Stage06OrchestrationManifest,
    build_event_markers,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestration-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    orchestration_summary_path = args.orchestration_summary.resolve()
    orchestration_summary = _load_json(orchestration_summary_path)
    manifest_path = PROJECT_ROOT / str(orchestration_summary["manifest_ref"])
    if _sha256(manifest_path) != orchestration_summary["manifest_sha256"]:
        raise ValueError("orchestration manifest hash differs from summary")
    manifest = Stage06OrchestrationManifest.model_validate(_load_json(manifest_path))
    _verify_manifest_sources(manifest)
    artifact_refs = {artifact.role: artifact.source_ref for artifact in manifest.artifacts}

    temporal_path = PROJECT_ROOT / artifact_refs[ArtifactRole.TEMPORAL_PRESENTATION]
    semantic_path = PROJECT_ROOT / artifact_refs[ArtifactRole.INTERACTION_TIMELINE]
    qwen_plan_path = PROJECT_ROOT / artifact_refs[ArtifactRole.QWEN_EVENT_PLAN]
    qwen_execution_path = PROJECT_ROOT / artifact_refs[ArtifactRole.QWEN_EVENT_RESULTS]
    temporal = TemporalPresentationRunSummary.model_validate(_load_json(temporal_path))
    semantic = SemanticInteractionRunSummary.model_validate(_load_json(semantic_path))
    qwen_plan = QwenEventJobPlanRunSummary.model_validate(_load_json(qwen_plan_path))
    qwen_execution = _load_json(qwen_execution_path)
    final_results_path = PROJECT_ROOT / str(qwen_execution["final_results_ref"])
    if _sha256(final_results_path) != qwen_execution["final_results_sha256"]:
        raise ValueError("accepted Qwen final-results hash differs")
    final_results_container = _load_json(final_results_path)
    final_results = final_results_container.get("results")
    if not isinstance(final_results, list):
        raise ValueError("accepted Qwen final results are missing")

    segments_path = PROJECT_ROOT / temporal.trajectory_segments_ref
    if _sha256(segments_path) != temporal.trajectory_segments_sha256:
        raise ValueError("accepted trajectory-segment hash differs")
    segment_container = _load_json(segments_path)
    segments = tuple(
        MeasuredTrajectorySegment.model_validate(item) for item in segment_container["segments"]
    )
    if segments != temporal.measured_trajectory_segments:
        raise ValueError("trajectory segment file differs from temporal summary")

    jobs = [job.model_dump(mode="json") for job in qwen_plan.jobs]
    markers = build_event_markers(jobs, final_results)
    candidates_by_kind = {
        candidate.event_kind.value: candidate for candidate in semantic.event_candidates
    }
    results_by_kind = {str(result["job"]["event_kind"]): result for result in final_results}
    events = tuple(
        Stage06EventExportRecord.create(
            candidate=candidates_by_kind[marker.event_kind],
            marker=marker,
            qwen_job_id=str(results_by_kind[marker.event_kind]["job"]["job_id"]),
            qwen_outcome=str(results_by_kind[marker.event_kind]["outcome"]),
        )
        for marker in markers
    )

    output_dir.mkdir(parents=True)
    tracks_path = output_dir / "track_states.jsonl"
    segments_export_path = output_dir / "trajectory_segments.jsonl"
    events_path = output_dir / "events.jsonl"
    _write_jsonl(
        tracks_path,
        [record.model_dump(mode="json") for record in temporal.presentation_records],
    )
    _write_jsonl(
        segments_export_path,
        [segment.model_dump(mode="json") for segment in segments],
    )
    _write_jsonl(
        events_path,
        [event.model_dump(mode="json") for event in events],
    )
    state_counts = Counter(
        f"{record.target.value}:{record.state.value}" for record in temporal.presentation_records
    )
    summary = {
        "schema_version": 1,
        "stage": "S06",
        "work_package": 5,
        "status": "completed",
        "purpose": "dedicated_track_trajectory_event_exports",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_orchestration_summary_ref": _relative(orchestration_summary_path),
        "source_orchestration_summary_sha256": _sha256(orchestration_summary_path),
        "source_manifest_id": manifest.manifest_id,
        "source_temporal_summary_ref": _relative(temporal_path),
        "source_temporal_summary_sha256": _sha256(temporal_path),
        "source_semantic_summary_ref": _relative(semantic_path),
        "source_semantic_summary_sha256": _sha256(semantic_path),
        "source_qwen_plan_ref": _relative(qwen_plan_path),
        "source_qwen_plan_sha256": _sha256(qwen_plan_path),
        "source_qwen_execution_ref": _relative(qwen_execution_path),
        "source_qwen_execution_sha256": _sha256(qwen_execution_path),
        "track_states_ref": _relative(tracks_path),
        "track_states_sha256": _sha256(tracks_path),
        "track_state_count": len(temporal.presentation_records),
        "track_state_counts": dict(sorted(state_counts.items())),
        "trajectory_segments_ref": _relative(segments_export_path),
        "trajectory_segments_sha256": _sha256(segments_export_path),
        "trajectory_segment_count": len(segments),
        "events_ref": _relative(events_path),
        "events_sha256": _sha256(events_path),
        "event_count": len(events),
        "event_kinds": [event.event_kind for event in events],
        "carry_transition_frame_index": events[1].transition_frame_index,
        "carry_review_frame_index": events[1].review_frame_index,
        "invented_xyz_count": 0,
        "interpolated_segment_count": sum(segment.interpolation_performed for segment in segments),
        "stale_segment_count": sum(segment.stale_points_used for segment in segments),
        "qwen_spatial_write_count": sum(event.qwen_changed_spatial_facts for event in events),
        "limitations": [
            "Track states preserve measured, stale, occluded, and missing semantics.",
            "Trajectory segments remain disconnected and contain measured endpoints only.",
            "Qwen event review remains qualitative and cannot change spatial facts.",
        ],
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _verify_manifest_sources(manifest: Stage06OrchestrationManifest) -> None:
    for video in manifest.source_videos:
        if _sha256(PROJECT_ROOT / video.source_ref) != video.source_sha256:
            raise ValueError(f"source video hash differs: {video.source_ref}")
    for artifact in manifest.artifacts:
        if _sha256(PROJECT_ROOT / artifact.source_ref) != artifact.source_sha256:
            raise ValueError(f"accepted artifact hash differs: {artifact.source_ref}")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
