"""Independently regenerate and verify dedicated S06 track/event exports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from spatial_reconstruction.interaction import (
    QwenEventJobPlanRunSummary,
    SemanticInteractionRunSummary,
)
from spatial_reconstruction.localization import (
    MeasuredTrajectorySegment,
    TemporalPresentationRecord,
    TemporalPresentationRunSummary,
    TemporalPresentationState,
)
from spatial_reconstruction.orchestration import (
    Stage06EventExportRecord,
    build_event_markers,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ModelT = TypeVar("ModelT", bound=BaseModel)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite verification: {output_path}")
    summary_path = args.summary.resolve()
    summary = _load_json(summary_path)
    tracks_path = PROJECT_ROOT / str(summary["track_states_ref"])
    segments_path = PROJECT_ROOT / str(summary["trajectory_segments_ref"])
    events_path = PROJECT_ROOT / str(summary["events_ref"])
    _require_hash(tracks_path, str(summary["track_states_sha256"]))
    _require_hash(segments_path, str(summary["trajectory_segments_sha256"]))
    _require_hash(events_path, str(summary["events_sha256"]))

    tracks = _load_jsonl(tracks_path, TemporalPresentationRecord)
    segments = _load_jsonl(segments_path, MeasuredTrajectorySegment)
    events = _load_jsonl(events_path, Stage06EventExportRecord)
    temporal_path = PROJECT_ROOT / str(summary["source_temporal_summary_ref"])
    semantic_path = PROJECT_ROOT / str(summary["source_semantic_summary_ref"])
    qwen_plan_path = PROJECT_ROOT / str(summary["source_qwen_plan_ref"])
    qwen_execution_path = PROJECT_ROOT / str(summary["source_qwen_execution_ref"])
    _require_hash(temporal_path, str(summary["source_temporal_summary_sha256"]))
    _require_hash(semantic_path, str(summary["source_semantic_summary_sha256"]))
    _require_hash(qwen_plan_path, str(summary["source_qwen_plan_sha256"]))
    _require_hash(qwen_execution_path, str(summary["source_qwen_execution_sha256"]))
    temporal = TemporalPresentationRunSummary.model_validate(_load_json(temporal_path))
    semantic = SemanticInteractionRunSummary.model_validate(_load_json(semantic_path))
    qwen_plan = QwenEventJobPlanRunSummary.model_validate(_load_json(qwen_plan_path))
    qwen_execution = _load_json(qwen_execution_path)
    final_results_path = PROJECT_ROOT / str(qwen_execution["final_results_ref"])
    _require_hash(final_results_path, str(qwen_execution["final_results_sha256"]))
    final_results = _load_json(final_results_path)["results"]
    if not isinstance(final_results, list):
        raise ValueError("accepted Qwen final results are missing")

    if tracks != temporal.presentation_records:
        raise ValueError("S06 track export differs from accepted S04 records")
    if segments != temporal.measured_trajectory_segments:
        raise ValueError("S06 trajectory export differs from accepted S04 segments")
    markers = build_event_markers(
        [job.model_dump(mode="json") for job in qwen_plan.jobs],
        final_results,
    )
    candidates_by_kind = {
        candidate.event_kind.value: candidate for candidate in semantic.event_candidates
    }
    results_by_kind = {str(result["job"]["event_kind"]): result for result in final_results}
    regenerated_events = tuple(
        Stage06EventExportRecord.create(
            candidate=candidates_by_kind[marker.event_kind],
            marker=marker,
            qwen_job_id=str(results_by_kind[marker.event_kind]["job"]["job_id"]),
            qwen_outcome=str(results_by_kind[marker.event_kind]["outcome"]),
        )
        for marker in markers
    )
    if events != regenerated_events:
        raise ValueError("S06 event export does not regenerate from S05 evidence")

    non_measured_raw_xyz = sum(
        record.raw_world_xyz_m is not None
        for record in tracks
        if record.state is not TemporalPresentationState.MEASURED
    )
    unavailable_presentation_xyz = sum(
        record.presentation_world_xyz_m is not None
        for record in tracks
        if record.state in {TemporalPresentationState.MISSING, TemporalPresentationState.OCCLUDED}
    )
    interpolated_segments = sum(segment.interpolation_performed for segment in segments)
    stale_segments = sum(segment.stale_points_used for segment in segments)
    qwen_spatial_writes = sum(event.qwen_changed_spatial_facts for event in events)
    if any(
        (
            non_measured_raw_xyz,
            unavailable_presentation_xyz,
            interpolated_segments,
            stale_segments,
            qwen_spatial_writes,
        )
    ):
        raise ValueError("S06 exports violate missing/stale/spatial-authority rules")
    if tuple(event.event_kind for event in events) != ("pickup", "carry", "place"):
        raise ValueError("S06 event order differs from pickup-carry-place")

    verification = {
        "schema_version": 1,
        "stage": "S06",
        "work_package": 5,
        "status": "passed",
        "purpose": "dedicated_track_trajectory_event_export_verification",
        "source_summary_ref": _relative(summary_path),
        "source_summary_sha256": _sha256(summary_path),
        "source_manifest_id": summary["source_manifest_id"],
        "track_state_count": len(tracks),
        "trajectory_segment_count": len(segments),
        "event_count": len(events),
        "tracks_regenerated_exactly": True,
        "trajectory_segments_regenerated_exactly": True,
        "events_regenerated_exactly": True,
        "non_measured_raw_xyz_count": non_measured_raw_xyz,
        "missing_occluded_presentation_xyz_count": unavailable_presentation_xyz,
        "interpolated_segment_count": interpolated_segments,
        "stale_segment_count": stale_segments,
        "qwen_spatial_write_count": qwen_spatial_writes,
        "carry_transition_frame_index": events[1].transition_frame_index,
        "carry_review_frame_index": events[1].review_frame_index,
        "capture_time_authoritative": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


def _load_jsonl(path: Path, model_type: type[ModelT]) -> tuple[ModelT, ...]:
    return tuple(
        model_type.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def _require_hash(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"artifact hash differs for {path}: {actual} != {expected}")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


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
