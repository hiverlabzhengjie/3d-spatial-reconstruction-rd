"""Derive explicit per-target S03 timelines from a retained bounded replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, cast

import numpy as np

from spatial_reconstruction.contracts import PerceptionTarget
from spatial_reconstruction.perception import (
    PerceptionFrameResult,
    PerceptionPresenceState,
    PerceptionResultOutcome,
    PerceptionTargetFrameState,
    build_target_frame_states,
)
from spatial_reconstruction.perception.timeline import UInt8Array

CAMERA_IDS = ("camera_a", "camera_b")
CANDIDATE_STATES = {
    PerceptionPresenceState.OBSERVED,
    PerceptionPresenceState.UNTRACKED,
    PerceptionPresenceState.AMBIGUOUS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--bounded-replay-summary",
        type=Path,
        default=Path("artifacts/s03/bounded_replay_5fps_20260731/summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    replay_summary_path = _resolve_project_path(
        project_root, args.bounded_replay_summary
    )
    output_dir = _resolve_project_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    replay_summary = _read_object(replay_summary_path)
    _validate_replay_summary(replay_summary)

    camera_states: dict[str, tuple[PerceptionTargetFrameState, ...]] = {}
    camera_summaries: dict[str, Any] = {}
    for camera_id in CAMERA_IDS:
        camera_record = replay_summary["camera_summaries"][camera_id]
        result_path = _resolve_project_path(project_root, Path(camera_record["worker_results"]))
        results = tuple(
            PerceptionFrameResult.model_validate(item) for item in _read_list(result_path)
        )
        states: list[PerceptionTargetFrameState] = []
        for result in results:
            masks = _load_source_masks(project_root, result)
            states.extend(build_target_frame_states(result, masks))
        state_tuple = tuple(states)
        _validate_camera_states(camera_id, results, state_tuple)
        camera_states[camera_id] = state_tuple

        timeline_path = output_dir / f"{camera_id}_target_timeline.json"
        timeline_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stage": "S03",
                    "camera_id": camera_id,
                    "source_worker_results": str(result_path.relative_to(project_root)),
                    "records": [state.model_dump(mode="json") for state in state_tuple],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        camera_summaries[camera_id] = {
            "source_worker_result_count": len(results),
            "target_state_record_count": len(state_tuple),
            "timeline_ref": str(timeline_path.relative_to(project_root)),
            "targets": {
                target.value: _summarize_target(
                    tuple(state for state in state_tuple if state.target is target)
                )
                for target in PerceptionTarget
            },
        }

    cross_camera = _summarize_cross_camera(camera_states)
    summary = {
        "schema_version": 1,
        "stage": "S03",
        "purpose": "explicit_per_target_image_plane_timeline",
        "derivation_only": True,
        "inference_rerun": False,
        "occlusion_inference": False,
        "visibility_assessment_policy": {
            "detector_missing_implies_occlusion": False,
            "explicit_synchronized_video_evidence_required": True,
            "visibility_is_independent_of_detector_presence": True,
            "visibility_evidence_may_supply_xyz": False,
        },
        "missing_state_meaning": (
            "no selected candidate in this camera frame; occlusion is not inferred"
        ),
        "source": {
            "bounded_replay_summary_ref": str(replay_summary_path.relative_to(project_root)),
            "bounded_replay_summary_sha256": _sha256(replay_summary_path),
            "capture_session_id": replay_summary["capture_session_id"],
            "pose_version_id": replay_summary["pose_version_id"],
            "sampling": replay_summary["sampling"],
            "model": replay_summary["model"],
            "policy": replay_summary["policy"],
        },
        "camera_summaries": camera_summaries,
        "cross_camera": cross_camera,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _load_source_masks(
    project_root: Path,
    result: PerceptionFrameResult,
) -> UInt8Array | None:
    if result.outcome is PerceptionResultOutcome.FAILED:
        return None
    npz_refs = [ref for ref in result.raw_artifact_refs if ref.endswith("_raw.npz")]
    if len(npz_refs) != 1:
        raise ValueError("completed result must identify exactly one retained raw NPZ")
    artifact_path = _resolve_project_path(project_root, Path(npz_refs[0]))
    with np.load(artifact_path, allow_pickle=False) as arrays:
        if "source_sized_masks" not in arrays.files:
            raise ValueError(f"raw artifact lacks source_sized_masks: {artifact_path}")
        return cast(UInt8Array, np.asarray(arrays["source_sized_masks"]).copy())


def _validate_camera_states(
    camera_id: str,
    results: tuple[PerceptionFrameResult, ...],
    states: tuple[PerceptionTargetFrameState, ...],
) -> None:
    if len(states) != len(results) * len(PerceptionTarget):
        raise ValueError("target-state count does not match worker result count")
    expected_job_ids = [
        result.job.job_id for result in results for _target in PerceptionTarget
    ]
    if [state.job_id for state in states] != expected_job_ids:
        raise ValueError("target-state job order differs from worker capture order")
    if any(state.frame_identity.camera_id != camera_id for state in states):
        raise ValueError("camera timeline contains a state from another camera")
    frame_indices = [result.job.frame_identity.source_frame_index for result in results]
    if frame_indices != sorted(frame_indices) or len(frame_indices) != len(set(frame_indices)):
        raise ValueError("camera worker results are not unique capture-ordered frames")


def _summarize_target(
    states: tuple[PerceptionTargetFrameState, ...],
) -> dict[str, Any]:
    if not states:
        raise ValueError("target timeline must not be empty")
    state_counts = Counter(state.state.value for state in states)
    metrics = [metric for state in states for metric in state.candidate_metrics]
    track_ids = Counter(
        metric.candidate.source_detection.camera_local_track_id
        for metric in metrics
        if metric.candidate.source_detection.camera_local_track_id is not None
    )
    vendor_labels = Counter(
        metric.candidate.source_detection.class_name for metric in metrics
    )
    visibility = Counter(metric.visibility.value for metric in metrics)
    mask_areas = [metric.mask_area_pixels for metric in metrics]
    mask_fractions = [metric.mask_area_fraction for metric in metrics]
    return {
        "frame_count": len(states),
        "state_counts": {
            state.value: state_counts.get(state.value, 0)
            for state in PerceptionPresenceState
        },
        "candidate_count": len(metrics),
        "camera_local_track_observation_counts": dict(sorted(track_ids.items())),
        "vendor_label_counts": dict(sorted(vendor_labels.items())),
        "visibility_counts": {
            value: visibility.get(value, 0)
            for value in ("fully_in_frame", "frame_edge_truncated")
        },
        "mask_area_pixels": _numeric_summary(mask_areas),
        "mask_area_fraction": _numeric_summary(mask_fractions),
        "state_intervals": _state_intervals(states),
    }


def _state_intervals(
    states: tuple[PerceptionTargetFrameState, ...],
) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    start = 0
    for index in range(1, len(states) + 1):
        if index < len(states) and states[index].state is states[start].state:
            continue
        first = states[start]
        last = states[index - 1]
        intervals.append(
            {
                "state": first.state.value,
                "start_source_frame_index": first.frame_identity.source_frame_index,
                "end_source_frame_index": last.frame_identity.source_frame_index,
                "start_capture_timestamp_seconds": (
                    first.frame_identity.capture_timestamp_seconds
                ),
                "end_capture_timestamp_seconds": last.frame_identity.capture_timestamp_seconds,
                "processed_frame_count": index - start,
            }
        )
        start = index
    return intervals


def _summarize_cross_camera(
    camera_states: dict[str, tuple[PerceptionTargetFrameState, ...]],
) -> dict[str, Any]:
    by_camera_target = {
        camera_id: {
            target: tuple(state for state in states if state.target is target)
            for target in PerceptionTarget
        }
        for camera_id, states in camera_states.items()
    }
    summary: dict[str, Any] = {}
    for target in PerceptionTarget:
        camera_a = by_camera_target["camera_a"][target]
        camera_b = by_camera_target["camera_b"][target]
        indices_a = [state.frame_identity.source_frame_index for state in camera_a]
        indices_b = [state.frame_identity.source_frame_index for state in camera_b]
        if indices_a != indices_b:
            raise ValueError("camera timelines do not share the same processed frame indices")
        pairs = tuple(zip(camera_a, camera_b, strict=True))
        summary[target.value] = {
            "synchronized_processed_frame_count": len(pairs),
            "at_least_one_camera_observed_count": sum(
                PerceptionPresenceState.OBSERVED in {left.state, right.state}
                for left, right in pairs
            ),
            "at_least_one_camera_candidate_count": sum(
                left.state in CANDIDATE_STATES or right.state in CANDIDATE_STATES
                for left, right in pairs
            ),
            "both_cameras_missing_count": sum(
                left.state is PerceptionPresenceState.MISSING
                and right.state is PerceptionPresenceState.MISSING
                for left, right in pairs
            ),
            "both_cameras_failed_count": sum(
                left.state is PerceptionPresenceState.FAILED
                and right.state is PerceptionPresenceState.FAILED
                for left, right in pairs
            ),
        }
    return summary


def _numeric_summary(values: list[int] | list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "minimum": float(min(values)),
        "median": float(median(values)),
        "maximum": float(max(values)),
    }


def _validate_replay_summary(summary: dict[str, Any]) -> None:
    if summary.get("stage") != "S03" or summary.get("purpose") != (
        "bounded_d028_perception_replay"
    ):
        raise ValueError("input is not the accepted S03 bounded replay summary")
    if float(summary["sampling"]["nominal_fps_per_camera"]) != 5.0:
        raise ValueError("timeline derivation requires the accepted nominal 5 FPS replay")
    if set(summary["camera_summaries"]) != set(CAMERA_IDS):
        raise ValueError("bounded replay must contain exactly both accepted cameras")


def _resolve_project_path(project_root: Path, path: Path) -> Path:
    resolved = (project_root / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(project_root):
        raise ValueError(f"path escapes project root: {path}")
    return resolved


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_list(path: Path) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
