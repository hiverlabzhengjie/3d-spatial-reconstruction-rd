"""Independently verify the integrated S06 Rerun recording and source semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, cast

from PIL import Image

from spatial_reconstruction.orchestration import (
    ArtifactRole,
    Stage06OrchestrationManifest,
    build_event_markers,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_RRD_PATHS = (
    "/cameras/camera_a/video",
    "/cameras/camera_b/video",
    "/world/static_scene",
    "/world/cameras/camera_a",
    "/world/cameras/camera_b",
    "/world/zones/pickup_blue_bed",
    "/world/zones/dropoff_white_floor",
    "/world/dynamic/person/current",
    "/world/dynamic/backpack/current",
    "/world/trajectories/person",
    "/world/trajectories/backpack",
    "/timeline/interaction/phase_code",
    "/timeline/localization/person/state_code",
    "/timeline/localization/backpack/state_code",
    "/events/transitions/carry",
    "/events/qwen_reviews/carry",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-qa-dir", type=Path, required=True)
    parser.add_argument("--visual-qa-passed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_summary_path = args.export_summary.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite verification: {output_path}")
    if not args.visual_qa_passed:
        raise ValueError("integrated Rerun verification requires explicit visual QA")
    export = _load_json(export_summary_path)
    recording_path = PROJECT_ROOT / str(export["recording_ref"])
    if _sha256(recording_path) != export["recording_sha256"]:
        raise ValueError("Rerun recording hash differs from export summary")
    if recording_path.stat().st_size != export["recording_bytes"]:
        raise ValueError("Rerun recording size differs from export summary")

    orchestration_summary_path = PROJECT_ROOT / str(export["source_orchestration_summary_ref"])
    if _sha256(orchestration_summary_path) != export["source_orchestration_summary_sha256"]:
        raise ValueError("orchestration summary hash differs from export summary")
    orchestration_summary = _load_json(orchestration_summary_path)
    manifest_path = PROJECT_ROOT / str(orchestration_summary["manifest_ref"])
    manifest = Stage06OrchestrationManifest.model_validate(_load_json(manifest_path))
    if manifest.manifest_id != export["manifest_id"]:
        raise ValueError("export manifest identity differs")
    artifacts = {artifact.role: artifact for artifact in manifest.artifacts}
    for artifact in artifacts.values():
        if _sha256(PROJECT_ROOT / artifact.source_ref) != artifact.source_sha256:
            raise ValueError(f"source artifact hash differs: {artifact.source_ref}")

    temporal = _load_json(PROJECT_ROOT / artifacts[ArtifactRole.TEMPORAL_PRESENTATION].source_ref)
    interaction = _load_json(
        PROJECT_ROOT / artifacts[ArtifactRole.INTERACTION_TIMELINE].source_ref
    )
    perception = _load_json(PROJECT_ROOT / artifacts[ArtifactRole.PERCEPTION_TIMELINE].source_ref)
    qwen_plan = _load_json(PROJECT_ROOT / artifacts[ArtifactRole.QWEN_EVENT_PLAN].source_ref)
    qwen_execution = _load_json(
        PROJECT_ROOT / artifacts[ArtifactRole.QWEN_EVENT_RESULTS].source_ref
    )
    results_document = _load_json(PROJECT_ROOT / str(qwen_execution["final_results_ref"]))
    results = cast(list[dict[str, Any]], results_document["results"])
    markers = build_event_markers(qwen_plan["jobs"], results)

    expected_state_counts = Counter(
        f"{record['target']}:{record['state']}" for record in temporal["presentation_records"]
    )
    if dict(expected_state_counts) != export["presentation_state_counts"]:
        raise ValueError("presentation state counts differ from accepted temporal source")
    if len(temporal["presentation_records"]) != export["presentation_record_count"]:
        raise ValueError("presentation record count differs")
    if any(
        record["state"] in {"missing", "occluded"}
        and (
            record["raw_world_xyz_m"] is not None or record["presentation_world_xyz_m"] is not None
        )
        for record in temporal["presentation_records"]
    ):
        raise ValueError("missing/occluded source state contains XYZ")
    if any(
        record["state"] == "stale" and record["raw_world_xyz_m"] is not None
        for record in temporal["presentation_records"]
    ):
        raise ValueError("stale source state contains raw XYZ")

    expected_segments = Counter(
        segment["target"] for segment in temporal["measured_trajectory_segments"]
    )
    if dict(expected_segments) != export["measured_segment_counts"]:
        raise ValueError("measured segment counts differ")
    if any(
        segment["interpolation_performed"] or segment["stale_points_used"]
        for segment in temporal["measured_trajectory_segments"]
    ):
        raise ValueError("measured segment source uses interpolation or stale points")
    if len(interaction["records"]) != export["interaction_record_count"]:
        raise ValueError("interaction record count differs")

    expected_box_count = 0
    expected_mask_frames = 0
    for camera_id in ("camera_a", "camera_b"):
        timeline_ref = perception["camera_summaries"][camera_id]["timeline_ref"]
        timeline = _load_json(PROJECT_ROOT / str(timeline_ref))
        frames_with_candidates: set[int] = set()
        for record in timeline["records"]:
            candidate_count = len(record["candidate_metrics"])
            expected_box_count += candidate_count
            if candidate_count:
                frames_with_candidates.add(int(record["frame_identity"]["source_frame_index"]))
        expected_mask_frames += len(frames_with_candidates)
    if expected_box_count != export["perception_box_count"]:
        raise ValueError("perception box count differs")
    if expected_mask_frames != export["segmentation_frame_count"]:
        raise ValueError("segmentation frame count differs")

    expected_video_counts = {
        video.camera_id: video.decoded_frame_count for video in manifest.source_videos
    }
    if expected_video_counts != export["video_frame_reference_counts"]:
        raise ValueError("video frame-reference counts differ")
    if [marker.model_dump(mode="json") for marker in markers] != export["event_markers"]:
        raise ValueError("event markers differ from accepted jobs and results")
    carry = markers[1]
    if carry.transition_frame_index == carry.review_frame_index:
        raise ValueError("carry transition and review identities were not kept separate")

    rrd_print = _print_rrd(recording_path)
    missing_paths = tuple(path for path in REQUIRED_RRD_PATHS if path not in rrd_print)
    if missing_paths:
        raise ValueError(f"Rerun recording lacks required entity paths: {missing_paths}")
    if "capture_time" not in rrd_print:
        raise ValueError("Rerun recording lacks the capture_time timeline")
    visual_qa_dir = args.visual_qa_dir.resolve()
    visual_evidence = _verify_visual_evidence(visual_qa_dir)

    report = {
        "schema_version": 1,
        "stage": "S06",
        "status": "passed",
        "purpose": "integrated_file_backed_rerun_verification",
        "source_export_summary_ref": _relative(export_summary_path),
        "source_export_summary_sha256": _sha256(export_summary_path),
        "recording_ref": export["recording_ref"],
        "recording_sha256": export["recording_sha256"],
        "recording_bytes": export["recording_bytes"],
        "recording_parsed": True,
        "required_entity_path_count": len(REQUIRED_RRD_PATHS),
        "capture_time_timeline_present": True,
        "visual_qa_passed": True,
        "visual_evidence": visual_evidence,
        "video_frame_reference_counts": expected_video_counts,
        "perception_box_count": expected_box_count,
        "segmentation_frame_count": expected_mask_frames,
        "presentation_record_count": len(temporal["presentation_records"]),
        "presentation_state_counts": dict(expected_state_counts),
        "measured_segment_counts": dict(expected_segments),
        "missing_or_occluded_xyz_count": 0,
        "stale_raw_xyz_count": 0,
        "interpolated_or_stale_segment_count": 0,
        "interaction_record_count": len(interaction["records"]),
        "event_kinds": [marker.event_kind for marker in markers],
        "carry_transition_frame": carry.transition_frame_index,
        "carry_review_frame": carry.review_frame_index,
        "source_transition_times_preserved": True,
        "worker_completion_order_used": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _print_rrd(path: Path) -> str:
    rerun_cli = Path(sys.executable).with_name("rerun")
    result = subprocess.run(  # noqa: S603 - fixed local CLI and explicit argv
        [str(rerun_cli), "rrd", "print", "-vv", str(path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30.0,
    )
    return result.stdout


def _verify_visual_evidence(directory: Path) -> dict[str, dict[str, Any]]:
    filenames = (
        "web_camera_a.png",
        "web_camera_b.png",
        "web_metric_twin_v2.png",
        "web_state_timeline.png",
        "web_events.png",
    )
    evidence: dict[str, dict[str, Any]] = {}
    for filename in filenames:
        path = directory / filename
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        if width < 640 or height < 360:
            raise ValueError(f"visual QA screenshot is unexpectedly small: {path}")
        evidence[filename] = {
            "ref": _relative(path),
            "sha256": _sha256(path),
            "width": width,
            "height": height,
        }
    return evidence


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
