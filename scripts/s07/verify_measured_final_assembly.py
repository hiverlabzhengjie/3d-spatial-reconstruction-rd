"""Independently verify the measured S07 final Rerun assembly."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from spatial_reconstruction.finalization import Stage07FinalRunExecution

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
    "/world/measurements/person",
    "/world/measurements/backpack",
    "/coordinates/person",
    "/coordinates/backpack",
    "/timeline/interaction/phase_code",
    "/timeline/localization/person/state_code",
    "/timeline/localization/backpack/state_code",
    "/events/transitions/carry",
    "/events/qwen_reviews/carry",
)
STABLE_EXPORT_FIELDS = (
    "authoritative_timeline",
    "logged_static_point_count",
    "video_frame_reference_counts",
    "perception_box_count",
    "segmentation_frame_count",
    "presentation_record_count",
    "presentation_state_counts",
    "measured_segment_counts",
    "interaction_record_count",
    "event_markers",
    "source_transition_times_preserved",
    "qwen_review_times_logged_separately",
    "worker_completion_order_used",
    "model_inference_performed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--visual-qa-dir", type=Path, required=True)
    parser.add_argument("--visual-qa-passed", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.visual_qa_passed:
        raise ValueError("final Rerun verification requires explicit visual QA")
    summary_path = args.summary.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite verification: {output_path}")
    summary = _load_json(summary_path)
    execution = Stage07FinalRunExecution.model_validate(
        {field: summary[field] for field in Stage07FinalRunExecution.model_fields}
    )

    for step in execution.steps:
        if _sha256(PROJECT_ROOT / step.stdout_ref) != step.stdout_sha256:
            raise ValueError(f"measured stdout hash differs: {step.name}")
        if _sha256(PROJECT_ROOT / step.stderr_ref) != step.stderr_sha256:
            raise ValueError(f"measured stderr hash differs: {step.name}")
    recording_path = PROJECT_ROOT / execution.recording_ref
    export_summary_path = PROJECT_ROOT / execution.export_summary_ref
    if _sha256(recording_path) != execution.recording_sha256:
        raise ValueError("final Rerun hash differs from measured execution")
    if recording_path.stat().st_size != execution.recording_bytes:
        raise ValueError("final Rerun size differs from measured execution")
    if _sha256(export_summary_path) != execution.export_summary_sha256:
        raise ValueError("final export summary hash differs from measured execution")

    export = _load_json(export_summary_path)
    accepted_export = _load_json(
        PROJECT_ROOT
        / "artifacts/s06/integrated_rerun_20260805/"
        "digital_twin_stage06_v2_export_summary.json"
    )
    for field in STABLE_EXPORT_FIELDS:
        if export[field] != accepted_export[field]:
            raise ValueError(f"final export semantics differ from accepted S06: {field}")
    if export.get("presentation_video_mode") == "seekable_h264_proxy":
        if export.get("presentation_video_manifest_ref") is None:
            raise ValueError("seekable presentation mode lacks its proxy manifest")
        if export.get("trajectory_logging_mode") != "capture_time_progressive":
            raise ValueError("refined trajectories are not capture-time progressive")
        if export.get("full_trajectory_visible_at_start") is not False:
            raise ValueError("refined full trajectories remain visible at capture start")
        if export.get("coordinate_log_record_count") != export["presentation_record_count"]:
            raise ValueError("coordinate log does not cover every presentation record")
        if export.get("measured_observation_counts") != {"backpack": 17, "person": 16}:
            raise ValueError("measured observation trail counts differ")
    if export["recording_sha256"] != execution.recording_sha256:
        raise ValueError("final export summary recording hash differs")
    rrd_print = _print_rrd(recording_path)
    missing_paths = tuple(path for path in REQUIRED_RRD_PATHS if path not in rrd_print)
    if missing_paths:
        raise ValueError(f"final Rerun lacks required entity paths: {missing_paths}")
    if "capture_time" not in rrd_print:
        raise ValueError("final Rerun lacks the capture_time timeline")
    visual_evidence = _verify_visual_evidence(args.visual_qa_dir.resolve())

    report = {
        "schema_version": 1,
        "stage": "S07",
        "work_package": 2,
        "status": "passed",
        "purpose": "measured_final_rerun_assembly_verification",
        "source_summary_ref": _relative(summary_path),
        "source_summary_sha256": _sha256(summary_path),
        "source_final_run_manifest_id": execution.source_final_run_manifest_id,
        "recording_ref": execution.recording_ref,
        "recording_sha256": execution.recording_sha256,
        "recording_bytes": execution.recording_bytes,
        "recording_parsed": True,
        "required_entity_path_count": len(REQUIRED_RRD_PATHS),
        "capture_time_timeline_present": True,
        "visual_qa_passed": True,
        "visual_evidence": visual_evidence,
        "stable_export_fields_verified": list(STABLE_EXPORT_FIELDS),
        "presentation_video_mode": export.get("presentation_video_mode"),
        "presentation_video_manifest_ref": export.get(
            "presentation_video_manifest_ref"
        ),
        "coordinate_log_record_count": export.get("coordinate_log_record_count"),
        "measured_observation_counts": export.get("measured_observation_counts"),
        "trajectory_logging_mode": export.get("trajectory_logging_mode"),
        "full_trajectory_visible_at_start": export.get(
            "full_trajectory_visible_at_start"
        ),
        "measured_step_wall_seconds": {
            step.name.value: step.wall_seconds for step in execution.steps
        },
        "measured_total_wall_seconds": execution.total_wall_seconds,
        "capture_duration_seconds": execution.capture_duration_seconds,
        "assembly_realtime_factor": execution.assembly_realtime_factor,
        "capture_seconds_per_assembly_second": (
            execution.capture_seconds_per_assembly_second
        ),
        "model_inference_performed": execution.model_inference_performed,
        "evidence_kind": execution.evidence_kind,
        "demonstrated_live_capacity": execution.demonstrated_live_capacity,
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
        "final_camera_a.png",
        "final_camera_b.png",
        "final_metric_twin.png",
        "final_coordinates.png",
        "final_state_timeline.png",
        "final_events.png",
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
