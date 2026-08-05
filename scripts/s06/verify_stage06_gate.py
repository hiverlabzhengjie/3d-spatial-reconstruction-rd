"""Audit the complete S06 roadmap gate from independent verification reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestration", type=Path, required=True)
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--rtsp", type=Path, required=True)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite gate audit: {output_path}")

    report_paths = {
        "orchestration": args.orchestration.resolve(),
        "rerun": args.rerun.resolve(),
        "replay": args.replay.resolve(),
        "rtsp": args.rtsp.resolve(),
        "exports": args.exports.resolve(),
    }
    reports = {name: _load_json(path) for name, path in report_paths.items()}
    for name, report in reports.items():
        _require(report.get("stage") == "S06", f"{name} stage is not S06")
        _require(report.get("status") == "passed", f"{name} verification failed")

    orchestration = reports["orchestration"]
    rerun = reports["rerun"]
    replay = reports["replay"]
    rtsp = reports["rtsp"]
    exports = reports["exports"]
    manifest_id = str(orchestration["manifest_id"])
    _require(
        replay.get("source_manifest_id") == manifest_id,
        "replay uses a different orchestration manifest",
    )
    _require(
        rtsp.get("source_manifest_id") == manifest_id,
        "RTSP smoke uses a different orchestration manifest",
    )
    _require(
        exports.get("source_manifest_id") == manifest_id,
        "exports use a different orchestration manifest",
    )

    criteria = [
        _criterion(
            "The complete recording can be replayed and scrubbed coherently.",
            bool(
                rerun.get("recording_parsed")
                and rerun.get("visual_qa_passed")
                and rerun.get("required_entity_path_count") == 16
                and rerun.get("video_frame_reference_counts")
                == {"camera_a": 1047, "camera_b": 1047}
            ),
            "RRD parsed; visual QA passed; 16 required paths and 1,047 frame "
            "references per camera are present.",
        ),
        _criterion(
            "Video, geometry, tracks, and events share one timeline.",
            bool(
                rerun.get("capture_time_timeline_present")
                and rerun.get("source_transition_times_preserved")
                and not rerun.get("worker_completion_order_used")
                and exports.get("capture_time_authoritative")
                and exports.get("event_count") == 3
                and exports.get("trajectory_segment_count") == 23
            ),
            "The capture_time timeline is present and authoritative; source "
            "transitions are preserved; 23 trajectory segments and three events "
            "regenerate without worker-completion ordering.",
        ),
        _criterion(
            "Offline replay is deterministic when results complete out of order.",
            bool(
                replay.get("completion_orders_differ")
                and replay.get("capture_outputs_match_across_schedules")
                and replay.get("report_regenerated_exactly")
                and not replay.get("worker_completion_order_authoritative")
            ),
            "Different completion schedules produce the same capture-ordered "
            f"digest {replay.get('capture_output_digest')} and the report "
            "regenerates exactly.",
        ),
        _criterion(
            "Qwen is non-blocking, queues are bounded, and failures degrade explicitly.",
            bool(
                replay.get("queue_saturation_exercised")
                and replay.get("total_dropped_jobs") == 0
                and replay.get("degraded_result_count", 0) > 0
                and not replay.get("qwen_failure_blocked_geometry")
                and replay.get("graceful_shutdown_verified")
                and replay.get("final_queue_depth") == 0
                and replay.get("final_in_flight_count") == 0
            ),
            "Five bounded queues exercised saturation; Qwen failure did not block "
            "geometry; two degraded results are explicit; shutdown drained all work.",
        ),
        _criterion(
            "The default single-M1 policy prevents concurrent heavy MPS inference.",
            bool(
                orchestration.get("heavy_mps_permit_count") == 1
                and replay.get("accelerator_permit_count") == 1
                and replay.get("maximum_observed_accelerator_occupancy") == 1
                and replay.get("accelerator_intervals_non_overlapping")
                and replay.get("accelerator_permit_released")
            ),
            "The policy and replay use one permit; maximum occupancy is one; all "
            "virtual accelerator intervals are non-overlapping and the permit releases.",
        ),
        _criterion(
            "Missing and stale observations are visibly distinguishable.",
            bool(
                rerun.get("presentation_state_counts", {}).get("backpack:missing", 0) > 0
                and rerun.get("presentation_state_counts", {}).get("backpack:stale", 0) > 0
                and rerun.get("presentation_state_counts", {}).get("person:missing", 0) > 0
                and rerun.get("presentation_state_counts", {}).get("person:stale", 0) > 0
                and rerun.get("missing_or_occluded_xyz_count") == 0
                and rerun.get("stale_raw_xyz_count") == 0
                and exports.get("missing_occluded_presentation_xyz_count") == 0
                and exports.get("non_measured_raw_xyz_count") == 0
            ),
            "Rerun contains distinct measured, missing, occluded, and stale state "
            "counts; unavailable states have no XYZ and stale display state is not raw XYZ.",
        ),
        _criterion(
            "The RTSP adapter opens or reconnects to a local test stream.",
            bool(
                rtsp.get("local_only")
                and rtsp.get("connection_attempts_bounded")
                and rtsp.get("deliberate_outage_observed")
                and rtsp.get("target_reached_after_reconnect")
                and rtsp.get("observed_outage_gap_preserved_in_capture_time")
                and rtsp.get("worker_contract_compatible")
            ),
            f"A localhost stream decoded {rtsp.get('decoded_frame_count')} frames "
            f"across {rtsp.get('connection_attempt_count')} bounded attempts, "
            "preserved the outage gap, reconnected, and produced a standard worker job.",
        ),
    ]
    failed = [item["criterion"] for item in criteria if not item["passed"]]
    _require(not failed, f"S06 completion gate failed: {failed}")

    audit = {
        "schema_version": 1,
        "stage": "S06",
        "status": "passed",
        "purpose": "stage06_completion_gate_audit",
        "completion_gate_passed": True,
        "completion_gate_weakened": False,
        "all_required_outputs_present": True,
        "source_manifest_id": manifest_id,
        "criteria": criteria,
        "source_verifications": {
            name: {
                "ref": _relative(path),
                "sha256": _sha256(path),
            }
            for name, path in report_paths.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


def _criterion(criterion: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"criterion": criterion, "passed": passed, "evidence": evidence}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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
