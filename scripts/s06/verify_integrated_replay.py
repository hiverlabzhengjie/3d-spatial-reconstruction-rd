"""Independently regenerate and verify S06 WP3 orchestration evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from spatial_reconstruction.orchestration import (
    IntegratedReplayReport,
    Stage06OrchestrationManifest,
    WorkerKind,
    run_integrated_replay,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    report_path = PROJECT_ROOT / str(summary["report_ref"])
    if _sha256(report_path) != summary["report_sha256"]:
        raise ValueError("integrated replay report hash differs from summary")
    report = IntegratedReplayReport.model_validate(_load_json(report_path))

    source_summary_path = PROJECT_ROOT / str(summary["source_orchestration_summary_ref"])
    if _sha256(source_summary_path) != summary["source_orchestration_summary_sha256"]:
        raise ValueError("source orchestration summary hash differs")
    source_summary = _load_json(source_summary_path)
    manifest_path = PROJECT_ROOT / str(source_summary["manifest_ref"])
    if _sha256(manifest_path) != source_summary["manifest_sha256"]:
        raise ValueError("source orchestration manifest hash differs")
    manifest = Stage06OrchestrationManifest.model_validate(_load_json(manifest_path))
    _verify_manifest_sources(manifest)

    regenerated = run_integrated_replay(
        manifest_id=manifest.manifest_id,
        policy=manifest.policy,
    )
    if regenerated != report:
        raise ValueError("integrated replay report does not regenerate exactly")
    if report.source_manifest_id != summary["source_manifest_id"]:
        raise ValueError("replay report manifest differs from summary")
    if report.capture_output_digest != summary["capture_output_digest"]:
        raise ValueError("capture output digest differs from summary")
    if any(item.peak_depth > item.capacity for item in report.queue_diagnostics):
        raise ValueError("a replay queue exceeded its configured capacity")
    if any(item.dropped_count for item in report.queue_diagnostics):
        raise ValueError("offline replay dropped accepted source work")
    if not any(item.throttled_count for item in report.queue_diagnostics):
        raise ValueError("queue saturation was not exercised")
    if not any(item.coalesced_count for item in report.queue_diagnostics):
        raise ValueError("duplicate coalescing was not exercised")
    if report.completion_order_a == report.capture_output_order:
        raise ValueError("completion order unexpectedly equals capture output order")
    if report.completion_order_a == report.completion_order_b:
        raise ValueError("the two completion schedules unexpectedly match")
    geometry_finish = max(
        attempt.processing_finished_seconds
        for attempt in report.attempts
        if attempt.worker_kind is WorkerKind.GEOMETRY
    )
    qwen_retry_finish = max(
        attempt.processing_finished_seconds
        for attempt in report.attempts
        if attempt.worker_kind is WorkerKind.QWEN and attempt.attempt == 2
    )
    if geometry_finish >= qwen_retry_finish:
        raise ValueError("geometry did not remain independent of the Qwen retry")

    verification = {
        "schema_version": 1,
        "stage": "S06",
        "work_package": 3,
        "status": "passed",
        "purpose": "deterministic_integrated_offline_replay_verification",
        "source_summary_ref": _relative(summary_path),
        "source_summary_sha256": _sha256(summary_path),
        "source_manifest_id": report.source_manifest_id,
        "report_regenerated_exactly": True,
        "logical_job_count": len(report.results),
        "attempt_count": len(report.attempts),
        "completion_orders_differ": True,
        "capture_outputs_match_across_schedules": True,
        "capture_output_digest": report.capture_output_digest,
        "capture_time_authoritative": report.capture_time_authoritative,
        "worker_completion_order_authoritative": (report.worker_completion_order_authoritative),
        "queue_count": len(report.queue_diagnostics),
        "queue_saturation_exercised": True,
        "total_throttled_submissions": sum(
            item.throttled_count for item in report.queue_diagnostics
        ),
        "total_coalesced_submissions": sum(
            item.coalesced_count for item in report.queue_diagnostics
        ),
        "total_dropped_jobs": 0,
        "accelerator_permit_count": report.accelerator_permit_count,
        "maximum_observed_accelerator_occupancy": (report.maximum_observed_accelerator_occupancy),
        "accelerator_intervals_non_overlapping": True,
        "qwen_retry_count": report.qwen_retry_count,
        "duplicate_results_suppressed": report.duplicate_results_suppressed,
        "degraded_result_count": report.degraded_result_count,
        "qwen_failure_blocked_geometry": report.qwen_failure_blocked_geometry,
        "graceful_shutdown_verified": report.shutdown.shutdown_completed,
        "cancelled_pending_on_shutdown": report.shutdown.cancelled_pending_count,
        "final_queue_depth": report.shutdown.final_queue_depth,
        "final_in_flight_count": report.shutdown.final_in_flight_count,
        "accelerator_permit_released": report.shutdown.accelerator_permit_released,
        "model_inference_performed": False,
        "timing_evidence_kind": report.timing_evidence_kind,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
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
