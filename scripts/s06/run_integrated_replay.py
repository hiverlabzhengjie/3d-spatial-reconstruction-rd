"""Generate S06 WP3 deterministic orchestration and shutdown evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spatial_reconstruction.orchestration import (
    Stage06OrchestrationManifest,
    run_integrated_replay,
    summarize_attempt_outcomes,
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
    summary_path = args.orchestration_summary.resolve()
    source_summary = _load_json(summary_path)
    manifest_path = PROJECT_ROOT / str(source_summary["manifest_ref"])
    if _sha256(manifest_path) != source_summary["manifest_sha256"]:
        raise ValueError("orchestration manifest hash differs from source summary")
    manifest = Stage06OrchestrationManifest.model_validate(_load_json(manifest_path))
    if manifest.manifest_id != source_summary["manifest_id"]:
        raise ValueError("orchestration manifest ID differs from source summary")
    _verify_manifest_sources(manifest)

    report = run_integrated_replay(
        manifest_id=manifest.manifest_id,
        policy=manifest.policy,
    )
    output_dir.mkdir(parents=True)
    report_path = output_dir / "integrated_replay_report.json"
    _write_json(report_path, report.model_dump(mode="json"))

    queue_totals = {
        "accepted": sum(item.accepted_count for item in report.queue_diagnostics),
        "completed": sum(item.completed_count for item in report.queue_diagnostics),
        "failed": sum(item.failed_count for item in report.queue_diagnostics),
        "throttled": sum(item.throttled_count for item in report.queue_diagnostics),
        "coalesced": sum(item.coalesced_count for item in report.queue_diagnostics),
        "dropped": sum(item.dropped_count for item in report.queue_diagnostics),
        "cancelled_in_shutdown_exercise": report.shutdown.cancelled_pending_count,
    }
    summary = {
        "schema_version": 1,
        "stage": "S06",
        "work_package": 3,
        "status": "completed",
        "purpose": "deterministic_integrated_offline_replay_diagnostics",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_orchestration_summary_ref": _relative(summary_path),
        "source_orchestration_summary_sha256": _sha256(summary_path),
        "source_manifest_id": manifest.manifest_id,
        "report_ref": _relative(report_path),
        "report_sha256": _sha256(report_path),
        "logical_job_count": len(report.results),
        "attempt_count": len(report.attempts),
        "attempt_outcomes": summarize_attempt_outcomes(report.attempts),
        "queue_totals": queue_totals,
        "capture_output_digest": report.capture_output_digest,
        "completion_orders_differ": report.completion_orders_differ,
        "capture_outputs_match_across_schedules": (report.capture_outputs_match_across_schedules),
        "accelerator_permit_count": report.accelerator_permit_count,
        "maximum_observed_accelerator_occupancy": (report.maximum_observed_accelerator_occupancy),
        "qwen_retry_count": report.qwen_retry_count,
        "duplicate_results_suppressed": report.duplicate_results_suppressed,
        "degraded_result_count": report.degraded_result_count,
        "shutdown_completed": report.shutdown.shutdown_completed,
        "model_inference_performed": False,
        "timing_evidence_kind": report.timing_evidence_kind,
        "limitations": [
            (
                "Virtual-time diagnostics prove orchestration invariants and are not "
                "measured model throughput or latency."
            ),
            "Accepted S02-S05 model outputs are referenced, not recomputed.",
            "RTSP open and reconnect testing remains a later S06 work package.",
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
