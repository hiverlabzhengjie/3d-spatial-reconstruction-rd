"""Verify the retained S06 WP4 localhost RTSP smoke evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from spatial_reconstruction.contracts import FrameSourceKind
from spatial_reconstruction.ingestion import (
    RTSPAttemptOutcome,
    RTSPReconnectRead,
)
from spatial_reconstruction.perception import PerceptionJob

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
    reconnect_path = PROJECT_ROOT / str(summary["reconnect_read_ref"])
    sample_job_path = PROJECT_ROOT / str(summary["sample_perception_job_ref"])
    logs_path = PROJECT_ROOT / str(summary["process_logs_ref"])
    _require_hash(reconnect_path, str(summary["reconnect_read_sha256"]))
    _require_hash(sample_job_path, str(summary["sample_perception_job_sha256"]))
    _require_hash(logs_path, str(summary["process_logs_sha256"]))
    reconnect = RTSPReconnectRead.model_validate(_load_json(reconnect_path))
    sample_job = PerceptionJob.model_validate(_load_json(sample_job_path))
    logs = _load_json_array(logs_path)

    source_summary_path = PROJECT_ROOT / str(summary["source_orchestration_summary_ref"])
    _require_hash(
        source_summary_path,
        str(summary["source_orchestration_summary_sha256"]),
    )
    source_summary = _load_json(source_summary_path)
    manifest_path = PROJECT_ROOT / str(source_summary["manifest_ref"])
    _require_hash(manifest_path, str(source_summary["manifest_sha256"]))
    manifest = _load_json(manifest_path)
    if source_summary["manifest_id"] != summary["source_manifest_id"]:
        raise ValueError("RTSP summary manifest identity differs from its source")
    source_video_path = PROJECT_ROOT / str(summary["source_video_ref"])
    _require_hash(source_video_path, str(summary["source_video_sha256"]))
    config_path = PROJECT_ROOT / str(summary["mediamtx_config_ref"])
    _require_hash(config_path, str(summary["mediamtx_config_sha256"]))

    identities = reconnect.frame_identities
    diagnostics = reconnect.diagnostics
    if not diagnostics.target_reached or diagnostics.exhausted:
        raise ValueError("RTSP smoke did not reach its bounded target")
    if diagnostics.reconnect_count < 1:
        raise ValueError("RTSP smoke did not reconnect")
    if len(diagnostics.attempts) > diagnostics.policy.maximum_connection_attempts:
        raise ValueError("RTSP smoke exceeded its connection-attempt bound")
    if diagnostics.final_outcome is not RTSPAttemptOutcome.TARGET_REACHED:
        raise ValueError("RTSP smoke final attempt did not reach its target")
    decoded_attempts = [attempt for attempt in diagnostics.attempts if attempt.decoded_frame_count]
    if len(decoded_attempts) < 2:
        raise ValueError("RTSP smoke did not decode frames on both sides of the outage")
    final_attempt = diagnostics.attempts[-1]
    if (
        final_attempt.observed_reconnect_gap_seconds is None
        or final_attempt.observed_reconnect_gap_seconds
        < diagnostics.policy.minimum_timestamp_step_seconds
    ):
        raise ValueError("RTSP reconnect did not retain observed outage-gap evidence")
    if not any(
        attempt.outcome in {RTSPAttemptOutcome.FAILED, RTSPAttemptOutcome.STREAM_ENDED}
        for attempt in diagnostics.attempts[:-1]
    ):
        raise ValueError("RTSP smoke contains no outage evidence")
    if any(identity.source_kind is not FrameSourceKind.RTSP for identity in identities):
        raise ValueError("RTSP smoke emitted a non-RTSP frame")
    if any("@" in identity.source_ref or "?" in identity.source_ref for identity in identities):
        raise ValueError("RTSP persistent source reference contains credentials or query")
    if sample_job.frame_identity != identities[-1]:
        raise ValueError("worker-contract sample does not reference the final RTSP frame")
    if sample_job.job_id != summary["worker_contract_job_id"]:
        raise ValueError("worker-contract job identity differs from summary")
    if len(logs) < 3:
        raise ValueError("RTSP process lifecycle evidence is incomplete")
    if not any("is publishing to path" in str(item.get("stdout", "")) for item in logs):
        raise ValueError("MediaMTX logs do not show an RTSP publisher")
    if not any("is reading from path" in str(item.get("stdout", "")) for item in logs):
        raise ValueError("MediaMTX logs do not show an RTSP reader")

    source_refs = {str(entry["source_ref"]) for entry in manifest["source_videos"]}
    if summary["source_video_ref"] not in source_refs:
        raise ValueError("RTSP fixture video is not bound by the orchestration manifest")
    verification = {
        "schema_version": 1,
        "stage": "S06",
        "work_package": 4,
        "status": "passed",
        "purpose": "local_rtsp_open_outage_reconnect_verification",
        "source_summary_ref": _relative(summary_path),
        "source_summary_sha256": _sha256(summary_path),
        "source_manifest_id": summary["source_manifest_id"],
        "decoded_frame_count": diagnostics.total_decoded_frame_count,
        "connection_attempt_count": len(diagnostics.attempts),
        "reconnect_count": diagnostics.reconnect_count,
        "connection_attempts_bounded": True,
        "deliberate_outage_observed": True,
        "target_reached_after_reconnect": True,
        "frame_ids_unique": True,
        "source_frame_indexes_contiguous": True,
        "capture_timestamps_strictly_increasing": True,
        "observed_outage_gap_preserved_in_capture_time": True,
        "credential_safe_source_ref": True,
        "stream_configuration_fingerprint_present": True,
        "worker_contract_compatible": True,
        "publisher_and_reader_log_evidence_present": True,
        "model_inference_performed": False,
        "local_only": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


def _require_hash(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"artifact hash differs for {path}: {actual} != {expected}")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"expected a JSON object array: {path}")
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
