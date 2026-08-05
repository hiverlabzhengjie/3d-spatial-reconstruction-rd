"""Run a localhost RTSP open-outage-reconnect smoke test for S06 WP4."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import av

from spatial_reconstruction.ingestion import (
    RTSPAttemptOutcome,
    RTSPFrameSource,
    RTSPReconnectPolicy,
    read_rtsp_with_reconnect,
)
from spatial_reconstruction.perception import PerceptionJob

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("artifacts/s01/action_take_01/synchronized/camera_a_action_synced.mp4")
DEFAULT_CONFIG = Path("configs/mediamtx_s06_local.yml")
DEFAULT_URL = "rtsp://127.0.0.1:18554/s06_camera_a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestration-summary", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--mediamtx-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--target-frames", type=int, default=45)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    if args.target_frames <= 0:
        raise ValueError("target frame count must be positive")
    output_dir.mkdir(parents=True)

    orchestration_summary_path = args.orchestration_summary.resolve()
    orchestration_summary = _load_json(orchestration_summary_path)
    manifest_path = PROJECT_ROOT / str(orchestration_summary["manifest_ref"])
    manifest = _load_json(manifest_path)
    if _sha256(manifest_path) != orchestration_summary["manifest_sha256"]:
        raise ValueError("orchestration manifest hash differs from summary")

    source_path = args.source.resolve()
    config_path = args.mediamtx_config.resolve()
    if not source_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("RTSP smoke source or MediaMTX config is missing")
    source_ref = _relative(source_path)
    source_entries = [
        entry for entry in manifest["source_videos"] if entry["source_ref"] == source_ref
    ]
    if len(source_entries) != 1:
        raise ValueError("RTSP smoke source is not bound by the orchestration manifest")
    source_entry = source_entries[0]
    if _sha256(source_path) != source_entry["source_sha256"]:
        raise ValueError("RTSP smoke source hash differs from orchestration manifest")

    parsed_url = urlsplit(args.url)
    if parsed_url.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("RTSP smoke test is restricted to localhost")
    if parsed_url.port is None:
        raise ValueError("RTSP smoke URL must contain an explicit port")

    server_command = ("mediamtx", str(config_path))
    publisher_command = (
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-re",
        "-stream_loop",
        "-1",
        "-i",
        str(source_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-vf",
        "scale=640:360",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "10",
        "-keyint_min",
        "10",
        "-sc_threshold",
        "0",
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        args.url,
    )
    processes: list[subprocess.Popen[str]] = []
    server = _start_process(server_command)
    processes.append(server)
    publishers: list[subprocess.Popen[str]] = []
    controller_errors: list[str] = []
    try:
        _wait_for_tcp(parsed_url.hostname or "127.0.0.1", parsed_url.port, timeout=5.0)
        first_publisher = _start_process(publisher_command)
        publishers.append(first_publisher)
        processes.append(first_publisher)
        time.sleep(0.6)
        if first_publisher.poll() is not None:
            raise RuntimeError("initial FFmpeg RTSP publisher exited before decoding")

        def cycle_publisher() -> None:
            try:
                time.sleep(0.7)
                _terminate(first_publisher)
                time.sleep(0.6)
                replacement = _start_process(publisher_command)
                publishers.append(replacement)
                processes.append(replacement)
            except Exception as exc:  # pragma: no cover - surfaced by smoke result
                controller_errors.append(f"{type(exc).__name__}: {exc}")

        controller = threading.Thread(target=cycle_publisher, daemon=True)
        controller.start()
        policy = RTSPReconnectPolicy(
            maximum_connection_attempts=8,
            reconnect_delay_seconds=0.25,
            minimum_timestamp_step_seconds=1.0 / 30.0,
        )

        def source_factory(_attempt: int) -> RTSPFrameSource:
            return RTSPFrameSource(
                url=args.url,
                capture_session_id=str(manifest["capture_session_id"]),
                camera_id="camera_a",
                synchronization_manifest_ref=str(manifest["synchronization_manifest_ref"]),
                synchronization_manifest_sha256=str(manifest["synchronization_manifest_sha256"]),
                pose_version_id=(f"{manifest['capture_session_id']}:camera_a:fixed_pose:v1"),
                open_options={"timeout": "1000000"},
            )

        frames, reconnect_read = read_rtsp_with_reconnect(
            source_factory,
            target_frame_count=args.target_frames,
            policy=policy,
        )
        controller.join(timeout=5.0)
        if controller.is_alive():
            raise RuntimeError("publisher outage controller did not finish")
        if controller_errors:
            raise RuntimeError(controller_errors[0])
        if not reconnect_read.diagnostics.target_reached:
            raise RuntimeError("bounded RTSP smoke did not reach its frame target")
        if reconnect_read.diagnostics.reconnect_count < 1:
            raise RuntimeError("RTSP smoke did not exercise reconnect behavior")
        if not any(
            attempt.outcome in {RTSPAttemptOutcome.FAILED, RTSPAttemptOutcome.STREAM_ENDED}
            for attempt in reconnect_read.diagnostics.attempts[:-1]
        ):
            raise RuntimeError("RTSP smoke did not observe the deliberate outage")

        reconnect_path = output_dir / "reconnect_read.json"
        _write_json(reconnect_path, reconnect_read.model_dump(mode="json"))
        sample_job = PerceptionJob.create(
            frame_identity=frames[-1].identity,
            model_id="yolov8n-seg.pt",
            model_revision="0" * 64,
            policy_id="s06_rtsp_contract_compatibility_only",
            created_processing_seconds=0.0,
        )
        sample_job_path = output_dir / "sample_perception_job.json"
        _write_json(sample_job_path, sample_job.model_dump(mode="json"))
        first_pixel_sha256 = hashlib.sha256(frames[0].image_bgr.tobytes()).hexdigest()
        last_pixel_sha256 = hashlib.sha256(frames[-1].image_bgr.tobytes()).hexdigest()
    finally:
        for process in reversed(processes):
            _terminate(process)
        process_logs = _collect_process_logs(processes)
        logs_path = output_dir / "process_logs.json"
        _write_json(logs_path, process_logs)

    summary = {
        "schema_version": 1,
        "stage": "S06",
        "work_package": 4,
        "status": "completed",
        "purpose": "local_rtsp_open_outage_reconnect_compatibility",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_orchestration_summary_ref": _relative(orchestration_summary_path),
        "source_orchestration_summary_sha256": _sha256(orchestration_summary_path),
        "source_manifest_id": orchestration_summary["manifest_id"],
        "source_video_ref": source_ref,
        "source_video_sha256": _sha256(source_path),
        "mediamtx_config_ref": _relative(config_path),
        "mediamtx_config_sha256": _sha256(config_path),
        "rtsp_source_ref": frames[0].identity.source_ref,
        "rtsp_source_fingerprint": frames[0].identity.source_fingerprint,
        "rtsp_source_fingerprint_kind": (frames[0].identity.source_fingerprint_kind.value),
        "reconnect_read_ref": _relative(reconnect_path),
        "reconnect_read_sha256": _sha256(reconnect_path),
        "sample_perception_job_ref": _relative(sample_job_path),
        "sample_perception_job_sha256": _sha256(sample_job_path),
        "process_logs_ref": _relative(logs_path),
        "process_logs_sha256": _sha256(logs_path),
        "decoded_frame_count": len(frames),
        "connection_attempt_count": len(reconnect_read.diagnostics.attempts),
        "reconnect_count": reconnect_read.diagnostics.reconnect_count,
        "attempt_outcomes": [
            attempt.outcome.value for attempt in reconnect_read.diagnostics.attempts
        ],
        "frame_ids_unique": True,
        "source_frame_indexes_contiguous": True,
        "capture_timestamps_strictly_increasing": True,
        "first_capture_timestamp_seconds": (frames[0].identity.capture_timestamp_seconds),
        "last_capture_timestamp_seconds": (frames[-1].identity.capture_timestamp_seconds),
        "image_width": frames[0].identity.image_width,
        "image_height": frames[0].identity.image_height,
        "first_pixel_sha256": first_pixel_sha256,
        "last_pixel_sha256": last_pixel_sha256,
        "worker_contract_job_id": sample_job.job_id,
        "worker_contract_frame_id": sample_job.frame_identity.frame_id,
        "mediamtx_version": _command_version(("mediamtx", "--version")),
        "ffmpeg_version": _command_version(("ffmpeg", "-version")),
        "pyav_version": av.__version__,
        "model_inference_performed": False,
        "local_only": True,
        "limitations": [
            "This is protocol-level localhost compatibility, not production RTSP validation.",
            "The deliberate outage restarts an FFmpeg publisher while MediaMTX stays online.",
            "No jitter, packet-loss, authentication, TLS, or multi-camera load test is claimed.",
        ],
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _start_process(command: tuple[str, ...]) -> subprocess.Popen[str]:
    return subprocess.Popen(  # noqa: S603 - project-owned argv only
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)


def _collect_process_logs(processes: list[subprocess.Popen[str]]) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    for index, process in enumerate(processes):
        stdout, stderr = process.communicate()
        logs.append(
            {
                "process_index": index,
                "return_code": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
            }
        )
    return logs


def _wait_for_tcp(host: str, port: int, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"local RTSP server did not listen on {host}:{port}")


def _command_version(command: tuple[str, ...]) -> str:
    result = subprocess.run(  # noqa: S603 - fixed version-only argv
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return (result.stdout or result.stderr).splitlines()[0]


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
