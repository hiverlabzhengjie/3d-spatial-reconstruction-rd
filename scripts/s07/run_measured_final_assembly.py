"""Regenerate the final S07 Rerun and measure retained-output assembly time."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from spatial_reconstruction.finalization import (
    FinalRunStepName,
    MeasuredFinalRunStep,
    Stage07FinalRunExecution,
    Stage07FinalRunManifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
S06_ORCHESTRATION_SUMMARY = (
    "artifacts/s06/orchestration_contract_v2_20260805/summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-run-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--presentation-video-manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    final_run_summary_path = (PROJECT_ROOT / args.final_run_summary).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    final_run_summary = _load_json(final_run_summary_path)
    manifest_path = PROJECT_ROOT / str(final_run_summary["manifest_ref"])
    if _sha256(manifest_path) != final_run_summary["manifest_sha256"]:
        raise ValueError("final-run manifest hash differs from summary")
    manifest = Stage07FinalRunManifest.model_validate(_load_json(manifest_path))

    entry_verification_path = output_dir / "entry_verification.json"
    entry_command = (
        sys.executable,
        str(PROJECT_ROOT / "scripts/s07/verify_final_run_manifest.py"),
        "--summary",
        str(final_run_summary_path),
        "--output",
        str(entry_verification_path),
    )
    entry_step = _run_step(
        FinalRunStepName.ENTRY_VERIFICATION,
        entry_command,
        output_dir=output_dir,
        timeout_seconds=60.0,
    )

    recording_path = output_dir / "digital_twin_stage07_final.rrd"
    export_command: tuple[str, ...] = (
        sys.executable,
        str(PROJECT_ROOT / "scripts/s06/export_integrated_rerun.py"),
        "--orchestration-summary",
        S06_ORCHESTRATION_SUMMARY,
        "--output",
        _relative(recording_path),
    )
    presentation_video_manifest_path: Path | None = None
    if args.presentation_video_manifest is not None:
        presentation_video_manifest_path = (
            PROJECT_ROOT / args.presentation_video_manifest
        ).resolve()
        export_command += (
            "--presentation-video-manifest",
            _relative(presentation_video_manifest_path),
        )
    export_step = _run_step(
        FinalRunStepName.RERUN_EXPORT,
        export_command,
        output_dir=output_dir,
        timeout_seconds=120.0,
    )
    export_summary_path = recording_path.with_name(
        f"{recording_path.stem}_export_summary.json"
    )
    export_summary = _load_json(export_summary_path)
    if export_summary["recording_sha256"] != _sha256(recording_path):
        raise ValueError("final recording hash differs from generated export summary")

    total_wall_seconds = entry_step.wall_seconds + export_step.wall_seconds
    capture_duration_seconds = max(video.duration_seconds for video in manifest.source_videos)
    execution = Stage07FinalRunExecution(
        source_final_run_manifest_id=manifest.manifest_id,
        steps=(entry_step, export_step),
        total_wall_seconds=total_wall_seconds,
        capture_duration_seconds=capture_duration_seconds,
        assembly_realtime_factor=total_wall_seconds / capture_duration_seconds,
        capture_seconds_per_assembly_second=capture_duration_seconds / total_wall_seconds,
        recording_ref=_relative(recording_path),
        recording_sha256=_sha256(recording_path),
        recording_bytes=recording_path.stat().st_size,
        export_summary_ref=_relative(export_summary_path),
        export_summary_sha256=_sha256(export_summary_path),
    )
    summary = execution.model_dump(mode="json") | {
        "purpose": "measured_final_retained_output_assembly",
        "source_final_run_summary_ref": _relative(final_run_summary_path),
        "source_final_run_summary_sha256": _sha256(final_run_summary_path),
        "visual_qa_completed": False,
        "structural_verification_completed": False,
        "presentation_video_manifest_ref": (
            None
            if presentation_video_manifest_path is None
            else _relative(presentation_video_manifest_path)
        ),
        "presentation_video_manifest_sha256": (
            None
            if presentation_video_manifest_path is None
            else _sha256(presentation_video_manifest_path)
        ),
        "limitations": [
            "Measured time covers entry verification and retained-output Rerun assembly only.",
            "No YOLO, DA3, or Qwen model inference was repeated in this execution.",
            "The measurement does not demonstrate sustainable live throughput or latency.",
        ],
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _run_step(
    name: FinalRunStepName,
    command: tuple[str, ...],
    *,
    output_dir: Path,
    timeout_seconds: float,
) -> MeasuredFinalRunStep:
    started = time.perf_counter()
    result = subprocess.run(  # noqa: S603 - fixed project scripts and explicit argv
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    wall_seconds = time.perf_counter() - started
    stdout_path = output_dir / f"{name.value}_stdout.log"
    stderr_path = output_dir / f"{name.value}_stderr.log"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"{name.value} failed with return code {result.returncode}; "
            f"see {_relative(stderr_path)}"
        )
    return MeasuredFinalRunStep(
        name=name,
        command=command,
        wall_seconds=wall_seconds,
        return_code=0,
        stdout_ref=_relative(stdout_path),
        stdout_sha256=_sha256(stdout_path),
        stderr_ref=_relative(stderr_path),
        stderr_sha256=_sha256(stderr_path),
    )


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
