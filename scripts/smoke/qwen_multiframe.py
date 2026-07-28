"""Run the isolated S00 WP7 Qwen3-VL bounded multi-frame MPS smoke check."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

import torch
import transformers

from spatial_reconstruction.config import PROJECT_ROOT, load_project_config
from spatial_reconstruction.contracts import (
    MemoryObservation,
    ModelRunObservation,
    RunOutcome,
    TimingObservation,
)
from spatial_reconstruction.models import (
    Qwen3VLAdapter,
    QwenTextResponse,
    extract_uniform_video_frames,
)
from spatial_reconstruction.runtime import (
    DeviceSelection,
    DeviceUnavailableError,
    PeakRSSMonitor,
    PhaseTimer,
    SystemMemorySource,
    build_run_observation,
    sample_memory,
    select_device,
)

sys.dont_write_bytecode = True

DEFAULT_PROMPT = (
    "Describe the directly visible action across these ordered frames in temporal order. "
    "Use concise factual language. If the action is unclear, say unknown. Do not infer or "
    "report coordinates, identities, timestamps, or zone membership."
)


def parse_args() -> argparse.Namespace:
    config = load_project_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=config.paths.da3_vendor_dir / "assets/examples/robot_unitree.mp4",
        help="Local source video; defaults to the approved DA3 vendor fixture.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Artifact directory; defaults to a new timestamped S00 Qwen run.",
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=config.qwen.smoke_frame_count,
        help="Ordered uniform frames; the S00 bound is four through eight.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=config.qwen.max_new_tokens,
        help="Deterministic output-token bound.",
    )
    return parser.parse_args()


def main() -> int:
    """Run cold/warm ordered-frame generation and persist gate evidence."""

    args = parse_args()
    config = load_project_config()
    video_path = args.video.resolve()
    output_dir = args.output_dir or _timestamped_output_dir(config.paths.artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "summary.json"
    failure_path = output_dir / "failure.txt"

    selection: DeviceSelection | None = None
    adapter: Qwen3VLAdapter | None = None
    model_revision: str | None = None
    timings: list[TimingObservation] = []
    memory: list[MemoryObservation] = []
    warnings: list[str] = []
    artifact_refs: list[str] = [str(summary_path)]
    source = SystemMemorySource()

    try:
        extracted = extract_uniform_video_frames(video_path, frame_count=args.frame_count)
        frame_dir = output_dir / "frames"
        frame_dir.mkdir()
        frame_records: list[dict[str, object]] = []
        for frame in extracted.frames:
            frame_path = frame_dir / f"frame_{frame.sequence_index:02d}.jpg"
            frame.image.save(frame_path, quality=92)
            artifact_refs.append(str(frame_path))
            frame_records.append(
                {
                    "sequence_index": frame.sequence_index,
                    "source_frame_index": frame.source_frame_index,
                    "timestamp_seconds": frame.timestamp_seconds,
                    "width": frame.image.width,
                    "height": frame.image.height,
                    "artifact_path": str(frame_path),
                    "artifact_sha256": _sha256(frame_path),
                }
            )

        selection = select_device(
            config.runtime.preferred_device,
            allow_cpu_fallback=False,
        )
        if selection.actual != "mps":
            raise RuntimeError("WP7 requires an actual MPS run")

        memory.append(sample_memory("before_model_load", device="mps", source=source))
        load_monitor = PeakRSSMonitor(source=source)
        load_timer = PhaseTimer(phase="model_load", device="mps")
        with load_monitor, load_timer:
            adapter = Qwen3VLAdapter.from_pretrained(
                model_id=config.models.qwen,
                cache_dir=PROJECT_ROOT / ".cache" / "huggingface",
                device="mps",
                dtype=torch.float16,
            )
        _append_timing(timings, load_timer)
        model_revision = adapter.model_revision
        memory.append(
            sample_memory(
                "after_model_load",
                device="mps",
                source=source,
                process_peak_rss_bytes=load_monitor.peak_rss_bytes,
            )
        )
        if adapter.model_precision != "float16":
            raise RuntimeError(
                f"unexpected Qwen model precision: {adapter.model_precision}"
            )

        images = tuple(frame.image for frame in extracted.frames)
        cold = _timed_generation(
            adapter=adapter,
            images=images,
            prompt=DEFAULT_PROMPT,
            max_new_tokens=args.max_new_tokens,
            phase="cold_generation",
            timings=timings,
            memory=memory,
            source=source,
        )
        warm = _timed_generation(
            adapter=adapter,
            images=images,
            prompt=DEFAULT_PROMPT,
            max_new_tokens=args.max_new_tokens,
            phase="warm_generation",
            timings=timings,
            memory=memory,
            source=source,
        )

        source_sha256_after = _sha256(video_path)
        if source_sha256_after != extracted.source_sha256:
            raise RuntimeError("Qwen source video changed during WP7")
        if cold.text != warm.text:
            warnings.append(
                "deterministic cold and warm decoded responses differed; both are retained"
            )

        cold_path = output_dir / "cold_response.txt"
        warm_path = output_dir / "warm_response.txt"
        raw_path = output_dir / "raw_responses.json"
        prompt_path = output_dir / "prompt_manifest.json"
        manifest_path = output_dir / "manifest.json"
        cold_path.write_text(f"{cold.text}\n", encoding="utf-8")
        warm_path.write_text(f"{warm.text}\n", encoding="utf-8")
        raw_payload = {
            "cold": _response_payload(cold),
            "warm": _response_payload(warm),
        }
        raw_path.write_text(
            f"{json.dumps(raw_payload, indent=2)}\n",
            encoding="utf-8",
        )
        prompt_payload = {
            "prompt": DEFAULT_PROMPT,
            "ordered_frames": frame_records,
        }
        prompt_path.write_text(
            f"{json.dumps(prompt_payload, indent=2)}\n",
            encoding="utf-8",
        )
        artifact_refs.extend(
            str(path)
            for path in (cold_path, warm_path, raw_path, prompt_path, manifest_path)
        )

        manifest = {
            "model_id": config.models.qwen,
            "model_revision": model_revision,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "device": selection.actual,
            "precision": adapter.model_precision,
            "asynchronous_adapter": True,
            "deterministic_decoding": {
                "do_sample": False,
                "max_new_tokens": args.max_new_tokens,
            },
            "spatial_write_interface": False,
            "input": {
                "path": str(video_path),
                "sha256_before": extracted.source_sha256,
                "sha256_after": source_sha256_after,
                "size_bytes": extracted.source_size_bytes,
                "total_frame_count": extracted.total_frame_count,
                "fps": extracted.source_fps,
                "width": extracted.source_width,
                "height": extracted.source_height,
                "selected_frame_count": len(extracted.frames),
                "selected_source_frame_indices": [
                    frame.source_frame_index for frame in extracted.frames
                ],
            },
            "responses": {
                "cold_non_empty": bool(cold.text),
                "warm_non_empty": bool(warm.text),
                "cold_output_tokens": cold.output_token_count,
                "warm_output_tokens": warm.output_token_count,
                "within_token_bound": (
                    cold.output_token_count <= args.max_new_tokens
                    and warm.output_token_count <= args.max_new_tokens
                ),
            },
            "warnings": warnings,
        }
        manifest_path.write_text(
            f"{json.dumps(manifest, indent=2)}\n",
            encoding="utf-8",
        )

        summary = build_run_observation(
            model_id=config.models.qwen,
            model_revision=model_revision,
            selection=selection,
            precision=adapter.model_precision,
            input_description=(
                f"{len(extracted.frames)} ordered uniform frames from "
                "the local vendor robot_unitree.mp4 fixture"
            ),
            timings=timings,
            memory=memory,
            outcome=RunOutcome.PASSED,
            warnings=warnings,
            artifact_refs=artifact_refs,
        )
        _write_summary(summary, summary_path)
        print(summary.model_dump_json(indent=2))
        print(f"Cold response: {cold.text}")
        print(f"Warm response: {warm.text}")
        print(f"Artifacts: {output_dir}")
        return 0
    except Exception as exc:
        failure_path.write_text(traceback.format_exc(), encoding="utf-8")
        artifact_refs.append(str(failure_path))
        warnings.append(f"{type(exc).__name__}: {exc}")
        if selection is None and isinstance(exc, DeviceUnavailableError):
            summary = build_run_observation(
                model_id=config.models.qwen,
                model_revision=model_revision,
                selection=None,
                requested_device=config.runtime.preferred_device,
                device_selection_error=str(exc),
                precision="unknown",
                input_description="Qwen WP7 preflight failed before device selection",
                timings=timings,
                memory=memory,
                outcome=RunOutcome.FAILED,
                warnings=warnings,
                artifact_refs=artifact_refs,
            )
        else:
            summary = build_run_observation(
                model_id=config.models.qwen,
                model_revision=model_revision,
                selection=selection,
                requested_device=(config.runtime.preferred_device if selection is None else None),
                device_selection_error=str(exc) if selection is None else None,
                precision=(adapter.model_precision if adapter is not None else "unknown"),
                input_description="Qwen WP7 bounded multi-frame smoke attempt",
                timings=timings,
                memory=memory,
                outcome=RunOutcome.FAILED,
                warnings=warnings,
                artifact_refs=artifact_refs,
            )
        _write_summary(summary, summary_path)
        print(summary.model_dump_json(indent=2))
        print(f"Failure details: {failure_path}")
        return 1


def _timed_generation(
    *,
    adapter: Qwen3VLAdapter,
    images: tuple[object, ...],
    prompt: str,
    max_new_tokens: int,
    phase: str,
    timings: list[TimingObservation],
    memory: list[MemoryObservation],
    source: SystemMemorySource,
) -> QwenTextResponse:
    """Retain timing and peak memory evidence even when generation fails."""

    from PIL import Image

    typed_images = tuple(image for image in images if isinstance(image, Image.Image))
    if len(typed_images) != len(images):
        raise TypeError("Qwen smoke images must all be Pillow images")
    monitor = PeakRSSMonitor(source=source)
    timer = PhaseTimer(phase=phase, device="mps")
    try:
        with monitor, timer:
            return asyncio.run(
                adapter.generate(
                    images=typed_images,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                )
            )
    finally:
        if timer.observation is not None:
            timings.append(timer.observation)
        memory.append(
            sample_memory(
                f"after_{phase}",
                device="mps",
                source=source,
                process_peak_rss_bytes=monitor.peak_rss_bytes or None,
            )
        )


def _response_payload(response: QwenTextResponse) -> dict[str, object]:
    return {
        "text": response.text,
        "input_token_count": response.input_token_count,
        "output_token_count": response.output_token_count,
        "output_token_ids": list(response.output_token_ids),
        "input_shapes": {
            name: list(shape) for name, shape in response.input_shapes.items()
        },
    }


def _append_timing(timings: list[TimingObservation], timer: PhaseTimer) -> None:
    if timer.observation is None:
        raise RuntimeError(f"timer '{timer.phase}' did not emit an observation")
    timings.append(timer.observation)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timestamped_output_dir(artifacts_dir: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return artifacts_dir / "s00" / "qwen" / f"wp7_{timestamp}"


def _write_summary(summary: ModelRunObservation, path: Path) -> None:
    path.write_text(f"{summary.model_dump_json(indent=2)}\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
