"""Execute the verified S05 Qwen event jobs once on local Apple MPS."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import av
import torch
from PIL import Image, ImageDraw

from spatial_reconstruction.interaction import (
    BoundedQwenEventQueue,
    QwenEventJob,
    QwenEventJobPlanRunSummary,
    QwenEventProcessingOutput,
    QwenEventResult,
    QwenEventResultOutcome,
    make_qwen_retry_job,
    process_next_qwen_event,
)
from spatial_reconstruction.models.qwen_adapter import Qwen3VLAdapter

INFERENCE_MAX_DIMENSION_PIXELS = 768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--plan-summary",
        type=Path,
        default=Path("artifacts/s05/qwen_event_job_plan_v4_20260804/summary.json"),
    )
    parser.add_argument(
        "--plan-verification",
        type=Path,
        default=Path("artifacts/s05/qwen_event_job_plan_v4_20260804/verification.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    return parser.parse_args()


class LocalFrameProcessor:
    def __init__(
        self,
        adapter: Qwen3VLAdapter,
        images_by_job: dict[str, tuple[Image.Image, ...]],
    ) -> None:
        self._adapter = adapter
        self._images_by_job = images_by_job

    async def process(self, job: QwenEventJob) -> QwenEventProcessingOutput:
        response = await self._adapter.generate(
            images=self._images_by_job[job.deduplication_key],
            prompt=job.prompt.prompt_text,
            max_new_tokens=job.max_new_tokens,
            assistant_prefill=job.prompt.assistant_prefill,
        )
        return QwenEventProcessingOutput(
            raw_response_text=response.text,
            input_token_count=response.input_token_count,
            output_token_count=response.output_token_count,
            output_token_ids=response.output_token_ids,
            input_shapes=response.input_shapes,
        )


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    root = args.project_root.resolve()
    plan_path = _resolve(root, args.plan_summary)
    verification_path = _resolve(root, args.plan_verification)
    output_dir = _resolve(root, args.output_dir)
    cache_dir = _resolve(root, args.cache_dir)
    if output_dir.exists():
        raise FileExistsError(f"execution output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    plan = QwenEventJobPlanRunSummary.model_validate_json(
        plan_path.read_text(encoding="utf-8")
    )
    verification = _read_object(verification_path)
    if (
        verification.get("status") != "passed"
        or verification.get("source_summary_sha256") != _sha256(plan_path)
    ):
        raise ValueError("Qwen execution requires the matching passed plan verification")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("Qwen execution requires available Apple MPS")

    extraction_started = time.monotonic()
    images_by_job, frame_manifest = _extract_frames(root, output_dir, plan.jobs)
    contact_sheet_path = output_dir / "qwen_event_contact_sheet.jpg"
    _write_contact_sheet(contact_sheet_path, plan.jobs, images_by_job)
    extraction_seconds = time.monotonic() - extraction_started

    load_started = time.monotonic()
    adapter = Qwen3VLAdapter.from_pretrained(
        model_id=plan.policy.model_id,
        cache_dir=cache_dir,
        device="mps",
        dtype=torch.float16,
    )
    model_load_seconds = time.monotonic() - load_started
    if adapter.model_revision != plan.policy.model_revision:
        raise ValueError("loaded Qwen revision differs from the verified job plan")
    if adapter.model_precision != "float16":
        raise ValueError("loaded Qwen precision differs from the verified MPS policy")

    queue = BoundedQwenEventQueue(
        capacity=plan.policy.queue_capacity,
        overflow_policy=plan.policy.overflow_policy,
        maximum_attempts=plan.policy.maximum_attempts,
    )
    for job in plan.jobs:
        submission = queue.submit(job)
        if not submission.accepted:
            raise RuntimeError(f"initial Qwen job was not accepted: {job.job_id}")
    processor = LocalFrameProcessor(adapter, images_by_job)
    attempts: list[QwenEventResult] = []
    for _ in plan.jobs:
        result = asyncio.run(process_next_qwen_event(queue, processor))
        if result is None:
            raise RuntimeError("Qwen queue ended before all initial jobs ran")
        attempts.append(result)

    retry_jobs = [
        make_qwen_retry_job(result.job, created_processing_seconds=time.monotonic())
        for result in attempts
        if result.outcome is not QwenEventResultOutcome.COMPLETED
    ]
    for retry in retry_jobs:
        submission = queue.submit(retry)
        if not submission.accepted:
            raise RuntimeError(f"Qwen repair attempt was not accepted: {retry.job_id}")
    for _ in retry_jobs:
        result = asyncio.run(process_next_qwen_event(queue, processor))
        if result is None:
            raise RuntimeError("Qwen queue ended before all repair attempts ran")
        attempts.append(result)

    latest_by_key: dict[str, QwenEventResult] = {}
    for result in attempts:
        latest_by_key[result.job.deduplication_key] = result
    final_results = tuple(latest_by_key[job.deduplication_key] for job in plan.jobs)

    frames_path = output_dir / "frame_manifest.json"
    frames_path.write_text(json.dumps(frame_manifest, indent=2) + "\n", encoding="utf-8")
    attempts_path = output_dir / "qwen_attempt_results.json"
    attempts_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "S05",
                "attempts": [result.model_dump(mode="json") for result in attempts],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    final_path = output_dir / "qwen_final_results.json"
    final_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "S05",
                "results": [result.model_dump(mode="json") for result in final_results],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    outcome_counts = {
        outcome.value: sum(result.outcome is outcome for result in attempts)
        for outcome in QwenEventResultOutcome
    }
    summary = {
        "schema_version": 1,
        "stage": "S05",
        "status": "completed_pending_verification",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_plan_summary_ref": _relative(plan_path, root),
        "source_plan_summary_sha256": _sha256(plan_path),
        "source_plan_verification_ref": _relative(verification_path, root),
        "source_plan_verification_sha256": _sha256(verification_path),
        "model_id": adapter.model_id,
        "model_revision": adapter.model_revision,
        "device": adapter.device,
        "precision": adapter.model_precision,
        "model_load_count": 1,
        "model_load_seconds": model_load_seconds,
        "frame_extraction_seconds": extraction_seconds,
        "frame_artifact_count": len(frame_manifest["frames"]),
        "inference_max_dimension_pixels": INFERENCE_MAX_DIMENSION_PIXELS,
        "attempt_count": len(attempts),
        "repair_attempt_count": len(retry_jobs),
        "outcome_counts": outcome_counts,
        "final_event_labels": {
            result.job.event_kind.value: result.interpretation.event_label.value
            for result in final_results
        },
        "final_matches_candidate": {
            result.job.event_kind.value: result.interpretation.matches_candidate
            for result in final_results
        },
        "queue_diagnostics": queue.diagnostics.model_dump(mode="json"),
        "frame_manifest_ref": _relative(frames_path, root),
        "frame_manifest_sha256": _sha256(frames_path),
        "attempt_results_ref": _relative(attempts_path, root),
        "attempt_results_sha256": _sha256(attempts_path),
        "final_results_ref": _relative(final_path, root),
        "final_results_sha256": _sha256(final_path),
        "contact_sheet_ref": _relative(contact_sheet_path, root),
        "contact_sheet_sha256": _sha256(contact_sheet_path),
        "spatial_state_mutated": False,
        "limitations": [
            "Qwen supplies semantic review only and cannot modify spatial state.",
            "A valid response is qualitative evidence, not calibrated probability.",
            "Malformed, failed, or timed-out output remains unknown after one repair attempt.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _extract_frames(
    root: Path,
    output_dir: Path,
    jobs: tuple[QwenEventJob, ...],
) -> tuple[dict[str, tuple[Image.Image, ...]], dict[str, Any]]:
    wanted: dict[str, set[int]] = {"camera_a": set(), "camera_b": set()}
    source_by_camera: dict[str, tuple[Path, str]] = {}
    for job in jobs:
        for frame_input in job.frame_inputs:
            wanted[frame_input.camera_id].add(frame_input.source_frame_index)
            source_by_camera[frame_input.camera_id] = (
                _resolve(root, Path(frame_input.source_video_ref)),
                frame_input.source_video_sha256,
            )
    decoded: dict[tuple[str, int], tuple[Image.Image, float]] = {}
    for camera_id, indices in wanted.items():
        path, expected_hash = source_by_camera[camera_id]
        if _sha256(path) != expected_hash:
            raise ValueError(f"{camera_id} video hash differs before frame extraction")
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for index, frame in enumerate(container.decode(stream)):
                if index in indices:
                    timestamp = float(frame.time) if frame.time is not None else index / 30.0
                    image = frame.to_image().convert("RGB")  # type: ignore[no-untyped-call]
                    image.thumbnail(
                        (
                            INFERENCE_MAX_DIMENSION_PIXELS,
                            INFERENCE_MAX_DIMENSION_PIXELS,
                        ),
                        Image.Resampling.LANCZOS,
                    )
                    decoded[(camera_id, index)] = (image, timestamp)
                if len([key for key in decoded if key[0] == camera_id]) == len(indices):
                    break
        missing = indices.difference(index for cam, index in decoded if cam == camera_id)
        if missing:
            raise ValueError(f"missing decoded {camera_id} frames: {sorted(missing)}")

    frames_dir = output_dir / "frames"
    frames_dir.mkdir()
    images_by_job: dict[str, tuple[Image.Image, ...]] = {}
    records: list[dict[str, Any]] = []
    for job in jobs:
        job_images: list[Image.Image] = []
        for item in job.frame_inputs:
            image, decoded_time = decoded[(item.camera_id, item.source_frame_index)]
            if abs(decoded_time - item.capture_timestamp_seconds) > 0.05:
                raise ValueError("decoded frame timestamp differs from planned capture time")
            filename = (
                f"{job.event_kind.value}_{item.sequence_index:02d}_"
                f"{item.sample_role.value}_{item.camera_id}_f{item.source_frame_index}.jpg"
            )
            path = frames_dir / filename
            image.save(path, format="JPEG", quality=92, subsampling=0)
            records.append(
                {
                    "job_id": job.job_id,
                    "deduplication_key": job.deduplication_key,
                    "event_kind": job.event_kind.value,
                    "sequence_index": item.sequence_index,
                    "sample_role": item.sample_role.value,
                    "camera_id": item.camera_id,
                    "source_frame_index": item.source_frame_index,
                    "planned_capture_timestamp_seconds": item.capture_timestamp_seconds,
                    "decoded_timestamp_seconds": decoded_time,
                    "artifact_ref": _relative(path, root),
                    "artifact_sha256": _sha256(path),
                    "width": image.width,
                    "height": image.height,
                }
            )
            job_images.append(image)
        images_by_job[job.deduplication_key] = tuple(job_images)
    return images_by_job, {
        "schema_version": 1,
        "stage": "S05",
        "inference_max_dimension_pixels": INFERENCE_MAX_DIMENSION_PIXELS,
        "frames": records,
    }


def _write_contact_sheet(
    path: Path,
    jobs: tuple[QwenEventJob, ...],
    images_by_job: dict[str, tuple[Image.Image, ...]],
) -> None:
    tile_width, tile_height, label_height = 320, 180, 30
    sheet = Image.new("RGB", (tile_width * 6, (tile_height + label_height) * 3), "white")
    draw = ImageDraw.Draw(sheet)
    for row, job in enumerate(jobs):
        for column, (frame_input, image) in enumerate(
            zip(job.frame_inputs, images_by_job[job.deduplication_key], strict=True)
        ):
            tile = image.copy()
            tile.thumbnail((tile_width, tile_height))
            x = column * tile_width + (tile_width - tile.width) // 2
            y = row * (tile_height + label_height)
            sheet.paste(tile, (x, y))
            label = (
                f"{job.event_kind.value} | {frame_input.sample_role.value} | "
                f"{frame_input.camera_id[-1].upper()} | f{frame_input.source_frame_index}"
            )
            draw.text((column * tile_width + 5, y + tile_height + 7), label, fill="black")
    sheet.save(path, format="JPEG", quality=92)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
