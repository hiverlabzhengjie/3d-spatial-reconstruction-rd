"""Run the isolated S00 WP6 YOLOv8n-seg representative-image MPS smoke check."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from spatial_reconstruction.config import PROJECT_ROOT, load_project_config

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
sys.dont_write_bytecode = True

from spatial_reconstruction.contracts import (  # noqa: E402
    FrameRef,
    MemoryObservation,
    ModelRunObservation,
    RunOutcome,
    TimingObservation,
)
from spatial_reconstruction.models import (  # noqa: E402
    NormalizedYOLOResult,
    YOLOSegAdapter,
    load_first_image_rgb,
    normalize_yolo_result,
)
from spatial_reconstruction.runtime import (  # noqa: E402
    DeviceSelection,
    DeviceUnavailableError,
    PeakRSSMonitor,
    PhaseTimer,
    SystemMemorySource,
    build_run_observation,
    sample_memory,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Artifact directory; defaults to a new timestamped S00 YOLO run.",
    )
    return parser.parse_args()


def main() -> int:
    """Run cold/warm segmentation and persist model-gate evidence."""

    args = parse_args()
    config = load_project_config()
    image_path = args.image.resolve()
    output_dir = args.output_dir or _timestamped_output_dir(config.paths.artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "summary.json"
    failure_path = output_dir / "failure.txt"

    selection: DeviceSelection | None = None
    adapter: YOLOSegAdapter | None = None
    weight_sha256: str | None = None
    timings: list[TimingObservation] = []
    memory: list[MemoryObservation] = []
    warnings: list[str] = []
    artifact_refs: list[str] = [str(summary_path)]
    source = SystemMemorySource()

    try:
        input_sha256_before = hashlib.sha256(image_path.read_bytes()).hexdigest()
        loaded_image = load_first_image_rgb(image_path)
        height, width = loaded_image.rgb.shape[:2]
        frame = FrameRef(
            camera_id="wp6_representative_view",
            frame_index=0,
            timestamp_seconds=0.0,
            source_ref=str(image_path),
            image_width=width,
            image_height=height,
        )

        selection = select_device(
            config.runtime.preferred_device,
            allow_cpu_fallback=False,
        )
        if selection.actual != "mps":
            raise RuntimeError("WP6 requires an actual MPS run")

        memory.append(sample_memory("before_model_load", device="mps", source=source))
        load_monitor = PeakRSSMonitor(source=source)
        load_timer = PhaseTimer(phase="model_load", device="cpu")
        with load_monitor, load_timer:
            adapter = YOLOSegAdapter.from_pretrained(
                model_id=config.models.yolo,
                cache_dir=PROJECT_ROOT / ".cache" / "models",
            )
        _append_timing(timings, load_timer)
        weight_sha256 = adapter.weight_sha256
        memory.append(
            sample_memory(
                "after_model_load",
                device="mps",
                source=source,
                process_peak_rss_bytes=load_monitor.peak_rss_bytes,
            )
        )

        normalized: NormalizedYOLOResult | None = None
        for phase in ("cold_inference", "warm_inference_1", "warm_inference_2"):
            normalized = _timed_prediction(
                adapter=adapter,
                image_rgb=loaded_image.rgb,
                frame=frame,
                phase=phase,
                image_size=config.perception.inference_image_size,
                confidence_threshold=config.perception.detection_confidence_threshold,
                mask_artifact_ref=str(output_dir / "masks.npz"),
                timings=timings,
                memory=memory,
                source=source,
            )
        if normalized is None:
            raise RuntimeError("YOLO smoke did not produce a normalized result")
        actual_precision = adapter.model_precision
        if actual_precision != "float32":
            raise RuntimeError(f"unexpected YOLO model precision: {actual_precision}")

        input_sha256_after = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if input_sha256_after != input_sha256_before:
            raise RuntimeError("representative source image changed during WP6")

        masks_path = output_dir / "masks.npz"
        detections_path = output_dir / "detections.json"
        preview_path = output_dir / "annotated_preview.jpg"
        manifest_path = output_dir / "manifest.json"
        artifact_refs.extend(
            str(path) for path in (masks_path, detections_path, preview_path, manifest_path)
        )

        np.savez_compressed(
            masks_path,
            source_sized_masks=normalized.masks,
            raw_masks=normalized.raw_masks,
            raw_boxes_xyxy=normalized.raw_boxes_xyxy,
            raw_class_ids=normalized.raw_class_ids,
            raw_confidence=normalized.raw_confidence,
        )
        detection_payload = {
            "frame": normalized.frame.model_dump(mode="json"),
            "detections": [
                detection.model_dump(mode="json") for detection in normalized.detections
            ],
        }
        detections_path.write_text(
            f"{json.dumps(detection_payload, indent=2)}\n",
            encoding="utf-8",
        )
        Image.fromarray(normalized.annotated_rgb, mode="RGB").save(
            preview_path,
            quality=92,
        )

        class_counts: dict[str, int] = {}
        for detection in normalized.detections:
            class_counts[detection.class_name] = class_counts.get(detection.class_name, 0) + 1
        target_counts = {
            name: class_counts.get(name, 0) for name in config.perception.target_classes
        }
        if not any(target_counts.values()):
            warnings.append(
                "no configured target class was detected; the valid empty-target result "
                "does not fail the S00 structural gate"
            )

        manifest = {
            "model_id": config.models.yolo,
            "weight_sha256": weight_sha256,
            "ultralytics_version": _ultralytics_version(),
            "input": {
                "path": str(image_path),
                "sha256_before": input_sha256_before,
                "sha256_after": input_sha256_after,
                "format": loaded_image.format,
                "embedded_frame_count": loaded_image.embedded_frame_count,
                "embedded_frame_used": 0,
                "exif_orientation": loaded_image.exif_orientation,
                "width": width,
                "height": height,
            },
            "task": "segment",
            "tracking_enabled": False,
            "device": selection.actual,
            "precision": actual_precision,
            "inference_image_size": config.perception.inference_image_size,
            "confidence_threshold": config.perception.detection_confidence_threshold,
            "detection_count": len(normalized.detections),
            "class_counts": class_counts,
            "target_class_counts": target_counts,
            "mask_shape": list(normalized.masks.shape),
            "native_speed_ms_last_run": normalized.speed_ms,
            "warnings": warnings,
        }
        manifest_path.write_text(f"{json.dumps(manifest, indent=2)}\n", encoding="utf-8")

        summary = build_run_observation(
            model_id=config.models.yolo,
            model_revision=weight_sha256,
            selection=selection,
            precision="float32",
            input_description=(
                "first display frame of the user-supplied 4032x3024 living-room MPO/JPEG"
            ),
            timings=timings,
            memory=memory,
            outcome=RunOutcome.PASSED,
            warnings=warnings,
            artifact_refs=artifact_refs,
        )
        _write_summary(summary, summary_path)
        print(summary.model_dump_json(indent=2))
        print(f"Artifacts: {output_dir}")
        return 0
    except Exception as exc:
        failure_path.write_text(traceback.format_exc(), encoding="utf-8")
        artifact_refs.append(str(failure_path))
        warnings.append(f"{type(exc).__name__}: {exc}")
        if selection is None and isinstance(exc, DeviceUnavailableError):
            summary = build_run_observation(
                model_id=config.models.yolo,
                model_revision=weight_sha256,
                selection=None,
                requested_device=config.runtime.preferred_device,
                device_selection_error=str(exc),
                precision="unknown",
                input_description="YOLO WP6 preflight failed before device selection",
                timings=timings,
                memory=memory,
                outcome=RunOutcome.FAILED,
                warnings=warnings,
                artifact_refs=artifact_refs,
            )
        else:
            summary = build_run_observation(
                model_id=config.models.yolo,
                model_revision=weight_sha256,
                selection=selection,
                requested_device=(config.runtime.preferred_device if selection is None else None),
                device_selection_error=str(exc) if selection is None else None,
                precision="float32" if adapter is not None else "unknown",
                input_description="YOLO WP6 representative-image smoke attempt",
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


def _timed_prediction(
    *,
    adapter: YOLOSegAdapter,
    image_rgb: NDArray[np.uint8],
    frame: FrameRef,
    phase: str,
    image_size: int,
    confidence_threshold: float,
    mask_artifact_ref: str,
    timings: list[TimingObservation],
    memory: list[MemoryObservation],
    source: SystemMemorySource,
) -> NormalizedYOLOResult:
    """Retain timing and peak memory evidence even when prediction fails."""

    monitor = PeakRSSMonitor(source=source)
    timer = PhaseTimer(phase=phase, device="mps")
    try:
        with monitor, timer:
            result = adapter.predict(
                image_rgb=image_rgb,
                device="mps",
                image_size=image_size,
                confidence_threshold=confidence_threshold,
            )
        return normalize_yolo_result(
            result,
            frame=frame,
            mask_artifact_ref=mask_artifact_ref,
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


def _append_timing(timings: list[TimingObservation], timer: PhaseTimer) -> None:
    if timer.observation is None:
        raise RuntimeError(f"timer '{timer.phase}' did not emit an observation")
    timings.append(timer.observation)


def _ultralytics_version() -> str:
    import ultralytics

    return str(ultralytics.__version__)


def _timestamped_output_dir(artifacts_dir: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return artifacts_dir / "s00" / "yolo" / f"wp6_{timestamp}"


def _write_summary(summary: ModelRunObservation, path: Path) -> None:
    path.write_text(f"{summary.model_dump_json(indent=2)}\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
