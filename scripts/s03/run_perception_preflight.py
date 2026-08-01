"""Run deterministic two-camera YOLO preflight on accepted S03 action frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from spatial_reconstruction.config import load_project_config
from spatial_reconstruction.contracts import SynchronizedFrameBundle
from spatial_reconstruction.geometry import select_bundles_for_target_times
from spatial_reconstruction.ingestion import FileFrameSource, build_synchronized_bundles
from spatial_reconstruction.models import YOLOSegAdapter, normalize_yolo_result
from spatial_reconstruction.runtime import select_device

CAMERA_IDS = ("camera_a", "camera_b")
DEFAULT_TARGET_TIMES = (1.0, 7.0, 13.0, 19.0, 25.0, 31.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--synchronization-manifest",
        type=Path,
        default=Path(
            "artifacts/s01/action_take_01/synchronized/synchronization_manifest.json"
        ),
    )
    parser.add_argument(
        "--replay-evidence",
        type=Path,
        default=Path("artifacts/s01/ingestion/action_take_01_frame_bundle_replay.json"),
    )
    parser.add_argument(
        "--target-time-seconds",
        type=float,
        nargs="+",
        default=list(DEFAULT_TARGET_TIMES),
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        help="Explicit diagnostic detection floor; defaults to project configuration.",
    )
    parser.add_argument(
        "--purpose",
        default="representative_person_backpack_yolo_preflight",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    config = load_project_config(project_root / "configs/default.yaml")
    confidence_threshold = (
        config.perception.detection_confidence_threshold
        if args.confidence_threshold is None
        else float(args.confidence_threshold)
    )
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence threshold must be within [0, 1]")
    manifest_path = (project_root / args.synchronization_manifest).resolve()
    replay_path = (project_root / args.replay_evidence).resolve()
    manifest = _read_json(manifest_path)
    replay = _read_json(replay_path)
    manifest_ref = str(manifest_path.relative_to(project_root))
    manifest_sha256 = _sha256(manifest_path)
    recorded_manifest_hash = str(
        replay["input_provenance"]["synchronization_manifest_sha256"]
    )
    if manifest_sha256 != recorded_manifest_hash:
        raise ValueError("synchronization manifest differs from accepted S01 evidence")

    pose_version_id = str(replay["pose_version_id"])
    sources = _build_sources(
        project_root=project_root,
        manifest=manifest,
        manifest_ref=manifest_ref,
        manifest_sha256=manifest_sha256,
        pose_version_id=pose_version_id,
    )
    identities = {
        camera_id: tuple(source.iter_identities())
        for camera_id, source in sources.items()
    }
    bundles = tuple(
        build_synchronized_bundles(
            identities,
            expected_camera_ids=CAMERA_IDS,
            reference_camera_id="camera_a",
            pairing_tolerance_seconds=float(replay["pairing_policy"]["pairing_tolerance_seconds"]),
        )
    )
    _validate_replay(bundles, replay)

    target_times = tuple(float(value) for value in args.target_time_seconds)
    selected = select_bundles_for_target_times(
        bundles,
        target_times_seconds=target_times,
        accepted_start_seconds=0.0,
        accepted_end_seconds=float(bundles[-1].capture_timestamp_seconds),
        maximum_target_error_seconds=1.0 / 30.0,
    )
    pixels = _load_selected_pixels(sources=sources, bundles=selected)

    selection = select_device(
        config.runtime.preferred_device,
        allow_cpu_fallback=config.runtime.allow_cpu_fallback,
    )
    if selection.actual != "mps":
        raise RuntimeError("S03 entry preflight requires native Apple MPS")
    model_load_started = time.perf_counter()
    adapter = YOLOSegAdapter.from_pretrained(
        model_id=config.models.yolo,
        cache_dir=project_root / ".cache/models",
    )
    model_load_seconds = time.perf_counter() - model_load_started

    sample_records: list[dict[str, Any]] = []
    total_class_counts: Counter[str] = Counter()
    target_counts_by_camera = {
        camera_id: Counter({name: 0 for name in config.perception.target_classes})
        for camera_id in CAMERA_IDS
    }
    preview_items: list[tuple[float, str, Image.Image]] = []
    inference_seconds: list[float] = []

    for bundle in selected:
        for frame_identity in bundle.frames:
            camera_id = frame_identity.camera_id
            image_bgr = pixels[frame_identity.frame_id]
            image_rgb = image_bgr[..., ::-1].copy()
            stem = f"bundle_{bundle.bundle_index:04d}_{camera_id}"
            mask_path = output_dir / f"{stem}_masks.npz"
            detection_path = output_dir / f"{stem}_detections.json"
            preview_path = output_dir / f"{stem}_annotated.jpg"

            started = time.perf_counter()
            vendor_result = adapter.predict(
                image_rgb=image_rgb,
                device="mps",
                image_size=config.perception.inference_image_size,
                confidence_threshold=confidence_threshold,
            )
            normalized = normalize_yolo_result(
                vendor_result,
                frame=frame_identity.as_frame_ref(),
                mask_artifact_ref=str(mask_path.relative_to(project_root)),
            )
            elapsed = time.perf_counter() - started
            inference_seconds.append(elapsed)

            np.savez_compressed(
                mask_path,
                source_sized_masks=normalized.masks,
                raw_masks=normalized.raw_masks,
                raw_boxes_xyxy=normalized.raw_boxes_xyxy,
                raw_class_ids=normalized.raw_class_ids,
                raw_confidence=normalized.raw_confidence,
                raw_track_ids=normalized.raw_track_ids,
            )
            detections = [item.model_dump(mode="json") for item in normalized.detections]
            detection_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "bundle_id": bundle.bundle_id,
                        "frame_identity": frame_identity.model_dump(mode="json"),
                        "detections": detections,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            preview = Image.fromarray(normalized.annotated_rgb, mode="RGB")
            preview.save(preview_path, quality=92)
            preview_items.append((bundle.capture_timestamp_seconds, camera_id, preview))

            class_counts = Counter(item.class_name for item in normalized.detections)
            total_class_counts.update(class_counts)
            for target_name in config.perception.target_classes:
                target_counts_by_camera[camera_id][target_name] += class_counts[target_name]
            sample_records.append(
                {
                    "bundle_index": bundle.bundle_index,
                    "bundle_id": bundle.bundle_id,
                    "capture_timestamp_seconds": bundle.capture_timestamp_seconds,
                    "camera_id": camera_id,
                    "frame_id": frame_identity.frame_id,
                    "source_frame_index": frame_identity.source_frame_index,
                    "detection_count": len(normalized.detections),
                    "class_counts": dict(sorted(class_counts.items())),
                    "target_class_counts": {
                        name: class_counts[name]
                        for name in config.perception.target_classes
                    },
                    "inference_seconds": elapsed,
                    "native_speed_ms": normalized.speed_ms,
                    "artifacts": {
                        "detections": str(detection_path.relative_to(project_root)),
                        "masks": str(mask_path.relative_to(project_root)),
                        "annotated_preview": str(preview_path.relative_to(project_root)),
                    },
                }
            )

    contact_sheet_path = output_dir / "annotated_pair_contact_sheet.jpg"
    _write_contact_sheet(preview_items, contact_sheet_path)
    source_hashes_after = {
        camera_id: _sha256(source_path)
        for camera_id, source_path in _source_paths(project_root, manifest).items()
    }
    summary = {
        "schema_version": 1,
        "stage": "S03",
        "purpose": str(args.purpose),
        "capture_session_id": manifest["capture_session_id"],
        "pose_version_id": pose_version_id,
        "synchronization_manifest_ref": manifest_ref,
        "synchronization_manifest_sha256": manifest_sha256,
        "model": {
            "model_id": config.models.yolo,
            "weight_sha256": adapter.weight_sha256,
            "device": selection.actual,
            "precision": adapter.model_precision,
            "inference_image_size": config.perception.inference_image_size,
            "confidence_threshold": confidence_threshold,
            "tracking_enabled": False,
        },
        "selection": {
            "target_times_seconds": target_times,
            "selected_bundles": [bundle.model_dump(mode="json") for bundle in selected],
        },
        "source_sha256_after": source_hashes_after,
        "target_classes": list(config.perception.target_classes),
        "total_class_counts": dict(sorted(total_class_counts.items())),
        "target_counts_by_camera": {
            camera_id: dict(counts) for camera_id, counts in target_counts_by_camera.items()
        },
        "samples": sample_records,
        "timings": {
            "model_load_seconds": model_load_seconds,
            "inference_count": len(inference_seconds),
            "inference_total_seconds": sum(inference_seconds),
            "inference_median_seconds": float(np.median(inference_seconds)),
            "inference_max_seconds": max(inference_seconds),
        },
        "artifacts": {
            "contact_sheet": str(contact_sheet_path.relative_to(project_root)),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


def _build_sources(
    *,
    project_root: Path,
    manifest: dict[str, Any],
    manifest_ref: str,
    manifest_sha256: str,
    pose_version_id: str,
) -> dict[str, FileFrameSource]:
    result: dict[str, FileFrameSource] = {}
    for camera_id, path in _source_paths(project_root, manifest).items():
        record = manifest["derived_outputs"][camera_id]
        result[camera_id] = FileFrameSource(
            path=path,
            capture_session_id=str(manifest["capture_session_id"]),
            camera_id=camera_id,
            source_ref=str(record["path"]),
            expected_sha256=str(record["sha256"]),
            synchronization_manifest_ref=manifest_ref,
            synchronization_manifest_sha256=manifest_sha256,
            pose_version_id=pose_version_id,
            expected_width=1920,
            expected_height=1080,
        )
    return result


def _source_paths(project_root: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    return {
        camera_id: (project_root / manifest["derived_outputs"][camera_id]["path"]).resolve()
        for camera_id in CAMERA_IDS
    }


def _validate_replay(
    bundles: tuple[SynchronizedFrameBundle, ...], replay: dict[str, Any]
) -> None:
    expected = replay["replay_results"]
    if len(bundles) != int(expected["bundle_count"]):
        raise ValueError("replayed bundle count differs from accepted S01 evidence")
    ordered_payload = ("\n".join(bundle.bundle_id for bundle in bundles) + "\n").encode()
    digest = hashlib.sha256(ordered_payload).hexdigest()
    if digest != expected["ordered_bundle_id_sha256"]:
        raise ValueError("replayed bundle identity/order differs from accepted S01 evidence")
    if any(bundle.missing_camera_ids for bundle in bundles):
        raise ValueError("S03 preflight requires complete two-camera bundles")


def _load_selected_pixels(
    *,
    sources: dict[str, FileFrameSource],
    bundles: tuple[SynchronizedFrameBundle, ...],
) -> dict[str, np.ndarray[Any, np.dtype[np.uint8]]]:
    requested = {frame.frame_id for bundle in bundles for frame in bundle.frames}
    pixels: dict[str, np.ndarray[Any, np.dtype[np.uint8]]] = {}
    for source in sources.values():
        for decoded in source.iter_frames():
            if decoded.identity.frame_id in requested:
                pixels[decoded.identity.frame_id] = decoded.image_bgr
    if set(pixels) != requested:
        raise RuntimeError("failed to decode every selected immutable frame")
    return pixels


def _write_contact_sheet(
    items: list[tuple[float, str, Image.Image]], output_path: Path
) -> None:
    width, height = 640, 360
    rows = len(items) // len(CAMERA_IDS)
    canvas = Image.new("RGB", (width * 2, (height + 26) * rows), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (timestamp, camera_id, image) in enumerate(items):
        row, column = divmod(index, 2)
        resized = image.resize((width, height), Image.Resampling.LANCZOS)
        x, y = column * width, row * (height + 26)
        canvas.paste(resized, (x, y + 26))
        draw.text((x + 8, y + 6), f"t={timestamp:.3f}s | {camera_id}", fill="white")
    canvas.save(output_path, quality=92)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
