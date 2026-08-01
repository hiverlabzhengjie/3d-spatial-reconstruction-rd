"""Run capture-ordered camera-local ByteTrack smoke checks for S03."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from spatial_reconstruction.config import load_project_config
from spatial_reconstruction.contracts import PerceptionCandidate, SynchronizedFrameBundle
from spatial_reconstruction.ingestion import FileFrameSource, build_synchronized_bundles
from spatial_reconstruction.models import (
    YOLOSegAdapter,
    normalize_yolo_result,
    select_perception_candidates,
)
from spatial_reconstruction.runtime import select_device

CAMERA_IDS = ("camera_a", "camera_b")
TRACKED_CLASS_IDS = (0, 24, 26)


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
    parser.add_argument("--start-seconds", type=float, default=1.0)
    parser.add_argument("--end-seconds", type=float, default=33.0)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=6,
        help="Process every Nth synchronized bundle; 6 is approximately 5 FPS.",
    )
    parser.add_argument("--preview-interval-frames", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = (project_root / args.output_dir).resolve()
    if args.frame_stride <= 0 or args.preview_interval_frames <= 0:
        raise ValueError("frame stride and preview interval must be positive")
    if args.start_seconds < 0 or args.end_seconds <= args.start_seconds:
        raise ValueError("tracking interval must have positive non-negative extent")
    output_dir.mkdir(parents=True, exist_ok=False)

    config = load_project_config(
        project_root / "configs/default.yaml", project_root=project_root
    )
    selection = select_device(
        config.runtime.preferred_device,
        allow_cpu_fallback=config.runtime.allow_cpu_fallback,
    )
    if selection.actual != "mps":
        raise RuntimeError("S03 tracking smoke requires native Apple MPS")

    manifest_path = (project_root / args.synchronization_manifest).resolve()
    replay_path = (project_root / args.replay_evidence).resolve()
    manifest = _read_object(manifest_path)
    replay = _read_object(replay_path)
    manifest_ref = str(manifest_path.relative_to(project_root))
    manifest_sha256 = _sha256(manifest_path)
    if manifest_sha256 != replay["input_provenance"]["synchronization_manifest_sha256"]:
        raise ValueError("synchronization manifest differs from accepted S01 evidence")
    sources = _build_sources(
        project_root=project_root,
        manifest=manifest,
        manifest_ref=manifest_ref,
        manifest_sha256=manifest_sha256,
        pose_version_id=str(replay["pose_version_id"]),
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
    selected_bundles = tuple(
        bundle
        for bundle in bundles
        if args.start_seconds <= bundle.capture_timestamp_seconds <= args.end_seconds
        and bundle.bundle_index % args.frame_stride == 0
    )
    if not selected_bundles:
        raise ValueError("tracking interval selected no synchronized bundles")
    selected_by_frame_id = {
        frame.frame_id: bundle
        for bundle in selected_bundles
        for frame in bundle.frames
    }

    camera_summaries: dict[str, Any] = {}
    weight_sha256: str | None = None
    preview_items: list[tuple[float, str, Image.Image]] = []
    for camera_id in CAMERA_IDS:
        camera_dir = output_dir / camera_id
        camera_dir.mkdir()
        load_started = time.perf_counter()
        adapter = YOLOSegAdapter.from_pretrained(
            model_id=config.models.yolo,
            cache_dir=project_root / ".cache/models",
        )
        load_seconds = time.perf_counter() - load_started
        if weight_sha256 is None:
            weight_sha256 = adapter.weight_sha256
        elif adapter.weight_sha256 != weight_sha256:
            raise RuntimeError("camera trackers loaded different checkpoint bytes")

        frame_records: list[dict[str, Any]] = []
        inference_seconds: list[float] = []
        processed_index = 0
        for decoded in sources[camera_id].iter_frames():
            bundle = selected_by_frame_id.get(decoded.identity.frame_id)
            if bundle is None:
                continue
            frame_ref = decoded.identity.as_frame_ref()
            started = time.perf_counter()
            vendor_result = adapter.track(
                image_rgb=decoded.image_bgr[..., ::-1].copy(),
                frame=frame_ref,
                device="mps",
                image_size=config.perception.inference_image_size,
                confidence_threshold=config.perception.detection_confidence_threshold,
                class_ids=TRACKED_CLASS_IDS,
            )
            normalized = normalize_yolo_result(
                vendor_result,
                frame=frame_ref,
                mask_artifact_ref=str(
                    (camera_dir / f"frame_{frame_ref.frame_index:04d}_masks.npz").relative_to(
                        project_root
                    )
                ),
                require_track_ids=False,
            )
            elapsed = time.perf_counter() - started
            inference_seconds.append(elapsed)
            candidates = select_perception_candidates(
                normalized,
                bag_class_aliases=config.perception.bag_class_aliases,
                excluded_bag_classes=config.perception.excluded_bag_classes,
                policy_id=config.perception.bag_policy_id,
            )
            mask_path = camera_dir / f"frame_{frame_ref.frame_index:04d}_masks.npz"
            np.savez_compressed(
                mask_path,
                source_sized_masks=normalized.masks,
                raw_masks=normalized.raw_masks,
                raw_boxes_xyxy=normalized.raw_boxes_xyxy,
                raw_class_ids=normalized.raw_class_ids,
                raw_confidence=normalized.raw_confidence,
                raw_track_ids=normalized.raw_track_ids,
            )
            frame_record = {
                "processed_index": processed_index,
                "bundle_id": bundle.bundle_id,
                "bundle_index": bundle.bundle_index,
                "capture_timestamp_seconds": bundle.capture_timestamp_seconds,
                "frame_identity": decoded.identity.model_dump(mode="json"),
                "detections": [
                    detection.model_dump(mode="json")
                    for detection in normalized.detections
                ],
                "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
                "candidate_counts": _candidate_counts(candidates),
                "inference_seconds": elapsed,
                "native_speed_ms": normalized.speed_ms,
                "mask_artifact": str(mask_path.relative_to(project_root)),
            }
            frame_records.append(frame_record)
            if processed_index % args.preview_interval_frames == 0:
                preview_path = camera_dir / f"frame_{frame_ref.frame_index:04d}_annotated.jpg"
                preview = Image.fromarray(normalized.annotated_rgb, mode="RGB")
                preview.save(preview_path, quality=92)
                preview_items.append((bundle.capture_timestamp_seconds, camera_id, preview))
                frame_record["annotated_preview"] = str(
                    preview_path.relative_to(project_root)
                )
            processed_index += 1

        expected_camera_frames = sum(
            1
            for bundle in selected_bundles
            for frame in bundle.frames
            if frame.camera_id == camera_id
        )
        if len(frame_records) != expected_camera_frames:
            raise RuntimeError(f"{camera_id} did not process every selected frame")
        results_path = camera_dir / "frame_results.json"
        results_path.write_text(json.dumps(frame_records, indent=2) + "\n", encoding="utf-8")
        camera_summaries[camera_id] = _summarize_camera(
            frame_records,
            model_load_seconds=load_seconds,
            inference_seconds=inference_seconds,
            results_ref=str(results_path.relative_to(project_root)),
        )

    contact_sheet_path = output_dir / "tracking_preview_contact_sheet.jpg"
    _write_contact_sheet(preview_items, contact_sheet_path)
    summary = {
        "schema_version": 1,
        "stage": "S03",
        "purpose": "camera_local_bytetrack_smoke",
        "capture_session_id": manifest["capture_session_id"],
        "pose_version_id": replay["pose_version_id"],
        "synchronization_manifest_ref": manifest_ref,
        "synchronization_manifest_sha256": manifest_sha256,
        "model": {
            "model_id": config.models.yolo,
            "weight_sha256": weight_sha256,
            "device": selection.actual,
            "precision": "float32",
            "image_size": config.perception.inference_image_size,
            "confidence_threshold": config.perception.detection_confidence_threshold,
            "tracker": "bytetrack.yaml",
            "tracked_class_ids": TRACKED_CLASS_IDS,
        },
        "policy": {
            "policy_id": config.perception.bag_policy_id,
            "bag_class_aliases": config.perception.bag_class_aliases,
            "excluded_bag_classes": config.perception.excluded_bag_classes,
        },
        "sampling": {
            "start_seconds": args.start_seconds,
            "end_seconds": args.end_seconds,
            "frame_stride": args.frame_stride,
            "nominal_processed_fps": 30.0 / args.frame_stride,
            "selected_bundle_count": len(selected_bundles),
            "first_bundle_index": selected_bundles[0].bundle_index,
            "last_bundle_index": selected_bundles[-1].bundle_index,
        },
        "camera_summaries": camera_summaries,
        "source_sha256_after": {
            camera_id: _sha256(path)
            for camera_id, path in _source_paths(project_root, manifest).items()
        },
        "artifacts": {
            "contact_sheet": str(contact_sheet_path.relative_to(project_root)),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


def _candidate_counts(candidates: tuple[PerceptionCandidate, ...]) -> dict[str, int]:
    return {
        target: sum(candidate.target.value == target for candidate in candidates)
        for target in ("person", "backpack")
    }


def _summarize_camera(
    frame_records: list[dict[str, Any]],
    *,
    model_load_seconds: float,
    inference_seconds: list[float],
    results_ref: str,
) -> dict[str, Any]:
    tracks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_counts = {"person": 0, "backpack": 0}
    untracked_candidate_counts = {"person": 0, "backpack": 0}
    ambiguous_backpack_frames: list[int] = []
    for record in frame_records:
        counts = record["candidate_counts"]
        for target in missing_counts:
            if counts[target] == 0:
                missing_counts[target] += 1
        if counts["backpack"] > 1:
            ambiguous_backpack_frames.append(int(record["bundle_index"]))
        for candidate in record["candidates"]:
            detection = candidate["source_detection"]
            track_id = detection["camera_local_track_id"]
            if track_id is None:
                untracked_candidate_counts[str(candidate["target"])] += 1
                continue
            tracks[str(track_id)].append(
                {
                    "processed_index": record["processed_index"],
                    "bundle_index": record["bundle_index"],
                    "time_seconds": record["capture_timestamp_seconds"],
                    "target": candidate["target"],
                    "vendor_class": detection["class_name"],
                    "confidence": detection["confidence"],
                }
            )
    track_summaries: dict[str, Any] = {}
    for track_id, observations in sorted(tracks.items()):
        ordered = sorted(observations, key=lambda item: item["processed_index"])
        indices = [int(item["processed_index"]) for item in ordered]
        track_summaries[track_id] = {
            "target_values": sorted({item["target"] for item in ordered}),
            "vendor_classes": sorted({item["vendor_class"] for item in ordered}),
            "observation_count": len(ordered),
            "first_time_seconds": ordered[0]["time_seconds"],
            "last_time_seconds": ordered[-1]["time_seconds"],
            "longest_consecutive_processed_run": _longest_consecutive_run(indices),
            "gap_count": sum(
                current != previous + 1
                for previous, current in zip(indices, indices[1:], strict=False)
            ),
        }
    return {
        "processed_frame_count": len(frame_records),
        "missing_candidate_frame_count": missing_counts,
        "untracked_candidate_count": untracked_candidate_counts,
        "ambiguous_backpack_bundle_indices": ambiguous_backpack_frames,
        "unique_track_count": len(track_summaries),
        "tracks": track_summaries,
        "timings": {
            "model_load_seconds": model_load_seconds,
            "inference_total_seconds": sum(inference_seconds),
            "inference_median_seconds": float(np.median(inference_seconds)),
            "inference_max_seconds": max(inference_seconds),
        },
        "frame_results": results_ref,
    }


def _longest_consecutive_run(indices: list[int]) -> int:
    if not indices:
        return 0
    longest = current = 1
    for previous, value in zip(indices, indices[1:], strict=False):
        current = current + 1 if value == previous + 1 else 1
        longest = max(longest, current)
    return longest


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
    payload = ("\n".join(bundle.bundle_id for bundle in bundles) + "\n").encode()
    if hashlib.sha256(payload).hexdigest() != expected["ordered_bundle_id_sha256"]:
        raise ValueError("replayed bundle order differs from accepted S01 evidence")
    if any(bundle.missing_camera_ids for bundle in bundles):
        raise ValueError("tracking smoke requires complete synchronized bundles")


def _write_contact_sheet(
    items: list[tuple[float, str, Image.Image]], output_path: Path
) -> None:
    grouped: dict[int, dict[str, tuple[float, Image.Image]]] = defaultdict(dict)
    for timestamp, camera_id, image in items:
        grouped[round(timestamp)][camera_id] = (timestamp, image)
    complete = [grouped[key] for key in sorted(grouped) if set(grouped[key]) == set(CAMERA_IDS)]
    width, height = 640, 360
    canvas = Image.new("RGB", (width * 2, (height + 26) * len(complete)), "black")
    draw = ImageDraw.Draw(canvas)
    for row, pair in enumerate(complete):
        for column, camera_id in enumerate(CAMERA_IDS):
            timestamp, image = pair[camera_id]
            resized = image.resize((width, height), Image.Resampling.LANCZOS)
            x, y = column * width, row * (height + 26)
            canvas.paste(resized, (x, y + 26))
            draw.text((x + 8, y + 6), f"t={timestamp:.3f}s | {camera_id}", fill="white")
    canvas.save(output_path, quality=92)


def _read_object(path: Path) -> dict[str, Any]:
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
