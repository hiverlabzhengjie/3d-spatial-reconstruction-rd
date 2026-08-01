"""Run the real D028 tracker through bounded deterministic S03 queues."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from spatial_reconstruction.config import load_project_config
from spatial_reconstruction.contracts import SynchronizedFrameBundle
from spatial_reconstruction.ingestion import FileFrameSource, build_synchronized_bundles
from spatial_reconstruction.models import YOLOSegAdapter
from spatial_reconstruction.perception import (
    BoundedPerceptionQueue,
    PerceptionFrameResult,
    PerceptionJob,
    PerceptionWorkItem,
    QueueOverflowPolicy,
    YOLOByteTrackProcessor,
    process_next_perception_item,
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
    parser.add_argument("--frame-stride", type=int, default=6)
    parser.add_argument("--queue-capacity", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frame_stride <= 0 or args.queue_capacity <= 0:
        raise ValueError("frame stride and queue capacity must be positive")
    if args.start_seconds < 0 or args.end_seconds <= args.start_seconds:
        raise ValueError("replay interval must have positive non-negative extent")
    project_root = args.project_root.resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    config = load_project_config(
        project_root / "configs/default.yaml", project_root=project_root
    )
    device = select_device(
        config.runtime.preferred_device,
        allow_cpu_fallback=config.runtime.allow_cpu_fallback,
    )
    if device.actual != "mps":
        raise RuntimeError("bounded S03 replay requires native Apple MPS")

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
    selected = tuple(
        bundle
        for bundle in bundles
        if args.start_seconds <= bundle.capture_timestamp_seconds <= args.end_seconds
        and bundle.bundle_index % args.frame_stride == 0
    )
    selected_by_frame_id = {
        frame.frame_id: bundle for bundle in selected for frame in bundle.frames
    }
    if not selected:
        raise ValueError("bounded replay selected no bundles")

    camera_summaries: dict[str, Any] = {}
    checkpoint_sha256: str | None = None
    for camera_id in CAMERA_IDS:
        camera_dir = output_dir / camera_id
        adapter = YOLOSegAdapter.from_pretrained(
            model_id=config.models.yolo,
            cache_dir=project_root / ".cache/models",
        )
        if checkpoint_sha256 is None:
            checkpoint_sha256 = adapter.weight_sha256
        elif adapter.weight_sha256 != checkpoint_sha256:
            raise RuntimeError("camera workers loaded different checkpoint bytes")
        processor = YOLOByteTrackProcessor(
            adapter=adapter,
            project_root=project_root,
            output_dir=camera_dir / "raw",
            device="mps",
            image_size=config.perception.inference_image_size,
            confidence_threshold=config.perception.detection_confidence_threshold,
            tracked_class_ids=TRACKED_CLASS_IDS,
            bag_class_aliases=config.perception.bag_class_aliases,
            excluded_bag_classes=config.perception.excluded_bag_classes,
            policy_id=config.perception.bag_policy_id,
        )
        queue = BoundedPerceptionQueue(
            capacity=args.queue_capacity,
            overflow_policy=QueueOverflowPolicy.THROTTLE,
        )
        submissions: list[dict[str, Any]] = []
        results: list[PerceptionFrameResult] = []
        stable_job_id_checks: list[bool] = []

        for decoded in sources[camera_id].iter_frames():
            bundle = selected_by_frame_id.get(decoded.identity.frame_id)
            if bundle is None:
                continue
            created = time.monotonic()
            job = PerceptionJob.create(
                frame_identity=decoded.identity,
                model_id=config.models.yolo,
                model_revision=adapter.weight_sha256,
                policy_id=config.perception.bag_policy_id,
                created_processing_seconds=created,
            )
            replay_job = PerceptionJob.create(
                frame_identity=decoded.identity,
                model_id=config.models.yolo,
                model_revision=adapter.weight_sha256,
                policy_id=config.perception.bag_policy_id,
                created_processing_seconds=created + 1000.0,
            )
            stable_job_id_checks.append(job.job_id == replay_job.job_id)
            item = PerceptionWorkItem(
                job=job,
                image_rgb=decoded.image_bgr[..., ::-1],
            )
            submission = queue.submit(item)
            submissions.append(submission.model_dump(mode="json"))
            if not submission.accepted:
                result = process_next_perception_item(queue, processor)
                if result is None:
                    raise RuntimeError("throttled queue could not drain one item")
                results.append(result)
                retry = queue.submit(item)
                submissions.append(retry.model_dump(mode="json"))
                if not retry.accepted:
                    raise RuntimeError("offline queue retry remained throttled after drain")

        while queue.diagnostics.current_depth:
            result = process_next_perception_item(queue, processor)
            if result is None:
                raise RuntimeError("non-empty queue returned no work during drain")
            results.append(result)
        diagnostics = queue.diagnostics
        expected_frame_ids = [
            frame.frame_id
            for bundle in selected
            for frame in bundle.frames
            if frame.camera_id == camera_id
        ]
        result_frame_ids = [result.job.frame_identity.frame_id for result in results]
        if result_frame_ids != expected_frame_ids:
            raise RuntimeError("bounded results differ from authoritative capture order")
        if not all(stable_job_id_checks):
            raise RuntimeError("processing creation time changed deterministic job identity")

        result_path = camera_dir / "worker_results.json"
        submission_path = camera_dir / "queue_submissions.json"
        result_path.write_text(
            json.dumps([result.model_dump(mode="json") for result in results], indent=2)
            + "\n",
            encoding="utf-8",
        )
        submission_path.write_text(
            json.dumps(submissions, indent=2) + "\n",
            encoding="utf-8",
        )
        camera_summaries[camera_id] = _summarize_camera(
            results,
            diagnostics=diagnostics.model_dump(mode="json"),
            submission_count=len(submissions),
            stable_job_ids=all(stable_job_id_checks),
            result_order_matches=result_frame_ids == expected_frame_ids,
            result_ref=str(result_path.relative_to(project_root)),
            submission_ref=str(submission_path.relative_to(project_root)),
        )

    summary = {
        "schema_version": 1,
        "stage": "S03",
        "purpose": "bounded_d028_perception_replay",
        "capture_session_id": manifest["capture_session_id"],
        "pose_version_id": replay["pose_version_id"],
        "synchronization_manifest_ref": manifest_ref,
        "synchronization_manifest_sha256": manifest_sha256,
        "model": {
            "model_id": config.models.yolo,
            "weight_sha256": checkpoint_sha256,
            "device": device.actual,
            "confidence_threshold": config.perception.detection_confidence_threshold,
            "image_size": config.perception.inference_image_size,
            "tracker": "bytetrack.yaml",
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
            "nominal_fps_per_camera": 30.0 / args.frame_stride,
            "selected_bundle_count": len(selected),
        },
        "queue": {
            "capacity": args.queue_capacity,
            "overflow_policy": QueueOverflowPolicy.THROTTLE,
        },
        "camera_summaries": camera_summaries,
        "source_sha256_after": {
            camera_id: _sha256(path)
            for camera_id, path in _source_paths(project_root, manifest).items()
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


def _summarize_camera(
    results: list[PerceptionFrameResult],
    *,
    diagnostics: dict[str, Any],
    submission_count: int,
    stable_job_ids: bool,
    result_order_matches: bool,
    result_ref: str,
    submission_ref: str,
) -> dict[str, Any]:
    completed = [result for result in results if result.outcome.value == "completed"]
    failed = [result for result in results if result.outcome.value == "failed"]
    target_candidate_frames = {"person": 0, "backpack": 0}
    empty_candidate_frames = 0
    untracked_candidates = {"person": 0, "backpack": 0}
    for result in completed:
        if not result.candidates:
            empty_candidate_frames += 1
        for target in target_candidate_frames:
            matching = [
                candidate
                for candidate in result.candidates
                if candidate.target.value == target
            ]
            if matching:
                target_candidate_frames[target] += 1
            untracked_candidates[target] += sum(
                candidate.source_detection.camera_local_track_id is None
                for candidate in matching
            )
    queue_wait = [
        float(result.processing_started_seconds - result.job.created_processing_seconds)
        for result in results
    ]
    processing = [
        float(result.processing_finished_seconds - result.processing_started_seconds)
        for result in results
    ]
    return {
        "result_count": len(results),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "empty_candidate_frame_count": empty_candidate_frames,
        "target_candidate_frame_count": target_candidate_frames,
        "untracked_candidate_count": untracked_candidates,
        "stable_job_identity_replay": stable_job_ids,
        "result_order_matches_capture_order": result_order_matches,
        "submission_record_count": submission_count,
        "queue_diagnostics": diagnostics,
        "timings": {
            "queue_wait_median_seconds": float(np.median(queue_wait)),
            "queue_wait_max_seconds": max(queue_wait),
            "processing_median_seconds": float(np.median(processing)),
            "processing_max_seconds": max(processing),
            "processing_total_seconds": sum(processing),
        },
        "worker_results": result_ref,
        "queue_submissions": submission_ref,
    }


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
        raise ValueError("bounded replay requires complete synchronized bundles")


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
