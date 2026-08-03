"""Run raw pose-conditioned DA3 depth on deterministic S04 action keyframes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import cv2
import matplotlib
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from spatial_reconstruction.config import load_project_config
from spatial_reconstruction.contracts import (
    CameraIntrinsics,
    CameraPose,
    SynchronizedFrameBundle,
)
from spatial_reconstruction.ingestion import (
    FileFrameSource,
    TimestampTransform,
    build_synchronized_bundles,
)
from spatial_reconstruction.localization import (
    ActionDepthKeyframeSpec,
    ActionDepthRunSummary,
    select_action_depth_jobs,
)
from spatial_reconstruction.models import (
    EXPECTED_DA3_VENDOR_FINGERPRINT,
    DA3Adapter,
    compute_vendor_fingerprint,
)
from spatial_reconstruction.models.da3_mps import DA3Precision
from spatial_reconstruction.perception import PerceptionTargetFrameState
from spatial_reconstruction.runtime import (
    PhaseTimer,
    SystemMemorySource,
    sample_memory,
    select_device,
)

Float32Array = NDArray[np.float32]
UInt8Array = NDArray[np.uint8]
MODEL_REVISION = "b2359bdf726fb44ef62acca04d629dcf158053e7"
CAMERA_IDS = ("camera_a", "camera_b")


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
        "--pose-calibration",
        type=Path,
        default=Path(
            "artifacts/s01/calibration/action_take_01_pose/camera_calibration.json"
        ),
    )
    parser.add_argument(
        "--scene-metadata",
        type=Path,
        default=Path("artifacts/s01/scene_metadata.json"),
    )
    parser.add_argument(
        "--bounded-replay-summary",
        type=Path,
        default=Path("artifacts/s03/bounded_replay_5fps_20260731/summary.json"),
    )
    parser.add_argument(
        "--camera-a-timeline",
        type=Path,
        default=Path(
            "artifacts/s03/target_timeline_5fps_20260801/"
            "camera_a_target_timeline.json"
        ),
    )
    parser.add_argument(
        "--camera-b-timeline",
        type=Path,
        default=Path(
            "artifacts/s03/target_timeline_5fps_20260801/"
            "camera_b_target_timeline.json"
        ),
    )
    parser.add_argument(
        "--selection-config",
        type=Path,
        default=Path("configs/s04_action_keyframes.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--process-resolution", type=int, default=504)
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = _resolve(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    config = load_project_config(project_root=project_root)

    paths = {
        "synchronization_manifest": _resolve(
            project_root, args.synchronization_manifest
        ),
        "replay_evidence": _resolve(project_root, args.replay_evidence),
        "pose_calibration": _resolve(project_root, args.pose_calibration),
        "scene_metadata": _resolve(project_root, args.scene_metadata),
        "bounded_replay_summary": _resolve(
            project_root, args.bounded_replay_summary
        ),
        "camera_a_timeline": _resolve(project_root, args.camera_a_timeline),
        "camera_b_timeline": _resolve(project_root, args.camera_b_timeline),
        "selection_config": _resolve(project_root, args.selection_config),
    }
    payloads = {name: _read_object(path) for name, path in paths.items()}
    _validate_inputs(payloads, paths=paths)

    manifest = payloads["synchronization_manifest"]
    calibration = payloads["pose_calibration"]
    replay = payloads["replay_evidence"]
    manifest_sha256 = _sha256(paths["synchronization_manifest"])
    sources = _build_sources(
        project_root=project_root,
        manifest=manifest,
        manifest_ref=_relative(paths["synchronization_manifest"], project_root),
        manifest_sha256=manifest_sha256,
        pose_version_id=str(calibration["pose_version_id"]),
    )
    bundles = tuple(
        build_synchronized_bundles(
            {
                camera_id: sources[camera_id].iter_identities()
                for camera_id in CAMERA_IDS
            },
            expected_camera_ids=CAMERA_IDS,
            reference_camera_id="camera_a",
            pairing_tolerance_seconds=float(
                cast(dict[str, Any], replay["pairing_policy"])[
                    "pairing_tolerance_seconds"
                ]
            ),
        )
    )
    _validate_replay(bundles, replay)

    states = _load_timeline_states(
        payloads["camera_a_timeline"],
        payloads["camera_b_timeline"],
    )
    selection = payloads["selection_config"]
    specs = tuple(
        ActionDepthKeyframeSpec.model_validate(item)
        for item in cast(list[dict[str, Any]], selection["keyframes"])
    )
    jobs = select_action_depth_jobs(
        bundles=bundles,
        states=states,
        specs=specs,
        model_id=config.models.da3,
        model_revision=MODEL_REVISION,
        process_resolution=int(args.process_resolution),
    )
    _validate_jobs_against_s03(jobs, payloads["bounded_replay_summary"])
    selection_path = output_dir / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "S04",
                "purpose": selection["purpose"],
                "selection_config_ref": _relative(
                    paths["selection_config"], project_root
                ),
                "selection_config_sha256": _sha256(paths["selection_config"]),
                "jobs": [job.model_dump(mode="json") for job in jobs],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    intrinsics, poses = _camera_contracts(calibration)
    keyframes = _extract_undistorted_keyframes(
        output_dir=output_dir,
        sources=sources,
        jobs=jobs,
        camera_intrinsics=intrinsics,
    )
    vendor_fingerprint = compute_vendor_fingerprint(config.paths.da3_vendor_dir)
    if vendor_fingerprint != EXPECTED_DA3_VENDOR_FINGERPRINT:
        raise RuntimeError(
            "DA3 vendor fingerprint changed: "
            f"expected {EXPECTED_DA3_VENDOR_FINGERPRINT}, got {vendor_fingerprint}"
        )
    device = select_device(config.runtime.preferred_device, allow_cpu_fallback=False)
    if device.actual != "mps":
        raise RuntimeError("S04 action-depth preflight requires native Apple MPS")

    memory_source = SystemMemorySource()
    timings: list[dict[str, object]] = []
    memory: list[dict[str, object]] = [
        sample_memory("before_model_load", device="mps", source=memory_source).model_dump(
            mode="json"
        )
    ]
    load_timer = PhaseTimer(phase="model_load", device="mps")
    with load_timer:
        adapter = DA3Adapter.from_pretrained(
            vendor_dir=config.paths.da3_vendor_dir,
            model_id=config.models.da3,
            model_revision=MODEL_REVISION,
            device=torch.device("mps"),
            precision=cast(DA3Precision, str(args.precision)),
        )
    timings.append(_timer_payload(load_timer))
    memory.append(
        sample_memory("after_model_load", device="mps", source=memory_source).model_dump(
            mode="json"
        )
    )

    intrinsics_by_id = {item.camera_id: item for item in intrinsics}
    poses_by_id = {item.camera_id: item for item in poses}
    predictions_dir = output_dir / "predictions"
    previews_dir = output_dir / "previews"
    predictions_dir.mkdir()
    previews_dir.mkdir()
    prediction_records: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        image_paths = tuple(
            keyframes[(job.bundle.bundle_id, camera_id)] for camera_id in CAMERA_IDS
        )
        inference_timer = PhaseTimer(
            phase=f"inference_job_{index:02d}", device="mps"
        )
        with inference_timer:
            output = adapter.infer_pose_conditioned(
                image_paths=image_paths,
                camera_intrinsics=tuple(
                    intrinsics_by_id[camera_id] for camera_id in CAMERA_IDS
                ),
                camera_poses=tuple(poses_by_id[camera_id] for camera_id in CAMERA_IDS),
                process_resolution=job.process_resolution,
        )
        timings.append(_timer_payload(inference_timer))
        memory.append(
            sample_memory(
                f"after_inference_job_{index:02d}",
                device="mps",
                source=memory_source,
            ).model_dump(mode="json")
        )
        _validate_da3_output(output, poses_by_id=poses_by_id)
        processed_images = cast(UInt8Array, output.processed_images)
        raw_path = predictions_dir / f"{index:02d}_{job.job_id[:12]}.npz"
        np.savez_compressed(
            raw_path,
            depth_m=output.depth_m,
            confidence=output.confidence,
            processed_images_rgb=processed_images,
            returned_T_camera_from_world=output.T_camera_from_world,
            returned_intrinsics=output.intrinsics,
            job_id=np.asarray(job.job_id),
            bundle_id=np.asarray(job.bundle.bundle_id),
            capture_timestamp_seconds=np.asarray(
                job.bundle.capture_timestamp_seconds, dtype=np.float64
            ),
            frame_ids=np.asarray([frame.frame_id for frame in job.bundle.frames]),
            camera_ids=np.asarray(CAMERA_IDS),
            model_id=np.asarray(config.models.da3),
            model_revision=np.asarray(MODEL_REVISION),
            process_resolution=np.asarray(job.process_resolution, dtype=np.int32),
            is_metric=np.asarray(output.is_metric),
            depth_scale_applied=np.asarray(1.0, dtype=np.float32),
            s02_corrections_applied=np.asarray(False),
        )
        preview_path = previews_dir / f"{index:02d}_{job.job_id[:12]}_depth.png"
        _save_depth_confidence_preview(
            depth_m=output.depth_m,
            confidence=output.confidence,
            processed_images_rgb=processed_images,
            camera_ids=CAMERA_IDS,
            phase_id=job.phase_id,
            path=preview_path,
        )
        prediction_records.append(
            {
                "job": job.model_dump(mode="json"),
                "raw_prediction_ref": _relative(raw_path, project_root),
                "raw_prediction_sha256": _sha256(raw_path),
                "depth_confidence_preview_ref": _relative(
                    preview_path, project_root
                ),
                "depth_confidence_preview_sha256": _sha256(preview_path),
                "cameras": {
                    camera_id: _camera_prediction_record(
                        camera_index=camera_index,
                        camera_id=camera_id,
                        job=job,
                        image_path=image_paths[camera_index],
                        depth_m=output.depth_m,
                        confidence=output.confidence,
                        processed_intrinsics=output.intrinsics,
                        project_root=project_root,
                    )
                    for camera_index, camera_id in enumerate(CAMERA_IDS)
                },
            }
        )

    result = {
        "schema_version": 1,
        "status": "completed_pending_mask_depth_qa",
        "stage": "S04",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "capture_session_id": manifest["capture_session_id"],
        "pose_version_id": calibration["pose_version_id"],
        "input_provenance": {
            name + "_ref": _relative(path, project_root)
            for name, path in paths.items()
        }
        | {
            name + "_sha256": _sha256(path) for name, path in paths.items()
        }
        | {
            "selection_ref": _relative(selection_path, project_root),
            "selection_sha256": _sha256(selection_path),
            "vendor_fingerprint": vendor_fingerprint,
        },
        "model": {
            "model_id": config.models.da3,
            "model_revision": MODEL_REVISION,
            "device": device.actual,
            "precision": adapter.autocast_policy.reported_precision,
            "is_metric": True,
            "two_view_alignment_policy": (
                "preserve_nested_metric_depth_and_return_supplied_poses"
            ),
        },
        "selection_config_ref": _relative(paths["selection_config"], project_root),
        "selection_config_sha256": _sha256(paths["selection_config"]),
        "processing": {
            "process_resolution": int(args.process_resolution),
            "source_frames_undistorted_before_inference": True,
            "raw_da3_metric_depth_preserved": True,
            "s02_marker_scale_applied": False,
            "s02_static_confidence_policy_applied": False,
            "s02_door_supplement_applied": False,
            "mask_resampling_or_localization_performed": False,
        },
        "predictions": prediction_records,
        "runtime": {
            "platform_machine": platform.machine(),
            "torch_version": torch.__version__,
            "timings": timings,
            "memory": memory,
        },
        "limitations": [
            "This preflight retains raw action-frame depth and confidence only; "
            "no mask-to-depth aggregation or XYZ localization has been performed.",
            "Source-sized S03 masks still require the same undistortion and DA3 "
            "preprocessing transform before sampling processed depth.",
            "No S02 marker scale, static confidence percentile, or door supplement is applied.",
            "Selected frames require observed masks and do not fill the accepted "
            "two-camera backpack absence interval.",
        ],
    }
    validated = ActionDepthRunSummary.model_validate(result).model_dump(mode="json")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(validated, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validated, indent=2))
    return 0


def _validate_inputs(
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    paths: Mapping[str, Path],
) -> None:
    manifest = payloads["synchronization_manifest"]
    replay = payloads["replay_evidence"]
    calibration = payloads["pose_calibration"]
    scene = payloads["scene_metadata"]
    bounded = payloads["bounded_replay_summary"]
    selection = payloads["selection_config"]
    expected_pose = "s01_capture_20260729:action_take_01:v1"
    if manifest.get("purpose") != "preferred_dynamic_pickup_carry_place_input":
        raise ValueError("synchronization manifest is not the accepted action input")
    sync_validation = cast(dict[str, Any], manifest["synchronization_validation"])
    if not sync_validation.get("accepted"):
        raise ValueError("action synchronization is not accepted")
    content_qa = cast(dict[str, Any], manifest["content_qa"])
    if not content_qa.get("complete_pickup_carry_place_sequence"):
        raise ValueError("action content does not contain the accepted sequence")
    if replay.get("status") != "passed" or replay.get("pose_version_id") != expected_pose:
        raise ValueError("S01 replay evidence is not accepted for the action pose")
    if calibration.get("pose_version_id") != expected_pose or not str(
        calibration.get("calibration_status", "")
    ).startswith("accepted_"):
        raise ValueError("action pose calibration is not accepted")
    accepted = cast(dict[str, str], scene["accepted_pose_versions"])
    if accepted.get("action_take_01") != expected_pose:
        raise ValueError("scene metadata does not accept the action pose version")
    if bounded.get("stage") != "S03" or bounded.get("pose_version_id") != expected_pose:
        raise ValueError("bounded perception evidence is not the accepted S03 action run")
    manifest_hash = _sha256(paths["synchronization_manifest"])
    if bounded.get("synchronization_manifest_sha256") != manifest_hash:
        raise ValueError("S03 perception and S04 action manifest differ")
    if selection.get("purpose") != "deterministic_action_spanning_raw_da3_depth_preflight":
        raise ValueError("selection config has the wrong S04 purpose")
    if selection.get("schema_version") != 1 or not selection.get("keyframes"):
        raise ValueError("selection config must contain versioned keyframes")


def _build_sources(
    *,
    project_root: Path,
    manifest: Mapping[str, Any],
    manifest_ref: str,
    manifest_sha256: str,
    pose_version_id: str,
) -> dict[str, FileFrameSource]:
    derived = cast(dict[str, dict[str, Any]], manifest["derived_outputs"])
    source_records = cast(dict[str, dict[str, Any]], manifest["sources"])
    return {
        camera_id: FileFrameSource(
            path=project_root / str(derived[camera_id]["path"]),
            capture_session_id=str(manifest["capture_session_id"]),
            camera_id=camera_id,
            source_ref=str(derived[camera_id]["path"]),
            expected_sha256=str(derived[camera_id]["sha256"]),
            synchronization_manifest_ref=manifest_ref,
            synchronization_manifest_sha256=manifest_sha256,
            pose_version_id=pose_version_id,
            timestamp_transform=TimestampTransform(),
            expected_width=int(source_records[camera_id]["image_width"]),
            expected_height=int(source_records[camera_id]["image_height"]),
        )
        for camera_id in CAMERA_IDS
    }


def _validate_replay(
    bundles: Sequence[SynchronizedFrameBundle], replay: Mapping[str, Any]
) -> None:
    record = cast(dict[str, Any], replay["replay_results"])
    ids = [bundle.bundle_id for bundle in bundles]
    ordered_digest = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    if len(bundles) != int(record["bundle_count"]):
        raise ValueError("action bundle count differs from accepted replay")
    if any(bundle.missing_camera_ids for bundle in bundles):
        raise ValueError("S04 action-depth input contains an incomplete bundle")
    if ordered_digest != record["ordered_bundle_id_sha256"]:
        raise ValueError("action bundle identity/order differs from accepted replay")


def _load_timeline_states(
    camera_a: Mapping[str, Any], camera_b: Mapping[str, Any]
) -> tuple[PerceptionTargetFrameState, ...]:
    states: list[PerceptionTargetFrameState] = []
    for expected_camera, timeline in zip(CAMERA_IDS, (camera_a, camera_b), strict=True):
        if timeline.get("stage") != "S03" or timeline.get("camera_id") != expected_camera:
            raise ValueError("perception timeline camera/stage identity is invalid")
        records = cast(list[dict[str, Any]], timeline["records"])
        parsed = tuple(PerceptionTargetFrameState.model_validate(item) for item in records)
        if any(item.frame_identity.camera_id != expected_camera for item in parsed):
            raise ValueError("perception timeline contains another camera")
        states.extend(parsed)
    return tuple(states)


def _validate_jobs_against_s03(
    jobs: Sequence[Any], bounded_summary: Mapping[str, Any]
) -> None:
    sampling = cast(dict[str, Any], bounded_summary["sampling"])
    stride = int(sampling["frame_stride"])
    if any(job.bundle.frames[0].source_frame_index % stride for job in jobs):
        raise ValueError("action-depth job is outside the retained S03 cadence")
    if any(job.bundle.capture_session_id != bounded_summary["capture_session_id"] for job in jobs):
        raise ValueError("action-depth job capture session differs from S03")


def _camera_contracts(
    calibration: Mapping[str, Any],
) -> tuple[tuple[CameraIntrinsics, ...], tuple[CameraPose, ...]]:
    records = cast(dict[str, dict[str, Any]], calibration["cameras"])
    intrinsics: list[CameraIntrinsics] = []
    poses: list[CameraPose] = []
    for camera_id in CAMERA_IDS:
        intrinsic = cast(dict[str, Any], records[camera_id]["intrinsics"])
        pose = cast(dict[str, Any], records[camera_id]["pose"])
        intrinsics.append(
            CameraIntrinsics(
                camera_id=camera_id,
                fx=float(intrinsic["fx"]),
                fy=float(intrinsic["fy"]),
                cx=float(intrinsic["cx"]),
                cy=float(intrinsic["cy"]),
                image_width=int(intrinsic["image_width"]),
                image_height=int(intrinsic["image_height"]),
                distortion_coefficients=tuple(
                    float(value)
                    for value in cast(list[float], intrinsic["distortion_coefficients"])
                ),
            )
        )
        poses.append(CameraPose.model_validate(pose))
    return tuple(intrinsics), tuple(poses)


def _extract_undistorted_keyframes(
    *,
    output_dir: Path,
    sources: Mapping[str, FileFrameSource],
    jobs: Sequence[Any],
    camera_intrinsics: Sequence[CameraIntrinsics],
) -> dict[tuple[str, str], Path]:
    keyframe_dir = output_dir / "keyframes"
    keyframe_dir.mkdir()
    intrinsics_by_id = {item.camera_id: item for item in camera_intrinsics}
    selected: dict[str, dict[int, tuple[str, str]]] = {
        camera_id: {} for camera_id in CAMERA_IDS
    }
    for job in jobs:
        for frame in job.bundle.frames:
            selected[frame.camera_id][frame.source_frame_index] = (
                job.bundle.bundle_id,
                frame.frame_id,
            )
    result: dict[tuple[str, str], Path] = {}
    for camera_id in CAMERA_IDS:
        intrinsic = intrinsics_by_id[camera_id]
        matrix = np.array(
            [
                [intrinsic.fx, 0.0, intrinsic.cx],
                [0.0, intrinsic.fy, intrinsic.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.asarray(intrinsic.distortion_coefficients, dtype=np.float64)
        for decoded in sources[camera_id].iter_frames():
            expected = selected[camera_id].get(decoded.identity.source_frame_index)
            if expected is None:
                continue
            bundle_id, expected_frame_id = expected
            if decoded.identity.frame_id != expected_frame_id:
                raise RuntimeError("decoded action keyframe identity changed")
            undistorted = cv2.undistort(
                decoded.image_bgr, matrix, distortion, None, matrix
            )
            rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
            path = keyframe_dir / (
                f"{bundle_id[:12]}_{camera_id}_frame_"
                f"{decoded.identity.source_frame_index:04d}.png"
            )
            Image.fromarray(rgb).save(path)
            result[(bundle_id, camera_id)] = path
    expected_count = len(jobs) * len(CAMERA_IDS)
    if len(result) != expected_count:
        raise RuntimeError(
            f"decoded {len(result)} action keyframes, expected {expected_count}"
        )
    return result


def _validate_da3_output(
    output: Any, *, poses_by_id: Mapping[str, CameraPose]
) -> None:
    if not output.is_metric:
        raise RuntimeError("DA3 action prediction is not metric")
    if output.processed_images is None:
        raise RuntimeError("DA3 action prediction lacks processed images")
    supplied = np.asarray(
        [poses_by_id[camera_id].T_camera_from_world for camera_id in CAMERA_IDS],
        dtype=np.float32,
    )
    if not np.allclose(output.T_camera_from_world, supplied, atol=1e-5):
        raise RuntimeError("DA3 returned poses differ from calibrated action inputs")
    if output.depth_m.shape != output.confidence.shape:
        raise RuntimeError("DA3 action depth/confidence shapes differ")
    if output.depth_m.shape[0] != len(CAMERA_IDS):
        raise RuntimeError("DA3 action prediction does not contain two views")
    if not np.any(np.isfinite(output.depth_m) & (output.depth_m > 0)):
        raise RuntimeError("DA3 action prediction has no finite positive depth")


def _camera_prediction_record(
    *,
    camera_index: int,
    camera_id: str,
    job: Any,
    image_path: Path,
    depth_m: Float32Array,
    confidence: Float32Array,
    processed_intrinsics: Float32Array,
    project_root: Path,
) -> dict[str, Any]:
    frame = job.bundle.frames[camera_index]
    camera_evidence = [
        item.model_dump(mode="json")
        for item in job.mask_evidence
        if item.frame_identity.camera_id == camera_id
    ]
    depth = depth_m[camera_index]
    conf = confidence[camera_index]
    valid_depth = np.isfinite(depth) & (depth > 0)
    return {
        "frame_id": frame.frame_id,
        "source_frame_index": frame.source_frame_index,
        "capture_timestamp_seconds": frame.capture_timestamp_seconds,
        "undistorted_keyframe_ref": _relative(image_path, project_root),
        "undistorted_keyframe_sha256": _sha256(image_path),
        "processed_intrinsics": processed_intrinsics[camera_index].tolist(),
        "processed_shape": list(depth.shape),
        "finite_positive_depth_count": int(np.count_nonzero(valid_depth)),
        "finite_confidence_count": int(np.count_nonzero(np.isfinite(conf))),
        "depth_range_m": _finite_range(depth[valid_depth]),
        "confidence_range": _finite_range(conf[np.isfinite(conf)]),
        "observed_mask_evidence": camera_evidence,
    }


def _save_depth_confidence_preview(
    *,
    depth_m: Float32Array,
    confidence: Float32Array,
    processed_images_rgb: UInt8Array,
    camera_ids: tuple[str, str],
    phase_id: str,
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    figure.suptitle(phase_id)
    for index, camera_id in enumerate(camera_ids):
        axes[index, 0].imshow(processed_images_rgb[index])
        axes[index, 0].set_title(f"{camera_id} processed RGB")
        valid = depth_m[index][np.isfinite(depth_m[index]) & (depth_m[index] > 0)]
        depth_image = axes[index, 1].imshow(
            depth_m[index],
            cmap="turbo",
            vmin=float(np.percentile(valid, 2)),
            vmax=float(np.percentile(valid, 98)),
        )
        axes[index, 1].set_title(f"{camera_id} raw metric depth")
        figure.colorbar(depth_image, ax=axes[index, 1], fraction=0.046)
        confidence_image = axes[index, 2].imshow(confidence[index], cmap="viridis")
        axes[index, 2].set_title(f"{camera_id} confidence")
        figure.colorbar(confidence_image, ax=axes[index, 2], fraction=0.046)
        for axis in axes[index]:
            axis.axis("off")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _finite_range(values: NDArray[np.generic]) -> dict[str, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        raise RuntimeError("raw DA3 array contains no finite values")
    return {
        "minimum": float(np.min(finite)),
        "p02": float(np.percentile(finite, 2)),
        "median": float(np.median(finite)),
        "p98": float(np.percentile(finite, 98)),
        "maximum": float(np.max(finite)),
    }


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _relative(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root).as_posix()


def _timer_payload(timer: PhaseTimer) -> dict[str, object]:
    if timer.observation is None:
        raise RuntimeError(f"timer {timer.phase} produced no observation")
    return timer.observation.model_dump(mode="json")


if __name__ == "__main__":
    raise SystemExit(main())
