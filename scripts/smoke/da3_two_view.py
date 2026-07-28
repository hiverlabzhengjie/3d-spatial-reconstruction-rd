"""Run the isolated S00 WP5 DA3 pose-conditioned two-view MPS smoke check."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import torch
from huggingface_hub import HfApi
from numpy.typing import NDArray
from PIL import Image

from spatial_reconstruction.config import load_project_config
from spatial_reconstruction.contracts import (
    CameraIntrinsics,
    CameraPose,
    MemoryObservation,
    ModelRunObservation,
    RunOutcome,
    TimingObservation,
)
from spatial_reconstruction.models import (
    EXPECTED_DA3_VENDOR_FINGERPRINT,
    DA3Adapter,
    DA3Output,
    build_da3_camera_arrays,
    compute_vendor_fingerprint,
    make_synthetic_two_view_cameras,
)
from spatial_reconstruction.models.da3_adapter import TWO_VIEW_ALIGNMENT_POLICY
from spatial_reconstruction.models.da3_mps import DA3Precision
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Artifact directory; defaults to a new timestamped S00 DA3 run.",
    )
    parser.add_argument(
        "--revision",
        help="Exact Hugging Face commit; resolved from the model repository when omitted.",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default=None,
        help="Override the configured DA3 precision policy.",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        help="Optional ordered patch-aligned resolution ladder.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_project_config()
    output_dir = args.output_dir or _timestamped_output_dir(config.paths.artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "summary.json"
    failure_path = output_dir / "failure.txt"

    selection: DeviceSelection | None = None
    model_revision: str | None = args.revision
    adapter: DA3Adapter | None = None
    timings: list[TimingObservation] = []
    memory: list[MemoryObservation] = []
    artifact_refs: list[str] = [str(summary_path)]
    warnings: list[str] = []
    requested_precision = _resolve_precision(args.precision, config.runtime.precision)
    source = SystemMemorySource()

    try:
        selection = select_device(
            config.runtime.preferred_device,
            allow_cpu_fallback=False,
        )
        if selection.actual != "mps":
            raise RuntimeError("WP5 requires an actual MPS run")

        image_paths = (
            config.paths.da3_vendor_dir / "assets/examples/SOH/000.png",
            config.paths.da3_vendor_dir / "assets/examples/SOH/010.png",
        )
        _require_input_images(image_paths)

        fingerprint = compute_vendor_fingerprint(config.paths.da3_vendor_dir)
        if fingerprint != EXPECTED_DA3_VENDOR_FINGERPRINT:
            raise RuntimeError(
                "DA3 vendor fingerprint changed: "
                f"expected {EXPECTED_DA3_VENDOR_FINGERPRINT}, got {fingerprint}"
            )

        model_revision = model_revision or _resolve_model_revision(config.models.da3)
        resolutions = tuple(args.resolutions or config.da3.process_resolutions)
        _validate_resolutions(resolutions)
        camera_intrinsics, camera_poses = make_synthetic_two_view_cameras(image_paths)
        supplied_T_camera_from_world, supplied_intrinsics = build_da3_camera_arrays(
            camera_intrinsics,
            camera_poses,
        )

        memory.append(sample_memory("before_model_load", device="mps", source=source))
        load_monitor = PeakRSSMonitor(source=source)
        load_timer = PhaseTimer(phase="model_load", device="mps")
        with load_monitor, load_timer:
            adapter = DA3Adapter.from_pretrained(
                vendor_dir=config.paths.da3_vendor_dir,
                model_id=config.models.da3,
                model_revision=model_revision,
                device=_torch_mps_device(),
                precision=requested_precision,
            )
        _append_timing(timings, load_timer)
        memory.append(
            sample_memory(
                "after_model_load",
                device="mps",
                source=source,
                process_peak_rss_bytes=load_monitor.peak_rss_bytes,
            )
        )

        outputs: dict[int, DA3Output] = {}
        attempted: list[int] = []
        lowest = resolutions[0]
        for suffix in ("cold", "warm_1", "warm_2"):
            outputs[lowest] = _timed_inference(
                adapter=adapter,
                image_paths=image_paths,
                camera_intrinsics=camera_intrinsics,
                camera_poses=camera_poses,
                resolution=lowest,
                phase=f"inference_{lowest}_{suffix}",
                timings=timings,
                memory=memory,
                source=source,
            )
        attempted.append(lowest)

        for resolution in resolutions[1:]:
            if not _has_conservative_mps_headroom(memory[-1]):
                warnings.append(
                    f"stopped before resolution {resolution}: MPS driver allocation "
                    "exceeded the 80% conservative headroom threshold"
                )
                break
            outputs[resolution] = _timed_inference(
                adapter=adapter,
                image_paths=image_paths,
                camera_intrinsics=camera_intrinsics,
                camera_poses=camera_poses,
                resolution=resolution,
                phase=f"inference_{resolution}_probe",
                timings=timings,
                memory=memory,
                source=source,
            )
            attempted.append(resolution)

        selected_resolution = attempted[-1]
        if selected_resolution != lowest:
            for suffix in ("repeat_1", "repeat_2"):
                outputs[selected_resolution] = _timed_inference(
                    adapter=adapter,
                    image_paths=image_paths,
                    camera_intrinsics=camera_intrinsics,
                    camera_poses=camera_poses,
                    resolution=selected_resolution,
                    phase=f"inference_{selected_resolution}_{suffix}",
                    timings=timings,
                    memory=memory,
                    source=source,
                )

        selected = outputs[selected_resolution]
        _assert_supplied_pose_returned(selected, supplied_T_camera_from_world)
        npz_path = output_dir / "prediction.npz"
        depth_preview_path = output_dir / "depth_preview.png"
        confidence_preview_path = output_dir / "confidence_preview.png"
        manifest_path = output_dir / "manifest.json"
        artifact_refs.extend(
            str(path)
            for path in (
                npz_path,
                depth_preview_path,
                confidence_preview_path,
                manifest_path,
            )
        )

        np.savez_compressed(
            npz_path,
            depth_m=selected.depth_m,
            confidence=selected.confidence,
            returned_T_camera_from_world=selected.T_camera_from_world,
            returned_intrinsics=selected.intrinsics,
            supplied_T_camera_from_world=supplied_T_camera_from_world,
            supplied_intrinsics=supplied_intrinsics,
            model_id=np.asarray(config.models.da3),
            model_revision=np.asarray(model_revision),
            process_resolution=np.asarray(selected_resolution, dtype=np.int32),
            image_paths=np.asarray([str(path) for path in image_paths]),
        )
        _save_array_preview(selected.depth_m, depth_preview_path, percentile_scale=True)
        _save_array_preview(
            selected.confidence,
            confidence_preview_path,
            percentile_scale=False,
        )

        selected_seconds = _selected_inference_seconds(timings, selected_resolution)
        keyframe_interval = _recommend_keyframe_interval(
            selected_seconds,
            config.da3.keyframe_interval_candidates_seconds,
        )
        manifest = {
            "model_id": config.models.da3,
            "model_revision": model_revision,
            "vendor_fingerprint": fingerprint,
            "vendor_fingerprint_expected": EXPECTED_DA3_VENDOR_FINGERPRINT,
            "inputs": [_image_manifest(path) for path in image_paths],
            "supplied_camera_parameters": True,
            "camera_convention": "OpenCV T_camera_from_world",
            "synthetic_baseline_m": 0.20,
            "synthetic_fixture_only": True,
            "two_view_alignment_policy": TWO_VIEW_ALIGNMENT_POLICY,
            "infer_gs": False,
            "use_ray_pose": False,
            "process_method": "upper_bound_resize",
            "resolutions_attempted": attempted,
            "selected_resolution": selected_resolution,
            "output_depth_shape": list(selected.depth_m.shape),
            "output_confidence_shape": list(selected.confidence.shape),
            "positive_depth_fraction": float(np.mean(selected.depth_m > 0)),
            "autocast_precision": adapter.autocast_policy.reported_precision,
            "selected_pair_seconds": selected_seconds,
            "provisional_keyframe_interval_seconds": keyframe_interval,
            "provisional_keyframe_interval_frames_at_30fps": int(keyframe_interval * 30),
            "warnings": warnings,
        }
        manifest_path.write_text(f"{json.dumps(manifest, indent=2)}\n", encoding="utf-8")

        summary = build_run_observation(
            model_id=config.models.da3,
            model_revision=model_revision,
            selection=selection,
            precision=adapter.autocast_policy.reported_precision,
            input_description=(
                "two vendor SOH PNGs with supplied synthetic OpenCV intrinsics "
                "and T_camera_from_world poses"
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
                model_id=config.models.da3,
                model_revision=model_revision,
                selection=None,
                requested_device=config.runtime.preferred_device,
                device_selection_error=str(exc),
                precision="unknown",
                input_description="DA3 WP5 preflight failed before device selection",
                timings=timings,
                memory=memory,
                outcome=RunOutcome.FAILED,
                warnings=warnings,
                artifact_refs=artifact_refs,
            )
        else:
            precision = (
                adapter.autocast_policy.reported_precision if adapter is not None else "unknown"
            )
            summary = build_run_observation(
                model_id=config.models.da3,
                model_revision=model_revision,
                selection=selection,
                requested_device=(config.runtime.preferred_device if selection is None else None),
                device_selection_error=(str(exc) if selection is None else None),
                precision=precision,
                input_description="DA3 WP5 two-view smoke attempt",
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


def _timed_inference(
    *,
    adapter: DA3Adapter,
    image_paths: tuple[Path, Path],
    camera_intrinsics: tuple[CameraIntrinsics, ...],
    camera_poses: tuple[CameraPose, ...],
    resolution: int,
    phase: str,
    timings: list[TimingObservation],
    memory: list[MemoryObservation],
    source: SystemMemorySource,
) -> DA3Output:
    monitor = PeakRSSMonitor(source=source)
    timer = PhaseTimer(phase=phase, device="mps")
    try:
        with monitor, timer:
            output = adapter.infer_pose_conditioned(
                image_paths=image_paths,
                camera_intrinsics=camera_intrinsics,
                camera_poses=camera_poses,
                process_resolution=resolution,
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
    return output


def _append_timing(timings: list[TimingObservation], timer: PhaseTimer) -> None:
    if timer.observation is None:
        raise RuntimeError(f"timer '{timer.phase}' did not emit an observation")
    timings.append(timer.observation)


def _has_conservative_mps_headroom(observation: MemoryObservation) -> bool:
    driver = observation.mps_driver_allocated_bytes
    recommended = observation.mps_recommended_max_bytes
    if driver is None or recommended is None or recommended == 0:
        return False
    return driver / recommended < 0.80


def _assert_supplied_pose_returned(
    output: DA3Output,
    supplied_T_camera_from_world: NDArray[np.float32],
) -> None:
    if not np.allclose(
        output.T_camera_from_world,
        supplied_T_camera_from_world,
        atol=1e-5,
    ):
        raise RuntimeError("returned DA3 poses do not match supplied pose-conditioned inputs")


def _selected_inference_seconds(
    timings: list[TimingObservation],
    selected_resolution: int,
) -> float:
    prefix = f"inference_{selected_resolution}_"
    values = [item.seconds for item in timings if item.phase.startswith(prefix)]
    if not values:
        raise RuntimeError("selected DA3 resolution has no timing observations")
    warm_values = values[1:] if len(values) > 1 else values
    return float(sum(warm_values) / len(warm_values))


def _recommend_keyframe_interval(
    seconds_per_pair: float,
    candidates: tuple[float, ...],
) -> float:
    for candidate in candidates:
        if seconds_per_pair <= candidate:
            return float(candidate)
    return float(candidates[-1])


def _save_array_preview(
    arrays: NDArray[np.float32],
    path: Path,
    *,
    percentile_scale: bool,
) -> None:
    finite = arrays[np.isfinite(arrays)]
    if finite.size == 0:
        raise RuntimeError("cannot preview an array without finite values")
    if percentile_scale:
        low, high = np.percentile(finite, (2.0, 98.0))
    else:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        high = low + 1.0
    normalized = np.clip((arrays - low) / (high - low), 0.0, 1.0)
    view_images = [(view * 255.0).astype(np.uint8) for view in normalized]
    combined = np.concatenate(view_images, axis=1)
    Image.fromarray(combined, mode="L").save(path)


def _resolve_model_revision(model_id: str) -> str:
    info = HfApi().model_info(model_id, files_metadata=False)
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a commit for {model_id}")
    return info.sha


def _torch_mps_device() -> torch.device:
    return torch.device("mps")


def _resolve_precision(cli_value: str | None, configured: str) -> DA3Precision:
    value = cli_value or configured
    if value not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError(f"unsupported DA3 precision: {value}")
    return cast(DA3Precision, value)


def _validate_resolutions(resolutions: tuple[int, ...]) -> None:
    if (
        not resolutions
        or tuple(sorted(set(resolutions))) != resolutions
        or any(value <= 0 or value % 14 != 0 for value in resolutions)
    ):
        raise ValueError("resolutions must be unique ascending positive multiples of 14")


def _require_input_images(image_paths: tuple[Path, Path]) -> None:
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing DA3 smoke images: {missing}")


def _image_manifest(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        width, height = image.size
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "width": width,
        "height": height,
    }


def _timestamped_output_dir(artifacts_dir: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return artifacts_dir / "s00" / "da3" / f"wp5_{timestamp}"


def _write_summary(summary: ModelRunObservation, path: Path) -> None:
    path.write_text(f"{summary.model_dump_json(indent=2)}\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
