"""Create non-destructive D025 marker-scaled S04 action-pair depth artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from spatial_reconstruction.localization import (
    ActionDepthRunSummary,
    ActionMarkerScaleObservation,
    ActionPairScalePolicy,
    estimate_action_pair_scale,
    sample_floor_marker_scale,
)

CAMERA_IDS = ("camera_a", "camera_b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--raw-summary",
        type=Path,
        default=Path("artifacts/s04/action_depth_preflight_20260801/summary.json"),
    )
    parser.add_argument(
        "--pose-calibration",
        type=Path,
        default=Path(
            "artifacts/s01/calibration/action_take_01_pose/camera_calibration.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    output_dir = _resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    corrected_dir = output_dir / "corrected_predictions"
    diagnostics_dir = output_dir / "diagnostics"
    corrected_dir.mkdir()
    diagnostics_dir.mkdir()

    raw_summary_path = _resolve(root, args.raw_summary)
    calibration_path = _resolve(root, args.pose_calibration)
    raw_summary = ActionDepthRunSummary.model_validate_json(
        raw_summary_path.read_text(encoding="utf-8")
    )
    calibration = cast(
        dict[str, Any], json.loads(calibration_path.read_text(encoding="utf-8"))
    )
    policy = ActionPairScalePolicy()
    marker_model = cast(dict[str, Any], calibration["marker_model"])
    if tuple(marker_model["pose_anchor_marker_ids"]) != policy.marker_ids:
        raise ValueError("D025 action marker IDs differ from accepted pose anchors")
    if float(marker_model["marker_length_m"]) != policy.marker_length_m:
        raise ValueError("D025 marker length differs from accepted calibration")
    centers = {
        int(item["marker_id"]): tuple(float(value) for value in item["center_world_m"])
        for item in cast(list[dict[str, Any]], marker_model["placements"])
    }

    job_records: list[dict[str, Any]] = []
    csv_rows: list[dict[str, object]] = []
    contact_images: list[Path] = []
    source_indices: list[int] = []
    for job_index, prediction in enumerate(raw_summary.predictions):
        raw_path = _resolve(root, Path(prediction.raw_prediction_ref))
        _require_hash(raw_path, prediction.raw_prediction_sha256)
        with np.load(raw_path, allow_pickle=False) as arrays:
            raw_depth = np.asarray(arrays["depth_m"], dtype=np.float32)
            confidence = np.asarray(arrays["confidence"])
        raw_confidence_sha256 = hashlib.sha256(confidence.tobytes()).hexdigest()
        observations: list[ActionMarkerScaleObservation] = []
        annotated: list[np.ndarray[Any, Any]] = []
        camera_records: dict[str, dict[str, Any]] = {}
        calibration_cameras = cast(dict[str, dict[str, Any]], calibration["cameras"])
        for camera_index, camera_id in enumerate(CAMERA_IDS):
            camera = prediction.cameras[camera_id]
            image_path = _resolve(root, Path(str(camera["undistorted_keyframe_ref"])))
            _require_hash(image_path, str(camera["undistorted_keyframe_sha256"]))
            image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
            detected = _detect_markers(image_rgb, policy.marker_ids)
            camera_calibration = calibration_cameras[camera_id]
            intrinsic_record = cast(dict[str, Any], camera_calibration["intrinsics"])
            source_intrinsics = _intrinsic_matrix(intrinsic_record)
            pose = cast(dict[str, Any], camera_calibration["pose"])
            world_from_camera = np.asarray(pose["T_world_from_camera"], dtype=np.float64)
            camera_from_world = np.asarray(pose["T_camera_from_world"], dtype=np.float64)
            processed_intrinsics = np.asarray(camera["processed_intrinsics"], dtype=np.float64)
            overlay = image_rgb.copy()
            camera_marker_ids: list[int] = []
            for marker_id in policy.marker_ids:
                corners = detected.get(marker_id)
                if corners is None:
                    continue
                center_world = centers[marker_id]
                projected_center = _project_world_point(
                    center_world, source_intrinsics, camera_from_world
                )
                center_values = np.mean(corners, axis=0)
                detected_center = (float(center_values[0]), float(center_values[1]))
                reprojection_error = float(
                    np.linalg.norm(np.asarray(detected_center) - projected_center)
                )
                (
                    sample_count,
                    expected_median,
                    raw_median,
                    ratio,
                    ratio_mad,
                    _,
                ) = sample_floor_marker_scale(
                    depth_m=raw_depth[camera_index],
                    processed_intrinsics=processed_intrinsics,
                    T_world_from_camera=world_from_camera,
                    marker_center_world_m=cast(tuple[float, float, float], center_world),
                    marker_length_m=policy.marker_length_m,
                    protected_inner_fraction=policy.protected_inner_fraction,
                )
                observation = ActionMarkerScaleObservation(
                    camera_id=cast(Any, camera_id),
                    marker_id=marker_id,
                    detected_center_uv=detected_center,
                    projected_center_uv=(
                        float(projected_center[0]),
                        float(projected_center[1]),
                    ),
                    reprojection_error_px=reprojection_error,
                    valid_sample_count=sample_count,
                    expected_camera_depth_median=expected_median,
                    raw_da3_depth_median=raw_median,
                    expected_over_raw_ratio=ratio,
                    ratio_mad=ratio_mad,
                )
                observations.append(observation)
                camera_marker_ids.append(marker_id)
                cv2.polylines(
                    overlay,
                    [np.rint(corners).astype(np.int32)],
                    True,
                    (0, 255, 0),
                    5,
                )
                cv2.drawMarker(
                    overlay,
                    tuple(np.rint(projected_center).astype(int)),
                    (255, 0, 255),
                    cv2.MARKER_CROSS,
                    28,
                    4,
                )
                cv2.putText(
                    overlay,
                    f"M{marker_id} r={ratio:.4f}",
                    tuple(np.rint(corners[0]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 0),
                    3,
                    cv2.LINE_AA,
                )
            annotated.append(overlay)
            camera_records[camera_id] = {
                "frame_id": camera["frame_id"],
                "source_frame_index": camera["source_frame_index"],
                "undistorted_keyframe_ref": camera["undistorted_keyframe_ref"],
                "undistorted_keyframe_sha256": camera["undistorted_keyframe_sha256"],
                "processed_intrinsics": camera["processed_intrinsics"],
                "accepted_marker_ids": camera_marker_ids,
            }

        estimate = estimate_action_pair_scale(observations, policy=policy)
        corrected = raw_depth * np.float32(estimate.scale)
        corrected_path = corrected_dir / f"{job_index:02d}_{prediction.job.job_id[:12]}.npz"
        np.savez_compressed(
            corrected_path,
            corrected_depth_m=corrected,
            scale=np.float64(estimate.scale),
            policy_id=np.asarray(policy.policy_id),
            job_id=np.asarray(prediction.job.job_id),
            bundle_id=np.asarray(prediction.job.bundle.bundle_id),
            frame_ids=np.asarray(
                [frame.frame_id for frame in prediction.job.bundle.frames]
            ),
            camera_ids=np.asarray(CAMERA_IDS),
            raw_prediction_ref=np.asarray(prediction.raw_prediction_ref),
            raw_prediction_sha256=np.asarray(prediction.raw_prediction_sha256),
            raw_confidence_sha256=np.asarray(raw_confidence_sha256),
            raw_da3_depth_preserved=np.asarray(True),
            confidence_unchanged=np.asarray(True),
        )
        diagnostic_path = diagnostics_dir / f"{job_index:02d}_markers.png"
        _save_diagnostic(
            images=annotated,
            camera_ids=CAMERA_IDS,
            frame_index=prediction.job.bundle.frames[0].source_frame_index,
            phase_id=prediction.job.phase_id,
            scale=estimate.scale,
            maximum_deviation=estimate.maximum_relative_deviation,
            path=diagnostic_path,
        )
        contact_images.append(diagnostic_path)
        for observation in estimate.observations:
            csv_rows.append(
                {
                    "job_id": prediction.job.job_id,
                    "source_frame_index": prediction.job.bundle.frames[0].source_frame_index,
                    "phase_id": prediction.job.phase_id,
                    "pair_scale": estimate.scale,
                    "maximum_relative_deviation": estimate.maximum_relative_deviation,
                    **observation.model_dump(mode="json"),
                }
            )
        source_index = prediction.job.bundle.frames[0].source_frame_index
        source_indices.append(source_index)
        job_records.append(
            {
                "job_id": prediction.job.job_id,
                "bundle_id": prediction.job.bundle.bundle_id,
                "source_frame_index": source_index,
                "capture_timestamp_seconds": prediction.job.bundle.capture_timestamp_seconds,
                "phase_id": prediction.job.phase_id,
                "raw_prediction_ref": prediction.raw_prediction_ref,
                "raw_prediction_sha256": prediction.raw_prediction_sha256,
                "raw_confidence_sha256": raw_confidence_sha256,
                "scale_estimate": estimate.model_dump(mode="json"),
                "corrected_prediction_ref": _relative(corrected_path, root),
                "corrected_prediction_sha256": _sha256(corrected_path),
                "diagnostic_ref": _relative(diagnostic_path, root),
                "diagnostic_sha256": _sha256(diagnostic_path),
                "corrected_depth_range_m": [
                    float(np.min(corrected)),
                    float(np.max(corrected)),
                ],
                "cameras": camera_records,
            }
        )
    if source_indices != sorted(source_indices):
        raise RuntimeError("action-pair scale jobs are not in capture order")

    csv_path = output_dir / "marker_scale_observations.csv"
    _write_csv(csv_path, csv_rows)
    contact_path = output_dir / "marker_scale_contact_sheet.png"
    _save_contact_sheet(contact_images, contact_path)
    summary = {
        "schema_version": 1,
        "stage": "S04",
        "status": "passed",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "d025_non_destructive_action_pair_marker_scaling",
        "policy": policy.model_dump(mode="json"),
        "raw_action_depth_summary_ref": _relative(raw_summary_path, root),
        "raw_action_depth_summary_sha256": _sha256(raw_summary_path),
        "pose_calibration_ref": _relative(calibration_path, root),
        "pose_calibration_sha256": _sha256(calibration_path),
        "marker_center_measurement_uncertainty_m": marker_model[
            "centre_measurement_uncertainty_m"
        ],
        "raw_da3_arrays_modified": False,
        "confidence_modified": False,
        "camera_specific_scale_applied": False,
        "jobs": job_records,
        "aggregate": {
            "job_count": len(job_records),
            "marker_observation_count": len(csv_rows),
            "scale_minimum": min(item["scale_estimate"]["scale"] for item in job_records),
            "scale_median": float(
                np.median([item["scale_estimate"]["scale"] for item in job_records])
            ),
            "scale_maximum": max(item["scale_estimate"]["scale"] for item in job_records),
            "maximum_marker_relative_deviation": max(
                item["scale_estimate"]["maximum_relative_deviation"]
                for item in job_records
            ),
        },
        "marker_observations_csv_ref": _relative(csv_path, root),
        "marker_observations_csv_sha256": _sha256(csv_path),
        "contact_sheet_ref": _relative(contact_path, root),
        "contact_sheet_sha256": _sha256(contact_path),
        "limitations": [
            "This scalar corrects derived S04 depth only; raw DA3 depth and "
            "confidence remain immutable.",
            "Marker centres retain the calibration's +/-0.05 m measurement uncertainty.",
            "The correction does not by itself revalidate D030-D033 localization "
            "or fusion artifacts.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2))
    return 0


def _detect_markers(
    image_rgb: np.ndarray[Any, Any], marker_ids: tuple[int, ...]
) -> dict[int, np.ndarray[Any, Any]]:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY))
    found: dict[int, np.ndarray[Any, Any]] = {}
    if ids is None:
        return found
    for item_corners, item_id in zip(corners, ids.reshape(-1), strict=True):
        marker_id = int(item_id)
        if marker_id in marker_ids:
            found[marker_id] = np.asarray(item_corners[0], dtype=np.float64)
    return found


def _project_world_point(
    point_world: tuple[float, ...],
    intrinsics: np.ndarray[Any, Any],
    camera_from_world: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    point = np.asarray((*point_world, 1.0), dtype=np.float64)
    camera = camera_from_world @ point
    if camera[2] <= 0:
        raise ValueError("marker center is behind camera")
    pixel = intrinsics @ camera[:3]
    result: np.ndarray[Any, Any] = pixel[:2] / pixel[2]
    return result


def _intrinsic_matrix(record: dict[str, Any]) -> np.ndarray[Any, Any]:
    return np.asarray(
        [[record["fx"], 0, record["cx"]], [0, record["fy"], record["cy"]], [0, 0, 1]],
        dtype=np.float64,
    )


def _save_diagnostic(
    *, images: list[np.ndarray[Any, Any]], camera_ids: tuple[str, str], frame_index: int,
    phase_id: str, scale: float, maximum_deviation: float, path: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    figure.suptitle(
        f"frame {frame_index} | {phase_id} | shared scale {scale:.6f} | "
        f"max deviation {maximum_deviation:.2%}"
    )
    for axis, image, camera_id in zip(axes, images, camera_ids, strict=True):
        axis.imshow(image)
        axis.set_title(f"{camera_id}: green detection, magenta calibrated center")
        axis.axis("off")
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _save_contact_sheet(paths: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    thumb_height = max(1, int(images[0].height * width / images[0].width))
    canvas = Image.new("RGB", (width, thumb_height * len(images)), "white")
    for index, image in enumerate(images):
        resized = image.resize((width, thumb_height))
        canvas.paste(resized, (0, index * thumb_height))
    canvas.save(output)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("no marker observations were accepted")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _require_hash(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"source artifact hash changed: {path}: {actual} != {expected}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
