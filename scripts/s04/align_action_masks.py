"""Align retained S03 masks to the exact DA3 action-depth grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import cv2
import matplotlib
import numpy as np
from numpy.typing import NDArray
from PIL import Image

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from spatial_reconstruction.contracts import CameraIntrinsics, PerceptionTarget
from spatial_reconstruction.localization import (
    ActionDepthRunSummary,
    AlignedMask,
    AlignedMaskRecord,
    MaskAlignmentRunSummary,
    align_source_mask_to_da3_grid,
    resize_intrinsics_for_da3_grid,
    transform_undistorted_rgb_to_da3_grid,
)

UInt8Array = NDArray[np.uint8]
Int16Array = NDArray[np.int16]
CAMERA_IDS = ("camera_a", "camera_b")
TARGET_COLORS_RGB = {
    PerceptionTarget.PERSON: np.array([0, 220, 255], dtype=np.float64),
    PerceptionTarget.BACKPACK: np.array([255, 40, 160], dtype=np.float64),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--action-depth-summary",
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
    project_root = args.project_root.resolve()
    summary_path = _resolve(project_root, args.action_depth_summary)
    calibration_path = _resolve(project_root, args.pose_calibration)
    output_dir = _resolve(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    action_summary = ActionDepthRunSummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    calibration = _read_object(calibration_path)
    if calibration.get("pose_version_id") != action_summary.pose_version_id:
        raise ValueError("mask alignment calibration differs from action-depth pose")
    intrinsics_by_id = _camera_intrinsics(calibration)

    masks_dir = output_dir / "aligned_masks"
    overlays_dir = output_dir / "overlays"
    masks_dir.mkdir()
    overlays_dir.mkdir()
    rgb_checks: list[dict[str, Any]] = []
    aligned_records: list[AlignedMaskRecord] = []
    overlay_records: list[dict[str, Any]] = []
    contact_rows: list[tuple[str, dict[str, UInt8Array]]] = []

    for job_index, prediction in enumerate(action_summary.predictions):
        raw_depth_path = _resolve(project_root, Path(prediction.raw_prediction_ref))
        _require_hash(raw_depth_path, prediction.raw_prediction_sha256)
        with np.load(raw_depth_path, allow_pickle=False) as arrays:
            processed_images = cast(
                UInt8Array, np.asarray(arrays["processed_images_rgb"]).copy()
            )
            processed_intrinsics = np.asarray(
                arrays["returned_intrinsics"], dtype=np.float64
            )
            depth_shape = tuple(int(value) for value in arrays["depth_m"].shape)
        if depth_shape != (2, 280, 504):
            raise ValueError("action-depth grid differs from accepted 2-by-280-by-504")

        masks: list[UInt8Array] = []
        mask_metadata: list[dict[str, Any]] = []
        pending_records: list[dict[str, Any]] = []
        overlays_by_camera = {
            camera_id: processed_images[camera_index].copy()
            for camera_index, camera_id in enumerate(CAMERA_IDS)
        }
        for camera_index, camera_id in enumerate(CAMERA_IDS):
            camera_record = prediction.cameras[camera_id]
            keyframe_path = _resolve(
                project_root, Path(str(camera_record["undistorted_keyframe_ref"]))
            )
            _require_hash(
                keyframe_path, str(camera_record["undistorted_keyframe_sha256"])
            )
            source_rgb = np.asarray(Image.open(keyframe_path).convert("RGB"))
            reproduced_rgb, transform = transform_undistorted_rgb_to_da3_grid(
                source_rgb,
                process_resolution=prediction.job.process_resolution,
            )
            rgb_difference = np.abs(
                reproduced_rgb.astype(np.int16)
                - processed_images[camera_index].astype(np.int16)
            )
            expected_intrinsics = resize_intrinsics_for_da3_grid(
                intrinsics_by_id[camera_id], transform
            )
            intrinsic_error = float(
                np.max(
                    np.abs(expected_intrinsics - processed_intrinsics[camera_index])
                )
            )
            check = {
                "action_depth_job_id": prediction.job.job_id,
                "camera_id": camera_id,
                "source_frame_index": prediction.job.bundle.frames[
                    camera_index
                ].source_frame_index,
                "reproduced_shape": list(reproduced_rgb.shape[:2]),
                "maximum_absolute_channel_difference": int(
                    np.max(rgb_difference)
                ),
                "mean_absolute_channel_difference": float(
                    np.mean(rgb_difference)
                ),
                "exact_channel_fraction": float(np.mean(rgb_difference == 0)),
                "processed_intrinsics_maximum_absolute_error": intrinsic_error,
                "passed": bool(
                    int(np.max(rgb_difference)) <= 1 and intrinsic_error <= 1e-4
                ),
                "transform": transform.model_dump(mode="json"),
            }
            if not check["passed"]:
                raise RuntimeError(
                    f"DA3 preprocessing reproduction failed for {camera_id} "
                    f"job {prediction.job.job_id}"
                )
            rgb_checks.append(check)

        for evidence in prediction.job.mask_evidence:
            metric = evidence.candidate_metrics[0]
            candidate = metric.candidate
            detection = candidate.source_detection
            camera_id = evidence.frame_identity.camera_id
            source_mask = _load_source_mask(
                project_root,
                detection.mask_ref,
                expected_detection_index=candidate.detection_index,
            )
            if int(np.count_nonzero(source_mask)) != metric.mask_area_pixels:
                raise ValueError("retained source mask area differs from S03 timeline")
            aligned = align_source_mask_to_da3_grid(
                source_mask,
                intrinsics=intrinsics_by_id[camera_id],
                process_resolution=prediction.job.process_resolution,
            )
            aligned_index = len(masks)
            masks.append(aligned.processed_mask)
            mask_metadata.append(
                {
                    "camera_id": camera_id,
                    "target": evidence.target.value,
                    "frame_id": evidence.frame_identity.frame_id,
                    "perception_job_id": evidence.job_id,
                    "source_mask_ref": detection.mask_ref,
                    "detection_index": candidate.detection_index,
                }
            )
            pending_records.append(
                _pending_mask_record(
                    prediction=prediction,
                    evidence=evidence,
                    aligned=aligned,
                    aligned_index=aligned_index,
                )
            )
            overlays_by_camera[camera_id] = _overlay_mask(
                overlays_by_camera[camera_id],
                aligned.processed_mask,
                target=evidence.target,
            )

        mask_path = masks_dir / f"{job_index:02d}_{prediction.job.job_id[:12]}.npz"
        np.savez_compressed(
            mask_path,
            masks=np.stack(masks, axis=0),
            metadata_json=np.asarray(json.dumps(mask_metadata, sort_keys=True)),
            action_depth_job_id=np.asarray(prediction.job.job_id),
            bundle_id=np.asarray(prediction.job.bundle.bundle_id),
            processed_shape=np.asarray([280, 504], dtype=np.int32),
            mask_interpolation=np.asarray("nearest"),
            localization_performed=np.asarray(False),
        )
        mask_sha256 = _sha256(mask_path)
        for item in pending_records:
            item["aligned_mask_artifact_ref"] = _relative(mask_path, project_root)
            item["aligned_mask_artifact_sha256"] = mask_sha256
            aligned_records.append(AlignedMaskRecord.model_validate(item))

        overlay_path = overlays_dir / (
            f"{job_index:02d}_{prediction.job.job_id[:12]}_alignment.png"
        )
        _save_pair_overlay(
            overlays_by_camera=overlays_by_camera,
            phase_id=prediction.job.phase_id,
            frame_index=prediction.job.bundle.frames[0].source_frame_index,
            path=overlay_path,
        )
        overlay_records.append(
            {
                "action_depth_job_id": prediction.job.job_id,
                "phase_id": prediction.job.phase_id,
                "source_frame_index": prediction.job.bundle.frames[
                    0
                ].source_frame_index,
                "overlay_ref": _relative(overlay_path, project_root),
                "overlay_sha256": _sha256(overlay_path),
            }
        )
        contact_rows.append((prediction.job.phase_id, overlays_by_camera))

    contact_sheet_path = output_dir / "mask_alignment_contact_sheet.png"
    _save_contact_sheet(contact_rows, contact_sheet_path)
    result = {
        "schema_version": 1,
        "status": "completed_pending_visual_qa",
        "stage": "S04",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_action_depth_summary_ref": _relative(summary_path, project_root),
        "source_action_depth_summary_sha256": _sha256(summary_path),
        "pose_calibration_ref": _relative(calibration_path, project_root),
        "pose_calibration_sha256": _sha256(calibration_path),
        "processing": {
            "source_mask_grid": "distorted 1920x1080 S03 source frame",
            "undistortion": (
                "OpenCV initUndistortRectifyMap with the accepted camera matrix "
                "as the output matrix"
            ),
            "mask_undistortion_interpolation": "nearest",
            "da3_process_res_method": "upper_bound_resize",
            "da3_resize_sequence": ["1920x1080", "504x284", "504x280"],
            "mask_resize_interpolation": "nearest",
            "batch_shape_unification": "not required; both views are 504x280",
            "crop_or_padding_applied": False,
            "mask_to_depth_localization_performed": False,
        },
        "rgb_reproduction_checks": rgb_checks,
        "aligned_masks": [item.model_dump(mode="json") for item in aligned_records],
        "job_overlays": overlay_records,
        "contact_sheet_ref": _relative(contact_sheet_path, project_root),
        "contact_sheet_sha256": _sha256(contact_sheet_path),
        "limitations": [
            "Overlay review checks mask/image registration qualitatively; it is "
            "not a segmentation ground-truth measurement.",
            "Nearest-neighbour mask sampling preserves binary membership but can "
            "change small-mask area at boundaries.",
            "No depth confidence threshold, robust mask aggregation, spatial "
            "anchor, back-projection, fusion, or XYZ output is selected here.",
        ],
    }
    validated = MaskAlignmentRunSummary.model_validate(result).model_dump(mode="json")
    summary_output = output_dir / "summary.json"
    summary_output.write_text(
        json.dumps(validated, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validated, indent=2))
    return 0


def _pending_mask_record(
    *,
    prediction: Any,
    evidence: Any,
    aligned: AlignedMask,
    aligned_index: int,
) -> dict[str, Any]:
    metric = evidence.candidate_metrics[0]
    detection = metric.candidate.source_detection
    processed_area = int(np.count_nonzero(aligned.processed_mask))
    return {
        "action_depth_job_id": prediction.job.job_id,
        "bundle_id": prediction.job.bundle.bundle_id,
        "camera_id": evidence.frame_identity.camera_id,
        "frame_id": evidence.frame_identity.frame_id,
        "source_frame_index": evidence.frame_identity.source_frame_index,
        "target": evidence.target.value,
        "perception_job_id": evidence.job_id,
        "source_mask_ref": detection.mask_ref,
        "detection_index": metric.candidate.detection_index,
        "vendor_class_name": detection.class_name,
        "camera_local_track_id": detection.camera_local_track_id,
        "source_mask_area_pixels": metric.mask_area_pixels,
        "undistorted_mask_area_pixels": int(
            np.count_nonzero(aligned.undistorted_source_mask)
        ),
        "processed_mask_area_pixels": processed_area,
        "processed_mask_fraction": processed_area
        / (aligned.transform.processed_width * aligned.transform.processed_height),
        "aligned_mask_artifact_ref": "pending",
        "aligned_mask_artifact_sha256": "0" * 64,
        "aligned_mask_index": aligned_index,
        "transform": aligned.transform.model_dump(mode="json"),
    }


def _load_source_mask(
    project_root: Path,
    mask_ref: str,
    *,
    expected_detection_index: int,
) -> UInt8Array:
    path_text, separator, fragment = mask_ref.partition("#mask_")
    if not separator or int(fragment) != expected_detection_index:
        raise ValueError("S03 mask reference fragment differs from detection index")
    path = _resolve(project_root, Path(path_text))
    with np.load(path, allow_pickle=False) as arrays:
        masks = np.asarray(arrays["source_sized_masks"])
        if (
            masks.dtype != np.uint8
            or masks.ndim != 3
            or expected_detection_index >= masks.shape[0]
        ):
            raise ValueError("S03 source mask artifact is invalid")
        return cast(UInt8Array, masks[expected_detection_index].copy())


def _overlay_mask(
    image_rgb: UInt8Array,
    mask: UInt8Array,
    *,
    target: PerceptionTarget,
) -> UInt8Array:
    if image_rgb.shape[:2] != mask.shape:
        raise ValueError("overlay mask and processed image shapes differ")
    result = image_rgb.astype(np.float64)
    foreground = mask.astype(bool)
    color = TARGET_COLORS_RGB[target]
    result[foreground] = result[foreground] * 0.52 + color * 0.48
    output = np.asarray(np.clip(result, 0, 255), dtype=np.uint8)
    contours, _hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(
        output,
        contours,
        -1,
        tuple(int(value) for value in color),
        2,
    )
    return output


def _save_pair_overlay(
    *,
    overlays_by_camera: dict[str, UInt8Array],
    phase_id: str,
    frame_index: int,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
    figure.suptitle(f"{phase_id} · source frame {frame_index}")
    for axis, camera_id in zip(axes, CAMERA_IDS, strict=True):
        axis.imshow(overlays_by_camera[camera_id])
        axis.set_title(camera_id)
        axis.axis("off")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_contact_sheet(
    rows: list[tuple[str, dict[str, UInt8Array]]], path: Path
) -> None:
    figure, axes = plt.subplots(
        len(rows),
        2,
        figsize=(14, 4.1 * len(rows)),
        constrained_layout=True,
    )
    for row_index, (phase_id, images) in enumerate(rows):
        for camera_index, camera_id in enumerate(CAMERA_IDS):
            axis = axes[row_index, camera_index]
            axis.imshow(images[camera_id])
            axis.set_title(f"{phase_id} · {camera_id}")
            axis.axis("off")
    figure.suptitle(
        "S04 source-mask alignment to DA3 504x280 grid\n"
        "cyan = person, magenta = backpack",
        fontsize=16,
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _camera_intrinsics(
    calibration: dict[str, Any],
) -> dict[str, CameraIntrinsics]:
    records = cast(dict[str, dict[str, Any]], calibration["cameras"])
    result: dict[str, CameraIntrinsics] = {}
    for camera_id in CAMERA_IDS:
        item = cast(dict[str, Any], records[camera_id]["intrinsics"])
        result[camera_id] = CameraIntrinsics(
            camera_id=camera_id,
            fx=float(item["fx"]),
            fy=float(item["fy"]),
            cx=float(item["cx"]),
            cy=float(item["cy"]),
            image_width=int(item["image_width"]),
            image_height=int(item["image_height"]),
            distortion_coefficients=tuple(
                float(value)
                for value in cast(list[float], item["distortion_coefficients"])
            ),
        )
    return result


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], payload)


def _require_hash(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"artifact hash changed for {path}: {actual} != {expected}")


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


if __name__ == "__main__":
    raise SystemExit(main())
