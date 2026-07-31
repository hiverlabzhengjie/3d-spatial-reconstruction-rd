"""Estimate fixed S01 camera poses and retain reprojection diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import av
import cv2
import numpy as np
from numpy.typing import NDArray

from spatial_reconstruction.calibration.fixed_pose import (
    PoseSolution,
    camera_optical_axis_world,
    marker_corners_world,
    optical_axis_floor_intersection,
    project_world_points,
    rotation_difference_degrees,
    solve_fixed_camera_pose,
)
from spatial_reconstruction.contracts import CameraIntrinsics

FloatArray = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]

CAMERA_IDS: Final = ("camera_a", "camera_b")
DICTIONARY_IDS: Final = {"DICT_5X5_100": cv2.aruco.DICT_5X5_100}


@dataclass(frozen=True)
class MarkerObservation:
    """One marker detection tied to an immutable decoded source frame."""

    frame_index: int
    timestamp_seconds: float
    marker_id: int
    corners_px: FloatArray


@dataclass(frozen=True)
class RepresentativeFrame:
    """Representative sampled frame used for a reprojection preview."""

    frame_index: int
    timestamp_seconds: float
    image_bgr: UInt8Array
    corners_by_marker: dict[int, FloatArray]


@dataclass(frozen=True)
class VideoScan:
    """Detections and video facts collected during deterministic sampling."""

    observations: tuple[MarkerObservation, ...]
    sampled_frame_count: int
    decoded_frame_count: int
    image_width: int
    image_height: int
    representative: RepresentativeFrame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], payload)


def _scan_video(
    video_path: Path,
    *,
    dictionary_name: str,
    marker_ids: set[int],
    start_seconds: float,
    end_seconds: float,
    sample_step: int,
    representative_seconds: float,
) -> VideoScan:
    if dictionary_name not in DICTIONARY_IDS:
        raise ValueError(f"unsupported ArUco dictionary: {dictionary_name}")
    if sample_step <= 0:
        raise ValueError("sample_step must be positive")

    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY_IDS[dictionary_name])
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    observations: list[MarkerObservation] = []
    representative: RepresentativeFrame | None = None
    representative_score: tuple[int, float] | None = None
    sampled_frame_count = 0
    decoded_frame_count = 0
    image_width = 0
    image_height = 0

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        image_width = int(stream.codec_context.width)
        image_height = int(stream.codec_context.height)
        for frame_index, frame in enumerate(container.decode(stream)):
            decoded_frame_count += 1
            if frame.pts is None or frame.time_base is None:
                continue
            timestamp_seconds = float(frame.pts * frame.time_base)
            if timestamp_seconds < start_seconds or timestamp_seconds > end_seconds:
                continue
            if frame_index % sample_step != 0:
                continue

            sampled_frame_count += 1
            image = cast(UInt8Array, frame.to_ndarray(format="bgr24"))
            detected_corners, detected_ids, _ = detector.detectMarkers(image)
            corners_by_marker: dict[int, FloatArray] = {}
            if detected_ids is not None:
                for raw_corners, raw_id in zip(
                    detected_corners,
                    detected_ids.reshape(-1),
                    strict=True,
                ):
                    marker_id = int(raw_id)
                    if marker_id not in marker_ids:
                        continue
                    corners = np.asarray(raw_corners, dtype=np.float64).reshape(4, 2)
                    corners_by_marker[marker_id] = corners
                    observations.append(
                        MarkerObservation(
                            frame_index=frame_index,
                            timestamp_seconds=timestamp_seconds,
                            marker_id=marker_id,
                            corners_px=corners.copy(),
                        )
                    )

            score = (len(corners_by_marker), -abs(timestamp_seconds - representative_seconds))
            if representative_score is None or score > representative_score:
                representative_score = score
                representative = RepresentativeFrame(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    image_bgr=image.copy(),
                    corners_by_marker={
                        marker_id: corners.copy()
                        for marker_id, corners in corners_by_marker.items()
                    },
                )

    if representative is None:
        raise RuntimeError(f"no sampled frames decoded from {video_path}")
    return VideoScan(
        observations=tuple(observations),
        sampled_frame_count=sampled_frame_count,
        decoded_frame_count=decoded_frame_count,
        image_width=image_width,
        image_height=image_height,
        representative=representative,
    )


def _observations_by_marker(
    observations: tuple[MarkerObservation, ...],
) -> dict[int, list[MarkerObservation]]:
    grouped: dict[int, list[MarkerObservation]] = {}
    for observation in observations:
        grouped.setdefault(observation.marker_id, []).append(observation)
    return grouped


def _median_marker_corners(
    observations: tuple[MarkerObservation, ...],
) -> tuple[dict[int, FloatArray], dict[int, dict[str, float]]]:
    medians: dict[int, FloatArray] = {}
    stability: dict[int, dict[str, float]] = {}
    for marker_id, marker_observations in _observations_by_marker(observations).items():
        samples = np.stack([observation.corners_px for observation in marker_observations])
        median = np.median(samples, axis=0)
        deviations = np.linalg.norm(samples - median[None, :, :], axis=2).reshape(-1)
        medians[marker_id] = np.asarray(median, dtype=np.float64)
        stability[marker_id] = {
            "detection_count": len(marker_observations),
            "corner_jitter_median_px": float(np.median(deviations)),
            "corner_jitter_p95_px": float(np.percentile(deviations, 95)),
            "corner_jitter_max_px": float(np.max(deviations)),
        }
    return medians, stability


def _marker_object_points(
    marker_centers: dict[int, FloatArray],
    marker_ids: tuple[int, ...],
    *,
    marker_length_m: float,
) -> FloatArray:
    return np.concatenate(
        [
            marker_corners_world(
                marker_centers[marker_id],
                marker_length_m=marker_length_m,
            )
            for marker_id in marker_ids
        ]
    )


def _marker_image_points(
    median_corners: dict[int, FloatArray],
    marker_ids: tuple[int, ...],
) -> FloatArray:
    return np.concatenate([median_corners[marker_id] for marker_id in marker_ids])


def _summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "median": None,
            "p95": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _per_frame_pose_stability(
    observations: tuple[MarkerObservation, ...],
    anchor_marker_ids: tuple[int, ...],
    marker_centers: dict[int, FloatArray],
    marker_length_m: float,
    camera_matrix: FloatArray,
    distortion: FloatArray,
    final_solution: PoseSolution,
    *,
    camera_id: str,
) -> dict[str, Any]:
    by_frame: dict[int, dict[int, MarkerObservation]] = {}
    for observation in observations:
        if observation.marker_id in anchor_marker_ids:
            by_frame.setdefault(observation.frame_index, {})[observation.marker_id] = observation

    object_points = _marker_object_points(
        marker_centers,
        anchor_marker_ids,
        marker_length_m=marker_length_m,
    )
    final_transform = np.asarray(final_solution.pose.T_world_from_camera, dtype=np.float64)
    final_center = final_solution.camera_center_world_m
    position_differences: list[float] = []
    rotation_differences: list[float] = []
    reprojection_rms_values: list[float] = []
    failed_frame_count = 0

    for frame_markers in by_frame.values():
        if any(marker_id not in frame_markers for marker_id in anchor_marker_ids):
            continue
        image_points = np.concatenate(
            [frame_markers[marker_id].corners_px for marker_id in anchor_marker_ids]
        )
        try:
            solution = solve_fixed_camera_pose(
                object_points,
                image_points,
                camera_matrix,
                distortion,
                camera_id=camera_id,
            )
        except (RuntimeError, ValueError):
            failed_frame_count += 1
            continue
        transform = np.asarray(solution.pose.T_world_from_camera, dtype=np.float64)
        position_differences.append(
            float(np.linalg.norm(solution.camera_center_world_m - final_center))
        )
        rotation_differences.append(
            rotation_difference_degrees(final_transform, transform)
        )
        reprojection_rms_values.append(solution.rms_reprojection_error_px)

    return {
        "successful_frame_pose_count": len(position_differences),
        "failed_frame_pose_count": failed_frame_count,
        "position_difference_from_final_m": _summarize(position_differences),
        "rotation_difference_from_final_degrees": _summarize(rotation_differences),
        "per_frame_reprojection_rms_px": _summarize(reprojection_rms_values),
    }


def _camera_diagnostics(
    scan: VideoScan,
    median_corners: dict[int, FloatArray],
    marker_stability: dict[int, dict[str, float]],
    marker_centers: dict[int, FloatArray],
    anchor_marker_ids: tuple[int, ...],
    marker_length_m: float,
    camera_matrix: FloatArray,
    distortion: FloatArray,
    thresholds: dict[str, Any],
    *,
    camera_id: str,
) -> tuple[PoseSolution, dict[str, Any]]:
    object_points = _marker_object_points(
        marker_centers,
        anchor_marker_ids,
        marker_length_m=marker_length_m,
    )
    image_points = _marker_image_points(median_corners, anchor_marker_ids)
    solution = solve_fixed_camera_pose(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        camera_id=camera_id,
    )

    per_marker_aggregate: dict[str, dict[str, float]] = {}
    predicted_by_marker: dict[int, FloatArray] = {}
    observation_errors: list[float] = []
    per_marker_observation_errors: dict[int, list[float]] = {
        marker_id: [] for marker_id in anchor_marker_ids
    }
    for marker_id in anchor_marker_ids:
        world_corners = marker_corners_world(
            marker_centers[marker_id],
            marker_length_m=marker_length_m,
        )
        predicted = project_world_points(
            world_corners,
            rvec=solution.rvec,
            tvec=solution.tvec,
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion,
        )
        predicted_by_marker[marker_id] = predicted
        aggregate_errors = np.linalg.norm(predicted - median_corners[marker_id], axis=1)
        per_marker_aggregate[str(marker_id)] = {
            "rms_px": float(np.sqrt(np.mean(np.square(aggregate_errors)))),
            "max_px": float(np.max(aggregate_errors)),
        }

    for observation in scan.observations:
        if observation.marker_id not in per_marker_observation_errors:
            continue
        errors = np.linalg.norm(
            predicted_by_marker[observation.marker_id] - observation.corners_px,
            axis=1,
        )
        values = [float(value) for value in errors]
        observation_errors.extend(values)
        per_marker_observation_errors[observation.marker_id].extend(values)

    pose_stability = _per_frame_pose_stability(
        scan.observations,
        anchor_marker_ids,
        marker_centers,
        marker_length_m,
        camera_matrix,
        distortion,
        solution,
        camera_id=camera_id,
    )
    center = solution.camera_center_world_m
    optical_axis = camera_optical_axis_world(solution.pose)
    floor_intersection = optical_axis_floor_intersection(solution.pose)
    margin = float(thresholds["floor_intersection_marker_envelope_margin_m"])
    all_centers = np.stack(list(marker_centers.values()))
    envelope_min = np.min(all_centers[:, :2], axis=0) - margin
    envelope_max = np.max(all_centers[:, :2], axis=0) + margin
    floor_intersection_inside_envelope = (
        floor_intersection is not None
        and bool(
            np.all(floor_intersection[:2] >= envelope_min)
            and np.all(floor_intersection[:2] <= envelope_max)
        )
    )

    observation_summary = _summarize(observation_errors)
    position_summary = cast(
        dict[str, float | int | None],
        pose_stability["position_difference_from_final_m"],
    )
    rotation_summary = cast(
        dict[str, float | int | None],
        pose_stability["rotation_difference_from_final_degrees"],
    )
    checks = {
        "aggregate_anchor_rms_below_threshold": bool(
            solution.rms_reprojection_error_px
            <= float(thresholds["aggregate_anchor_rms_px"])
        ),
        "sampled_observation_p95_below_threshold": bool(
            observation_summary["p95"] is not None
            and float(observation_summary["p95"])
            <= float(thresholds["sampled_observation_p95_px"])
        ),
        "pose_position_p95_below_threshold": bool(
            position_summary["p95"] is not None
            and float(position_summary["p95"])
            <= float(thresholds["pose_position_p95_m"])
        ),
        "pose_rotation_p95_below_threshold": bool(
            rotation_summary["p95"] is not None
            and float(rotation_summary["p95"])
            <= float(thresholds["pose_rotation_p95_degrees"])
        ),
        "camera_height_plausible": bool(
            float(thresholds["minimum_camera_height_m"])
            <= center[2]
            <= float(thresholds["maximum_camera_height_m"])
        ),
        "optical_axis_points_downward": bool(
            optical_axis[2]
            <= -float(thresholds["minimum_downward_optical_axis_z"])
        ),
        "optical_axis_intersects_marker_envelope": floor_intersection_inside_envelope,
    }

    extra_marker_diagnostics: dict[str, dict[str, Any]] = {}
    for marker_id, observed_median in sorted(median_corners.items()):
        if marker_id in anchor_marker_ids:
            continue
        expected = project_world_points(
            marker_corners_world(
                marker_centers[marker_id],
                marker_length_m=marker_length_m,
            ),
            rvec=solution.rvec,
            tvec=solution.tvec,
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion,
        )
        errors = np.linalg.norm(expected - observed_median, axis=1)
        extra_marker_diagnostics[str(marker_id)] = {
            "used_for_pose": False,
            "reason": "not complete in both cameras",
            "rms_reprojection_error_px": float(np.sqrt(np.mean(np.square(errors)))),
            "max_reprojection_error_px": float(np.max(errors)),
            "consistent_with_pose_at_anchor_threshold": bool(
                np.sqrt(np.mean(np.square(errors)))
                <= float(thresholds["aggregate_anchor_rms_px"])
            ),
        }

    diagnostics = {
        "sampled_frame_count": scan.sampled_frame_count,
        "decoded_frame_count": scan.decoded_frame_count,
        "detection_counts_and_jitter": {
            str(marker_id): values
            for marker_id, values in sorted(marker_stability.items())
        },
        "aggregate_anchor_reprojection": {
            "rms_px": solution.rms_reprojection_error_px,
            "max_px": solution.max_reprojection_error_px,
            "per_marker": per_marker_aggregate,
        },
        "all_sampled_anchor_corner_errors_px": observation_summary,
        "per_marker_sampled_corner_errors_px": {
            str(marker_id): _summarize(values)
            for marker_id, values in sorted(per_marker_observation_errors.items())
        },
        "pose_stability": pose_stability,
        "camera_center_world_m": [float(value) for value in center],
        "optical_axis_world_unit": [float(value) for value in optical_axis],
        "optical_axis_floor_intersection_world_m": (
            None
            if floor_intersection is None
            else [float(value) for value in floor_intersection]
        ),
        "floor_intersection_validation_envelope_xy_m": {
            "min": [float(value) for value in envelope_min],
            "max": [float(value) for value in envelope_max],
        },
        "extra_marker_diagnostics": extra_marker_diagnostics,
        "validation_checks": checks,
        "accepted": bool(all(checks.values())),
    }
    return solution, diagnostics


def _draw_reprojection_preview(
    representative: RepresentativeFrame,
    marker_centers: dict[int, FloatArray],
    anchor_marker_ids: tuple[int, ...],
    marker_length_m: float,
    camera_matrix: FloatArray,
    distortion: FloatArray,
    solution: PoseSolution,
    output_path: Path,
    *,
    camera_id: str,
) -> None:
    image = representative.image_bgr.copy()
    for marker_id, center in sorted(marker_centers.items()):
        expected = project_world_points(
            marker_corners_world(center, marker_length_m=marker_length_m),
            rvec=solution.rvec,
            tvec=solution.tvec,
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion,
        )
        expected_integer = np.rint(expected).astype(np.int32)
        is_anchor = marker_id in anchor_marker_ids
        predicted_color = (255, 255, 0) if is_anchor else (0, 0, 255)
        cv2.polylines(
            image,
            [expected_integer.reshape(-1, 1, 2)],
            isClosed=True,
            color=predicted_color,
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        for point in expected_integer:
            cv2.drawMarker(
                image,
                tuple(int(value) for value in point),
                predicted_color,
                markerType=cv2.MARKER_CROSS,
                markerSize=14,
                thickness=2,
                line_type=cv2.LINE_AA,
            )

        observed = representative.corners_by_marker.get(marker_id)
        if observed is not None:
            observed_integer = np.rint(observed).astype(np.int32)
            cv2.polylines(
                image,
                [observed_integer.reshape(-1, 1, 2)],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            label_origin = tuple(int(value) for value in observed_integer[0])
            cv2.putText(
                image,
                f"M{marker_id}{'' if is_anchor else ' excluded'}",
                (label_origin[0] + 8, label_origin[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0) if is_anchor else (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

    cv2.drawFrameAxes(
        image,
        camera_matrix,
        distortion,
        solution.rvec,
        solution.tvec,
        0.5,
        3,
    )
    center = solution.camera_center_world_m
    lines = [
        (
            f"{camera_id} frame {representative.frame_index} "
            f"t={representative.timestamp_seconds:.3f}s"
        ),
        (
            f"C_world=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) m  "
            f"anchor RMS={solution.rms_reprojection_error_px:.3f}px"
        ),
        "green=detected  cyan=anchor reprojection  red=non-anchor expected position",
    ]
    for index, text in enumerate(lines):
        y = 40 + index * 34
        cv2.putText(
            image,
            text,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"failed to write preview {output_path}")


def _intrinsic_records(
    intrinsic_summary: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
) -> tuple[dict[str, dict[str, Any]], FloatArray, FloatArray]:
    calibration = intrinsic_summary.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("intrinsic summary lacks calibration results")
    camera_matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(calibration["distortion_coefficients"], dtype=np.float64)
    records: dict[str, dict[str, Any]] = {}
    for camera_id in CAMERA_IDS:
        intrinsics = CameraIntrinsics(
            camera_id=camera_id,
            fx=float(camera_matrix[0, 0]),
            fy=float(camera_matrix[1, 1]),
            cx=float(camera_matrix[0, 2]),
            cy=float(camera_matrix[1, 2]),
            image_width=image_width,
            image_height=image_height,
            distortion_coefficients=tuple(float(value) for value in distortion),
        )
        records[camera_id] = {
            **intrinsics.model_dump(mode="json"),
            "numeric_parameter_source": "shared camera_ab_intrinsics.mp4 capture",
            "independently_calibrated": camera_id == "camera_a",
        }
    return records, camera_matrix, distortion


def _validate_non_collinear_markers(
    marker_centers: dict[int, FloatArray],
    marker_ids: tuple[int, ...],
) -> None:
    centers_xy = np.stack([marker_centers[marker_id][:2] for marker_id in marker_ids])
    if len(marker_ids) < 3 or np.linalg.matrix_rank(centers_xy - centers_xy.mean(axis=0)) < 2:
        raise RuntimeError("at least three non-collinear common markers are required")


def run(input_config_path: Path, output_dir: Path) -> Path:
    """Run deterministic pose estimation and return the calibration JSON path."""

    config = _load_json(input_config_path)
    calibration_role = str(config.get("calibration_role", "fixed_world_reference"))
    pose_version_id = str(
        config.get(
            "pose_version_id",
            f"{config['capture_session_id']}:{calibration_role}",
        )
    )
    sync_manifest_path = Path(str(config["synchronization_manifest"]))
    intrinsic_summary_path = Path(str(config["intrinsic_review_summary"]))
    intrinsic_source_path = Path(str(config["intrinsic_source_video"]))
    sync_manifest = _load_json(sync_manifest_path)
    intrinsic_summary = _load_json(intrinsic_summary_path)
    reference_calibration_path_value = config.get("reference_calibration")
    reference_calibration_path = (
        None
        if reference_calibration_path_value is None
        else Path(str(reference_calibration_path_value))
    )
    reference_calibration = (
        None
        if reference_calibration_path is None
        else _load_json(reference_calibration_path)
    )
    marker_config = cast(dict[str, Any], config["marker_set"])
    sampling = cast(dict[str, Any], config["sampling"])
    thresholds = cast(dict[str, Any], config["validation_thresholds"])
    marker_length_m = float(marker_config["marker_length_m"])
    marker_centers = {
        int(placement["marker_id"]): np.asarray(
            placement["center_world_m"],
            dtype=np.float64,
        )
        for placement in marker_config["placements"]
    }
    marker_ids = set(marker_centers)

    derived_outputs = cast(dict[str, dict[str, Any]], sync_manifest["derived_outputs"])
    video_paths = {
        camera_id: Path(str(derived_outputs[camera_id]["path"]))
        for camera_id in CAMERA_IDS
    }
    scans = {
        camera_id: _scan_video(
            video_paths[camera_id],
            dictionary_name=str(marker_config["dictionary"]),
            marker_ids=marker_ids,
            start_seconds=float(sampling["start_seconds"]),
            end_seconds=float(sampling["end_seconds"]),
            sample_step=int(sampling["sample_every_nth_decoded_frame"]),
            representative_seconds=float(sampling["representative_timestamp_seconds"]),
        )
        for camera_id in CAMERA_IDS
    }
    widths = {scan.image_width for scan in scans.values()}
    heights = {scan.image_height for scan in scans.values()}
    if len(widths) != 1 or len(heights) != 1:
        raise RuntimeError("both fixed-pose videos must have the same dimensions")
    image_width = widths.pop()
    image_height = heights.pop()

    intrinsic_records, camera_matrix, distortion = _intrinsic_records(
        intrinsic_summary,
        image_width=image_width,
        image_height=image_height,
    )
    medians_and_stability = {
        camera_id: _median_marker_corners(scan.observations)
        for camera_id, scan in scans.items()
    }
    minimum_detections = int(sampling["minimum_marker_detections_per_camera"])
    common_marker_ids = tuple(
        sorted(
            marker_id
            for marker_id in marker_ids
            if all(
                marker_id in medians_and_stability[camera_id][1]
                and int(
                    medians_and_stability[camera_id][1][marker_id]["detection_count"]
                )
                >= minimum_detections
                for camera_id in CAMERA_IDS
            )
        )
    )
    _validate_non_collinear_markers(marker_centers, common_marker_ids)

    output_dir.mkdir(parents=True, exist_ok=True)
    camera_results: dict[str, dict[str, Any]] = {}
    solutions: dict[str, PoseSolution] = {}
    preview_paths: dict[str, Path] = {}
    for camera_id in CAMERA_IDS:
        median_corners, marker_stability = medians_and_stability[camera_id]
        solution, diagnostics = _camera_diagnostics(
            scans[camera_id],
            median_corners,
            marker_stability,
            marker_centers,
            common_marker_ids,
            marker_length_m,
            camera_matrix,
            distortion,
            thresholds,
            camera_id=camera_id,
        )
        solutions[camera_id] = solution
        preview_path = output_dir / f"{camera_id}_reprojection_preview.jpg"
        preview_paths[camera_id] = preview_path
        _draw_reprojection_preview(
            scans[camera_id].representative,
            marker_centers,
            common_marker_ids,
            marker_length_m,
            camera_matrix,
            distortion,
            solution,
            preview_path,
            camera_id=camera_id,
        )
        camera_results[camera_id] = {
            "intrinsics": intrinsic_records[camera_id],
            "pose": solution.pose.model_dump(mode="json"),
            "opencv_pose_parameters": {
                "rvec_world_to_camera": [
                    float(value) for value in solution.rvec.reshape(-1)
                ],
                "tvec_world_to_camera_m": [
                    float(value) for value in solution.tvec.reshape(-1)
                ],
            },
            "diagnostics": diagnostics,
            "reprojection_preview": str(preview_path),
        }

    reference_pose_acceptance = {camera_id: True for camera_id in CAMERA_IDS}
    if reference_calibration is not None:
        reference_thresholds = cast(
            dict[str, Any],
            config["reference_validation_thresholds"],
        )
        for camera_id in CAMERA_IDS:
            reference_transform = np.asarray(
                reference_calibration["cameras"][camera_id]["pose"][
                    "T_world_from_camera"
                ],
                dtype=np.float64,
            )
            capture_transform = np.asarray(
                solutions[camera_id].pose.T_world_from_camera,
                dtype=np.float64,
            )
            center_delta = capture_transform[:3, 3] - reference_transform[:3, 3]
            center_delta_norm = float(np.linalg.norm(center_delta))
            rotation_delta = rotation_difference_degrees(
                reference_transform,
                capture_transform,
            )
            checks = {
                "camera_center_delta_within_threshold": bool(
                    center_delta_norm
                    <= float(reference_thresholds["maximum_camera_center_delta_m"])
                ),
                "rotation_delta_within_threshold": bool(
                    rotation_delta
                    <= float(reference_thresholds["maximum_rotation_delta_degrees"])
                ),
            }
            reference_pose_acceptance[camera_id] = bool(all(checks.values()))
            camera_results[camera_id]["reference_pose_comparison"] = {
                "reference_pose_version_id": reference_calibration.get(
                    "pose_version_id",
                    "fixed_world_reference",
                ),
                "camera_center_delta_world_m": [
                    float(value) for value in center_delta
                ],
                "camera_center_delta_norm_m": center_delta_norm,
                "rotation_delta_degrees": rotation_delta,
                "validation_checks": checks,
                "accepted": reference_pose_acceptance[camera_id],
            }

    preview_images = [
        cv2.imread(str(preview_paths[camera_id]), cv2.IMREAD_COLOR)
        for camera_id in CAMERA_IDS
    ]
    if any(image is None for image in preview_images):
        raise RuntimeError("could not reload per-camera reprojection previews")
    preview_pair = cv2.hconcat(
        [
            cv2.resize(cast(UInt8Array, image), (960, 540), interpolation=cv2.INTER_AREA)
            for image in preview_images
        ]
    )
    pair_preview_path = output_dir / "camera_pair_reprojection_preview.jpg"
    if not cv2.imwrite(
        str(pair_preview_path),
        preview_pair,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    ):
        raise RuntimeError(f"failed to write {pair_preview_path}")

    camera_acceptance = {
        camera_id: bool(
            camera_results[camera_id]["diagnostics"]["accepted"]
            and reference_pose_acceptance[camera_id]
        )
        for camera_id in CAMERA_IDS
    }
    non_anchor_markers = sorted(marker_ids - set(common_marker_ids))
    if all(camera_acceptance.values()):
        calibration_status = (
            "accepted_capture_specific_pose_correction"
            if reference_calibration is not None
            else "accepted_with_documented_marker_exclusion"
        )
    else:
        calibration_status = "failed_validation"
    output = {
        "schema_version": 1,
        "capture_session_id": config["capture_session_id"],
        "calibration_role": calibration_role,
        "pose_version_id": pose_version_id,
        "calibration_status": calibration_status,
        "world_frame": config["world_frame"],
        "marker_model": {
            **marker_config,
            "pose_anchor_marker_ids": list(common_marker_ids),
            "markers_not_used_for_pose": non_anchor_markers,
            "anchor_selection_rule": (
                "measured markers with at least the configured detection count "
                "in both synchronized cameras"
            ),
        },
        "input_provenance": {
            "input_config": str(input_config_path),
            "input_config_sha256": _sha256(input_config_path),
            "synchronization_manifest": str(sync_manifest_path),
            "synchronization_manifest_sha256": _sha256(sync_manifest_path),
            "intrinsic_review_summary": str(intrinsic_summary_path),
            "intrinsic_review_summary_sha256": _sha256(intrinsic_summary_path),
            "intrinsic_source_video": str(intrinsic_source_path),
            "intrinsic_source_video_sha256": _sha256(intrinsic_source_path),
            "synchronized_videos": {
                camera_id: {
                    "path": str(video_paths[camera_id]),
                    "sha256": _sha256(video_paths[camera_id]),
                }
                for camera_id in CAMERA_IDS
            },
            "raw_recording_sources": sync_manifest["sources"],
            "reference_calibration": (
                None
                if reference_calibration_path is None
                else {
                    "path": str(reference_calibration_path),
                    "sha256": _sha256(reference_calibration_path),
                }
            ),
        },
        "sampling": sampling,
        "validation_thresholds": thresholds,
        "reference_validation_thresholds": config.get(
            "reference_validation_thresholds"
        ),
        "cameras": camera_results,
        "pair_diagnostics": {
            "camera_centres_distance_m": float(
                np.linalg.norm(
                    solutions["camera_a"].camera_center_world_m
                    - solutions["camera_b"].camera_center_world_m
                )
            ),
            "both_camera_poses_accepted": all(camera_acceptance.values()),
            "shared_camera_b_intrinsics_accepted_by_world_marker_reprojection": (
                camera_acceptance["camera_b"]
            ),
            "pair_preview": str(pair_preview_path),
        },
        "limitations": [
            (
                "Marker centre measurements have stated +/-0.05 m uncertainty, "
                "so the resulting camera positions are prototype-grade rather "
                "than survey-grade."
            ),
            (
                "M43 is not a pose anchor because it is not completely detected "
                "in Camera B; its Camera A reprojection is retained as an "
                "independent consistency diagnostic."
            ),
            (
                "Camera B uses the shared Camera A intrinsic estimate under D021; "
                "the world-marker validation recorded here is the required check."
            ),
        ],
        "reproduction_command": (
            ".venv/bin/python scripts/calibration/estimate_fixed_poses.py "
            f"--input-config {input_config_path} --output-dir {output_dir}"
        ),
    }
    output_path = output_dir / "camera_calibration.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-config",
        type=Path,
        default=Path("artifacts/s01/calibration/world_pose_inputs.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/s01/calibration/fixed_pose"),
    )
    args = parser.parse_args()
    output_path = run(args.input_config, args.output_dir)
    print(output_path)


if __name__ == "__main__":
    main()
