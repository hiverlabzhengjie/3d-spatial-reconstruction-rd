"""Estimate S01 pickup/drop-off rope-circle metadata from calibrated video views."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Final, cast

import av
import cv2
import numpy as np
from numpy.typing import NDArray

from spatial_reconstruction.calibration.zones import (
    CameraGeometry,
    closest_point_to_rays,
    ellipse_center,
    fit_ellipse_boundary,
    fit_horizontal_circle_center,
    intersect_ray_with_horizontal_plane,
    world_ray_from_pixel,
)
from spatial_reconstruction.contracts import CameraPose

FloatArray = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]
CAMERA_IDS: Final = ("camera_a", "camera_b")


def _load_json(path: Path) -> dict[str, Any]:
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


def _extract_frame(video_path: Path, timestamp_seconds: float) -> UInt8Array:
    best_frame: UInt8Array | None = None
    best_delta = float("inf")
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                continue
            timestamp = float(frame.pts * frame.time_base)
            delta = abs(timestamp - timestamp_seconds)
            if delta < best_delta:
                best_delta = delta
                best_frame = cast(UInt8Array, frame.to_ndarray(format="bgr24"))
            if timestamp > timestamp_seconds and best_frame is not None:
                break
    if best_frame is None:
        raise RuntimeError(f"could not decode a frame from {video_path}")
    return best_frame


def _camera_geometry(payload: dict[str, Any], camera_id: str) -> CameraGeometry:
    camera_payload = cast(dict[str, Any], payload["cameras"][camera_id])
    intrinsics = cast(dict[str, Any], camera_payload["intrinsics"])
    pose = CameraPose.model_validate(camera_payload["pose"])
    camera_matrix = np.array(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.asarray(
        intrinsics["distortion_coefficients"],
        dtype=np.float64,
    )
    return CameraGeometry(
        pose=pose,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
    )


def _point_to_ray_distance(
    point_world_m: FloatArray,
    origin_world_m: FloatArray,
    direction_world: FloatArray,
) -> float:
    direction = direction_world / np.linalg.norm(direction_world)
    offset = point_world_m - origin_world_m
    return float(np.linalg.norm(offset - np.dot(offset, direction) * direction))


def _boundary_rms_px(observed: FloatArray, projected: FloatArray) -> float:
    pairwise = np.sum(
        np.square(observed[:, None, :] - projected[None, :, :]),
        axis=2,
    )
    distances = np.concatenate([np.min(pairwise, axis=1), np.min(pairwise, axis=0)])
    return float(np.sqrt(np.mean(distances)))


def _draw_overlay(
    image: UInt8Array,
    *,
    zone_results: dict[str, dict[str, Any]],
    camera_id: str,
) -> UInt8Array:
    overlay = image.copy()
    colours = {
        "pickup_blue_bed": (255, 180, 0),
        "dropoff_white_floor": (255, 0, 255),
    }
    for zone_id, result in zone_results.items():
        colour = colours[zone_id]
        observations = np.asarray(
            result["boundary_annotations_px"][camera_id],
            dtype=np.float64,
        )
        observed_ellipse = np.asarray(
            result["observed_ellipse_px"][camera_id],
            dtype=np.float64,
        )
        projected = np.asarray(
            result["projected_boundary_px"][camera_id],
            dtype=np.float64,
        )
        for point in observations:
            cv2.circle(overlay, tuple(np.rint(point).astype(int)), 5, (0, 255, 0), -1)
        cv2.polylines(
            overlay,
            [np.rint(observed_ellipse).astype(np.int32)],
            True,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.polylines(
            overlay,
            [np.rint(projected).astype(np.int32)],
            True,
            colour,
            4,
            cv2.LINE_AA,
        )
        projected_center = np.mean(projected, axis=0)
        center_pixel = tuple(np.rint(projected_center).astype(int))
        cv2.drawMarker(
            overlay,
            center_pixel,
            colour,
            cv2.MARKER_CROSS,
            28,
            3,
            cv2.LINE_AA,
        )
        center = result["center_world_m"]
        label = (
            f"{zone_id}: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) m"
        )
        label_origin = (max(15, center_pixel[0] - 220), max(35, center_pixel[1] - 35))
        cv2.putText(
            overlay,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colour,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        overlay,
        (
            "green: annotated rope centreline | yellow: fitted image ellipse | "
            "coloured: projected 0.30 m world circle"
        ),
        (24, 1045),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        (
            "green: annotated rope centreline | yellow: fitted image ellipse | "
            "coloured: projected 0.30 m world circle"
        ),
        (24, 1045),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return overlay


def estimate_zones(
    project_root: Path,
    input_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = _load_json(input_path)
    calibration_path = project_root / str(config["pose_calibration"])
    calibration = _load_json(calibration_path)
    if calibration["pose_version_id"] != config["pose_version_id"]:
        raise ValueError("pose version does not match the declared zone-estimation input")

    videos = {
        camera_id: project_root / str(config["synchronized_videos"][camera_id])
        for camera_id in CAMERA_IDS
    }
    cameras = {
        camera_id: _camera_geometry(calibration, camera_id)
        for camera_id in CAMERA_IDS
    }
    frames = {
        camera_id: _extract_frame(
            videos[camera_id],
            float(config["representative_timestamp_seconds"]),
        )
        for camera_id in CAMERA_IDS
    }
    lower = np.asarray(
        config["world_plausibility_bounds_m"]["lower"],
        dtype=np.float64,
    )
    upper = np.asarray(
        config["world_plausibility_bounds_m"]["upper"],
        dtype=np.float64,
    )
    thresholds = cast(dict[str, Any], config["validation_thresholds"])
    user_validation = cast(dict[str, Any], config.get("user_validation", {}))
    user_accepted = bool(user_validation.get("accepted", False))
    zone_results: dict[str, dict[str, Any]] = {}

    for zone_id, raw_zone in cast(dict[str, dict[str, Any]], config["zones"]).items():
        boundaries = {
            camera_id: np.asarray(
                raw_zone["boundary_annotations_px"][camera_id],
                dtype=np.float64,
            )
            for camera_id in CAMERA_IDS
        }
        ellipse_centers = {
            camera_id: ellipse_center(boundaries[camera_id])
            for camera_id in CAMERA_IDS
        }
        rays = {
            camera_id: world_ray_from_pixel(
                ellipse_centers[camera_id],
                camera=cameras[camera_id],
            )
            for camera_id in CAMERA_IDS
        }
        fixed_z = raw_zone["fixed_z_m"]
        diagnostics: dict[str, Any]
        if fixed_z is not None:
            floor_points = {
                camera_id: intersect_ray_with_horizontal_plane(
                    *rays[camera_id],
                    plane_z_m=float(fixed_z),
                )
                for camera_id in CAMERA_IDS
            }
            disagreement_m = float(
                np.linalg.norm(floor_points["camera_a"] - floor_points["camera_b"])
            )
            initial_center = np.mean(np.stack(list(floor_points.values())), axis=0)
            diagnostics = {
                "independent_floor_intersections_world_m": {
                    camera_id: floor_points[camera_id].tolist()
                    for camera_id in CAMERA_IDS
                },
                "independent_camera_disagreement_m": disagreement_m,
            }
        else:
            initial_center, signed_distances = closest_point_to_rays(tuple(rays.values()))
            diagnostics = {
                "initial_ray_triangulation_world_m": initial_center.tolist(),
                "initial_ray_signed_distances_m": signed_distances.tolist(),
                "initial_ray_perpendicular_residuals_m": {
                    camera_id: _point_to_ray_distance(
                        initial_center,
                        *rays[camera_id],
                    )
                    for camera_id in CAMERA_IDS
                },
            }

        fit = fit_horizontal_circle_center(
            boundaries,
            cameras,
            initial_center,
            radius_m=float(raw_zone["radius_m"]),
            fixed_z_m=None if fixed_z is None else float(fixed_z),
            lower_bounds_world_m=lower,
            upper_bounds_world_m=upper,
            initial_step_m=0.05,
            minimum_step_m=0.00025,
        )
        observed_ellipses = {
            camera_id: fit_ellipse_boundary(boundaries[camera_id])
            for camera_id in CAMERA_IDS
        }
        per_camera_rms = {
            camera_id: _boundary_rms_px(
                observed_ellipses[camera_id],
                fit.projected_boundaries_px[camera_id],
            )
            for camera_id in CAMERA_IDS
        }
        within_bounds = bool(
            np.all(fit.center_world_m >= lower)
            and np.all(fit.center_world_m <= upper)
        )
        boundary_pass = all(
            value <= float(thresholds["maximum_ring_boundary_rms_px"])
            for value in per_camera_rms.values()
        )
        floor_agreement_pass = (
            True
            if fixed_z is None
            else diagnostics["independent_camera_disagreement_m"]
            <= float(thresholds["maximum_floor_camera_disagreement_m"])
        )
        zone_results[zone_id] = {
            "role": raw_zone["role"],
            "visual_cue": raw_zone["visual_cue"],
            "surface": raw_zone["surface"],
            "boundary_model": raw_zone["boundary_model"],
            "center_world_m": fit.center_world_m.tolist(),
            "radius_m": fit.radius_m,
            "coordinate_source": "video_estimated",
            "validation_status": (
                "accepted_video_estimated"
                if user_accepted
                else "pending_user_visual_sanity_check"
            ),
            "boundary_annotations_px": {
                camera_id: boundaries[camera_id].tolist()
                for camera_id in CAMERA_IDS
            },
            "ellipse_centers_px": {
                camera_id: ellipse_centers[camera_id].tolist()
                for camera_id in CAMERA_IDS
            },
            "observed_ellipse_px": {
                camera_id: observed_ellipses[camera_id].tolist()
                for camera_id in CAMERA_IDS
            },
            "projected_boundary_px": {
                camera_id: fit.projected_boundaries_px[camera_id].tolist()
                for camera_id in CAMERA_IDS
            },
            "ring_boundary_rms_px": per_camera_rms,
            "joint_ring_boundary_rms_px": fit.objective_rms_px,
            "diagnostics": diagnostics,
            "automated_checks": {
                "center_within_declared_room_bounds": within_bounds,
                "ring_boundary_reprojection_within_threshold": boundary_pass,
                "floor_camera_agreement_within_threshold": floor_agreement_pass,
            },
        }

    automated_pass = all(
        all(cast(dict[str, bool], zone["automated_checks"]).values())
        for zone in zone_results.values()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_paths: dict[str, str] = {}
    overlays: dict[str, UInt8Array] = {}
    for camera_id in CAMERA_IDS:
        overlay = _draw_overlay(
            frames[camera_id],
            zone_results=zone_results,
            camera_id=camera_id,
        )
        overlays[camera_id] = overlay
        overlay_path = output_dir / f"{camera_id}_zone_estimation_overlay.jpg"
        if not cv2.imwrite(str(overlay_path), overlay):
            raise RuntimeError(f"failed to write {overlay_path}")
        overlay_paths[camera_id] = str(overlay_path.relative_to(project_root))

    pair = np.hstack(
        [
            cv2.resize(overlays[camera_id], (960, 540), interpolation=cv2.INTER_AREA)
            for camera_id in CAMERA_IDS
        ]
    )
    pair_path = output_dir / "camera_pair_zone_estimation_overlay.jpg"
    if not cv2.imwrite(str(pair_path), pair):
        raise RuntimeError(f"failed to write {pair_path}")

    result: dict[str, Any] = {
        "schema_version": 1,
        "capture_session_id": config["capture_session_id"],
        "estimation_status": (
            "automated_checks_failed"
            if not automated_pass
            else (
                "accepted_video_estimated"
                if user_accepted
                else "automated_checks_passed_pending_user_validation"
            )
        ),
        "coordinate_convention": calibration["world_frame"],
        "pose_version_id": config["pose_version_id"],
        "representative_timestamp_seconds": config[
            "representative_timestamp_seconds"
        ],
        "input_provenance": {
            "input_config": str(input_path.relative_to(project_root)),
            "input_config_sha256": _sha256(input_path),
            "pose_calibration": str(calibration_path.relative_to(project_root)),
            "pose_calibration_sha256": _sha256(calibration_path),
            "synchronized_videos": {
                camera_id: {
                    "path": str(videos[camera_id].relative_to(project_root)),
                    "sha256": _sha256(videos[camera_id]),
                }
                for camera_id in CAMERA_IDS
            },
        },
        "boundary_interpretation": (
            "The blue and white ropes are thin circular boundaries. Pixel "
            "annotations follow the visible rope centreline; the enclosed areas "
            "are not treated as painted or colour-filled surfaces."
        ),
        "validation_thresholds": thresholds,
        "user_validation": user_validation,
        "zones": zone_results,
        "automated_checks_passed": automated_pass,
        "required_remaining_validation": (
            None
            if user_accepted
            else (
                "User must compare the projected coloured circles with both "
                "visible rope boundaries and sanity-check the estimated XYZ centres."
            )
        ),
        "diagnostic_overlays": {
            **overlay_paths,
            "camera_pair": str(pair_path.relative_to(project_root)),
        },
    }
    output_path = output_dir / "estimated_zones.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/s01/zones/zone_estimation_inputs.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/s01/zones"),
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    input_path = (project_root / args.input).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    result = estimate_zones(project_root, input_path, output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
