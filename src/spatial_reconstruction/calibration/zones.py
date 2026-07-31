"""Video-based zone geometry from calibrated camera rays and rope boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from spatial_reconstruction.contracts import CameraPose

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CameraGeometry:
    """OpenCV camera model required by the zone-estimation workflow."""

    pose: CameraPose
    camera_matrix: FloatArray
    distortion_coefficients: FloatArray


@dataclass(frozen=True)
class ZoneFit:
    """One deterministic horizontal circle fit."""

    center_world_m: FloatArray
    radius_m: float
    objective_rms_px: float
    projected_boundaries_px: dict[str, FloatArray]


def world_ray_from_pixel(
    pixel_px: ArrayLike,
    *,
    camera: CameraGeometry,
) -> tuple[FloatArray, FloatArray]:
    """Return one world-space ray origin and unit direction from a distorted pixel."""

    pixel = np.asarray(pixel_px, dtype=np.float64)
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        raise ValueError("pixel_px must be a finite vector with shape (2,)")
    undistorted = cv2.undistortPoints(
        pixel.reshape(1, 1, 2),
        camera.camera_matrix,
        camera.distortion_coefficients,
    ).reshape(2)
    direction_camera = np.array(
        [undistorted[0], undistorted[1], 1.0],
        dtype=np.float64,
    )
    world_from_camera = np.asarray(camera.pose.T_world_from_camera, dtype=np.float64)
    origin_world = world_from_camera[:3, 3].copy()
    direction_world = world_from_camera[:3, :3] @ direction_camera
    direction_world /= np.linalg.norm(direction_world)
    return origin_world, direction_world


def intersect_ray_with_horizontal_plane(
    origin_world_m: ArrayLike,
    direction_world: ArrayLike,
    *,
    plane_z_m: float,
) -> FloatArray:
    """Intersect a forward ray with a horizontal world-Z plane."""

    origin = np.asarray(origin_world_m, dtype=np.float64)
    direction = np.asarray(direction_world, dtype=np.float64)
    if origin.shape != (3,) or direction.shape != (3,):
        raise ValueError("ray origin and direction must both have shape (3,)")
    if not np.isfinite(origin).all() or not np.isfinite(direction).all():
        raise ValueError("ray origin and direction must be finite")
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        raise ValueError("ray direction must be non-zero")
    direction = direction / norm
    if abs(direction[2]) < 1e-9:
        raise ValueError("ray is parallel to the horizontal plane")
    distance = (float(plane_z_m) - origin[2]) / direction[2]
    if distance <= 0:
        raise ValueError("horizontal plane intersection lies behind the camera")
    point = origin + distance * direction
    point[2] = float(plane_z_m)
    return np.asarray(point, dtype=np.float64)


def closest_point_to_rays(
    rays: tuple[tuple[ArrayLike, ArrayLike], ...],
) -> tuple[FloatArray, FloatArray]:
    """Return the least-squares point closest to at least two world-space rays."""

    if len(rays) < 2:
        raise ValueError("at least two rays are required")
    normal_matrix = np.zeros((3, 3), dtype=np.float64)
    right_hand_side = np.zeros(3, dtype=np.float64)
    normalized_rays: list[tuple[FloatArray, FloatArray]] = []
    for raw_origin, raw_direction in rays:
        origin = np.asarray(raw_origin, dtype=np.float64)
        direction = np.asarray(raw_direction, dtype=np.float64)
        if origin.shape != (3,) or direction.shape != (3,):
            raise ValueError("ray origin and direction must both have shape (3,)")
        if not np.isfinite(origin).all() or not np.isfinite(direction).all():
            raise ValueError("ray origin and direction must be finite")
        direction = direction / np.linalg.norm(direction)
        projector = np.eye(3, dtype=np.float64) - np.outer(direction, direction)
        normal_matrix += projector
        right_hand_side += projector @ origin
        normalized_rays.append((origin, direction))
    if np.linalg.matrix_rank(normal_matrix) < 3:
        raise ValueError("rays do not constrain a unique 3D point")
    point = np.linalg.solve(normal_matrix, right_hand_side)
    signed_distances = np.asarray(
        [np.dot(point - origin, direction) for origin, direction in normalized_rays],
        dtype=np.float64,
    )
    if np.any(signed_distances <= 0):
        raise ValueError("triangulated point lies behind at least one camera")
    return np.asarray(point, dtype=np.float64), signed_distances


def fit_ellipse_boundary(points_px: ArrayLike, *, samples: int = 180) -> FloatArray:
    """Fit a dense ellipse boundary through sparse rope-boundary annotations."""

    points = np.asarray(points_px, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 5:
        raise ValueError("points_px must have shape (N, 2) with N >= 5")
    if not np.isfinite(points).all():
        raise ValueError("points_px must be finite")
    if samples < 16:
        raise ValueError("ellipse samples must be at least 16")
    ellipse = cv2.fitEllipse(points.astype(np.float32).reshape(-1, 1, 2))
    (center_x, center_y), (diameter_a, diameter_b), angle_degrees = ellipse
    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    local = np.column_stack(
        [
            (diameter_a / 2.0) * np.cos(angles),
            (diameter_b / 2.0) * np.sin(angles),
        ]
    )
    angle = np.deg2rad(angle_degrees)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    return local @ rotation.T + np.array([center_x, center_y], dtype=np.float64)


def ellipse_center(points_px: ArrayLike) -> FloatArray:
    """Return the centre of the ellipse fitted through rope annotations."""

    points = np.asarray(points_px, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 5:
        raise ValueError("points_px must have shape (N, 2) with N >= 5")
    ellipse = cv2.fitEllipse(points.astype(np.float32).reshape(-1, 1, 2))
    return np.asarray(ellipse[0], dtype=np.float64)


def project_horizontal_circle(
    center_world_m: ArrayLike,
    *,
    radius_m: float,
    camera: CameraGeometry,
    samples: int = 180,
) -> FloatArray:
    """Project a horizontal world-space circle through one distorted camera."""

    center = np.asarray(center_world_m, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("center_world_m must be finite with shape (3,)")
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("radius_m must be finite and positive")
    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    world_points = np.column_stack(
        [
            center[0] + radius_m * np.cos(angles),
            center[1] + radius_m * np.sin(angles),
            np.full(samples, center[2], dtype=np.float64),
        ]
    )
    camera_from_world = np.asarray(camera.pose.T_camera_from_world, dtype=np.float64)
    rotation = camera_from_world[:3, :3]
    translation = camera_from_world[:3, 3]
    camera_points = world_points @ rotation.T + translation
    if np.any(camera_points[:, 2] <= 0):
        raise ValueError("circle projects behind the camera")
    rvec, _ = cv2.Rodrigues(rotation)
    projected, _ = cv2.projectPoints(
        world_points,
        rvec,
        translation,
        camera.camera_matrix,
        camera.distortion_coefficients,
    )
    return np.asarray(projected, dtype=np.float64).reshape(-1, 2)


def fit_horizontal_circle_center(
    observed_boundaries_px: Mapping[str, ArrayLike],
    cameras: Mapping[str, CameraGeometry],
    initial_center_world_m: ArrayLike,
    *,
    radius_m: float,
    fixed_z_m: float | None,
    lower_bounds_world_m: ArrayLike,
    upper_bounds_world_m: ArrayLike,
    initial_step_m: float = 0.1,
    minimum_step_m: float = 0.0005,
) -> ZoneFit:
    """Fit a horizontal circle centre by deterministic projected-boundary search."""

    if set(observed_boundaries_px) != set(cameras):
        raise ValueError("observed boundary and camera IDs must match")
    observed = {
        camera_id: fit_ellipse_boundary(points)
        for camera_id, points in observed_boundaries_px.items()
    }
    center = np.asarray(initial_center_world_m, dtype=np.float64)
    lower = np.asarray(lower_bounds_world_m, dtype=np.float64)
    upper = np.asarray(upper_bounds_world_m, dtype=np.float64)
    if center.shape != (3,) or lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("initial centre and bounds must all have shape (3,)")
    if np.any(lower > upper) or np.any(center < lower) or np.any(center > upper):
        raise ValueError("initial centre must lie within ordered bounds")
    if fixed_z_m is not None:
        center[2] = float(fixed_z_m)
        lower[2] = float(fixed_z_m)
        upper[2] = float(fixed_z_m)

    active_axes = (0, 1) if fixed_z_m is not None else (0, 1, 2)

    def objective(candidate: FloatArray) -> float:
        squared_distances: list[FloatArray] = []
        for camera_id, camera in cameras.items():
            projected = project_horizontal_circle(
                candidate,
                radius_m=radius_m,
                camera=camera,
            )
            observed_points = observed[camera_id]
            pairwise = np.sum(
                np.square(observed_points[:, None, :] - projected[None, :, :]),
                axis=2,
            )
            squared_distances.append(np.min(pairwise, axis=1))
            squared_distances.append(np.min(pairwise, axis=0))
        return float(np.mean(np.concatenate(squared_distances)))

    best_score = objective(center)
    step = float(initial_step_m)
    while step >= minimum_step_m:
        improved = False
        for axis in active_axes:
            for direction in (-1.0, 1.0):
                candidate = center.copy()
                candidate[axis] = np.clip(
                    candidate[axis] + direction * step,
                    lower[axis],
                    upper[axis],
                )
                score = objective(candidate)
                if score + 1e-12 < best_score:
                    center = candidate
                    best_score = score
                    improved = True
        if not improved:
            step /= 2.0

    projected_boundaries = {
        camera_id: project_horizontal_circle(
            center,
            radius_m=radius_m,
            camera=camera,
        )
        for camera_id, camera in cameras.items()
    }
    return ZoneFit(
        center_world_m=center.copy(),
        radius_m=float(radius_m),
        objective_rms_px=float(np.sqrt(best_score)),
        projected_boundaries_px=projected_boundaries,
    )
