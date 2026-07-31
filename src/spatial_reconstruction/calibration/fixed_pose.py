"""Fixed-camera pose estimation from surveyed planar marker corners."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from spatial_reconstruction.contracts import CameraPose, Matrix4x4
from spatial_reconstruction.geometry.transforms import invert_rigid_transform

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PoseSolution:
    """One validated OpenCV world-to-camera pose and its reprojection evidence."""

    pose: CameraPose
    rvec: FloatArray
    tvec: FloatArray
    reprojected_image_points: FloatArray
    reprojection_errors_px: FloatArray

    @property
    def rms_reprojection_error_px(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.reprojection_errors_px))))

    @property
    def max_reprojection_error_px(self) -> float:
        return float(np.max(self.reprojection_errors_px))

    @property
    def camera_center_world_m(self) -> FloatArray:
        return np.asarray(self.pose.T_world_from_camera, dtype=np.float64)[:3, 3].copy()


def marker_corners_world(
    center_world_m: ArrayLike,
    *,
    marker_length_m: float,
) -> FloatArray:
    """Return OpenCV ArUco corner order for a floor marker aligned to +X/+Y.

    The printed marker's page top points toward world +Y and page right points
    toward world +X. OpenCV returns corners as top-left, top-right,
    bottom-right, bottom-left.
    """

    center = np.asarray(center_world_m, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("center_world_m must be a finite vector with shape (3,)")
    if not math.isfinite(marker_length_m) or marker_length_m <= 0:
        raise ValueError("marker_length_m must be finite and positive")

    half = marker_length_m / 2.0
    offsets = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )
    return center + offsets


def solve_fixed_camera_pose(
    object_points_world_m: ArrayLike,
    image_points_px: ArrayLike,
    camera_matrix: ArrayLike,
    distortion_coefficients: ArrayLike,
    *,
    camera_id: str,
) -> PoseSolution:
    """Solve and refine one fixed OpenCV camera pose from 3D/2D correspondences."""

    object_points = _point_matrix(
        object_points_world_m,
        columns=3,
        name="object_points_world_m",
    )
    image_points = _point_matrix(image_points_px, columns=2, name="image_points_px")
    if object_points.shape[0] != image_points.shape[0]:
        raise ValueError("object and image point counts must match")
    if object_points.shape[0] < 4:
        raise ValueError("at least four 3D/2D correspondences are required")
    if not camera_id.strip():
        raise ValueError("camera_id must not be blank")

    intrinsic_matrix = np.asarray(camera_matrix, dtype=np.float64)
    if intrinsic_matrix.shape != (3, 3) or not np.isfinite(intrinsic_matrix).all():
        raise ValueError("camera_matrix must be finite with shape (3, 3)")
    if intrinsic_matrix[0, 0] <= 0 or intrinsic_matrix[1, 1] <= 0:
        raise ValueError("camera focal lengths must be positive")

    distortion = np.asarray(distortion_coefficients, dtype=np.float64).reshape(-1)
    if not np.isfinite(distortion).all():
        raise ValueError("distortion_coefficients must be finite")

    solved, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        intrinsic_matrix,
        distortion,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not solved:
        raise RuntimeError(f"OpenCV could not solve a pose for camera '{camera_id}'")

    rvec, tvec = cv2.solvePnPRefineLM(
        object_points,
        image_points,
        intrinsic_matrix,
        distortion,
        rvec,
        tvec,
    )
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    rotation, _ = cv2.Rodrigues(rvec)

    T_camera_from_world = np.eye(4, dtype=np.float64)
    T_camera_from_world[:3, :3] = rotation
    T_camera_from_world[:3, 3] = tvec[:, 0]
    T_world_from_camera = invert_rigid_transform(T_camera_from_world)

    camera_points = object_points @ rotation.T + tvec[:, 0]
    if np.any(camera_points[:, 2] <= 0):
        raise RuntimeError(f"camera '{camera_id}' pose places marker points behind the camera")
    if T_world_from_camera[2, 3] <= 0:
        raise RuntimeError(f"camera '{camera_id}' pose places the camera below the floor")

    projected = project_world_points(
        object_points,
        rvec=rvec,
        tvec=tvec,
        camera_matrix=intrinsic_matrix,
        distortion_coefficients=distortion,
    )
    errors = np.linalg.norm(projected - image_points, axis=1)

    pose = CameraPose(
        camera_id=camera_id.strip(),
        T_world_from_camera=_matrix_tuple(T_world_from_camera),
        T_camera_from_world=_matrix_tuple(T_camera_from_world),
    )
    return PoseSolution(
        pose=pose,
        rvec=rvec.copy(),
        tvec=tvec.copy(),
        reprojected_image_points=projected,
        reprojection_errors_px=np.asarray(errors, dtype=np.float64),
    )


def project_world_points(
    object_points_world_m: ArrayLike,
    *,
    rvec: ArrayLike,
    tvec: ArrayLike,
    camera_matrix: ArrayLike,
    distortion_coefficients: ArrayLike,
) -> FloatArray:
    """Project world points through one OpenCV pose and distortion model."""

    object_points = _point_matrix(
        object_points_world_m,
        columns=3,
        name="object_points_world_m",
    )
    projected, _ = cv2.projectPoints(
        object_points,
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(distortion_coefficients, dtype=np.float64).reshape(-1),
    )
    return np.asarray(projected, dtype=np.float64).reshape(-1, 2)


def camera_optical_axis_world(pose: CameraPose) -> FloatArray:
    """Return the camera's OpenCV +Z optical axis expressed in world coordinates."""

    world_from_camera = np.asarray(pose.T_world_from_camera, dtype=np.float64)
    axis = world_from_camera[:3, 2]
    return axis / np.linalg.norm(axis)


def optical_axis_floor_intersection(pose: CameraPose) -> FloatArray | None:
    """Intersect the forward optical axis with world Z=0, if it points downward."""

    center = np.asarray(pose.T_world_from_camera, dtype=np.float64)[:3, 3]
    axis = camera_optical_axis_world(pose)
    if axis[2] >= -1e-9:
        return None
    distance = -center[2] / axis[2]
    if distance <= 0:
        return None
    point = center + distance * axis
    point[2] = 0.0
    return cast(FloatArray, point)


def rotation_difference_degrees(
    T_world_from_camera_a: ArrayLike,
    T_world_from_camera_b: ArrayLike,
) -> float:
    """Return the geodesic angular difference between two camera rotations."""

    first = np.asarray(T_world_from_camera_a, dtype=np.float64)
    second = np.asarray(T_world_from_camera_b, dtype=np.float64)
    if first.shape != (4, 4) or second.shape != (4, 4):
        raise ValueError("camera transforms must both have shape (4, 4)")
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    if math.isclose(cosine, 1.0, abs_tol=1e-12):
        return 0.0
    return math.degrees(math.acos(cosine))


def _point_matrix(points: ArrayLike, *, columns: int, name: str) -> FloatArray:
    matrix = np.asarray(points, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns}), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix.copy()


def _matrix_tuple(matrix: FloatArray) -> Matrix4x4:
    return cast(
        Matrix4x4,
        tuple(tuple(float(value) for value in row) for row in matrix),
    )
