"""Rigid-transform operations with explicit source and target conventions."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spatial_reconstruction.contracts import CameraPose

FloatArray = NDArray[np.float64]


def validate_rigid_transform(
    T_target_from_source: ArrayLike,
    *,
    name: str = "T_target_from_source",
) -> FloatArray:
    """Return a validated float64 copy of one right-handed rigid transform."""

    matrix = np.asarray(T_target_from_source, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-7):
        raise ValueError(f"{name} must have homogeneous final row [0, 0, 0, 1]")

    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise ValueError(f"{name} rotation determinant must be +1")
    return matrix.copy()


def invert_rigid_transform(T_target_from_source: ArrayLike) -> FloatArray:
    """Compute ``T_source_from_target`` from a rigid ``T_target_from_source``."""

    matrix = validate_rigid_transform(T_target_from_source)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]

    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def transform_points(
    points_source: ArrayLike,
    *,
    T_target_from_source: ArrayLike,
) -> FloatArray:
    """Transform one point or an ``(N, 3)`` array into the target frame."""

    points, was_single = _as_point_matrix(points_source, name="points_source")
    matrix = validate_rigid_transform(T_target_from_source)
    transformed = np.asarray(
        points @ matrix[:3, :3].T + matrix[:3, 3],
        dtype=np.float64,
    )
    return transformed[0].copy() if was_single else transformed


def camera_points_to_world(points_camera: ArrayLike, *, pose: CameraPose) -> FloatArray:
    """Transform OpenCV camera-frame points into the declared world frame."""

    return transform_points(
        points_camera,
        T_target_from_source=pose.T_world_from_camera,
    )


def world_points_to_camera(points_world: ArrayLike, *, pose: CameraPose) -> FloatArray:
    """Transform world-frame points into the OpenCV camera frame."""

    return transform_points(
        points_world,
        T_target_from_source=pose.T_camera_from_world,
    )


def _as_point_matrix(points: ArrayLike, *, name: str) -> tuple[FloatArray, bool]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim == 1:
        if array.shape != (3,):
            raise ValueError(f"{name} must have shape (3,) or (N, 3), got {array.shape}")
        matrix = array.reshape(1, 3)
        was_single = True
    elif array.ndim == 2 and array.shape[1] == 3:
        matrix = array
        was_single = False
    else:
        raise ValueError(f"{name} must have shape (3,) or (N, 3), got {array.shape}")

    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix.copy(), was_single
