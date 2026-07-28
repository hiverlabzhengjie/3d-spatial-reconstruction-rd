"""OpenCV pinhole projection and depth back-projection."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spatial_reconstruction.contracts import CameraIntrinsics

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class MissingCameraError(ValueError):
    """Raised when an operation lacks required camera intrinsics."""


def project_camera_points(
    points_camera: ArrayLike,
    *,
    intrinsics: CameraIntrinsics | None,
) -> FloatArray:
    """Project positive-Z OpenCV camera points to ``(u, v)`` pixels."""

    camera = _require_intrinsics(intrinsics)
    points, was_single = _as_sample_matrix(
        points_camera,
        columns=3,
        name="points_camera",
    )
    depths = points[:, 2]
    if np.any(depths <= 0):
        raise ValueError("camera points must have finite, strictly positive Z depth")

    pixels = np.empty((points.shape[0], 2), dtype=np.float64)
    pixels[:, 0] = camera.fx * points[:, 0] / depths + camera.cx
    pixels[:, 1] = camera.fy * points[:, 1] / depths + camera.cy
    return pixels[0] if was_single else pixels


def backproject_pixels(
    pixels_uv: ArrayLike,
    depths_m: ArrayLike,
    *,
    intrinsics: CameraIntrinsics | None,
) -> FloatArray:
    """Back-project pixels and matching positive metric depths into camera XYZ."""

    camera = _require_intrinsics(intrinsics)
    pixels, was_single = _as_sample_matrix(pixels_uv, columns=2, name="pixels_uv")
    depths = _as_depth_vector(depths_m, sample_count=pixels.shape[0])
    _require_valid_depths(depths)

    points = np.empty((pixels.shape[0], 3), dtype=np.float64)
    points[:, 0] = (pixels[:, 0] - camera.cx) * depths / camera.fx
    points[:, 1] = (pixels[:, 1] - camera.cy) * depths / camera.fy
    points[:, 2] = depths
    return points[0] if was_single else points


def depth_confidence_valid_mask(
    depths_m: ArrayLike,
    confidence: ArrayLike,
    *,
    minimum_confidence: float,
) -> BoolArray:
    """Select finite positive depths with finite confidence above a threshold."""

    depths = np.asarray(depths_m, dtype=np.float64)
    scores = np.asarray(confidence, dtype=np.float64)
    if depths.shape != scores.shape:
        raise ValueError(
            f"depth and confidence shapes must match, got {depths.shape} and {scores.shape}"
        )
    if not math.isfinite(minimum_confidence):
        raise ValueError("minimum_confidence must be finite")
    valid = (
        np.isfinite(depths)
        & (depths > 0)
        & np.isfinite(scores)
        & (scores >= minimum_confidence)
    )
    return np.asarray(valid, dtype=np.bool_)


def backproject_valid_pixels(
    pixels_uv: ArrayLike,
    depths_m: ArrayLike,
    confidence: ArrayLike,
    *,
    intrinsics: CameraIntrinsics | None,
    minimum_confidence: float,
) -> tuple[FloatArray, BoolArray]:
    """Back-project only valid samples, returning no placeholder XYZ rows."""

    camera = _require_intrinsics(intrinsics)
    pixels, _ = _as_sample_matrix(pixels_uv, columns=2, name="pixels_uv")
    depths = _as_depth_vector(depths_m, sample_count=pixels.shape[0])
    scores = _as_depth_vector(
        confidence,
        sample_count=pixels.shape[0],
        name="confidence",
    )
    valid_mask = depth_confidence_valid_mask(
        depths,
        scores,
        minimum_confidence=minimum_confidence,
    )
    if not np.any(valid_mask):
        return np.empty((0, 3), dtype=np.float64), valid_mask

    points = backproject_pixels(
        pixels[valid_mask],
        depths[valid_mask],
        intrinsics=camera,
    )
    return np.asarray(points, dtype=np.float64).reshape(-1, 3), valid_mask


def _require_intrinsics(intrinsics: CameraIntrinsics | None) -> CameraIntrinsics:
    if intrinsics is None:
        raise MissingCameraError("camera intrinsics are required")
    return intrinsics


def _as_sample_matrix(
    samples: ArrayLike,
    *,
    columns: int,
    name: str,
) -> tuple[FloatArray, bool]:
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim == 1:
        if array.shape != (columns,):
            raise ValueError(
                f"{name} must have shape ({columns},) or (N, {columns}), got {array.shape}"
            )
        matrix = array.reshape(1, columns)
        was_single = True
    elif array.ndim == 2 and array.shape[1] == columns:
        matrix = array
        was_single = False
    else:
        raise ValueError(
            f"{name} must have shape ({columns},) or (N, {columns}), got {array.shape}"
        )

    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix.copy(), was_single


def _as_depth_vector(
    values: ArrayLike,
    *,
    sample_count: int,
    name: str = "depths_m",
) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1 or array.shape[0] != sample_count:
        raise ValueError(f"{name} must have shape ({sample_count},), got {array.shape}")
    return array.copy()


def _require_valid_depths(depths: FloatArray) -> None:
    if not np.isfinite(depths).all() or np.any(depths <= 0):
        raise ValueError("depths_m must contain only finite, strictly positive metric depths")
