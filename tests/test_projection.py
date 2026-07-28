import numpy as np
import pytest

from spatial_reconstruction.contracts import CameraIntrinsics
from spatial_reconstruction.geometry.projection import (
    MissingCameraError,
    backproject_pixels,
    backproject_valid_pixels,
    depth_confidence_valid_mask,
    project_camera_points,
)


def make_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        camera_id="camera_01",
        fx=800.0,
        fy=820.0,
        cx=640.0,
        cy=360.0,
        image_width=1280,
        image_height=720,
    )


def test_projection_and_backprojection_round_trip() -> None:
    intrinsics = make_intrinsics()
    points_camera = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.5, -0.25, 4.0],
            [-0.75, 0.4, 3.0],
        ]
    )

    pixels = project_camera_points(points_camera, intrinsics=intrinsics)
    recovered = backproject_pixels(
        pixels,
        points_camera[:, 2],
        intrinsics=intrinsics,
    )

    np.testing.assert_allclose(recovered, points_camera, atol=1e-12)


def test_principal_point_backprojects_to_optical_axis() -> None:
    intrinsics = make_intrinsics()

    point = backproject_pixels(
        np.array([intrinsics.cx, intrinsics.cy]),
        2.5,
        intrinsics=intrinsics,
    )

    np.testing.assert_allclose(point, np.array([0.0, 0.0, 2.5]), atol=1e-12)


@pytest.mark.parametrize("depth", [0.0, -1.0, np.nan, np.inf])
def test_invalid_depth_is_rejected(depth: float) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        backproject_pixels(
            np.array([640.0, 360.0]),
            depth,
            intrinsics=make_intrinsics(),
        )


@pytest.mark.parametrize("z_depth", [0.0, -1.0])
def test_non_positive_camera_z_is_rejected(z_depth: float) -> None:
    with pytest.raises(ValueError, match="strictly positive Z"):
        project_camera_points(
            np.array([0.0, 0.0, z_depth]),
            intrinsics=make_intrinsics(),
        )


def test_depth_confidence_mask_filters_invalid_samples() -> None:
    depths = np.array([1.0, 0.0, -1.0, np.nan, 2.0, 3.0])
    confidence = np.array([0.9, 0.9, 0.9, 0.9, np.nan, 0.49])

    mask = depth_confidence_valid_mask(
        depths,
        confidence,
        minimum_confidence=0.5,
    )

    np.testing.assert_array_equal(mask, np.array([True, False, False, False, False, False]))


def test_valid_backprojection_returns_no_placeholder_xyz() -> None:
    intrinsics = make_intrinsics()
    pixels = np.array(
        [
            [640.0, 360.0],
            [650.0, 360.0],
            [660.0, 360.0],
            [670.0, 360.0],
        ]
    )
    depths = np.array([2.0, np.nan, 0.0, 4.0])
    confidence = np.array([0.9, 0.9, 0.9, 0.1])

    points, mask = backproject_valid_pixels(
        pixels,
        depths,
        confidence,
        intrinsics=intrinsics,
        minimum_confidence=0.5,
    )

    np.testing.assert_array_equal(mask, np.array([True, False, False, False]))
    assert points.shape == (1, 3)
    np.testing.assert_allclose(points[0], np.array([0.0, 0.0, 2.0]))


def test_no_valid_depth_returns_empty_observation_set() -> None:
    points, mask = backproject_valid_pixels(
        np.array([[640.0, 360.0], [650.0, 360.0]]),
        np.array([0.0, np.nan]),
        np.array([0.9, 0.9]),
        intrinsics=make_intrinsics(),
        minimum_confidence=0.5,
    )

    assert points.shape == (0, 3)
    np.testing.assert_array_equal(mask, np.array([False, False]))


def test_foreground_depth_is_not_interchangeable_with_static_depth() -> None:
    intrinsics = make_intrinsics()
    pixel = np.array([intrinsics.cx, intrinsics.cy])

    foreground_point = backproject_pixels(pixel, 1.5, intrinsics=intrinsics)
    static_background_point = backproject_pixels(pixel, 4.0, intrinsics=intrinsics)

    np.testing.assert_allclose(foreground_point, np.array([0.0, 0.0, 1.5]))
    np.testing.assert_allclose(static_background_point, np.array([0.0, 0.0, 4.0]))
    assert not np.allclose(foreground_point, static_background_point)


def test_missing_camera_intrinsics_raise_typed_error() -> None:
    with pytest.raises(MissingCameraError, match="required"):
        backproject_pixels(np.array([640.0, 360.0]), 2.0, intrinsics=None)


def test_sample_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        backproject_valid_pixels(
            np.array([[640.0, 360.0], [650.0, 360.0]]),
            np.array([2.0]),
            np.array([0.9, 0.9]),
            intrinsics=make_intrinsics(),
            minimum_confidence=0.5,
        )
