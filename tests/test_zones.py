import cv2
import numpy as np
import pytest

from spatial_reconstruction.calibration.zones import (
    CameraGeometry,
    closest_point_to_rays,
    fit_horizontal_circle_center,
    intersect_ray_with_horizontal_plane,
    project_horizontal_circle,
    world_ray_from_pixel,
)
from spatial_reconstruction.contracts import CameraPose
from spatial_reconstruction.geometry.transforms import invert_rigid_transform


def make_camera(camera_id: str, center: np.ndarray, target: np.ndarray) -> CameraGeometry:
    forward = target - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    world_from_camera = np.eye(4, dtype=np.float64)
    world_from_camera[:3, :3] = np.column_stack([right, down, forward])
    world_from_camera[:3, 3] = center
    camera_from_world = invert_rigid_transform(world_from_camera)
    pose = CameraPose(
        camera_id=camera_id,
        T_world_from_camera=world_from_camera,
        T_camera_from_world=camera_from_world,
    )
    return CameraGeometry(
        pose=pose,
        camera_matrix=np.array(
            [[900.0, 0.0, 960.0], [0.0, 900.0, 540.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        distortion_coefficients=np.zeros(5, dtype=np.float64),
    )


def project_point(point_world: np.ndarray, camera: CameraGeometry) -> np.ndarray:
    camera_from_world = np.asarray(camera.pose.T_camera_from_world)
    rotation = camera_from_world[:3, :3]
    translation = camera_from_world[:3, 3]
    rvec, _ = cv2.Rodrigues(rotation)
    pixel, _ = cv2.projectPoints(
        point_world.reshape(1, 3),
        rvec,
        translation,
        camera.camera_matrix,
        camera.distortion_coefficients,
    )
    return pixel.reshape(2)


def test_pixel_ray_intersects_known_floor_point() -> None:
    camera = make_camera(
        "camera_a",
        np.array([0.3, 4.0, 2.2]),
        np.array([0.8, 1.5, 0.0]),
    )
    expected = np.array([1.1, 1.8, 0.0])
    pixel = project_point(expected, camera)

    origin, direction = world_ray_from_pixel(pixel, camera=camera)
    intersection = intersect_ray_with_horizontal_plane(
        origin,
        direction,
        plane_z_m=0.0,
    )

    np.testing.assert_allclose(intersection, expected, atol=1e-9)


def test_closest_point_to_two_rays_recovers_world_point() -> None:
    expected = np.array([1.0, 1.7, 0.65])
    cameras = (
        make_camera("camera_a", np.array([0.3, 4.0, 2.2]), expected),
        make_camera("camera_b", np.array([2.3, 3.8, 2.2]), expected),
    )
    rays = tuple(
        world_ray_from_pixel(project_point(expected, camera), camera=camera)
        for camera in cameras
    )

    point, signed_distances = closest_point_to_rays(rays)

    np.testing.assert_allclose(point, expected, atol=1e-9)
    assert np.all(signed_distances > 0)


def test_horizontal_circle_fit_recovers_synthetic_zone() -> None:
    expected = np.array([1.0, 1.7, 0.65])
    cameras = {
        "camera_a": make_camera(
            "camera_a",
            np.array([0.3, 4.0, 2.2]),
            expected,
        ),
        "camera_b": make_camera(
            "camera_b",
            np.array([2.3, 3.8, 2.2]),
            expected,
        ),
    }
    boundaries = {
        camera_id: project_horizontal_circle(
            expected,
            radius_m=0.3,
            camera=camera,
            samples=24,
        )
        for camera_id, camera in cameras.items()
    }

    fit = fit_horizontal_circle_center(
        boundaries,
        cameras,
        np.array([0.92, 1.77, 0.72]),
        radius_m=0.3,
        fixed_z_m=None,
        lower_bounds_world_m=np.array([-0.5, -0.5, 0.2]),
        upper_bounds_world_m=np.array([3.0, 4.5, 1.5]),
        initial_step_m=0.05,
        minimum_step_m=0.0001,
    )

    np.testing.assert_allclose(fit.center_world_m, expected, atol=0.002)
    assert fit.objective_rms_px < 1.0


@pytest.mark.parametrize(
    "direction",
    [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ],
)
def test_invalid_floor_intersection_is_rejected(direction: np.ndarray) -> None:
    with pytest.raises(ValueError):
        intersect_ray_with_horizontal_plane(
            np.array([0.0, 0.0, 1.0]),
            direction,
            plane_z_m=0.0,
        )
