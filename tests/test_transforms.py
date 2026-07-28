import numpy as np
import pytest

from spatial_reconstruction.contracts import CameraPose
from spatial_reconstruction.geometry.transforms import (
    camera_points_to_world,
    invert_rigid_transform,
    transform_points,
    validate_rigid_transform,
    world_points_to_camera,
)


def make_pose() -> CameraPose:
    world_from_camera = (
        (0.0, -1.0, 0.0, 1.0),
        (1.0, 0.0, 0.0, 2.0),
        (0.0, 0.0, 1.0, 0.5),
        (0.0, 0.0, 0.0, 1.0),
    )
    camera_from_world = invert_rigid_transform(world_from_camera)
    return CameraPose(
        camera_id="camera_01",
        T_world_from_camera=world_from_camera,
        T_camera_from_world=camera_from_world,
    )


def test_rigid_transform_inversion_multiplies_to_identity() -> None:
    pose = make_pose()
    world_from_camera = np.asarray(pose.T_world_from_camera)
    camera_from_world = invert_rigid_transform(world_from_camera)

    np.testing.assert_allclose(world_from_camera @ camera_from_world, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(camera_from_world, pose.T_camera_from_world, atol=1e-12)


def test_camera_world_point_round_trip() -> None:
    pose = make_pose()
    points_camera = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.5, -0.25, 2.0],
            [-1.0, 0.75, 4.0],
        ]
    )

    points_world = camera_points_to_world(points_camera, pose=pose)
    recovered_camera = world_points_to_camera(points_world, pose=pose)

    np.testing.assert_allclose(recovered_camera, points_camera, atol=1e-12)


def test_transform_points_preserves_single_point_shape() -> None:
    point = np.array([1.0, 2.0, 3.0])
    transformed = transform_points(point, T_target_from_source=np.eye(4))

    assert transformed.shape == (3,)
    np.testing.assert_allclose(transformed, point)


@pytest.mark.parametrize(
    "invalid_transform",
    [
        np.eye(3),
        np.full((4, 4), np.nan),
        np.diag([2.0, 1.0, 1.0, 1.0]),
        np.diag([-1.0, 1.0, 1.0, 1.0]),
    ],
)
def test_invalid_rigid_transforms_are_rejected(invalid_transform: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_rigid_transform(invalid_transform)


@pytest.mark.parametrize(
    "invalid_points",
    [
        np.array([1.0, 2.0]),
        np.zeros((2, 2)),
        np.array([1.0, np.inf, 3.0]),
    ],
)
def test_invalid_point_arrays_are_rejected(invalid_points: np.ndarray) -> None:
    with pytest.raises(ValueError):
        transform_points(invalid_points, T_target_from_source=np.eye(4))
