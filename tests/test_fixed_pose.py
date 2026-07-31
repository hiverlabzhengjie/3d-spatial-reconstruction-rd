import cv2
import numpy as np
import pytest

from spatial_reconstruction.calibration.fixed_pose import (
    camera_optical_axis_world,
    marker_corners_world,
    optical_axis_floor_intersection,
    project_world_points,
    rotation_difference_degrees,
    solve_fixed_camera_pose,
)


def make_camera_transform() -> np.ndarray:
    center = np.array([1.0, 4.0, 2.2], dtype=np.float64)
    target = np.array([0.6, 1.6, 0.0], dtype=np.float64)
    forward = target - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)

    world_from_camera = np.eye(4, dtype=np.float64)
    world_from_camera[:3, :3] = np.column_stack([right, down, forward])
    world_from_camera[:3, 3] = center
    return world_from_camera


def test_marker_corners_follow_printed_page_axes() -> None:
    corners = marker_corners_world((1.0, 2.0, 0.0), marker_length_m=0.2)

    np.testing.assert_allclose(
        corners,
        np.array(
            [
                [0.9, 2.1, 0.0],
                [1.1, 2.1, 0.0],
                [1.1, 1.9, 0.0],
                [0.9, 1.9, 0.0],
            ]
        ),
    )


def test_synthetic_planar_pose_round_trip_and_reprojection() -> None:
    world_from_camera = make_camera_transform()
    camera_from_world = np.linalg.inv(world_from_camera)
    rotation = camera_from_world[:3, :3]
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = camera_from_world[:3, 3]
    camera_matrix = np.array(
        [[900.0, 0.0, 960.0], [0.0, 905.0, 540.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.zeros(5, dtype=np.float64)
    world_points = np.concatenate(
        [
            marker_corners_world((0.0, 0.0, 0.0), marker_length_m=0.18),
            marker_corners_world((1.2, 0.5, 0.0), marker_length_m=0.18),
            marker_corners_world((0.0, 2.2, 0.0), marker_length_m=0.18),
        ]
    )
    image_points = project_world_points(
        world_points,
        rvec=rvec,
        tvec=tvec,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
    )

    solution = solve_fixed_camera_pose(
        world_points,
        image_points,
        camera_matrix,
        distortion,
        camera_id="camera_a",
    )

    np.testing.assert_allclose(
        solution.pose.T_world_from_camera,
        world_from_camera,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(solution.pose.T_world_from_camera)
        @ np.asarray(solution.pose.T_camera_from_world),
        np.eye(4),
        atol=1e-10,
    )
    assert solution.rms_reprojection_error_px < 1e-7
    assert camera_optical_axis_world(solution.pose)[2] < 0
    assert optical_axis_floor_intersection(solution.pose) is not None


def test_rotation_difference_is_zero_for_same_pose() -> None:
    transform = make_camera_transform()
    assert rotation_difference_degrees(transform, transform) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("world_points", "image_points"),
    [
        (np.zeros((3, 3)), np.zeros((3, 2))),
        (np.zeros((4, 3)), np.zeros((5, 2))),
        (np.full((4, 3), np.nan), np.zeros((4, 2))),
    ],
)
def test_invalid_pose_inputs_are_rejected(
    world_points: np.ndarray,
    image_points: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        solve_fixed_camera_pose(
            world_points,
            image_points,
            np.eye(3),
            np.zeros(5),
            camera_id="camera_a",
        )
