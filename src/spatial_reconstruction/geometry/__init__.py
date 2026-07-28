"""Coordinate transforms and OpenCV projection utilities."""

from spatial_reconstruction.geometry.projection import (
    MissingCameraError,
    backproject_pixels,
    backproject_valid_pixels,
    depth_confidence_valid_mask,
    project_camera_points,
)
from spatial_reconstruction.geometry.transforms import (
    camera_points_to_world,
    invert_rigid_transform,
    transform_points,
    validate_rigid_transform,
    world_points_to_camera,
)

__all__ = [
    "MissingCameraError",
    "backproject_pixels",
    "backproject_valid_pixels",
    "camera_points_to_world",
    "depth_confidence_valid_mask",
    "invert_rigid_transform",
    "project_camera_points",
    "transform_points",
    "validate_rigid_transform",
    "world_points_to_camera",
]
