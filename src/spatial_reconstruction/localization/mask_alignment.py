"""Align source-sized S03 masks with DA3's processed action-depth grid."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import field_validator, model_validator

from spatial_reconstruction.contracts import (
    CameraIntrinsics,
    ContractModel,
    NonNegativeFloat,
    NonNegativeInt,
    PerceptionTarget,
    PositiveInt,
    Sha256Digest,
)

UInt8Array = NDArray[np.uint8]
Float64Array = NDArray[np.float64]


class DA3GridTransform(ContractModel):
    """Exact geometry of DA3's upper-bound resize preprocessing."""

    method: Literal["upper_bound_resize"] = "upper_bound_resize"
    patch_size: PositiveInt = 14
    process_resolution: PositiveInt
    source_width: PositiveInt
    source_height: PositiveInt
    boundary_width: PositiveInt
    boundary_height: PositiveInt
    processed_width: PositiveInt
    processed_height: PositiveInt
    batch_center_crop_left: NonNegativeInt = 0
    batch_center_crop_top: NonNegativeInt = 0
    image_boundary_interpolation: Literal["area", "cubic"]
    image_patch_interpolation: Literal["area", "cubic"]
    mask_interpolation: Literal["nearest"] = "nearest"

    @model_validator(mode="after")
    def validate_geometry(self) -> DA3GridTransform:
        if max(self.boundary_width, self.boundary_height) != self.process_resolution:
            raise ValueError("boundary resize must set the longest side to process resolution")
        if self.processed_width % self.patch_size or self.processed_height % self.patch_size:
            raise ValueError("processed dimensions must be divisible by DA3 patch size")
        if self.batch_center_crop_left or self.batch_center_crop_top:
            raise ValueError("current S04 equal-sized camera pair must not use batch cropping")
        return self


class AlignedMaskRecord(ContractModel):
    """Persistent identity and area metrics for one processed target mask."""

    action_depth_job_id: Sha256Digest
    bundle_id: Sha256Digest
    camera_id: Literal["camera_a", "camera_b"]
    frame_id: Sha256Digest
    source_frame_index: NonNegativeInt
    target: PerceptionTarget
    perception_job_id: Sha256Digest
    source_mask_ref: str
    detection_index: NonNegativeInt
    vendor_class_name: str
    camera_local_track_id: str
    source_mask_area_pixels: PositiveInt
    undistorted_mask_area_pixels: PositiveInt
    processed_mask_area_pixels: PositiveInt
    processed_mask_fraction: NonNegativeFloat
    aligned_mask_artifact_ref: str
    aligned_mask_artifact_sha256: Sha256Digest
    aligned_mask_index: NonNegativeInt
    transform: DA3GridTransform

    @field_validator(
        "source_mask_ref",
        "vendor_class_name",
        "camera_local_track_id",
        "aligned_mask_artifact_ref",
    )
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("aligned-mask text must be non-empty without outer whitespace")
        return value

    @model_validator(mode="after")
    def validate_fraction(self) -> AlignedMaskRecord:
        pixel_count = self.transform.processed_width * self.transform.processed_height
        expected = self.processed_mask_area_pixels / pixel_count
        if not np.isclose(self.processed_mask_fraction, expected, atol=1e-12):
            raise ValueError("processed mask fraction is inconsistent with its area")
        return self


class MaskAlignmentRunSummary(ContractModel):
    """Strict persistent summary for S04 mask-to-DA3-grid alignment."""

    schema_version: Literal[1]
    status: Literal["completed_pending_visual_qa"]
    stage: Literal["S04"]
    created_at_utc: datetime
    source_action_depth_summary_ref: str
    source_action_depth_summary_sha256: Sha256Digest
    pose_calibration_ref: str
    pose_calibration_sha256: Sha256Digest
    processing: dict[str, Any]
    rgb_reproduction_checks: tuple[dict[str, Any], ...]
    aligned_masks: tuple[AlignedMaskRecord, ...]
    job_overlays: tuple[dict[str, Any], ...]
    contact_sheet_ref: str
    contact_sheet_sha256: Sha256Digest
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_run(self) -> MaskAlignmentRunSummary:
        if not self.aligned_masks:
            raise ValueError("mask-alignment run must contain aligned masks")
        identity_keys = {
            (item.action_depth_job_id, item.camera_id, item.target)
            for item in self.aligned_masks
        }
        if len(identity_keys) != len(self.aligned_masks):
            raise ValueError("aligned masks cannot duplicate job/camera/target")
        job_ids = {item.action_depth_job_id for item in self.aligned_masks}
        if {str(item["action_depth_job_id"]) for item in self.job_overlays} != job_ids:
            raise ValueError("job overlays must cover every aligned-mask job")
        return self


@dataclass(frozen=True, slots=True)
class AlignedMask:
    """Runtime binary masks before and after DA3 preprocessing."""

    undistorted_source_mask: UInt8Array
    processed_mask: UInt8Array
    transform: DA3GridTransform

    def __post_init__(self) -> None:
        undistorted = _validated_binary_mask(self.undistorted_source_mask)
        processed = _validated_binary_mask(self.processed_mask)
        if undistorted.shape != (
            self.transform.source_height,
            self.transform.source_width,
        ):
            raise ValueError("undistorted mask shape differs from transform source")
        if processed.shape != (
            self.transform.processed_height,
            self.transform.processed_width,
        ):
            raise ValueError("processed mask shape differs from transform output")
        undistorted.setflags(write=False)
        processed.setflags(write=False)
        object.__setattr__(self, "undistorted_source_mask", undistorted)
        object.__setattr__(self, "processed_mask", processed)


def build_da3_upper_bound_resize_transform(
    *,
    source_width: int,
    source_height: int,
    process_resolution: int,
    patch_size: int = 14,
) -> DA3GridTransform:
    """Reproduce DA3 InputProcessor dimensions for upper_bound_resize."""

    if source_width <= 0 or source_height <= 0:
        raise ValueError("source dimensions must be positive")
    if process_resolution <= 0 or patch_size <= 0:
        raise ValueError("process resolution and patch size must be positive")
    scale = process_resolution / float(max(source_width, source_height))
    boundary_width = max(1, int(round(source_width * scale)))
    boundary_height = max(1, int(round(source_height * scale)))
    processed_width = max(1, _nearest_multiple(boundary_width, patch_size))
    processed_height = max(1, _nearest_multiple(boundary_height, patch_size))
    return DA3GridTransform(
        process_resolution=process_resolution,
        patch_size=patch_size,
        source_width=source_width,
        source_height=source_height,
        boundary_width=boundary_width,
        boundary_height=boundary_height,
        processed_width=processed_width,
        processed_height=processed_height,
        image_boundary_interpolation="cubic" if scale > 1.0 else "area",
        image_patch_interpolation=(
            "cubic"
            if processed_width > boundary_width or processed_height > boundary_height
            else "area"
        ),
    )


def resize_intrinsics_for_da3_grid(
    intrinsics: CameraIntrinsics,
    transform: DA3GridTransform,
) -> Float64Array:
    """Apply the same two DA3 resize steps to one OpenCV intrinsic matrix."""

    if (intrinsics.image_width, intrinsics.image_height) != (
        transform.source_width,
        transform.source_height,
    ):
        raise ValueError("intrinsic dimensions differ from DA3 grid transform")
    matrix = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    matrix[0] *= transform.boundary_width / transform.source_width
    matrix[1] *= transform.boundary_height / transform.source_height
    matrix[0] *= transform.processed_width / transform.boundary_width
    matrix[1] *= transform.processed_height / transform.boundary_height
    return matrix


def align_source_mask_to_da3_grid(
    mask: NDArray[np.generic],
    *,
    intrinsics: CameraIntrinsics,
    process_resolution: int,
) -> AlignedMask:
    """Undistort a source mask and reproduce DA3's two resize steps."""

    source = _validated_binary_mask(mask)
    if source.shape != (intrinsics.image_height, intrinsics.image_width):
        raise ValueError("source mask shape differs from camera intrinsics")
    matrix = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.asarray(intrinsics.distortion_coefficients, dtype=np.float64)
    map_x, map_y = cv2.initUndistortRectifyMap(
        matrix,
        distortion,
        np.eye(3, dtype=np.float64),
        matrix,
        (intrinsics.image_width, intrinsics.image_height),
        cv2.CV_32FC1,
    )
    undistorted = cv2.remap(
        source,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0,),
    )
    transform = build_da3_upper_bound_resize_transform(
        source_width=intrinsics.image_width,
        source_height=intrinsics.image_height,
        process_resolution=process_resolution,
    )
    boundary = cv2.resize(
        undistorted,
        (transform.boundary_width, transform.boundary_height),
        interpolation=cv2.INTER_NEAREST,
    )
    processed = cv2.resize(
        boundary,
        (transform.processed_width, transform.processed_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return AlignedMask(
        undistorted_source_mask=np.asarray(undistorted, dtype=np.uint8),
        processed_mask=np.asarray(processed, dtype=np.uint8),
        transform=transform,
    )


def transform_undistorted_rgb_to_da3_grid(
    image_rgb: NDArray[np.generic],
    *,
    process_resolution: int,
) -> tuple[UInt8Array, DA3GridTransform]:
    """Independently reproduce DA3's resize mapping for an undistorted RGB input."""

    image_array = np.asarray(image_rgb)
    if (
        image_array.dtype != np.uint8
        or image_array.ndim != 3
        or image_array.shape[2] != 3
    ):
        raise ValueError("DA3 RGB reproduction input must be uint8 H-by-W-by-3")
    image = cast(UInt8Array, image_array)
    transform = build_da3_upper_bound_resize_transform(
        source_width=int(image.shape[1]),
        source_height=int(image.shape[0]),
        process_resolution=process_resolution,
    )
    scale = process_resolution / float(max(image.shape[1], image.shape[0]))
    boundary = cv2.resize(
        image,
        (transform.boundary_width, transform.boundary_height),
        interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
    )
    upscale = (
        transform.processed_width > transform.boundary_width
        or transform.processed_height > transform.boundary_height
    )
    processed = cv2.resize(
        boundary,
        (transform.processed_width, transform.processed_height),
        interpolation=cv2.INTER_CUBIC if upscale else cv2.INTER_AREA,
    )
    return np.asarray(processed, dtype=np.uint8), transform


def _validated_binary_mask(mask: NDArray[np.generic]) -> UInt8Array:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    if array.dtype != np.uint8 and array.dtype != np.bool_:
        raise ValueError("mask must use uint8 or bool values")
    binary = np.ascontiguousarray(array != 0, dtype=np.uint8)
    if not np.any(binary):
        raise ValueError("mask must contain foreground pixels")
    return binary


def _nearest_multiple(value: int, patch: int) -> int:
    down = (value // patch) * patch
    up = down + patch
    return up if abs(up - value) <= abs(value - down) else down
