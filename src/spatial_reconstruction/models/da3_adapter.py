"""Pose-conditioned DA3 adapter with explicit project coordinate names."""

from __future__ import annotations

import hashlib
import importlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, ModuleType
from typing import Any, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import nn

from spatial_reconstruction.contracts import CameraIntrinsics, CameraPose
from spatial_reconstruction.geometry import invert_rigid_transform
from spatial_reconstruction.models.da3_mps import (
    AutocastPolicy,
    DA3Precision,
    install_device_safe_forward,
)

FloatArray = NDArray[np.float64]
Float32Array = NDArray[np.float32]
UInt8Array = NDArray[np.uint8]

EXPECTED_DA3_VENDOR_FINGERPRINT = (
    "683cad1fec1186cd2a22f2b6d083b73d4c83c7ab1140f45ba24876612bc51d43"
)
TWO_VIEW_ALIGNMENT_POLICY = "preserve_nested_metric_depth_and_return_supplied_poses"


class DA3PredictionError(ValueError):
    """Raised when DA3 returns an invalid or incomplete prediction."""


class _VendorPrediction(Protocol):
    depth: object
    conf: object
    extrinsics: object
    intrinsics: object
    processed_images: object
    is_metric: object


class _VendorModel(Protocol):
    def inference(self, image: list[str], **kwargs: object) -> _VendorPrediction: ...


@dataclass(frozen=True, slots=True)
class DA3Output:
    """Validated project-owned view of one DA3 prediction."""

    depth_m: Float32Array
    confidence: Float32Array
    T_camera_from_world: Float32Array
    intrinsics: Float32Array
    processed_images: UInt8Array | None
    is_metric: bool


class DA3Adapter:
    """Thin wrapper around the unmodified DA3 vendor API."""

    def __init__(
        self,
        *,
        model: _VendorModel,
        model_id: str,
        model_revision: str,
        device: torch.device,
        autocast_policy: AutocastPolicy,
    ) -> None:
        self._model = model
        self.model_id = model_id
        self.model_revision = model_revision
        self.device = device
        self.autocast_policy = autocast_policy

    @classmethod
    def from_pretrained(
        cls,
        *,
        vendor_dir: Path,
        model_id: str,
        model_revision: str,
        device: torch.device,
        precision: DA3Precision,
    ) -> DA3Adapter:
        """Load an exact DA3 revision and install the project MPS boundary."""

        da3_class = _load_vendor_da3_class(vendor_dir)
        loaded = da3_class.from_pretrained(
            model_id,
            revision=model_revision,
            map_location="cpu",
        )
        model = cast(nn.Module, loaded).to(device=device)
        model.eval()
        policy = install_device_safe_forward(model, precision=precision)
        _install_two_view_alignment(model)
        return cls(
            model=cast(_VendorModel, model),
            model_id=model_id,
            model_revision=model_revision,
            device=device,
            autocast_policy=policy,
        )

    def infer_pose_conditioned(
        self,
        *,
        image_paths: Sequence[Path],
        camera_intrinsics: Sequence[CameraIntrinsics],
        camera_poses: Sequence[CameraPose],
        process_resolution: int,
    ) -> DA3Output:
        """Run metric multi-view inference with supplied OpenCV camera parameters."""

        if process_resolution <= 0 or process_resolution % 14 != 0:
            raise ValueError("DA3 process resolution must be a positive multiple of 14")
        paths = [str(path) for path in image_paths]
        if not paths or any(not path.is_file() for path in image_paths):
            raise FileNotFoundError("every DA3 input image must exist")

        vendor_T_camera_from_world, vendor_intrinsics = build_da3_camera_arrays(
            camera_intrinsics,
            camera_poses,
        )
        prediction = self._model.inference(
            paths,
            extrinsics=vendor_T_camera_from_world,
            intrinsics=vendor_intrinsics,
            align_to_input_ext_scale=True,
            infer_gs=False,
            use_ray_pose=False,
            process_res=process_resolution,
            process_res_method="upper_bound_resize",
            export_dir=None,
        )
        return validate_da3_prediction(prediction, expected_views=len(paths))


def build_da3_camera_arrays(
    camera_intrinsics: Sequence[CameraIntrinsics],
    camera_poses: Sequence[CameraPose],
) -> tuple[Float32Array, Float32Array]:
    """Convert explicit project cameras to DA3's OpenCV world-to-camera arrays."""

    if not camera_intrinsics or len(camera_intrinsics) != len(camera_poses):
        raise ValueError("DA3 intrinsics and poses must have the same non-zero count")

    pose_by_id = {pose.camera_id: pose for pose in camera_poses}
    if len(pose_by_id) != len(camera_poses):
        raise ValueError("DA3 camera pose IDs must be unique")

    vendor_T_camera_from_world: list[FloatArray] = []
    vendor_intrinsics: list[FloatArray] = []
    for intrinsics in camera_intrinsics:
        pose = pose_by_id.get(intrinsics.camera_id)
        if pose is None:
            raise ValueError(f"missing pose for camera '{intrinsics.camera_id}'")

        T_camera_from_world = np.asarray(pose.T_camera_from_world, dtype=np.float64)
        T_world_from_camera = np.asarray(pose.T_world_from_camera, dtype=np.float64)
        recovered = invert_rigid_transform(T_camera_from_world)
        if not np.allclose(recovered, T_world_from_camera, atol=1e-6):
            raise ValueError(f"camera '{intrinsics.camera_id}' transforms do not round-trip")

        vendor_T_camera_from_world.append(T_camera_from_world)
        vendor_intrinsics.append(
            np.array(
                [
                    [intrinsics.fx, 0.0, intrinsics.cx],
                    [0.0, intrinsics.fy, intrinsics.cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        )

    return (
        np.asarray(vendor_T_camera_from_world, dtype=np.float32),
        np.asarray(vendor_intrinsics, dtype=np.float32),
    )


def make_synthetic_two_view_cameras(
    image_paths: Sequence[Path],
    *,
    baseline_m: float = 0.20,
) -> tuple[tuple[CameraIntrinsics, ...], tuple[CameraPose, ...]]:
    """Create labelled synthetic pinhole cameras for the S00 API/MPS gate."""

    if len(image_paths) != 2:
        raise ValueError("the S00 DA3 smoke requires exactly two images")
    if not np.isfinite(baseline_m) or baseline_m <= 0:
        raise ValueError("synthetic camera baseline must be finite and positive")

    intrinsics: list[CameraIntrinsics] = []
    poses: list[CameraPose] = []
    for index, path in enumerate(image_paths):
        with Image.open(path) as image:
            width, height = image.size
        camera_id = f"synthetic_camera_{index + 1:02d}"
        focal = 0.8 * width
        intrinsics.append(
            CameraIntrinsics(
                camera_id=camera_id,
                fx=focal,
                fy=focal,
                cx=width / 2.0,
                cy=height / 2.0,
                image_width=width,
                image_height=height,
            )
        )

        T_world_from_camera = np.eye(4, dtype=np.float64)
        T_world_from_camera[0, 3] = index * baseline_m
        T_camera_from_world = invert_rigid_transform(T_world_from_camera)
        poses.append(
            CameraPose(
                camera_id=camera_id,
                T_world_from_camera=_matrix_tuple(T_world_from_camera),
                T_camera_from_world=_matrix_tuple(T_camera_from_world),
            )
        )
    return tuple(intrinsics), tuple(poses)


def validate_da3_prediction(
    prediction: _VendorPrediction | object,
    *,
    expected_views: int,
) -> DA3Output:
    """Validate and normalize the DA3 arrays needed by later project stages."""

    vendor = cast(_VendorPrediction, prediction)
    depth = np.asarray(vendor.depth, dtype=np.float32)
    confidence = np.asarray(vendor.conf, dtype=np.float32)
    if depth.ndim != 3 or depth.shape[0] != expected_views:
        raise DA3PredictionError(
            f"depth must have shape ({expected_views}, H, W), got {depth.shape}"
        )
    if confidence.shape != depth.shape:
        raise DA3PredictionError(
            f"confidence shape {confidence.shape} must match depth {depth.shape}"
        )
    if not np.isfinite(depth).all() or not np.isfinite(confidence).all():
        raise DA3PredictionError("depth and confidence must contain only finite values")
    if not np.any(depth > 0):
        raise DA3PredictionError("DA3 depth contains no strictly positive samples")

    intrinsics = np.asarray(vendor.intrinsics, dtype=np.float32)
    if intrinsics.shape != (expected_views, 3, 3) or not np.isfinite(intrinsics).all():
        raise DA3PredictionError(f"intrinsics must be finite with shape ({expected_views}, 3, 3)")

    T_camera_from_world = _homogeneous_camera_poses(vendor.extrinsics, expected_views)
    is_metric = bool(vendor.is_metric)
    if not is_metric:
        raise DA3PredictionError("the approved nested DA3 output must be metric")

    processed_raw = vendor.processed_images
    processed_images: UInt8Array | None
    if processed_raw is None:
        processed_images = None
    else:
        processed_images = np.asarray(processed_raw, dtype=np.uint8)
        expected_image_shape = (expected_views, depth.shape[1], depth.shape[2], 3)
        if processed_images.shape != expected_image_shape:
            raise DA3PredictionError(
                f"processed images must have shape {expected_image_shape}, "
                f"got {processed_images.shape}"
            )

    return DA3Output(
        depth_m=depth.copy(),
        confidence=confidence.copy(),
        T_camera_from_world=T_camera_from_world,
        intrinsics=intrinsics.copy(),
        processed_images=None if processed_images is None else processed_images.copy(),
        is_metric=True,
    )


def _homogeneous_camera_poses(raw: object, expected_views: int) -> Float32Array:
    poses = np.asarray(raw, dtype=np.float32)
    if poses.shape == (expected_views, 4, 4):
        homogeneous = poses
    elif poses.shape == (expected_views, 3, 4):
        final_rows = np.zeros((expected_views, 1, 4), dtype=np.float32)
        final_rows[:, 0, 3] = 1.0
        homogeneous = np.concatenate((poses, final_rows), axis=1)
    else:
        raise DA3PredictionError(
            f"camera poses must have shape ({expected_views}, 3, 4) or "
            f"({expected_views}, 4, 4), got {poses.shape}"
        )
    if not np.isfinite(homogeneous).all():
        raise DA3PredictionError("camera poses must contain only finite values")
    return homogeneous.copy()


def compute_vendor_fingerprint(vendor_dir: Path) -> str:
    """Reproduce the canonical sorted per-file aggregate SHA-256."""

    if not vendor_dir.is_dir():
        raise FileNotFoundError(f"DA3 vendor directory does not exist: {vendor_dir}")
    aggregate = hashlib.sha256()
    files = sorted(path for path in vendor_dir.rglob("*") if path.is_file())
    if not files:
        raise ValueError("DA3 vendor directory contains no files")
    for path in files:
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        recorded_path = (Path(vendor_dir.name) / path.relative_to(vendor_dir)).as_posix()
        aggregate.update(f"{file_digest}  {recorded_path}\n".encode())
    return aggregate.hexdigest()


def _load_vendor_da3_class(vendor_dir: Path) -> Any:
    source_dir = vendor_dir / "src"
    package_dir = source_dir / "depth_anything_3"
    if not package_dir.is_dir():
        raise FileNotFoundError(f"DA3 vendor package is missing: {package_dir}")

    sys.dont_write_bytecode = True
    source_text = str(source_dir.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    _install_disabled_colmap_export()
    module = importlib.import_module("depth_anything_3.api")
    da3_class = getattr(module, "DepthAnything3", None)
    if da3_class is None:
        raise ImportError("DA3 vendor API does not expose DepthAnything3")
    return da3_class


def _install_two_view_alignment(model: nn.Module) -> None:
    """Avoid vendor Umeyama degeneracy for the exact two-view S00/S01 case."""

    original_alignment = cast(Any, model)._align_to_input_extrinsics_intrinsics

    def align_two_view_or_delegate(
        bound_model: nn.Module,
        extrinsics: torch.Tensor | None,
        intrinsics: torch.Tensor | None,
        prediction: Any,
        align_to_input_ext_scale: bool = True,
        ransac_view_thresh: int = 10,
    ) -> Any:
        del bound_model, align_to_input_ext_scale
        if extrinsics is None or len(extrinsics) != 2:
            return original_alignment(
                extrinsics,
                intrinsics,
                prediction,
                True,
                ransac_view_thresh,
            )
        if intrinsics is None:
            raise ValueError("pose-conditioned DA3 inference requires intrinsics")
        if not bool(prediction.is_metric):
            raise DA3PredictionError(
                "two-view alignment bypass is valid only for nested metric DA3 output"
            )

        # Nested DA3 has already metric-scaled depth and camera translation.
        # Two camera centres cannot define a full Umeyama Sim(3), so preserve
        # that metric depth and return the supplied pose-conditioned cameras.
        prediction.intrinsics = intrinsics.numpy()
        prediction.extrinsics = extrinsics[..., :3, :].numpy()
        return prediction

    cast(Any, model)._align_to_input_extrinsics_intrinsics = MethodType(
        align_two_view_or_delegate,
        model,
    )


def _install_disabled_colmap_export() -> None:
    module_name = "depth_anything_3.utils.export.colmap"
    if module_name in sys.modules:
        return
    stub = ModuleType(module_name)

    def export_to_colmap(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError(
            "COLMAP export is disabled in the DA3 MPS process; use an isolated "
            "optional COLMAP process only after an explicit project decision"
        )

    stub.export_to_colmap = export_to_colmap  # type: ignore[attr-defined]
    sys.modules[module_name] = stub


def _matrix_tuple(
    matrix: NDArray[np.float64],
) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    return cast(
        tuple[
            tuple[float, float, float, float],
            tuple[float, float, float, float],
            tuple[float, float, float, float],
            tuple[float, float, float, float],
        ],
        tuple(tuple(float(value) for value in row) for row in matrix),
    )
