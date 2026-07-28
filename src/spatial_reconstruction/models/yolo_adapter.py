"""Project-owned YOLOv8 segmentation adapter and result validation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import cv2
import numpy as np
import torch
from numpy.typing import DTypeLike, NDArray
from PIL import Image, ImageOps

from spatial_reconstruction.contracts import (
    FrameRef,
    PixelBox,
    SegmentationDetection,
)
from spatial_reconstruction.runtime import DeviceName

Float32Array = NDArray[np.float32]
Int64Array = NDArray[np.int64]
UInt8Array = NDArray[np.uint8]


class YOLOValidationError(ValueError):
    """Raised when a segmentation result is structurally invalid."""


class _YOLOModel(Protocol):
    task: str
    ckpt_path: str
    model: object

    def predict(self, **kwargs: object) -> Sequence[object]: ...


@dataclass(frozen=True, slots=True)
class LoadedImage:
    """First display frame decoded from an immutable image source."""

    rgb: UInt8Array
    format: str
    embedded_frame_count: int
    exif_orientation: int | None


@dataclass(frozen=True, slots=True)
class NormalizedYOLOResult:
    """Validated detections, source-sized masks, and retained raw arrays."""

    frame: FrameRef
    detections: tuple[SegmentationDetection, ...]
    masks: UInt8Array
    annotated_rgb: UInt8Array
    raw_boxes_xyxy: Float32Array
    raw_class_ids: Int64Array
    raw_confidence: Float32Array
    raw_masks: Float32Array
    class_names: dict[int, str]
    speed_ms: dict[str, float]


class YOLOSegAdapter:
    """Thin independent adapter around Ultralytics segmentation prediction."""

    def __init__(self, *, model: _YOLOModel, weight_path: Path) -> None:
        if model.task != "segment":
            raise ValueError(f"expected a segmentation model, got task '{model.task}'")
        self._model = model
        self.weight_path = weight_path.resolve()

    @classmethod
    def from_pretrained(cls, *, model_id: str, cache_dir: Path) -> YOLOSegAdapter:
        """Load the exact baseline filename into the ignored local model cache."""

        if model_id != "yolov8n-seg.pt":
            raise ValueError(f"unexpected YOLO model ID: {model_id}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        requested_path = cache_dir / model_id
        from ultralytics.models.yolo.model import YOLO

        loaded = YOLO(str(requested_path), task="segment")
        checkpoint = Path(str(loaded.ckpt_path))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"YOLO checkpoint was not resolved: {checkpoint}")
        return cls(model=cast(_YOLOModel, loaded), weight_path=checkpoint)

    @property
    def weight_sha256(self) -> str:
        """Fingerprint the exact local checkpoint bytes."""

        return hashlib.sha256(self.weight_path.read_bytes()).hexdigest()

    @property
    def model_precision(self) -> str:
        """Report the actual parameter dtype used by the loaded model."""

        module = cast(Any, self._model.model)
        dtypes = {parameter.dtype for parameter in module.parameters()}
        if dtypes == {torch.float32}:
            return "float32"
        if dtypes == {torch.float16}:
            return "float16"
        if dtypes == {torch.bfloat16}:
            return "bfloat16"
        return "mixed" if dtypes else "unknown"

    def predict(
        self,
        *,
        image_rgb: UInt8Array,
        device: DeviceName,
        image_size: int,
        confidence_threshold: float,
    ) -> object:
        """Run segmentation prediction without tracking or source mutation."""

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("YOLO input must be an RGB H-by-W-by-3 array")
        if image_size <= 0:
            raise ValueError("YOLO inference image size must be positive")
        if (
            not math.isfinite(confidence_threshold)
            or confidence_threshold < 0
            or confidence_threshold > 1
        ):
            raise ValueError("YOLO confidence threshold must be within [0, 1]")

        results = self._model.predict(
            source=image_rgb,
            device=device,
            imgsz=image_size,
            conf=confidence_threshold,
            retina_masks=True,
            verbose=False,
            save=False,
            stream=False,
        )
        if len(results) != 1:
            raise YOLOValidationError(f"expected one YOLO result, got {len(results)}")
        return results[0]


def load_first_image_rgb(path: Path) -> LoadedImage:
    """Decode the first frame with EXIF orientation without modifying the source."""

    if not path.is_file():
        raise FileNotFoundError(f"representative image does not exist: {path}")
    with Image.open(path) as image:
        source_format = image.format or "unknown"
        frame_count = int(getattr(image, "n_frames", 1))
        orientation = image.getexif().get(274)
        image.seek(0)
        displayed = ImageOps.exif_transpose(image).convert("RGB")
        rgb = np.asarray(displayed, dtype=np.uint8).copy()
    return LoadedImage(
        rgb=rgb,
        format=source_format,
        embedded_frame_count=frame_count,
        exif_orientation=int(orientation) if orientation is not None else None,
    )


def normalize_yolo_result(
    result: object,
    *,
    frame: FrameRef,
    mask_artifact_ref: str,
) -> NormalizedYOLOResult:
    """Validate an Ultralytics result and normalize masks to source dimensions."""

    vendor = cast(Any, result)
    original_shape = tuple(int(value) for value in vendor.orig_shape)
    expected_shape = (frame.image_height, frame.image_width)
    if original_shape != expected_shape:
        raise YOLOValidationError(
            f"YOLO original shape {original_shape} does not match frame {expected_shape}"
        )

    class_names = _normalize_class_names(vendor.names)
    boxes = vendor.boxes
    if boxes is None:
        raw_boxes = np.empty((0, 4), dtype=np.float32)
        raw_class_ids = np.empty((0,), dtype=np.int64)
        raw_confidence = np.empty((0,), dtype=np.float32)
    else:
        raw_boxes = _to_numpy(boxes.xyxy, dtype=np.float32).reshape(-1, 4)
        raw_classes_float = _to_numpy(boxes.cls, dtype=np.float32).reshape(-1)
        raw_confidence = _to_numpy(boxes.conf, dtype=np.float32).reshape(-1)
        if not np.equal(raw_classes_float, np.floor(raw_classes_float)).all():
            raise YOLOValidationError("YOLO class IDs must be integers")
        raw_class_ids = raw_classes_float.astype(np.int64)

    count = raw_boxes.shape[0]
    if raw_class_ids.shape != (count,) or raw_confidence.shape != (count,):
        raise YOLOValidationError("YOLO box, class, and confidence counts must match")

    masks_object = vendor.masks
    if masks_object is None:
        if count:
            raise YOLOValidationError("segmentation detections require masks")
        raw_masks = np.empty((0, frame.image_height, frame.image_width), dtype=np.float32)
    else:
        raw_masks = _to_numpy(masks_object.data, dtype=np.float32)
        if raw_masks.ndim != 3 or raw_masks.shape[0] != count:
            raise YOLOValidationError("YOLO mask count must match detection count")
        if not np.isfinite(raw_masks).all():
            raise YOLOValidationError("YOLO masks must contain only finite values")

    source_masks = np.empty(
        (count, frame.image_height, frame.image_width),
        dtype=np.uint8,
    )
    detections: list[SegmentationDetection] = []
    for index in range(count):
        box = raw_boxes[index]
        confidence = float(raw_confidence[index])
        class_id = int(raw_class_ids[index])
        _validate_box(box, width=frame.image_width, height=frame.image_height)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise YOLOValidationError("YOLO confidence must be finite and within [0, 1]")
        class_name = class_names.get(class_id)
        if class_name is None:
            raise YOLOValidationError(f"YOLO class ID {class_id} has no class name")

        resized = cv2.resize(
            raw_masks[index],
            (frame.image_width, frame.image_height),
            interpolation=cv2.INTER_NEAREST,
        )
        source_masks[index] = (resized >= 0.5).astype(np.uint8)
        detections.append(
            SegmentationDetection(
                frame=frame,
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                box=PixelBox(
                    x_min=float(box[0]),
                    y_min=float(box[1]),
                    x_max=float(box[2]),
                    y_max=float(box[3]),
                ),
                mask_ref=f"{mask_artifact_ref}#mask_{index:04d}",
            )
        )

    annotated_bgr = np.asarray(vendor.plot(), dtype=np.uint8)
    if annotated_bgr.shape != (frame.image_height, frame.image_width, 3):
        raise YOLOValidationError("YOLO annotated preview has an unexpected shape")
    annotated_rgb = annotated_bgr[..., ::-1].copy()
    speed_ms = _normalize_speed(vendor.speed)

    return NormalizedYOLOResult(
        frame=frame,
        detections=tuple(detections),
        masks=source_masks,
        annotated_rgb=annotated_rgb,
        raw_boxes_xyxy=raw_boxes.copy(),
        raw_class_ids=raw_class_ids.copy(),
        raw_confidence=raw_confidence.copy(),
        raw_masks=raw_masks.copy(),
        class_names=class_names,
        speed_ms=speed_ms,
    )


def _to_numpy(value: Any, *, dtype: DTypeLike) -> NDArray[Any]:
    detached = value.detach().cpu().numpy()
    return np.asarray(detached, dtype=dtype)


def _normalize_class_names(raw: object) -> dict[int, str]:
    if isinstance(raw, dict):
        names = {int(index): str(name).strip() for index, name in raw.items()}
    elif isinstance(raw, (list, tuple)):
        names = {int(index): str(name).strip() for index, name in enumerate(raw)}
    else:
        raise YOLOValidationError("YOLO class names must be a mapping or sequence")
    if not names or any(index < 0 or not name for index, name in names.items()):
        raise YOLOValidationError("YOLO class names must be non-empty")
    return names


def _validate_box(box: Float32Array, *, width: int, height: int) -> None:
    if not np.isfinite(box).all():
        raise YOLOValidationError("YOLO boxes must contain only finite values")
    x_min, y_min, x_max, y_max = (float(value) for value in box)
    if (
        x_min < 0
        or y_min < 0
        or x_max > width
        or y_max > height
        or x_max <= x_min
        or y_max <= y_min
    ):
        raise YOLOValidationError("YOLO box must have positive in-frame extent")


def _normalize_speed(raw: object) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise YOLOValidationError("YOLO speed metadata must be a mapping")
    speed = {str(name): float(value) for name, value in raw.items()}
    if any(not math.isfinite(value) or value < 0 for value in speed.values()):
        raise YOLOValidationError("YOLO speed values must be finite and non-negative")
    return speed
