"""Replaceable project-owned model adapters."""

from spatial_reconstruction.models.da3_adapter import (
    EXPECTED_DA3_VENDOR_FINGERPRINT,
    DA3Adapter,
    DA3Output,
    DA3PredictionError,
    build_da3_camera_arrays,
    compute_vendor_fingerprint,
    make_synthetic_two_view_cameras,
    validate_da3_prediction,
)
from spatial_reconstruction.models.qwen_adapter import (
    EXPECTED_QWEN_MODEL_ID,
    ExtractedVideoFrame,
    ExtractedVideoFrames,
    Qwen3VLAdapter,
    QwenTextResponse,
    QwenValidationError,
    build_multiframe_message,
    extract_uniform_video_frames,
    uniform_frame_indices,
)
from spatial_reconstruction.models.yolo_adapter import (
    NormalizedYOLOResult,
    YOLOSegAdapter,
    YOLOValidationError,
    load_first_image_rgb,
    normalize_yolo_result,
)

__all__ = [
    "DA3Adapter",
    "DA3Output",
    "DA3PredictionError",
    "EXPECTED_DA3_VENDOR_FINGERPRINT",
    "EXPECTED_QWEN_MODEL_ID",
    "ExtractedVideoFrame",
    "ExtractedVideoFrames",
    "NormalizedYOLOResult",
    "Qwen3VLAdapter",
    "QwenTextResponse",
    "QwenValidationError",
    "YOLOSegAdapter",
    "YOLOValidationError",
    "build_da3_camera_arrays",
    "build_multiframe_message",
    "compute_vendor_fingerprint",
    "extract_uniform_video_frames",
    "make_synthetic_two_view_cameras",
    "load_first_image_rgb",
    "normalize_yolo_result",
    "uniform_frame_indices",
    "validate_da3_prediction",
]
