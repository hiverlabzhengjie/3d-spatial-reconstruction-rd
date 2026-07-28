from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from spatial_reconstruction.contracts import FrameRef
from spatial_reconstruction.models.yolo_adapter import (
    YOLOSegAdapter,
    YOLOValidationError,
    load_first_image_rgb,
    normalize_yolo_result,
)


class FakeResult:
    def __init__(self, *, count: int = 1) -> None:
        self.orig_shape = (80, 120)
        self.names = {0: "person", 24: "backpack"}
        if count:
            self.boxes = SimpleNamespace(
                xyxy=torch.tensor([[10.0, 20.0, 70.0, 75.0]]),
                cls=torch.tensor([24.0]),
                conf=torch.tensor([0.8]),
            )
            self.masks = SimpleNamespace(data=torch.ones((1, 40, 60)))
        else:
            self.boxes = SimpleNamespace(
                xyxy=torch.empty((0, 4)),
                cls=torch.empty((0,)),
                conf=torch.empty((0,)),
            )
            self.masks = None
        self.speed = {"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5}

    def plot(self) -> np.ndarray:
        return np.zeros((80, 120, 3), dtype=np.uint8)


class FakeModel:
    task = "segment"
    ckpt_path = "yolov8n-seg.pt"

    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}
        self.model = nn.Linear(2, 2)

    def predict(self, **kwargs: object) -> list[FakeResult]:
        self.kwargs = kwargs
        return [FakeResult()]


def make_frame() -> FrameRef:
    return FrameRef(
        camera_id="wp6_camera",
        frame_index=0,
        timestamp_seconds=0.0,
        source_ref="/inputs/living-room.jpeg",
        image_width=120,
        image_height=80,
    )


def test_load_first_image_rgb_preserves_source_and_reports_embedded_frames(
    tmp_path: Path,
) -> None:
    path = tmp_path / "image.jpg"
    Image.new("RGB", (120, 80), color=(10, 20, 30)).save(path)
    before = path.read_bytes()

    loaded = load_first_image_rgb(path)

    assert loaded.rgb.shape == (80, 120, 3)
    assert loaded.embedded_frame_count == 1
    assert path.read_bytes() == before


def test_adapter_predict_disables_tracking_and_requests_source_masks(tmp_path: Path) -> None:
    weight_path = tmp_path / "yolov8n-seg.pt"
    weight_path.write_bytes(b"weights")
    model = FakeModel()
    adapter = YOLOSegAdapter(model=model, weight_path=weight_path)

    result = adapter.predict(
        image_rgb=np.zeros((80, 120, 3), dtype=np.uint8),
        device="mps",
        image_size=640,
        confidence_threshold=0.25,
    )

    assert isinstance(result, FakeResult)
    assert model.kwargs["device"] == "mps"
    assert model.kwargs["retina_masks"] is True
    assert "half" not in model.kwargs
    assert "tracker" not in model.kwargs
    assert adapter.model_precision == "float32"


def test_normalize_yolo_result_validates_and_resizes_masks() -> None:
    normalized = normalize_yolo_result(
        FakeResult(),
        frame=make_frame(),
        mask_artifact_ref="artifacts/s00/yolo/masks.npz",
    )

    assert len(normalized.detections) == 1
    assert normalized.detections[0].class_name == "backpack"
    assert normalized.detections[0].camera_local_track_id is None
    assert normalized.masks.shape == (1, 80, 120)
    assert normalized.masks.dtype == np.uint8
    assert normalized.raw_masks.shape == (1, 40, 60)
    assert normalized.annotated_rgb.shape == (80, 120, 3)


def test_zero_detection_result_is_valid_and_explicit() -> None:
    normalized = normalize_yolo_result(
        FakeResult(count=0),
        frame=make_frame(),
        mask_artifact_ref="artifacts/s00/yolo/masks.npz",
    )

    assert normalized.detections == ()
    assert normalized.masks.shape == (0, 80, 120)
    assert normalized.raw_boxes_xyxy.shape == (0, 4)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda result: setattr(result, "orig_shape", (40, 60)), "does not match frame"),
        (
            lambda result: setattr(
                result.boxes,
                "conf",
                torch.tensor([float("nan")]),
            ),
            "confidence",
        ),
        (
            lambda result: setattr(
                result.boxes,
                "xyxy",
                torch.tensor([[-1.0, 20.0, 70.0, 75.0]]),
            ),
            "positive in-frame",
        ),
        (lambda result: setattr(result, "masks", None), "require masks"),
        (
            lambda result: setattr(
                result.masks,
                "data",
                torch.ones((2, 40, 60)),
            ),
            "mask count",
        ),
    ],
)
def test_normalize_yolo_result_rejects_invalid_structure(
    mutation: object,
    message: str,
) -> None:
    result = FakeResult()
    mutation(result)  # type: ignore[operator]

    with pytest.raises(YOLOValidationError, match=message):
        normalize_yolo_result(
            result,
            frame=make_frame(),
            mask_artifact_ref="artifacts/s00/yolo/masks.npz",
        )
