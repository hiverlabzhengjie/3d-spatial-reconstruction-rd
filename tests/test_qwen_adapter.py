from __future__ import annotations

import asyncio
from pathlib import Path

import av
import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from spatial_reconstruction.models.qwen_adapter import (
    EXPECTED_QWEN_MODEL_ID,
    Qwen3VLAdapter,
    QwenValidationError,
    build_multiframe_message,
    extract_uniform_video_frames,
    uniform_frame_indices,
)


class FakeBatch(dict[str, torch.Tensor]):
    def to(self, device: str) -> FakeBatch:
        del device
        return self


class FakeProcessor:
    def __init__(
        self,
        *,
        decoded: str = "A robot moves forward.",
        expect_prefill: bool = False,
    ) -> None:
        self.decoded = decoded
        self.expect_prefill = expect_prefill
        self.messages: object = None

    def apply_chat_template(self, messages: object, **kwargs: object) -> FakeBatch:
        self.messages = messages
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is (not self.expect_prefill)
        assert kwargs["continue_final_message"] is self.expect_prefill
        return FakeBatch(
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "pixel_values": torch.ones((4, 3, 16, 16)),
            }
        )

    def batch_decode(self, outputs: object, **kwargs: object) -> list[str]:
        del outputs
        assert kwargs["skip_special_tokens"] is True
        return [self.decoded]


class FakeModel(nn.Module):
    def __init__(self, *, include_output: bool = True) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 2)
        self.include_output = include_output
        self.kwargs: dict[str, object] = {}

    def generate(self, **kwargs: object) -> torch.Tensor:
        self.kwargs = kwargs
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        if not self.include_output:
            return input_ids
        output = torch.tensor([[10, 11]])
        return torch.cat((input_ids, output), dim=1)


def make_images(count: int = 4) -> tuple[Image.Image, ...]:
    return tuple(
        Image.new("RGB", (64, 48), color=(index * 20, 30, 40))
        for index in range(count)
    )


def make_adapter(
    *,
    model: FakeModel | None = None,
    processor: FakeProcessor | None = None,
) -> Qwen3VLAdapter:
    return Qwen3VLAdapter(
        model=model or FakeModel(),
        processor=processor or FakeProcessor(),
        model_id=EXPECTED_QWEN_MODEL_ID,
        model_revision="abc123",
        device="mps",
    )


def write_test_video(path: Path, *, frame_count: int = 8) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for index in range(frame_count):
            array = np.full((48, 64, 3), index * 20, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_uniform_frame_indices_include_endpoints() -> None:
    assert uniform_frame_indices(total_frame_count=175, requested_count=4) == (
        0,
        58,
        116,
        174,
    )


@pytest.mark.parametrize(
    ("total", "requested", "message"),
    [
        (1, 4, "more than one"),
        (8, 3, r"\[4, 8\]"),
        (8, 9, r"\[4, 8\]"),
        (4, 5, "exceeds"),
    ],
)
def test_uniform_frame_indices_reject_invalid_bounds(
    total: int,
    requested: int,
    message: str,
) -> None:
    with pytest.raises(QwenValidationError, match=message):
        uniform_frame_indices(total_frame_count=total, requested_count=requested)


def test_video_extraction_is_ordered_and_preserves_source(tmp_path: Path) -> None:
    path = tmp_path / "fixture.mp4"
    write_test_video(path)
    before = path.read_bytes()

    extracted = extract_uniform_video_frames(path, frame_count=4)

    assert tuple(frame.source_frame_index for frame in extracted.frames) == (0, 2, 5, 7)
    assert tuple(frame.sequence_index for frame in extracted.frames) == (0, 1, 2, 3)
    assert all(frame.image.size == (64, 48) for frame in extracted.frames)
    assert extracted.source_fps == pytest.approx(4.0)
    assert path.read_bytes() == before


def test_multiframe_message_preserves_order_and_has_no_spatial_fields() -> None:
    images = make_images()
    messages = build_multiframe_message(images=images, prompt="Describe visible action.")

    content = messages[0]["content"]
    assert isinstance(content, list)
    labels = [
        item["text"]
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    assert labels[:4] == [
        "Ordered frame 1 of 4.",
        "Ordered frame 2 of 4.",
        "Ordered frame 3 of 4.",
        "Ordered frame 4 of 4.",
    ]
    serialized_keys = {
        str(key)
        for item in content
        if isinstance(item, dict)
        for key in item
    }
    assert serialized_keys == {"type", "text", "image"}


def test_async_generation_returns_bounded_text_only_contract() -> None:
    model = FakeModel()
    adapter = make_adapter(model=model)

    response = asyncio.run(
        adapter.generate(
            images=make_images(),
            prompt="Describe visible action or say unknown.",
            max_new_tokens=16,
        )
    )

    assert response.text == "A robot moves forward."
    assert response.input_token_count == 3
    assert response.output_token_count == 2
    assert response.output_token_ids == (10, 11)
    assert response.input_shapes["pixel_values"] == (4, 3, 16, 16)
    assert model.kwargs["do_sample"] is False
    assert model.kwargs["max_new_tokens"] == 16
    assert adapter.model_precision == "float32"
    assert set(response.__dataclass_fields__) == {
        "text",
        "input_token_count",
        "output_token_count",
        "output_token_ids",
        "input_shapes",
    }


def test_generation_continues_and_reconstructs_bounded_assistant_prefill() -> None:
    prefix = '{"event_label":"'
    processor = FakeProcessor(
        decoded='carry","evidence_strength":"strong"}',
        expect_prefill=True,
    )
    adapter = make_adapter(processor=processor)

    response = asyncio.run(
        adapter.generate(
            images=make_images(),
            prompt="Return compact event JSON.",
            max_new_tokens=32,
            assistant_prefill=prefix,
        )
    )

    assert response.text == (
        '{"event_label":"carry","evidence_strength":"strong"}'
    )
    assert isinstance(processor.messages, list)
    assert processor.messages[-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": prefix}],
    }


def test_generation_rejects_empty_model_output() -> None:
    adapter = make_adapter(model=FakeModel(include_output=False))

    with pytest.raises(QwenValidationError, match="no generated tokens"):
        asyncio.run(
            adapter.generate(
                images=make_images(),
                prompt="Describe visible action.",
                max_new_tokens=16,
            )
        )


@pytest.mark.parametrize(
    ("images", "prompt", "tokens", "message"),
    [
        ((Image.new("RGB", (8, 8)),), "valid", 16, "between two and eight"),
        (make_images(), " ", 16, "must not be empty"),
        (make_images(), "valid", 0, r"\[1, 256\]"),
        (make_images(), "valid", 257, r"\[1, 256\]"),
    ],
)
def test_generation_validates_bounded_inputs(
    images: tuple[Image.Image, ...],
    prompt: str,
    tokens: int,
    message: str,
) -> None:
    adapter = make_adapter()

    with pytest.raises(QwenValidationError, match=message):
        asyncio.run(
            adapter.generate(
                images=images,
                prompt=prompt,
                max_new_tokens=tokens,
            )
        )


def test_exact_model_identifier_is_enforced() -> None:
    with pytest.raises(ValueError, match="unexpected Qwen model ID"):
        Qwen3VLAdapter(
            model=FakeModel(),
            processor=FakeProcessor(),
            model_id="wrong/model",
            model_revision="abc123",
            device="mps",
        )
