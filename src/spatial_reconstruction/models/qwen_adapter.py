"""Project-owned asynchronous adapter for bounded Qwen3-VL multi-image text."""

from __future__ import annotations

import asyncio
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import av
import torch
from PIL import Image

from spatial_reconstruction.runtime import DeviceName, PrecisionName

EXPECTED_QWEN_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


class QwenValidationError(ValueError):
    """Raised when a Qwen input or generated response is structurally invalid."""


@dataclass(frozen=True, slots=True)
class ExtractedVideoFrame:
    """One immutable, ordered frame decoded from a local video."""

    sequence_index: int
    source_frame_index: int
    timestamp_seconds: float
    image: Image.Image


@dataclass(frozen=True, slots=True)
class ExtractedVideoFrames:
    """Bounded ordered frames plus immutable source-video provenance."""

    source_path: Path
    source_sha256: str
    source_size_bytes: int
    total_frame_count: int
    source_fps: float
    source_width: int
    source_height: int
    frames: tuple[ExtractedVideoFrame, ...]


@dataclass(frozen=True, slots=True)
class QwenTextResponse:
    """Bounded semantic text with token and tensor-shape diagnostics only."""

    text: str
    input_token_count: int
    output_token_count: int
    output_token_ids: tuple[int, ...]
    input_shapes: dict[str, tuple[int, ...]]


class Qwen3VLAdapter:
    """Exact Qwen3-VL adapter with no spatial-state write interface."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        model_id: str,
        model_revision: str,
        device: DeviceName,
    ) -> None:
        if model_id != EXPECTED_QWEN_MODEL_ID:
            raise ValueError(f"unexpected Qwen model ID: {model_id}")
        if not model_revision.strip():
            raise ValueError("Qwen model revision must not be empty")
        self._model = model
        self._processor = processor
        self.model_id = model_id
        self.model_revision = model_revision.strip()
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str,
        cache_dir: Path,
        device: DeviceName,
        dtype: torch.dtype = torch.float16,
    ) -> Qwen3VLAdapter:
        """Load the exact approved checkpoint and processor into a local cache."""

        if model_id != EXPECTED_QWEN_MODEL_ID:
            raise ValueError(f"unexpected Qwen model ID: {model_id}")
        if device not in {"mps", "cpu", "cuda"}:
            raise ValueError(f"unsupported Qwen device: {device}")
        cache_dir.mkdir(parents=True, exist_ok=True)

        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        processor_factory = cast(Any, AutoProcessor)
        model_factory = cast(Any, Qwen3VLForConditionalGeneration)
        processor = processor_factory.from_pretrained(
            model_id,
            cache_dir=str(cache_dir),
        )
        model = model_factory.from_pretrained(
            model_id,
            cache_dir=str(cache_dir),
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.eval()
        model.to(device)
        revision = _model_revision(model)
        return cls(
            model=model,
            processor=processor,
            model_id=model_id,
            model_revision=revision,
            device=device,
        )

    @property
    def model_precision(self) -> PrecisionName:
        """Report the actual model parameter dtype."""

        dtypes = {parameter.dtype for parameter in self._model.parameters()}
        if dtypes == {torch.float32}:
            return "float32"
        if dtypes == {torch.float16}:
            return "float16"
        if dtypes == {torch.bfloat16}:
            return "bfloat16"
        return "mixed" if dtypes else "unknown"

    async def generate(
        self,
        *,
        images: tuple[Image.Image, ...],
        prompt: str,
        max_new_tokens: int,
    ) -> QwenTextResponse:
        """Generate without blocking the caller's event loop."""

        return await asyncio.to_thread(
            self._generate_sync,
            images=images,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

    def _generate_sync(
        self,
        *,
        images: tuple[Image.Image, ...],
        prompt: str,
        max_new_tokens: int,
    ) -> QwenTextResponse:
        normalized_images = _validate_images(images)
        normalized_prompt = _validate_prompt(prompt)
        if max_new_tokens <= 0 or max_new_tokens > 256:
            raise QwenValidationError("max_new_tokens must be within [1, 256]")

        messages = build_multiframe_message(
            images=normalized_images,
            prompt=normalized_prompt,
        )
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        input_ids = cast(torch.Tensor, inputs["input_ids"])
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise QwenValidationError("Qwen processor must return one tokenized request")
        input_shapes = {
            str(name): tuple(int(value) for value in tensor.shape)
            for name, tensor in inputs.items()
            if isinstance(tensor, torch.Tensor)
        }

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        if not isinstance(generated, torch.Tensor):
            raise QwenValidationError("Qwen generation must return a tensor")
        if generated.ndim != 2 or generated.shape[0] != 1:
            raise QwenValidationError("Qwen generation must return one sequence")

        input_token_count = int(input_ids.shape[1])
        output_ids = generated[0, input_token_count:]
        output_token_ids = tuple(int(value) for value in output_ids.detach().cpu().tolist())
        if not output_token_ids:
            raise QwenValidationError("Qwen returned no generated tokens")
        if len(output_token_ids) > max_new_tokens:
            raise QwenValidationError("Qwen exceeded the configured token bound")

        decoded = self._processor.batch_decode(
            [output_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(decoded, list) or len(decoded) != 1:
            raise QwenValidationError("Qwen processor returned an invalid decoded response")
        text = str(decoded[0]).strip()
        if not text:
            raise QwenValidationError("Qwen returned empty decoded text")

        return QwenTextResponse(
            text=text,
            input_token_count=input_token_count,
            output_token_count=len(output_token_ids),
            output_token_ids=output_token_ids,
            input_shapes=input_shapes,
        )


def build_multiframe_message(
    *,
    images: tuple[Image.Image, ...],
    prompt: str,
) -> list[dict[str, object]]:
    """Build one ordered multi-image message without spatial-state fields."""

    normalized_images = _validate_images(images)
    normalized_prompt = _validate_prompt(prompt)
    content: list[dict[str, object]] = []
    frame_count = len(normalized_images)
    for index, image in enumerate(normalized_images, start=1):
        content.append(
            {
                "type": "text",
                "text": f"Ordered frame {index} of {frame_count}.",
            }
        )
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": normalized_prompt})
    return [{"role": "user", "content": content}]


def uniform_frame_indices(*, total_frame_count: int, requested_count: int) -> tuple[int, ...]:
    """Return deterministic, endpoint-inclusive uniform frame indices."""

    if total_frame_count <= 1:
        raise QwenValidationError("video must contain more than one frame")
    if requested_count < 4 or requested_count > 8:
        raise QwenValidationError("Qwen smoke frame count must be within [4, 8]")
    if requested_count > total_frame_count:
        raise QwenValidationError("requested frame count exceeds available video frames")
    denominator = requested_count - 1
    last_index = total_frame_count - 1
    indices = tuple(round(index * last_index / denominator) for index in range(requested_count))
    if len(set(indices)) != requested_count or indices != tuple(sorted(indices)):
        raise QwenValidationError("uniform frame sampling did not produce unique ordered indices")
    return indices


def extract_uniform_video_frames(
    path: Path,
    *,
    frame_count: int,
) -> ExtractedVideoFrames:
    """Decode selected frames without changing the local source video."""

    source_path = path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Qwen smoke video does not exist: {source_path}")
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()

    with av.open(str(source_path), mode="r") as container:
        if not container.streams.video:
            raise QwenValidationError("Qwen smoke input has no video stream")
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
        source_width = int(stream.width)
        source_height = int(stream.height)
        if not math.isfinite(fps) or fps <= 0:
            raise QwenValidationError("video must report a valid frame rate")
        decoded_frame_count = sum(1 for _ in container.decode(stream))
        if decoded_frame_count <= 1:
            raise QwenValidationError("video must contain more than one decodable frame")
        target_indices = uniform_frame_indices(
            total_frame_count=decoded_frame_count,
            requested_count=frame_count,
        )

    with av.open(str(source_path), mode="r") as container:
        stream = container.streams.video[0]
        targets = set(target_indices)
        selected: dict[int, ExtractedVideoFrame] = {}
        for decoded_index, frame in enumerate(container.decode(stream)):
            if decoded_index not in targets:
                continue
            timestamp = float(frame.time) if frame.time is not None else decoded_index / fps
            if not math.isfinite(timestamp) or timestamp < 0:
                raise QwenValidationError(
                    "decoded frame timestamp must be finite and non-negative"
                )
            selected[decoded_index] = ExtractedVideoFrame(
                sequence_index=len(selected),
                source_frame_index=decoded_index,
                timestamp_seconds=timestamp,
                image=cast(Any, frame).to_image().convert("RGB"),
            )

    if tuple(selected) != target_indices:
        raise QwenValidationError("video ended before all selected frames were decoded")
    frames = tuple(selected[index] for index in target_indices)
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != source_sha256:
        raise QwenValidationError("source video changed during frame extraction")
    return ExtractedVideoFrames(
        source_path=source_path,
        source_sha256=source_sha256,
        source_size_bytes=len(source_bytes),
        total_frame_count=decoded_frame_count,
        source_fps=fps,
        source_width=source_width,
        source_height=source_height,
        frames=frames,
    )


def _validate_images(images: tuple[Image.Image, ...]) -> tuple[Image.Image, ...]:
    if len(images) < 2 or len(images) > 8:
        raise QwenValidationError("Qwen requires between two and eight ordered images")
    normalized: list[Image.Image] = []
    for image in images:
        if not isinstance(image, Image.Image):
            raise QwenValidationError("Qwen images must be Pillow images")
        if image.width <= 0 or image.height <= 0:
            raise QwenValidationError("Qwen images must have positive dimensions")
        normalized.append(image.convert("RGB"))
    return tuple(normalized)


def _validate_prompt(prompt: str) -> str:
    normalized = prompt.strip()
    if not normalized:
        raise QwenValidationError("Qwen prompt must not be empty")
    return normalized


def _model_revision(model: Any) -> str:
    revision = getattr(model.config, "_commit_hash", None)
    if not isinstance(revision, str) or not revision.strip():
        raise RuntimeError("loaded Qwen model did not expose a resolved repository revision")
    return revision.strip()
