"""Project-owned device/autocast boundary for the unmodified DA3 API."""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Literal, Protocol, cast

import torch
from torch import nn

DA3Precision = Literal["auto", "float32", "float16", "bfloat16"]


@dataclass(frozen=True, slots=True)
class AutocastPolicy:
    """Resolved device-specific autocast behavior."""

    enabled: bool
    dtype: torch.dtype | None
    reported_precision: Literal["float32", "float16", "bfloat16", "mixed"]


def resolve_autocast_policy(
    device: torch.device,
    precision: DA3Precision,
) -> AutocastPolicy:
    """Resolve precision without querying a different accelerator backend."""

    if precision == "float32":
        return AutocastPolicy(enabled=False, dtype=None, reported_precision="float32")
    if precision == "float16":
        return AutocastPolicy(enabled=True, dtype=torch.float16, reported_precision="float16")
    if precision == "bfloat16":
        return AutocastPolicy(enabled=True, dtype=torch.bfloat16, reported_precision="bfloat16")

    if device.type == "mps":
        return AutocastPolicy(enabled=True, dtype=torch.float16, reported_precision="float16")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        label: Literal["bfloat16", "float16"] = (
            "bfloat16" if dtype is torch.bfloat16 else "float16"
        )
        return AutocastPolicy(enabled=True, dtype=dtype, reported_precision=label)
    return AutocastPolicy(enabled=False, dtype=None, reported_precision="float32")


class _DA3Core(Protocol):
    def __call__(
        self,
        image: torch.Tensor,
        T_camera_from_world: torch.Tensor | None,
        intrinsics: torch.Tensor | None,
        export_feat_layers: list[int] | None,
        infer_gs: bool,
        use_ray_pose: bool,
        ref_view_strategy: str,
    ) -> dict[str, torch.Tensor]: ...


def install_device_safe_forward(
    model: nn.Module,
    *,
    precision: DA3Precision,
) -> AutocastPolicy:
    """Replace only DA3's high-level autocast boundary on this model instance."""

    device = _module_device(model)
    policy = resolve_autocast_policy(device, precision)

    @torch.inference_mode()
    def device_safe_forward(
        bound_model: nn.Module,
        image: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = None,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> dict[str, torch.Tensor]:
        core = cast(_DA3Core, bound_model.model)
        if policy.enabled:
            if policy.dtype is None:
                raise RuntimeError("enabled autocast policy requires a dtype")
            with torch.autocast(
                device_type=image.device.type,
                dtype=policy.dtype,
                enabled=True,
            ):
                return core(
                    image,
                    extrinsics,
                    intrinsics,
                    export_feat_layers,
                    infer_gs,
                    use_ray_pose,
                    ref_view_strategy,
                )
        with torch.autocast(device_type=image.device.type, enabled=False):
            return core(
                image,
                extrinsics,
                intrinsics,
                export_feat_layers,
                infer_gs,
                use_ray_pose,
                ref_view_strategy,
            )

    model.forward = MethodType(device_safe_forward, model)
    return policy


def _module_device(model: nn.Module) -> torch.device:
    for parameter in model.parameters():
        return parameter.device
    for buffer in model.buffers():
        return buffer.device
    raise ValueError("DA3 model has no parameter or buffer from which to determine its device")
