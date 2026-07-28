from __future__ import annotations

import torch
from torch import nn

from spatial_reconstruction.models.da3_mps import (
    install_device_safe_forward,
    resolve_autocast_policy,
)


class FakeCore(nn.Module):
    def forward(
        self,
        image: torch.Tensor,
        extrinsics: torch.Tensor | None,
        intrinsics: torch.Tensor | None,
        export_feat_layers: list[int] | None,
        infer_gs: bool,
        use_ray_pose: bool,
        ref_view_strategy: str,
    ) -> dict[str, torch.Tensor]:
        del extrinsics, intrinsics, export_feat_layers, infer_gs, use_ray_pose
        assert ref_view_strategy == "saddle_balanced"
        return {"depth": image.mean(dim=2, keepdim=False)}


class FakeDA3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.model = FakeCore()


def test_mps_auto_policy_uses_float16_without_cuda_query() -> None:
    policy = resolve_autocast_policy(torch.device("mps"), "auto")

    assert policy.enabled is True
    assert policy.dtype is torch.float16
    assert policy.reported_precision == "float16"


def test_float32_policy_disables_autocast() -> None:
    policy = resolve_autocast_policy(torch.device("mps"), "float32")

    assert policy.enabled is False
    assert policy.dtype is None
    assert policy.reported_precision == "float32"


def test_installed_cpu_forward_calls_core_without_autocast() -> None:
    model = FakeDA3()
    policy = install_device_safe_forward(model, precision="auto")
    image = torch.ones((1, 2, 3, 4, 6), dtype=torch.float32)

    output = model(
        image,
        extrinsics=None,
        intrinsics=None,
        infer_gs=False,
    )

    assert policy.enabled is False
    assert output["depth"].shape == (1, 2, 4, 6)
    assert output["depth"].dtype is torch.float32
