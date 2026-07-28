from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from spatial_reconstruction.models.da3_adapter import (
    DA3PredictionError,
    _install_two_view_alignment,
    build_da3_camera_arrays,
    compute_vendor_fingerprint,
    make_synthetic_two_view_cameras,
    validate_da3_prediction,
)


def make_images(tmp_path: Path) -> tuple[Path, Path]:
    from PIL import Image

    paths = (tmp_path / "one.png", tmp_path / "two.png")
    for path in paths:
        Image.new("RGB", (120, 80), color=(10, 20, 30)).save(path)
    return paths


def make_prediction() -> SimpleNamespace:
    depth = np.ones((2, 4, 6), dtype=np.float32)
    return SimpleNamespace(
        depth=depth,
        conf=np.full_like(depth, 0.75),
        extrinsics=np.array(
            [
                np.eye(4, dtype=np.float32)[:3],
                np.array(
                    [
                        [1.0, 0.0, 0.0, -0.2],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
            ]
        ),
        intrinsics=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
        processed_images=np.zeros((2, 4, 6, 3), dtype=np.uint8),
        is_metric=1,
    )


def test_synthetic_camera_arrays_use_explicit_world_to_camera_poses(tmp_path: Path) -> None:
    images = make_images(tmp_path)
    intrinsics, poses = make_synthetic_two_view_cameras(images, baseline_m=0.2)

    T_camera_from_world, camera_matrices = build_da3_camera_arrays(intrinsics, poses)

    assert T_camera_from_world.shape == (2, 4, 4)
    assert T_camera_from_world[0] == pytest.approx(np.eye(4))
    assert T_camera_from_world[1, 0, 3] == pytest.approx(-0.2)
    assert camera_matrices.shape == (2, 3, 3)
    assert camera_matrices[0, 0, 0] == pytest.approx(96.0)
    assert camera_matrices[0, 0, 2] == pytest.approx(60.0)

    for pose, vendor_pose in zip(poses, T_camera_from_world, strict=True):
        assert vendor_pose == pytest.approx(np.asarray(pose.T_camera_from_world))
        assert np.linalg.inv(vendor_pose) == pytest.approx(np.asarray(pose.T_world_from_camera))


def test_camera_array_conversion_rejects_missing_or_duplicate_pose(tmp_path: Path) -> None:
    images = make_images(tmp_path)
    intrinsics, poses = make_synthetic_two_view_cameras(images)

    with pytest.raises(ValueError, match="same non-zero count"):
        build_da3_camera_arrays(intrinsics, poses[:1])

    with pytest.raises(ValueError, match="unique"):
        build_da3_camera_arrays(intrinsics, (poses[0], poses[0]))


def test_synthetic_camera_factory_validates_inputs(tmp_path: Path) -> None:
    images = make_images(tmp_path)

    with pytest.raises(ValueError, match="exactly two"):
        make_synthetic_two_view_cameras(images[:1])
    with pytest.raises(ValueError, match="finite and positive"):
        make_synthetic_two_view_cameras(images, baseline_m=0.0)


def test_validate_da3_prediction_normalizes_three_by_four_poses() -> None:
    output = validate_da3_prediction(make_prediction(), expected_views=2)

    assert output.depth_m.shape == (2, 4, 6)
    assert output.confidence.shape == output.depth_m.shape
    assert output.T_camera_from_world.shape == (2, 4, 4)
    assert output.T_camera_from_world[:, 3, :] == pytest.approx(
        np.array([[0.0, 0.0, 0.0, 1.0]] * 2)
    )
    assert output.intrinsics.shape == (2, 3, 3)
    assert output.processed_images is not None
    assert output.is_metric is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("depth", np.ones((1, 4, 6)), "depth must have shape"),
        ("conf", np.ones((2, 3, 6)), "confidence shape"),
        ("depth", np.full((2, 4, 6), np.nan), "only finite"),
        ("depth", np.zeros((2, 4, 6)), "no strictly positive"),
        ("intrinsics", np.ones((2, 4, 4)), "intrinsics must be finite"),
        ("extrinsics", np.ones((2, 2, 4)), "camera poses must have shape"),
        ("is_metric", 0, "must be metric"),
    ],
)
def test_validate_da3_prediction_rejects_invalid_output(
    field: str,
    value: object,
    message: str,
) -> None:
    prediction = make_prediction()
    setattr(prediction, field, value)

    with pytest.raises(DA3PredictionError, match=message):
        validate_da3_prediction(prediction, expected_views=2)


def test_vendor_fingerprint_matches_documented_shell_algorithm(tmp_path: Path) -> None:
    vendor_dir = tmp_path / "vendor"
    nested = vendor_dir / "nested"
    nested.mkdir(parents=True)
    (vendor_dir / "a.txt").write_text("alpha", encoding="utf-8")
    (nested / "b.txt").write_text("beta", encoding="utf-8")

    lines = []
    for relative in (Path("a.txt"), Path("nested/b.txt")):
        digest = hashlib.sha256((vendor_dir / relative).read_bytes()).hexdigest()
        lines.append(f"{digest}  vendor/{relative.as_posix()}\n")
    expected = hashlib.sha256("".join(lines).encode()).hexdigest()

    assert compute_vendor_fingerprint(vendor_dir) == expected


class FakeAlignmentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.delegated = False

    def _align_to_input_extrinsics_intrinsics(
        self,
        extrinsics: torch.Tensor | None,
        intrinsics: torch.Tensor | None,
        prediction: SimpleNamespace,
        align_to_input_ext_scale: bool = True,
        ransac_view_thresh: int = 10,
    ) -> SimpleNamespace:
        del extrinsics, intrinsics, align_to_input_ext_scale, ransac_view_thresh
        self.delegated = True
        return prediction


def test_two_view_alignment_preserves_metric_depth_and_returns_supplied_poses() -> None:
    model = FakeAlignmentModel()
    _install_two_view_alignment(model)
    prediction = SimpleNamespace(
        depth=np.full((2, 4, 6), 2.5, dtype=np.float32),
        is_metric=1,
    )
    supplied_poses = torch.from_numpy(np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0))
    supplied_intrinsics = torch.from_numpy(np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0))

    result = model._align_to_input_extrinsics_intrinsics(
        supplied_poses,
        supplied_intrinsics,
        prediction,
    )

    assert result.depth == pytest.approx(np.full((2, 4, 6), 2.5))
    assert result.extrinsics == pytest.approx(supplied_poses[:, :3].numpy())
    assert result.intrinsics == pytest.approx(supplied_intrinsics.numpy())
    assert model.delegated is False


def test_two_view_alignment_rejects_non_metric_output() -> None:
    model = FakeAlignmentModel()
    _install_two_view_alignment(model)
    prediction = SimpleNamespace(depth=np.ones((2, 2, 2)), is_metric=0)
    supplied_poses = torch.from_numpy(np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0))
    supplied_intrinsics = torch.from_numpy(np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0))

    with pytest.raises(DA3PredictionError, match="nested metric"):
        model._align_to_input_extrinsics_intrinsics(
            supplied_poses,
            supplied_intrinsics,
            prediction,
        )
