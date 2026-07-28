from pathlib import Path

import pytest
from pydantic import ValidationError

from spatial_reconstruction.config import (
    DEFAULT_CONFIG_PATH,
    ProjectConfig,
    load_project_config,
)


def test_default_config_loads_with_exact_baseline_models() -> None:
    config = load_project_config()

    assert config.config_version == 1
    assert config.models.da3 == "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    assert config.models.yolo == "yolov8n-seg.pt"
    assert config.models.qwen == "Qwen/Qwen3-VL-2B-Instruct"
    assert config.paths.da3_vendor_dir.is_absolute()
    assert config.paths.da3_vendor_dir.name == "Depth-Anything-3-main"
    assert config.da3.process_resolutions == (336, 420, 504)
    assert config.perception.inference_image_size == 640
    assert config.qwen.smoke_frame_count == 4
    assert config.qwen.max_new_tokens == 64


def test_config_round_trips_through_json() -> None:
    config = load_project_config()

    restored = ProjectConfig.model_validate_json(config.model_dump_json())

    assert restored == config


def test_relative_paths_resolve_from_explicit_project_root(tmp_path: Path) -> None:
    config = load_project_config(project_root=tmp_path)

    assert config.paths.da3_vendor_dir == (tmp_path / "Depth-Anything-3-main").resolve()
    assert config.paths.artifacts_dir == (tmp_path / "artifacts").resolve()


def test_unknown_config_field_is_rejected(tmp_path: Path) -> None:
    contents = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    path = tmp_path / "unknown.yaml"
    path.write_text(f"{contents}\nunexpected: true\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="unexpected"):
        load_project_config(path)


def test_non_mapping_config_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="root must be a mapping"):
        load_project_config(path)


def test_da3_resolution_must_be_patch_aligned() -> None:
    payload = load_project_config().model_dump(mode="json")
    payload["da3"]["process_resolutions"] = [335]

    with pytest.raises(ValidationError, match="positive multiples of 14"):
        ProjectConfig.model_validate(payload)


def test_baseline_model_change_requires_a_decision() -> None:
    payload = load_project_config().model_dump(mode="json")
    payload["models"]["qwen"] = "different/model"

    with pytest.raises(ValidationError, match="baseline model identifiers changed"):
        ProjectConfig.model_validate(payload)
