"""Validated project configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Confidence = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class ConfigModel(BaseModel):
    """Shared strict configuration behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class ProjectPaths(ConfigModel):
    """Project-owned and vendor paths."""

    da3_vendor_dir: Path
    artifacts_dir: Path

    def resolved(self, project_root: Path) -> Self:
        """Return a copy with relative paths resolved from the project root."""

        root = project_root.resolve()
        return self.model_copy(
            update={
                "da3_vendor_dir": _resolve_from(root, self.da3_vendor_dir),
                "artifacts_dir": _resolve_from(root, self.artifacts_dir),
            }
        )


class ModelIdentifiers(ConfigModel):
    """Exact baseline model identifiers."""

    da3: str
    yolo: str
    qwen: str

    @field_validator("da3", "yolo", "qwen")
    @classmethod
    def require_non_empty_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("model identifiers must not be empty")
        return normalized


class RuntimeConfig(ConfigModel):
    """Device and precision policy for heavy model processes."""

    preferred_device: Literal["mps", "cpu", "cuda"] = "mps"
    precision: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    allow_cpu_fallback: bool = False


class DA3Config(ConfigModel):
    """DA3 smoke-test operating candidates."""

    process_resolutions: tuple[int, ...] = (336, 420, 504)
    keyframe_interval_candidates_seconds: tuple[PositiveFloat, ...] = (1.0, 2.0, 5.0)

    @field_validator("process_resolutions")
    @classmethod
    def validate_process_resolutions(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if not values:
            raise ValueError("at least one DA3 process resolution is required")
        if any(value <= 0 or value % 14 != 0 for value in values):
            raise ValueError("DA3 process resolutions must be positive multiples of 14")
        if len(set(values)) != len(values) or tuple(sorted(values)) != values:
            raise ValueError("DA3 process resolutions must be unique and ascending")
        return values

    @field_validator("keyframe_interval_candidates_seconds")
    @classmethod
    def validate_keyframe_intervals(
        cls, values: tuple[PositiveFloat, ...]
    ) -> tuple[PositiveFloat, ...]:
        if not values:
            raise ValueError("at least one keyframe interval candidate is required")
        numeric_values = tuple(float(value) for value in values)
        if len(set(numeric_values)) != len(numeric_values):
            raise ValueError("keyframe interval candidates must be unique")
        if tuple(sorted(numeric_values)) != numeric_values:
            raise ValueError("keyframe interval candidates must be ascending")
        return values


class PerceptionConfig(ConfigModel):
    """Baseline detector configuration."""

    detection_confidence_threshold: Confidence = 0.25
    target_classes: tuple[str, ...] = ("person", "backpack")

    @field_validator("target_classes")
    @classmethod
    def validate_target_classes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if not normalized or any(not value for value in normalized):
            raise ValueError("target classes must be non-empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("target classes must be unique")
        return normalized


class OutputConfig(ConfigModel):
    """Persistent-output policy."""

    preserve_raw_model_outputs: bool = True


class ProjectConfig(ConfigModel):
    """Top-level project configuration."""

    config_version: Literal[1]
    paths: ProjectPaths
    models: ModelIdentifiers
    runtime: RuntimeConfig
    da3: DA3Config
    perception: PerceptionConfig
    outputs: OutputConfig

    @model_validator(mode="after")
    def enforce_baseline_model_identities(self) -> Self:
        expected = {
            "da3": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
            "yolo": "yolov8n-seg.pt",
            "qwen": "Qwen/Qwen3-VL-2B-Instruct",
        }
        actual = self.models.model_dump()
        mismatches = [name for name, value in expected.items() if actual[name] != value]
        if mismatches:
            joined = ", ".join(mismatches)
            raise ValueError(f"baseline model identifiers changed without a decision: {joined}")
        return self


def load_project_config(
    path: Path = DEFAULT_CONFIG_PATH,
    *,
    project_root: Path = PROJECT_ROOT,
) -> ProjectConfig:
    """Load, validate, and resolve a project YAML configuration."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")

    config = ProjectConfig.model_validate(raw)
    return config.model_copy(update={"paths": config.paths.resolved(project_root)})


def _resolve_from(project_root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()
