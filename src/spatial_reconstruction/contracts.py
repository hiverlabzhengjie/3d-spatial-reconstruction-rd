"""Typed persistent contracts shared across pipeline stages."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Confidence = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]

Vector3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
Vector4 = tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
Matrix4x4 = tuple[Vector4, Vector4, Vector4, Vector4]


class ContractModel(BaseModel):
    """Shared strict and immutable contract behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class FrameRef(ContractModel):
    """Immutable reference to one camera frame."""

    camera_id: str
    frame_index: NonNegativeInt
    timestamp_seconds: NonNegativeFloat
    source_ref: str
    image_width: PositiveInt
    image_height: PositiveInt

    @field_validator("camera_id", "source_ref")
    @classmethod
    def require_non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("frame text fields must not be empty")
        return normalized


class CameraIntrinsics(ContractModel):
    """OpenCV pinhole intrinsics in pixel units."""

    camera_id: str
    fx: PositiveFloat
    fy: PositiveFloat
    cx: FiniteFloat
    cy: FiniteFloat
    image_width: PositiveInt
    image_height: PositiveInt
    distortion_coefficients: tuple[FiniteFloat, ...] = ()
    convention: Literal["opencv"] = "opencv"
    units: Literal["pixels"] = "pixels"

    @field_validator("camera_id")
    @classmethod
    def require_camera_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("camera_id must not be empty")
        return normalized


class CameraPose(ContractModel):
    """Explicit mutually inverse rigid camera/world transforms."""

    camera_id: str
    T_world_from_camera: Matrix4x4
    T_camera_from_world: Matrix4x4
    world_handedness: Literal["right"] = "right"
    world_units: Literal["metres"] = "metres"
    camera_convention: Literal["opencv"] = "opencv"

    @field_validator("camera_id")
    @classmethod
    def require_camera_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("camera_id must not be empty")
        return normalized

    @field_validator("T_world_from_camera", "T_camera_from_world")
    @classmethod
    def validate_rigid_transform(cls, value: Matrix4x4) -> Matrix4x4:
        matrix = np.asarray(value, dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError("camera transforms must contain only finite values")
        if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-7):
            raise ValueError("camera transforms must have homogeneous final row [0, 0, 0, 1]")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("camera transform rotation must be orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
            raise ValueError("camera transform rotation determinant must be +1")
        return value

    @model_validator(mode="after")
    def validate_mutual_inverse(self) -> Self:
        world_from_camera = np.asarray(self.T_world_from_camera, dtype=np.float64)
        camera_from_world = np.asarray(self.T_camera_from_world, dtype=np.float64)
        if not np.allclose(world_from_camera @ camera_from_world, np.eye(4), atol=1e-6):
            raise ValueError("camera transforms must be mutual inverses")
        return self


class InvalidDepthPolicy(StrEnum):
    """Declared handling for invalid depth samples."""

    REJECT = "reject"
    FILTER = "filter"


class DepthPrediction(ContractModel):
    """References to retained metric depth and confidence outputs."""

    frame: FrameRef
    depth_ref: str
    confidence_ref: str
    raw_output_ref: str
    model_id: str
    process_width: PositiveInt
    process_height: PositiveInt
    invalid_depth_policy: InvalidDepthPolicy = InvalidDepthPolicy.FILTER
    metric_units: Literal["metres"] = "metres"

    @field_validator("depth_ref", "confidence_ref", "raw_output_ref", "model_id")
    @classmethod
    def require_non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("depth prediction references must not be empty")
        return normalized


class PixelBox(ContractModel):
    """Axis-aligned pixel bounding box using exclusive maximum coordinates."""

    x_min: NonNegativeFloat
    y_min: NonNegativeFloat
    x_max: PositiveFloat
    y_max: PositiveFloat

    @model_validator(mode="after")
    def validate_extent(self) -> Self:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("pixel box must have positive width and height")
        return self


class SegmentationDetection(ContractModel):
    """One camera-local segmentation detection."""

    frame: FrameRef
    class_id: NonNegativeInt
    class_name: str
    confidence: Confidence
    box: PixelBox
    mask_ref: str
    camera_local_track_id: str | None = None

    @field_validator("class_name", "mask_ref")
    @classmethod
    def require_non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("detection text fields must not be empty")
        return normalized

    @field_validator("camera_local_track_id")
    @classmethod
    def validate_optional_track_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("camera_local_track_id must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_box_within_frame(self) -> Self:
        if self.box.x_max > self.frame.image_width or self.box.y_max > self.frame.image_height:
            raise ValueError("pixel box must remain within the source frame")
        return self


class ObservationState(StrEnum):
    """Availability state for one raw spatial observation."""

    OBSERVED = "observed"
    MISSING = "missing"
    OCCLUDED = "occluded"
    STALE = "stale"


class SpatialObservation(ContractModel):
    """Raw world-space entity observation with honest missing-data behavior."""

    entity_type: str
    entity_id: str
    timestamp_seconds: NonNegativeFloat
    state: ObservationState
    raw_world_xyz_m: Vector3 | None = None
    source_camera_ids: tuple[str, ...] = ()
    confidence: Confidence | None = None
    last_observed_timestamp_seconds: NonNegativeFloat | None = None
    provenance: str

    @field_validator("entity_type", "entity_id", "provenance")
    @classmethod
    def require_non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("spatial observation text fields must not be empty")
        return normalized

    @field_validator("source_camera_ids")
    @classmethod
    def validate_camera_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("source camera IDs must not be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("source camera IDs must be unique")
        return normalized

    @model_validator(mode="after")
    def enforce_state_semantics(self) -> Self:
        if self.state is ObservationState.OBSERVED:
            if self.raw_world_xyz_m is None:
                raise ValueError("observed state requires raw_world_xyz_m")
            if not self.source_camera_ids:
                raise ValueError("observed state requires at least one source camera")
            if self.confidence is None:
                raise ValueError("observed state requires confidence")
            if self.last_observed_timestamp_seconds is not None:
                raise ValueError("observed state must not carry a stale timestamp")
            return self

        if self.raw_world_xyz_m is not None:
            raise ValueError(f"{self.state.value} state must not carry raw_world_xyz_m")

        if self.state is ObservationState.STALE:
            if self.last_observed_timestamp_seconds is None:
                raise ValueError("stale state requires last_observed_timestamp_seconds")
            if self.last_observed_timestamp_seconds > self.timestamp_seconds:
                raise ValueError("last observed timestamp cannot be in the future")
        elif self.last_observed_timestamp_seconds is not None:
            raise ValueError("only stale state may carry last_observed_timestamp_seconds")

        return self


class TimingObservation(ContractModel):
    """One named non-negative runtime measurement."""

    phase: str
    seconds: NonNegativeFloat

    @field_validator("phase")
    @classmethod
    def require_phase(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("timing phase must not be empty")
        return normalized


class MemoryObservation(ContractModel):
    """One named non-negative memory measurement in bytes."""

    phase: str
    process_rss_bytes: NonNegativeInt
    mps_allocated_bytes: NonNegativeInt | None = None
    mps_driver_allocated_bytes: NonNegativeInt | None = None

    @field_validator("phase")
    @classmethod
    def require_phase(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("memory phase must not be empty")
        return normalized


class RunOutcome(StrEnum):
    """Outcome of one isolated model run."""

    PASSED = "passed"
    FAILED = "failed"


class ModelRunObservation(ContractModel):
    """Persistent diagnostic summary for one isolated model run."""

    model_id: str
    model_revision: str | None = None
    device: Literal["mps", "cpu", "cuda"]
    precision: Literal["float32", "float16", "bfloat16", "mixed", "unknown"]
    input_description: str
    timings: tuple[TimingObservation, ...]
    memory: tuple[MemoryObservation, ...]
    outcome: RunOutcome
    warnings: tuple[str, ...] = ()
    fallback: str | None = None
    artifact_refs: tuple[str, ...] = ()

    @field_validator("model_id", "input_description")
    @classmethod
    def require_non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("model run text fields must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_measurements(self) -> Self:
        timing_phases = [item.phase for item in self.timings]
        memory_phases = [item.phase for item in self.memory]
        if len(set(timing_phases)) != len(timing_phases):
            raise ValueError("timing phases must be unique")
        if len(set(memory_phases)) != len(memory_phases):
            raise ValueError("memory phases must be unique")
        return self
