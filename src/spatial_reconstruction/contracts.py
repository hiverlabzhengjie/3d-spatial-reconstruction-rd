"""Typed persistent contracts shared across pipeline stages."""

from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

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
Sha256Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def _stable_identity_digest(payload: dict[str, Any]) -> str:
    """Hash one canonical JSON identity payload."""

    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=lambda value: value.value if isinstance(value, StrEnum) else str(value),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


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


class FrameSourceKind(StrEnum):
    """Supported frame-source transport boundaries."""

    FILE = "file"
    RTSP = "rtsp"


class SourceFingerprintKind(StrEnum):
    """Meaning of the immutable source fingerprint."""

    CONTENT_SHA256 = "content_sha256"
    STREAM_CONFIGURATION_SHA256 = "stream_configuration_sha256"


class FrameIdentity(ContractModel):
    """Persistent immutable identity for one decoded source frame."""

    schema_version: Literal[1] = 1
    capture_session_id: str
    camera_id: str
    source_kind: FrameSourceKind
    source_frame_index: NonNegativeInt
    source_timestamp_seconds: NonNegativeFloat
    capture_timestamp_seconds: NonNegativeFloat
    source_ref: str
    source_fingerprint: Sha256Digest
    source_fingerprint_kind: SourceFingerprintKind
    synchronization_manifest_ref: str
    synchronization_manifest_sha256: Sha256Digest
    pose_version_id: str
    image_width: PositiveInt
    image_height: PositiveInt
    frame_id: Sha256Digest

    @field_validator(
        "capture_session_id",
        "camera_id",
        "source_ref",
        "synchronization_manifest_ref",
        "pose_version_id",
    )
    @classmethod
    def require_identity_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("frame identity text fields must not be empty")
        if normalized != value:
            raise ValueError("frame identity text fields must not have outer whitespace")
        return value

    @classmethod
    def create(
        cls,
        *,
        capture_session_id: str,
        camera_id: str,
        source_kind: FrameSourceKind,
        source_frame_index: int,
        source_timestamp_seconds: float,
        capture_timestamp_seconds: float,
        source_ref: str,
        source_fingerprint: str,
        source_fingerprint_kind: SourceFingerprintKind,
        synchronization_manifest_ref: str,
        synchronization_manifest_sha256: str,
        pose_version_id: str,
        image_width: int,
        image_height: int,
    ) -> Self:
        """Create an identity with its deterministic content-derived ID."""

        payload: dict[str, Any] = {
            "schema_version": 1,
            "capture_session_id": capture_session_id,
            "camera_id": camera_id,
            "source_kind": source_kind,
            "source_frame_index": source_frame_index,
            "source_timestamp_seconds": source_timestamp_seconds,
            "capture_timestamp_seconds": capture_timestamp_seconds,
            "source_ref": source_ref,
            "source_fingerprint": source_fingerprint,
            "source_fingerprint_kind": source_fingerprint_kind,
            "synchronization_manifest_ref": synchronization_manifest_ref,
            "synchronization_manifest_sha256": synchronization_manifest_sha256,
            "pose_version_id": pose_version_id,
            "image_width": image_width,
            "image_height": image_height,
        }
        payload["frame_id"] = _stable_identity_digest(payload)
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def validate_frame_id(self) -> Self:
        expected = _stable_identity_digest(
            self.model_dump(mode="json", exclude={"frame_id"})
        )
        if self.frame_id != expected:
            raise ValueError("frame_id does not match the immutable frame identity")
        return self

    def as_frame_ref(self) -> FrameRef:
        """Convert to the existing model-worker frame reference."""

        return FrameRef(
            camera_id=self.camera_id,
            frame_index=self.source_frame_index,
            timestamp_seconds=self.capture_timestamp_seconds,
            source_ref=self.source_ref,
            image_width=self.image_width,
            image_height=self.image_height,
        )


class FrameBundleStatus(StrEnum):
    """Availability state for a synchronized multi-camera bundle."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class SynchronizedFrameBundle(ContractModel):
    """Persistent capture-time bundle independent of worker completion order."""

    schema_version: Literal[1] = 1
    bundle_index: NonNegativeInt
    bundle_id: Sha256Digest
    capture_session_id: str
    capture_timestamp_seconds: NonNegativeFloat
    reference_camera_id: str
    expected_camera_ids: tuple[str, ...]
    frames: tuple[FrameIdentity, ...]
    missing_camera_ids: tuple[str, ...]
    status: FrameBundleStatus
    pairing_tolerance_seconds: PositiveFloat
    max_frame_time_difference_seconds: NonNegativeFloat
    synchronization_manifest_ref: str
    synchronization_manifest_sha256: Sha256Digest

    @field_validator(
        "capture_session_id",
        "reference_camera_id",
        "synchronization_manifest_ref",
    )
    @classmethod
    def require_bundle_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("bundle text fields must not be empty")
        if normalized != value:
            raise ValueError("bundle text fields must not have outer whitespace")
        return value

    @classmethod
    def create(
        cls,
        *,
        bundle_index: int,
        capture_session_id: str,
        capture_timestamp_seconds: float,
        reference_camera_id: str,
        expected_camera_ids: tuple[str, ...],
        frames: tuple[FrameIdentity, ...],
        pairing_tolerance_seconds: float,
        synchronization_manifest_ref: str,
        synchronization_manifest_sha256: str,
    ) -> Self:
        """Create and validate one deterministic frame bundle."""

        present = {frame.camera_id for frame in frames}
        missing = tuple(
            camera_id for camera_id in expected_camera_ids if camera_id not in present
        )
        status = (
            FrameBundleStatus.COMPLETE
            if not missing
            else FrameBundleStatus.INCOMPLETE
        )
        frame_times = [frame.capture_timestamp_seconds for frame in frames]
        max_difference = (
            float(max(frame_times) - min(frame_times)) if len(frame_times) > 1 else 0.0
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "bundle_index": bundle_index,
            "capture_session_id": capture_session_id,
            "capture_timestamp_seconds": capture_timestamp_seconds,
            "reference_camera_id": reference_camera_id,
            "expected_camera_ids": expected_camera_ids,
            "frames": frames,
            "missing_camera_ids": missing,
            "status": status,
            "pairing_tolerance_seconds": pairing_tolerance_seconds,
            "max_frame_time_difference_seconds": max_difference,
            "synchronization_manifest_ref": synchronization_manifest_ref,
            "synchronization_manifest_sha256": synchronization_manifest_sha256,
        }
        digest_payload = {
            **payload,
            "frames": [frame.frame_id for frame in frames],
        }
        payload["bundle_id"] = _stable_identity_digest(digest_payload)
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def validate_bundle_semantics(self) -> Self:
        if not self.expected_camera_ids:
            raise ValueError("expected_camera_ids must not be empty")
        if len(set(self.expected_camera_ids)) != len(self.expected_camera_ids):
            raise ValueError("expected_camera_ids must be unique")
        if self.reference_camera_id not in self.expected_camera_ids:
            raise ValueError("reference_camera_id must be expected")
        if not self.frames:
            raise ValueError("a frame bundle must contain at least one frame")

        frame_camera_ids = tuple(frame.camera_id for frame in self.frames)
        if len(set(frame_camera_ids)) != len(frame_camera_ids):
            raise ValueError("a frame bundle cannot contain duplicate cameras")
        if any(camera_id not in self.expected_camera_ids for camera_id in frame_camera_ids):
            raise ValueError("bundle frame camera is not in expected_camera_ids")
        expected_present_order = tuple(
            camera_id
            for camera_id in self.expected_camera_ids
            if camera_id in set(frame_camera_ids)
        )
        if frame_camera_ids != expected_present_order:
            raise ValueError("bundle frames must follow expected_camera_ids order")

        expected_missing = tuple(
            camera_id
            for camera_id in self.expected_camera_ids
            if camera_id not in set(frame_camera_ids)
        )
        if self.missing_camera_ids != expected_missing:
            raise ValueError("missing_camera_ids does not match available frames")
        expected_status = (
            FrameBundleStatus.COMPLETE
            if not expected_missing
            else FrameBundleStatus.INCOMPLETE
        )
        if self.status is not expected_status:
            raise ValueError("bundle status does not match missing cameras")

        for frame in self.frames:
            if frame.capture_session_id != self.capture_session_id:
                raise ValueError("bundle frames must share capture_session_id")
            if (
                frame.synchronization_manifest_ref
                != self.synchronization_manifest_ref
                or frame.synchronization_manifest_sha256
                != self.synchronization_manifest_sha256
            ):
                raise ValueError("bundle frames must share synchronization provenance")

        frame_times = [frame.capture_timestamp_seconds for frame in self.frames]
        expected_difference = (
            float(max(frame_times) - min(frame_times)) if len(frame_times) > 1 else 0.0
        )
        if not math.isclose(
            self.max_frame_time_difference_seconds,
            expected_difference,
            abs_tol=1e-12,
        ):
            raise ValueError("max frame-time difference is inconsistent")
        if self.max_frame_time_difference_seconds > self.pairing_tolerance_seconds:
            raise ValueError("bundle exceeds the pairing tolerance")

        reference_frames = [
            frame
            for frame in self.frames
            if frame.camera_id == self.reference_camera_id
        ]
        expected_timestamp = (
            reference_frames[0].capture_timestamp_seconds
            if reference_frames
            else min(frame_times)
        )
        if not math.isclose(
            self.capture_timestamp_seconds,
            expected_timestamp,
            abs_tol=1e-12,
        ):
            raise ValueError("bundle capture timestamp is inconsistent")

        digest_payload = self.model_dump(
            mode="json",
            exclude={"bundle_id"},
        )
        digest_payload["frames"] = [frame.frame_id for frame in self.frames]
        if self.bundle_id != _stable_identity_digest(digest_payload):
            raise ValueError("bundle_id does not match the immutable bundle identity")
        return self


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


class PerceptionTarget(StrEnum):
    """Canonical S03 target without erasing the vendor class label."""

    PERSON = "person"
    BACKPACK = "backpack"


class PerceptionCandidate(ContractModel):
    """One vendor detection selected as a canonical perception candidate."""

    detection_index: NonNegativeInt
    target: PerceptionTarget
    source_detection: SegmentationDetection
    policy_id: str

    @field_validator("policy_id")
    @classmethod
    def validate_policy_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or normalized != value:
            raise ValueError("perception policy ID must be non-empty without whitespace")
        return value

    @model_validator(mode="after")
    def validate_target_source_class(self) -> Self:
        source_class = self.source_detection.class_name
        if self.target is PerceptionTarget.PERSON and source_class != "person":
            raise ValueError("person candidates must retain the person vendor class")
        if self.target is PerceptionTarget.BACKPACK and source_class not in {
            "backpack",
            "handbag",
        }:
            raise ValueError(
                "backpack candidates must retain a D028 backpack or handbag class"
            )
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
    process_peak_rss_bytes: NonNegativeInt | None = None
    mps_allocated_bytes: NonNegativeInt | None = None
    mps_driver_allocated_bytes: NonNegativeInt | None = None
    mps_recommended_max_bytes: NonNegativeInt | None = None

    @field_validator("phase")
    @classmethod
    def require_phase(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("memory phase must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_peak_rss(self) -> Self:
        if (
            self.process_peak_rss_bytes is not None
            and self.process_peak_rss_bytes < self.process_rss_bytes
        ):
            raise ValueError("peak process RSS cannot be below the phase RSS")
        return self


class RunOutcome(StrEnum):
    """Outcome of one isolated model run."""

    PASSED = "passed"
    FAILED = "failed"


class ModelRunObservation(ContractModel):
    """Persistent diagnostic summary for one isolated model run."""

    model_id: str
    model_revision: str | None = None
    requested_device: Literal["mps", "cpu", "cuda"]
    device: Literal["mps", "cpu", "cuda"] | None
    device_selection_error: str | None = None
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

    @field_validator("device_selection_error")
    @classmethod
    def validate_optional_device_error(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("device_selection_error must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_measurements(self) -> Self:
        timing_phases = [item.phase for item in self.timings]
        memory_phases = [item.phase for item in self.memory]
        if len(set(timing_phases)) != len(timing_phases):
            raise ValueError("timing phases must be unique")
        if len(set(memory_phases)) != len(memory_phases):
            raise ValueError("memory phases must be unique")
        if self.device is None:
            if self.outcome is not RunOutcome.FAILED or self.device_selection_error is None:
                raise ValueError(
                    "an unavailable device requires a failed outcome and selection error"
                )
        elif self.device_selection_error is not None:
            raise ValueError("a selected device cannot carry a device selection error")
        if self.outcome is RunOutcome.PASSED and self.device is None:
            raise ValueError("a passed run requires an actual device")
        return self
