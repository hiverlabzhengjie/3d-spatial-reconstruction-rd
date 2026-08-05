"""Hash-bound S06 offline orchestration and Rerun timeline contracts."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import field_validator, model_validator

from spatial_reconstruction.contracts import (
    ContractModel,
    PositiveFloat,
    PositiveInt,
    Sha256Digest,
)


class WorkerKind(StrEnum):
    """Integrated logical workers; heavy MPS execution remains serialized."""

    PERCEPTION = "perception"
    DA3 = "da3"
    QWEN = "qwen"
    GEOMETRY = "geometry"
    RERUN = "rerun"


class ArtifactRole(StrEnum):
    """Required accepted inputs for the first S06 offline assembly."""

    ACTION_SYNCHRONIZATION = "action_synchronization"
    ACTION_CALIBRATION = "action_calibration"
    SCENE_METADATA = "scene_metadata"
    STATIC_SCENE = "static_scene"
    PERCEPTION_TIMELINE = "perception_timeline"
    TEMPORAL_PRESENTATION = "temporal_presentation"
    INTERACTION_TIMELINE = "interaction_timeline"
    QWEN_EVENT_PLAN = "qwen_event_plan"
    QWEN_EVENT_RESULTS = "qwen_event_results"


class OrchestrationArtifact(ContractModel):
    """One immutable accepted artifact used by integrated orchestration."""

    role: ArtifactRole
    source_ref: str
    source_sha256: Sha256Digest

    @field_validator("source_ref")
    @classmethod
    def validate_ref(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("orchestration artifact reference must be non-empty and trimmed")
        return value


class SourceVideo(ContractModel):
    """One exact synchronized file source used by the offline Rerun assembly."""

    camera_id: Literal["camera_a", "camera_b"]
    source_ref: str
    source_sha256: Sha256Digest
    decoded_frame_count: PositiveInt
    duration_seconds: PositiveFloat
    nominal_frame_rate_fps: PositiveFloat

    @field_validator("source_ref")
    @classmethod
    def validate_ref(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("source video reference must be non-empty and trimmed")
        return value


class OrchestrationPolicy(ContractModel):
    """Versioned deterministic offline policy for S06 WP1."""

    policy_id: Literal["s06_integrated_offline_orchestration_v1"] = (
        "s06_integrated_offline_orchestration_v1"
    )
    source_mode: Literal["file"] = "file"
    authoritative_timeline: Literal["capture_time"] = "capture_time"
    rerun_timeline_name: Literal["capture_time"] = "capture_time"
    worker_completion_order_is_authoritative: Literal[False] = False
    perception_queue_capacity_per_camera: PositiveInt = 8
    da3_queue_capacity: PositiveInt = 2
    qwen_queue_capacity: PositiveInt = 3
    offline_overflow_policy: Literal["throttle_and_drain"] = "throttle_and_drain"
    heavy_mps_permit_count: Literal[1] = 1
    qwen_hard_timeout_seconds: PositiveFloat = 45.0
    qwen_maximum_process_attempts: Literal[2] = 2
    qwen_failure_blocks_geometry: Literal[False] = False


class Stage06OrchestrationManifest(ContractModel):
    """Stable entry manifest binding video and accepted S02-S05 evidence."""

    schema_version: Literal[1] = 1
    stage: Literal["S06"] = "S06"
    manifest_id: Sha256Digest
    capture_session_id: str
    synchronization_manifest_ref: str
    synchronization_manifest_sha256: Sha256Digest
    source_videos: tuple[SourceVideo, SourceVideo]
    artifacts: tuple[OrchestrationArtifact, ...]
    policy: OrchestrationPolicy

    @field_validator("capture_session_id", "synchronization_manifest_ref")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("orchestration identity text must be non-empty and trimmed")
        return value

    @classmethod
    def create(
        cls,
        *,
        capture_session_id: str,
        synchronization_manifest_ref: str,
        synchronization_manifest_sha256: str,
        source_videos: tuple[SourceVideo, SourceVideo],
        artifacts: tuple[OrchestrationArtifact, ...],
        policy: OrchestrationPolicy | None = None,
    ) -> Self:
        selected_policy = policy or OrchestrationPolicy()
        identity = {
            "schema_version": 1,
            "stage": "S06",
            "capture_session_id": capture_session_id,
            "synchronization_manifest_ref": synchronization_manifest_ref,
            "synchronization_manifest_sha256": synchronization_manifest_sha256,
            "source_videos": source_videos,
            "artifacts": artifacts,
            "policy": selected_policy,
        }
        return cls(
            manifest_id=_stable_digest(identity),
            capture_session_id=capture_session_id,
            synchronization_manifest_ref=synchronization_manifest_ref,
            synchronization_manifest_sha256=synchronization_manifest_sha256,
            source_videos=source_videos,
            artifacts=artifacts,
            policy=selected_policy,
        )

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if tuple(video.camera_id for video in self.source_videos) != (
            "camera_a",
            "camera_b",
        ):
            raise ValueError("offline orchestration requires Camera A then Camera B")
        roles = tuple(artifact.role for artifact in self.artifacts)
        if len(set(roles)) != len(roles):
            raise ValueError("orchestration artifact roles must be unique")
        if set(roles) != set(ArtifactRole):
            raise ValueError("orchestration manifest must bind every required artifact role")
        expected = _stable_digest(self.model_dump(mode="json", exclude={"manifest_id"}))
        if self.manifest_id != expected:
            raise ValueError("orchestration manifest ID differs from bound inputs")
        return self


def _stable_digest(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=lambda value: value.model_dump(mode="json"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
