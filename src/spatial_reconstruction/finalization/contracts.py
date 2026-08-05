"""Hash-bound contracts for the selected S07 final demonstration run."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import field_validator, model_validator

from spatial_reconstruction.contracts import ContractModel, Sha256Digest
from spatial_reconstruction.orchestration import SourceVideo


class FinalArtifactRole(StrEnum):
    """Accepted S06 evidence required at the S07 final-run boundary."""

    ORCHESTRATION_SUMMARY = "orchestration_summary"
    INTEGRATED_RERUN = "integrated_rerun"
    RERUN_EXPORT_SUMMARY = "rerun_export_summary"
    INTEGRATED_REPLAY_SUMMARY = "integrated_replay_summary"
    RTSP_SMOKE_SUMMARY = "rtsp_smoke_summary"
    TRACK_EVENT_EXPORT_SUMMARY = "track_event_export_summary"
    STAGE06_GATE_AUDIT = "stage06_gate_audit"


class FinalRunArtifact(ContractModel):
    """One immutable accepted artifact used by the final demonstration run."""

    role: FinalArtifactRole
    source_ref: str
    source_sha256: Sha256Digest

    @field_validator("source_ref")
    @classmethod
    def validate_ref(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("final-run artifact reference must be non-empty and trimmed")
        return value


class FinalRecordingSelection(ContractModel):
    """User-approved physical recording selection for S07."""

    selection_id: Literal["s07_action_take_01_final_v1"] = "s07_action_take_01_final_v1"
    capture_session_id: Literal["s01_capture_20260729"] = "s01_capture_20260729"
    recording_name: Literal["action_take_01"] = "action_take_01"
    selected_by_user: Literal[True] = True
    selected_on: Literal["2026-08-05"] = "2026-08-05"
    recapture_required: Literal[False] = False
    recalibration_required: Literal[False] = False


class FinalRunPolicy(ContractModel):
    """Boundaries for regenerating the final demonstration from retained evidence."""

    policy_id: Literal["s07_reproducible_final_run_v1"] = "s07_reproducible_final_run_v1"
    source_mode: Literal["recorded_mp4"] = "recorded_mp4"
    authoritative_timeline: Literal["capture_time"] = "capture_time"
    model_inference_required_for_final_assembly: Literal[False] = False
    reuse_verified_model_outputs: Literal[True] = True
    preserve_null_unavailable_xyz: Literal[True] = True
    preserve_disconnected_measured_trajectories: Literal[True] = True
    qwen_has_spatial_authority: Literal[False] = False
    demonstrated_live_capacity: Literal[False] = False


class Stage07FinalRunManifest(ContractModel):
    """Stable S07 entry manifest binding the selected recording and S06 outputs."""

    schema_version: Literal[1] = 1
    stage: Literal["S07"] = "S07"
    manifest_id: Sha256Digest
    source_stage06_manifest_id: Sha256Digest
    recording: FinalRecordingSelection
    source_videos: tuple[SourceVideo, SourceVideo]
    artifacts: tuple[FinalRunArtifact, ...]
    policy: FinalRunPolicy

    @classmethod
    def create(
        cls,
        *,
        source_stage06_manifest_id: str,
        source_videos: tuple[SourceVideo, SourceVideo],
        artifacts: tuple[FinalRunArtifact, ...],
        recording: FinalRecordingSelection | None = None,
        policy: FinalRunPolicy | None = None,
    ) -> Self:
        selected_recording = recording or FinalRecordingSelection()
        selected_policy = policy or FinalRunPolicy()
        identity = {
            "schema_version": 1,
            "stage": "S07",
            "source_stage06_manifest_id": source_stage06_manifest_id,
            "recording": selected_recording,
            "source_videos": source_videos,
            "artifacts": artifacts,
            "policy": selected_policy,
        }
        return cls(
            manifest_id=_stable_digest(identity),
            source_stage06_manifest_id=source_stage06_manifest_id,
            recording=selected_recording,
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
            raise ValueError("final run requires Camera A then Camera B")
        if any("action_take_01" not in video.source_ref for video in self.source_videos):
            raise ValueError("final source videos must belong to action_take_01")
        if any(video.decoded_frame_count != 1047 for video in self.source_videos):
            raise ValueError("final source videos must retain 1,047 decoded frames")
        roles = tuple(artifact.role for artifact in self.artifacts)
        if len(set(roles)) != len(roles):
            raise ValueError("final-run artifact roles must be unique")
        if set(roles) != set(FinalArtifactRole):
            raise ValueError("final-run manifest must bind every required artifact role")
        expected = _stable_digest(self.model_dump(mode="json", exclude={"manifest_id"}))
        if self.manifest_id != expected:
            raise ValueError("final-run manifest ID differs from bound inputs")
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
