"""Deterministic S04 action-depth selection and persistent contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Literal, Self

from pydantic import field_validator, model_validator

from spatial_reconstruction.contracts import (
    ContractModel,
    NonNegativeInt,
    PerceptionTarget,
    PositiveInt,
    Sha256Digest,
    SynchronizedFrameBundle,
)
from spatial_reconstruction.perception import (
    PerceptionPresenceState,
    PerceptionTargetFrameState,
)


def _stable_digest(payload: Mapping[str, object]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class ActionDepthKeyframeSpec(ContractModel):
    """One explicit action-spanning processed-frame selection."""

    source_frame_index: NonNegativeInt
    phase_id: str
    selection_reason: str
    required_backpack_camera_id: Literal["camera_a", "camera_b"]

    @field_validator("phase_id", "selection_reason")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("keyframe text must be non-empty without outer whitespace")
        return value


class ActionDepthJob(ContractModel):
    """Stable DA3 action-keyframe job tied to exact mask evidence."""

    schema_version: Literal[1] = 1
    job_id: Sha256Digest
    bundle: SynchronizedFrameBundle
    phase_id: str
    selection_reason: str
    mask_evidence: tuple[PerceptionTargetFrameState, ...]
    model_id: str
    model_revision: str
    process_resolution: PositiveInt
    attempt: PositiveInt = 1

    @field_validator("phase_id", "selection_reason", "model_id")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("action-depth text must be non-empty without outer whitespace")
        return value

    @field_validator("model_revision")
    @classmethod
    def validate_model_revision(cls, value: str) -> str:
        if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("DA3 revision must be a 40-character lowercase hex ID")
        return value

    @field_validator("process_resolution")
    @classmethod
    def validate_process_resolution(cls, value: int) -> int:
        if value % 14 != 0:
            raise ValueError("DA3 process resolution must be a multiple of 14")
        return value

    @classmethod
    def create(
        cls,
        *,
        bundle: SynchronizedFrameBundle,
        phase_id: str,
        selection_reason: str,
        mask_evidence: Sequence[PerceptionTargetFrameState],
        model_id: str,
        model_revision: str,
        process_resolution: int,
        attempt: int = 1,
    ) -> Self:
        ordered_evidence = tuple(
            sorted(
                mask_evidence,
                key=lambda item: (item.frame_identity.camera_id, item.target.value),
            )
        )
        identity = {
            "schema_version": 1,
            "bundle_id": bundle.bundle_id,
            "phase_id": phase_id,
            "mask_evidence": [
                {
                    "camera_id": item.frame_identity.camera_id,
                    "frame_id": item.frame_identity.frame_id,
                    "perception_job_id": item.job_id,
                    "target": item.target.value,
                    "state": item.state.value,
                }
                for item in ordered_evidence
            ],
            "model_id": model_id,
            "model_revision": model_revision,
            "process_resolution": process_resolution,
            "attempt": attempt,
        }
        return cls(
            job_id=_stable_digest(identity),
            bundle=bundle,
            phase_id=phase_id,
            selection_reason=selection_reason,
            mask_evidence=ordered_evidence,
            model_id=model_id,
            model_revision=model_revision,
            process_resolution=process_resolution,
            attempt=attempt,
        )

    @model_validator(mode="after")
    def validate_semantics_and_identity(self) -> Self:
        if self.bundle.missing_camera_ids:
            raise ValueError("action-depth jobs require a complete camera bundle")
        if tuple(frame.camera_id for frame in self.bundle.frames) != (
            "camera_a",
            "camera_b",
        ):
            raise ValueError("action-depth jobs require ordered camera A/B frames")
        frame_by_camera = {frame.camera_id: frame for frame in self.bundle.frames}
        evidence_keys: set[tuple[str, PerceptionTarget]] = set()
        for item in self.mask_evidence:
            camera_id = item.frame_identity.camera_id
            frame = frame_by_camera.get(camera_id)
            if frame is None or item.frame_identity.frame_id != frame.frame_id:
                raise ValueError("mask evidence must match an exact job bundle frame")
            if item.state is not PerceptionPresenceState.OBSERVED:
                raise ValueError("action-depth preflight uses only observed mask evidence")
            if len(item.candidate_metrics) != 1:
                raise ValueError("observed mask evidence must contain one candidate mask")
            key = (camera_id, item.target)
            if key in evidence_keys:
                raise ValueError("action-depth mask evidence cannot duplicate camera/target")
            evidence_keys.add(key)
        if not self.mask_evidence:
            raise ValueError("action-depth job requires retained observed mask evidence")
        targets = {item.target for item in self.mask_evidence}
        if targets != {PerceptionTarget.PERSON, PerceptionTarget.BACKPACK}:
            raise ValueError("action-depth job requires observed person and backpack evidence")

        identity = {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle.bundle_id,
            "phase_id": self.phase_id,
            "mask_evidence": [
                {
                    "camera_id": item.frame_identity.camera_id,
                    "frame_id": item.frame_identity.frame_id,
                    "perception_job_id": item.job_id,
                    "target": item.target.value,
                    "state": item.state.value,
                }
                for item in self.mask_evidence
            ],
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "process_resolution": self.process_resolution,
            "attempt": self.attempt,
        }
        if self.job_id != _stable_digest(identity):
            raise ValueError("job_id does not match action-depth source identity")
        return self


class ActionDepthProcessingPolicy(ContractModel):
    """Explicit boundary preventing S02 policies from leaking into S04."""

    process_resolution: PositiveInt
    source_frames_undistorted_before_inference: Literal[True]
    raw_da3_metric_depth_preserved: Literal[True]
    s02_marker_scale_applied: Literal[False]
    s02_static_confidence_policy_applied: Literal[False]
    s02_door_supplement_applied: Literal[False]
    mask_resampling_or_localization_performed: Literal[False]

    @field_validator("process_resolution")
    @classmethod
    def validate_process_resolution(cls, value: int) -> int:
        if value % 14 != 0:
            raise ValueError("DA3 process resolution must be a multiple of 14")
        return value


class ActionDepthPredictionRecord(ContractModel):
    """One retained raw two-camera action-depth prediction."""

    job: ActionDepthJob
    raw_prediction_ref: str
    raw_prediction_sha256: Sha256Digest
    depth_confidence_preview_ref: str
    depth_confidence_preview_sha256: Sha256Digest
    cameras: dict[str, dict[str, Any]]

    @model_validator(mode="after")
    def require_camera_pair(self) -> Self:
        if set(self.cameras) != {"camera_a", "camera_b"}:
            raise ValueError("action-depth prediction must contain camera A and B")
        return self


class ActionDepthRunSummary(ContractModel):
    """Strict persistent summary for the raw S04 action-depth preflight."""

    schema_version: Literal[1]
    status: Literal["completed_pending_mask_depth_qa"]
    stage: Literal["S04"]
    created_at_utc: datetime
    capture_session_id: str
    pose_version_id: str
    input_provenance: dict[str, Any]
    model: dict[str, Any]
    selection_config_ref: str
    selection_config_sha256: Sha256Digest
    processing: ActionDepthProcessingPolicy
    predictions: tuple[ActionDepthPredictionRecord, ...]
    runtime: dict[str, Any]
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        if not self.predictions:
            raise ValueError("action-depth run must retain at least one prediction")
        jobs = tuple(record.job for record in self.predictions)
        if len({job.job_id for job in jobs}) != len(jobs):
            raise ValueError("action-depth jobs must be unique")
        frame_indices = tuple(job.bundle.frames[0].source_frame_index for job in jobs)
        if frame_indices != tuple(sorted(frame_indices)):
            raise ValueError("action-depth predictions must follow capture order")
        if any(job.process_resolution != self.processing.process_resolution for job in jobs):
            raise ValueError("action-depth job resolution differs from run policy")
        return self


def select_action_depth_jobs(
    *,
    bundles: Sequence[SynchronizedFrameBundle],
    states: Sequence[PerceptionTargetFrameState],
    specs: Sequence[ActionDepthKeyframeSpec],
    model_id: str,
    model_revision: str,
    process_resolution: int,
) -> tuple[ActionDepthJob, ...]:
    """Select exact processed frames with observed person and backpack masks."""

    indices = tuple(spec.source_frame_index for spec in specs)
    if not indices or indices != tuple(sorted(set(indices))):
        raise ValueError("action-depth frame indices must be unique and increasing")
    if len({spec.phase_id for spec in specs}) != len(specs):
        raise ValueError("action-depth phase IDs must be unique")

    bundles_by_index: dict[int, SynchronizedFrameBundle] = {}
    for bundle in bundles:
        camera_indices = {frame.source_frame_index for frame in bundle.frames}
        if len(camera_indices) == 1:
            (source_index,) = tuple(camera_indices)
            if source_index in bundles_by_index:
                raise ValueError("multiple bundles share one source frame index")
            bundles_by_index[source_index] = bundle

    state_lookup: dict[tuple[int, str, PerceptionTarget], PerceptionTargetFrameState] = {}
    for state in states:
        key = (
            state.frame_identity.source_frame_index,
            state.frame_identity.camera_id,
            state.target,
        )
        if key in state_lookup:
            raise ValueError("duplicate perception state for frame/camera/target")
        state_lookup[key] = state

    jobs: list[ActionDepthJob] = []
    for spec in specs:
        selected_bundle = bundles_by_index.get(spec.source_frame_index)
        if selected_bundle is None:
            raise ValueError(
                f"no synchronized same-index bundle for frame {spec.source_frame_index}"
            )
        evidence = tuple(
            state_lookup[key]
            for camera_id in ("camera_a", "camera_b")
            for target in PerceptionTarget
            if (
                key := (spec.source_frame_index, camera_id, target)
            ) in state_lookup
            and state_lookup[key].state is PerceptionPresenceState.OBSERVED
        )
        required_bag = state_lookup.get(
            (
                spec.source_frame_index,
                spec.required_backpack_camera_id,
                PerceptionTarget.BACKPACK,
            )
        )
        if required_bag is None or required_bag.state is not PerceptionPresenceState.OBSERVED:
            raise ValueError(
                "selected frame lacks the required observed backpack mask in "
                f"{spec.required_backpack_camera_id}"
            )
        jobs.append(
            ActionDepthJob.create(
                bundle=selected_bundle,
                phase_id=spec.phase_id,
                selection_reason=spec.selection_reason,
                mask_evidence=evidence,
                model_id=model_id,
                model_revision=model_revision,
                process_resolution=process_resolution,
            )
        )
    return tuple(jobs)
