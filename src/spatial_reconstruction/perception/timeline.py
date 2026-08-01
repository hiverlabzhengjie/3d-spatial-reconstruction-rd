"""Typed per-target S03 states derived from retained perception outputs."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import field_validator, model_validator

from spatial_reconstruction.contracts import (
    Confidence,
    ContractModel,
    FrameIdentity,
    NonNegativeInt,
    PerceptionCandidate,
    PerceptionTarget,
    Sha256Digest,
)
from spatial_reconstruction.perception.worker import (
    PerceptionFrameResult,
    PerceptionResultOutcome,
)

UInt8Array = NDArray[np.uint8]


class PerceptionPresenceState(StrEnum):
    """Honest image-plane availability for one canonical target."""

    OBSERVED = "observed"
    UNTRACKED = "untracked"
    AMBIGUOUS = "ambiguous"
    MISSING = "missing"
    FAILED = "failed"


class ImagePlaneVisibility(StrEnum):
    """Whether a selected mask reaches the known image boundary."""

    FULLY_IN_FRAME = "fully_in_frame"
    FRAME_EDGE_TRUNCATED = "frame_edge_truncated"


class CandidateMaskMetrics(ContractModel):
    """One retained candidate plus source-image mask and visibility metrics."""

    schema_version: Literal[1] = 1
    candidate: PerceptionCandidate
    mask_area_pixels: NonNegativeInt
    mask_area_fraction: Confidence
    touches_frame_border: bool
    visibility: ImagePlaneVisibility

    @model_validator(mode="after")
    def validate_visibility(self) -> Self:
        expected = (
            ImagePlaneVisibility.FRAME_EDGE_TRUNCATED
            if self.touches_frame_border
            else ImagePlaneVisibility.FULLY_IN_FRAME
        )
        if self.visibility is not expected:
            raise ValueError("visibility must agree with frame-border contact")
        return self


class PerceptionTargetFrameState(ContractModel):
    """Persistent per-frame state for person or physical-backpack perception."""

    schema_version: Literal[1] = 1
    job_id: Sha256Digest
    frame_identity: FrameIdentity
    target: PerceptionTarget
    state: PerceptionPresenceState
    candidate_metrics: tuple[CandidateMaskMetrics, ...] = ()
    error_type: str | None = None
    error_message: str | None = None

    @field_validator("error_type", "error_message")
    @classmethod
    def validate_optional_error_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or normalized != value:
            raise ValueError("target-state error text must be non-empty without whitespace")
        return value

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        expected_frame = self.frame_identity.as_frame_ref()
        for metric in self.candidate_metrics:
            if metric.candidate.target is not self.target:
                raise ValueError("target-state candidate has a different canonical target")
            if metric.candidate.source_detection.frame != expected_frame:
                raise ValueError("target-state candidate has a different source frame")

        candidate_count = len(self.candidate_metrics)
        tracked_count = sum(
            metric.candidate.source_detection.camera_local_track_id is not None
            for metric in self.candidate_metrics
        )
        if self.state is PerceptionPresenceState.OBSERVED:
            if candidate_count != 1 or tracked_count != 1:
                raise ValueError("observed state requires exactly one tracked candidate")
        elif self.state is PerceptionPresenceState.UNTRACKED:
            if candidate_count != 1 or tracked_count != 0:
                raise ValueError("untracked state requires exactly one untracked candidate")
        elif self.state is PerceptionPresenceState.AMBIGUOUS:
            if candidate_count < 2:
                raise ValueError("ambiguous state requires multiple candidates")
        elif self.state is PerceptionPresenceState.MISSING:
            if candidate_count or self.error_type is not None or self.error_message is not None:
                raise ValueError("missing state cannot contain candidates or errors")
        elif candidate_count or not self.error_type or not self.error_message:
            raise ValueError("failed state requires errors and cannot contain candidates")

        if self.state is not PerceptionPresenceState.FAILED and (
            self.error_type is not None or self.error_message is not None
        ):
            raise ValueError("only failed state can contain error details")
        return self


def build_target_frame_states(
    result: PerceptionFrameResult,
    source_sized_masks: NDArray[np.generic] | None,
) -> tuple[PerceptionTargetFrameState, PerceptionTargetFrameState]:
    """Derive deterministic person/backpack states without inferring occlusion."""

    if result.outcome is PerceptionResultOutcome.FAILED:
        if source_sized_masks is not None:
            raise ValueError("failed perception result cannot have retained masks")
        if result.error_type is None or result.error_message is None:
            raise ValueError("failed perception result lacks explicit error details")
        person = PerceptionTargetFrameState(
            job_id=result.job.job_id,
            frame_identity=result.job.frame_identity,
            target=PerceptionTarget.PERSON,
            state=PerceptionPresenceState.FAILED,
            error_type=result.error_type,
            error_message=result.error_message,
        )
        backpack = PerceptionTargetFrameState(
            job_id=result.job.job_id,
            frame_identity=result.job.frame_identity,
            target=PerceptionTarget.BACKPACK,
            state=PerceptionPresenceState.FAILED,
            error_type=result.error_type,
            error_message=result.error_message,
        )
        return person, backpack

    masks = _validate_source_masks(result, source_sized_masks)
    records: list[PerceptionTargetFrameState] = []
    for target in PerceptionTarget:
        matching = tuple(
            candidate for candidate in result.candidates if candidate.target is target
        )
        metrics = tuple(
            _measure_candidate(candidate, masks, result.job.frame_identity)
            for candidate in matching
        )
        if not metrics:
            state = PerceptionPresenceState.MISSING
        elif len(metrics) > 1:
            state = PerceptionPresenceState.AMBIGUOUS
        elif metrics[0].candidate.source_detection.camera_local_track_id is None:
            state = PerceptionPresenceState.UNTRACKED
        else:
            state = PerceptionPresenceState.OBSERVED
        records.append(
            PerceptionTargetFrameState(
                job_id=result.job.job_id,
                frame_identity=result.job.frame_identity,
                target=target,
                state=state,
                candidate_metrics=metrics,
            )
        )
    return records[0], records[1]


def _validate_source_masks(
    result: PerceptionFrameResult,
    source_sized_masks: NDArray[np.generic] | None,
) -> UInt8Array:
    if source_sized_masks is None:
        raise ValueError("completed perception result requires retained source-sized masks")
    masks = np.asarray(source_sized_masks)
    identity = result.job.frame_identity
    expected_tail = (identity.image_height, identity.image_width)
    if masks.dtype != np.uint8 or masks.ndim != 3 or masks.shape[1:] != expected_tail:
        raise ValueError("source-sized masks must be uint8 N-by-height-by-width")
    if any(candidate.detection_index >= masks.shape[0] for candidate in result.candidates):
        raise ValueError("perception candidate mask index is outside retained masks")
    return cast(UInt8Array, masks)


def _measure_candidate(
    candidate: PerceptionCandidate,
    masks: UInt8Array,
    identity: FrameIdentity,
) -> CandidateMaskMetrics:
    mask = masks[candidate.detection_index]
    mask_area = int(np.count_nonzero(mask))
    pixel_count = identity.image_width * identity.image_height
    box = candidate.source_detection.box
    touches_border = bool(
        np.any(mask[0, :])
        or np.any(mask[-1, :])
        or np.any(mask[:, 0])
        or np.any(mask[:, -1])
        or box.x_min <= 0.5
        or box.y_min <= 0.5
        or box.x_max >= identity.image_width - 0.5
        or box.y_max >= identity.image_height - 0.5
    )
    return CandidateMaskMetrics(
        candidate=candidate,
        mask_area_pixels=mask_area,
        mask_area_fraction=mask_area / pixel_count,
        touches_frame_border=touches_border,
        visibility=(
            ImagePlaneVisibility.FRAME_EDGE_TRUNCATED
            if touches_border
            else ImagePlaneVisibility.FULLY_IN_FRAME
        ),
    )
