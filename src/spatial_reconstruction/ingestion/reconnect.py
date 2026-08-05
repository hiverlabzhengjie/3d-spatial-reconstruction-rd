"""Bounded RTSP reconnect runtime preserving monotonic frame identity."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Literal, Self

from pydantic import model_validator

from spatial_reconstruction.contracts import (
    ContractModel,
    FrameIdentity,
    FrameSourceKind,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)
from spatial_reconstruction.ingestion.sources import DecodedFrame, FrameSource


class RTSPAttemptOutcome(StrEnum):
    """Observable outcome of one RTSP connection attempt."""

    TARGET_REACHED = "target_reached"
    STREAM_ENDED = "stream_ended"
    FAILED = "failed"


class RTSPReconnectPolicy(ContractModel):
    """Bounded local-prototype reconnect policy."""

    policy_id: Literal["s06_rtsp_bounded_reconnect_v1"] = "s06_rtsp_bounded_reconnect_v1"
    maximum_connection_attempts: PositiveInt = 6
    reconnect_delay_seconds: NonNegativeFloat = 0.25
    minimum_timestamp_step_seconds: PositiveFloat = 1.0 / 30.0


class RTSPConnectionAttempt(ContractModel):
    """One connection lifecycle with its global frame-index range."""

    attempt: PositiveInt
    outcome: RTSPAttemptOutcome
    processing_started_seconds: NonNegativeFloat
    processing_finished_seconds: NonNegativeFloat
    decoded_frame_count: NonNegativeInt
    first_global_frame_index: NonNegativeInt | None = None
    last_global_frame_index: NonNegativeInt | None = None
    source_timestamp_offset_seconds: NonNegativeFloat | None = None
    capture_timestamp_offset_seconds: NonNegativeFloat | None = None
    observed_reconnect_gap_seconds: NonNegativeFloat | None = None
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_attempt(self) -> Self:
        if self.processing_finished_seconds < self.processing_started_seconds:
            raise ValueError("RTSP attempt finish cannot precede start")
        if self.decoded_frame_count:
            if self.first_global_frame_index is None or self.last_global_frame_index is None:
                raise ValueError("decoded RTSP attempt requires a frame-index range")
            if (
                self.last_global_frame_index - self.first_global_frame_index + 1
                != self.decoded_frame_count
            ):
                raise ValueError("RTSP attempt frame range differs from decoded count")
            if (
                self.source_timestamp_offset_seconds is None
                or self.capture_timestamp_offset_seconds is None
                or self.observed_reconnect_gap_seconds is None
            ):
                raise ValueError("decoded RTSP attempt requires timestamp-rebase evidence")
        elif self.first_global_frame_index is not None or self.last_global_frame_index is not None:
            raise ValueError("empty RTSP attempt cannot claim a frame-index range")
        elif (
            self.source_timestamp_offset_seconds is not None
            or self.capture_timestamp_offset_seconds is not None
            or self.observed_reconnect_gap_seconds is not None
        ):
            raise ValueError("empty RTSP attempt cannot claim timestamp-rebase evidence")
        if self.outcome is RTSPAttemptOutcome.FAILED:
            if not self.error_type or not self.error_message:
                raise ValueError("failed RTSP attempt requires explicit error details")
        elif self.error_type is not None or self.error_message is not None:
            raise ValueError("non-failed RTSP attempt cannot contain an error")
        return self


class RTSPReconnectDiagnostics(ContractModel):
    """Bounded reconnect result and complete attempt history."""

    policy: RTSPReconnectPolicy
    target_frame_count: PositiveInt
    total_decoded_frame_count: NonNegativeInt
    reconnect_count: NonNegativeInt
    attempts: tuple[RTSPConnectionAttempt, ...]
    target_reached: bool
    exhausted: bool
    final_outcome: RTSPAttemptOutcome

    @model_validator(mode="after")
    def validate_diagnostics(self) -> Self:
        if not self.attempts or len(self.attempts) > self.policy.maximum_connection_attempts:
            raise ValueError("RTSP reconnect history must be non-empty and bounded")
        if tuple(attempt.attempt for attempt in self.attempts) != tuple(
            range(1, len(self.attempts) + 1)
        ):
            raise ValueError("RTSP attempt numbers must be contiguous")
        if self.total_decoded_frame_count != sum(
            attempt.decoded_frame_count for attempt in self.attempts
        ):
            raise ValueError("RTSP total frame count differs from attempt history")
        if self.reconnect_count != len(self.attempts) - 1:
            raise ValueError("RTSP reconnect count differs from attempt history")
        if self.final_outcome is not self.attempts[-1].outcome:
            raise ValueError("RTSP final outcome differs from final attempt")
        expected_target = self.total_decoded_frame_count == self.target_frame_count
        if self.target_reached != expected_target:
            raise ValueError("RTSP target state differs from decoded frame count")
        expected_exhausted = not expected_target and (
            len(self.attempts) == self.policy.maximum_connection_attempts
        )
        if self.exhausted != expected_exhausted:
            raise ValueError("RTSP exhausted state differs from bounded attempts")
        return self


class RTSPReconnectRead(ContractModel):
    """Persistable identities plus reconnect diagnostics; pixels stay runtime-only."""

    frame_identities: tuple[FrameIdentity, ...]
    diagnostics: RTSPReconnectDiagnostics

    @model_validator(mode="after")
    def validate_frames(self) -> Self:
        if len(self.frame_identities) != self.diagnostics.total_decoded_frame_count:
            raise ValueError("RTSP identities differ from decoded diagnostics")
        if any(
            identity.source_kind is not FrameSourceKind.RTSP for identity in self.frame_identities
        ):
            raise ValueError("RTSP reconnect output contains a non-RTSP identity")
        indexes = tuple(identity.source_frame_index for identity in self.frame_identities)
        if indexes != tuple(range(len(indexes))):
            raise ValueError("RTSP reconnect frame indexes must be contiguous")
        timestamps = tuple(
            identity.capture_timestamp_seconds for identity in self.frame_identities
        )
        if any(
            current >= following
            for current, following in zip(timestamps, timestamps[1:], strict=False)
        ):
            raise ValueError("RTSP reconnect capture timestamps must be increasing")
        if len(set(identity.frame_id for identity in self.frame_identities)) != len(
            self.frame_identities
        ):
            raise ValueError("RTSP reconnect emitted duplicate frame identities")
        return self


def read_rtsp_with_reconnect(
    source_factory: Callable[[int], FrameSource],
    *,
    target_frame_count: int,
    policy: RTSPReconnectPolicy | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[tuple[DecodedFrame, ...], RTSPReconnectRead]:
    """Read a bounded number of frames across bounded RTSP reconnects."""

    selected_policy = policy or RTSPReconnectPolicy()
    if target_frame_count <= 0:
        raise ValueError("RTSP target frame count must be positive")
    frames: list[DecodedFrame] = []
    attempts: list[RTSPConnectionAttempt] = []
    previous_source_timestamp: float | None = None
    previous_capture_timestamp: float | None = None
    previous_arrival_seconds: float | None = None

    for attempt_number in range(1, selected_policy.maximum_connection_attempts + 1):
        started = _clock_value(clock)
        first_global_index = len(frames)
        error_type: str | None = None
        error_message: str | None = None
        outcome = RTSPAttemptOutcome.STREAM_ENDED
        source_offset: float | None = None
        capture_offset: float | None = None
        observed_reconnect_gap: float | None = None
        try:
            source = source_factory(attempt_number)
            if source.source_kind is not FrameSourceKind.RTSP:
                raise ValueError("reconnect source factory must return an RTSP source")
            for raw_frame in source.iter_frames():
                raw_identity = raw_frame.identity
                arrival_seconds = _clock_value(clock)
                if source_offset is None:
                    source_offset = _timestamp_offset(
                        previous_source_timestamp,
                        raw_identity.source_timestamp_seconds,
                        selected_policy.minimum_timestamp_step_seconds,
                    )
                    observed_reconnect_gap = (
                        0.0
                        if previous_arrival_seconds is None
                        else max(
                            selected_policy.minimum_timestamp_step_seconds,
                            arrival_seconds - previous_arrival_seconds,
                        )
                    )
                    capture_offset = _timestamp_offset_with_gap(
                        previous_capture_timestamp,
                        raw_identity.capture_timestamp_seconds,
                        observed_reconnect_gap,
                    )
                assert capture_offset is not None
                identity = FrameIdentity.create(
                    capture_session_id=raw_identity.capture_session_id,
                    camera_id=raw_identity.camera_id,
                    source_kind=FrameSourceKind.RTSP,
                    source_frame_index=len(frames),
                    source_timestamp_seconds=(
                        raw_identity.source_timestamp_seconds + source_offset
                    ),
                    capture_timestamp_seconds=(
                        raw_identity.capture_timestamp_seconds + capture_offset
                    ),
                    source_ref=raw_identity.source_ref,
                    source_fingerprint=raw_identity.source_fingerprint,
                    source_fingerprint_kind=raw_identity.source_fingerprint_kind,
                    synchronization_manifest_ref=(raw_identity.synchronization_manifest_ref),
                    synchronization_manifest_sha256=(raw_identity.synchronization_manifest_sha256),
                    pose_version_id=raw_identity.pose_version_id,
                    image_width=raw_identity.image_width,
                    image_height=raw_identity.image_height,
                )
                if (
                    previous_source_timestamp is not None
                    and identity.source_timestamp_seconds <= previous_source_timestamp
                ):
                    raise RuntimeError("RTSP source timestamp regressed within a connection")
                if (
                    previous_capture_timestamp is not None
                    and identity.capture_timestamp_seconds <= previous_capture_timestamp
                ):
                    raise RuntimeError("RTSP capture timestamp regressed within a connection")
                frames.append(DecodedFrame(identity=identity, image_bgr=raw_frame.image_bgr))
                previous_source_timestamp = identity.source_timestamp_seconds
                previous_capture_timestamp = identity.capture_timestamp_seconds
                previous_arrival_seconds = arrival_seconds
                if len(frames) == target_frame_count:
                    outcome = RTSPAttemptOutcome.TARGET_REACHED
                    break
        except Exception as exc:
            outcome = RTSPAttemptOutcome.FAILED
            error_type = type(exc).__name__
            error_message = str(exc) or repr(exc)
        finished = _clock_value(clock)
        decoded_count = len(frames) - first_global_index
        attempts.append(
            RTSPConnectionAttempt(
                attempt=attempt_number,
                outcome=outcome,
                processing_started_seconds=started,
                processing_finished_seconds=finished,
                decoded_frame_count=decoded_count,
                first_global_frame_index=(first_global_index if decoded_count else None),
                last_global_frame_index=(len(frames) - 1 if decoded_count else None),
                source_timestamp_offset_seconds=(source_offset if decoded_count else None),
                capture_timestamp_offset_seconds=(capture_offset if decoded_count else None),
                observed_reconnect_gap_seconds=(observed_reconnect_gap if decoded_count else None),
                error_type=error_type,
                error_message=error_message,
            )
        )
        if outcome is RTSPAttemptOutcome.TARGET_REACHED:
            break
        if attempt_number < selected_policy.maximum_connection_attempts:
            sleeper(selected_policy.reconnect_delay_seconds)

    target_reached = len(frames) == target_frame_count
    diagnostics = RTSPReconnectDiagnostics(
        policy=selected_policy,
        target_frame_count=target_frame_count,
        total_decoded_frame_count=len(frames),
        reconnect_count=len(attempts) - 1,
        attempts=tuple(attempts),
        target_reached=target_reached,
        exhausted=(
            not target_reached and len(attempts) == selected_policy.maximum_connection_attempts
        ),
        final_outcome=attempts[-1].outcome,
    )
    persisted = RTSPReconnectRead(
        frame_identities=tuple(frame.identity for frame in frames),
        diagnostics=diagnostics,
    )
    return tuple(frames), persisted


def _timestamp_offset(
    previous: float | None,
    current: float,
    minimum_step: float,
) -> float:
    if previous is None:
        return 0.0
    return max(0.0, previous + minimum_step - current)


def _timestamp_offset_with_gap(
    previous: float | None,
    current: float,
    observed_gap: float,
) -> float:
    if previous is None:
        return 0.0
    return max(0.0, previous + observed_gap - current)


def _clock_value(clock: Callable[[], float]) -> float:
    value = float(clock())
    if not math.isfinite(value) or value < 0:
        raise ValueError("RTSP reconnect clock must return finite non-negative seconds")
    return value
