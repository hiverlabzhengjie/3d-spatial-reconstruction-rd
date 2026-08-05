"""Process-level hard timeout, termination, and bounded restart supervision."""

from __future__ import annotations

import math
import subprocess
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Self

from pydantic import field_validator, model_validator

from spatial_reconstruction.contracts import (
    ContractModel,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    Sha256Digest,
)
from spatial_reconstruction.orchestration.contracts import WorkerKind


class ProcessOutcome(StrEnum):
    """Terminal outcome of one supervised process attempt."""

    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class ProcessWorkerSpec(ContractModel):
    """Bounded process command for one immutable worker job."""

    worker_id: str
    worker_kind: WorkerKind
    job_id: Sha256Digest
    command: tuple[str, ...]
    hard_timeout_seconds: PositiveFloat
    termination_grace_seconds: PositiveFloat = 1.0
    maximum_attempts: PositiveInt = 2

    @field_validator("worker_id")
    @classmethod
    def validate_worker_id(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("worker ID must be non-empty and trimmed")
        return value

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not item or "\x00" in item for item in value):
            raise ValueError("worker command must contain valid argv entries")
        return value


class SupervisedAttemptResult(ContractModel):
    """Observable lifecycle and output for one hard-supervised attempt."""

    worker_id: str
    worker_kind: WorkerKind
    job_id: Sha256Digest
    attempt: PositiveInt
    outcome: ProcessOutcome
    processing_started_seconds: NonNegativeFloat
    processing_finished_seconds: NonNegativeFloat
    exit_code: int | None
    hard_timeout_seconds: PositiveFloat
    terminate_sent: bool
    kill_sent: bool
    stdout: str
    stderr: str
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.processing_finished_seconds < self.processing_started_seconds:
            raise ValueError("worker finish cannot precede start")
        if self.outcome is ProcessOutcome.COMPLETED:
            if self.exit_code != 0 or self.terminate_sent or self.kill_sent:
                raise ValueError("completed worker attempt has inconsistent lifecycle")
            if self.error_message is not None:
                raise ValueError("completed worker attempt cannot contain an error")
        elif self.outcome is ProcessOutcome.TIMED_OUT:
            if not self.terminate_sent or not self.error_message:
                raise ValueError("timed-out worker must record hard termination")
        elif self.exit_code in (None, 0) or not self.error_message:
            raise ValueError("failed worker must record a non-zero exit and error")
        return self


class SupervisedWorkerRun(ContractModel):
    """Bounded attempt history for one worker job."""

    worker_id: str
    worker_kind: WorkerKind
    job_id: Sha256Digest
    maximum_attempts: PositiveInt
    attempts: tuple[SupervisedAttemptResult, ...]
    restart_count: NonNegativeInt
    final_outcome: ProcessOutcome
    degraded: bool

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        if not self.attempts or len(self.attempts) > self.maximum_attempts:
            raise ValueError("supervised run requires bounded attempt history")
        if tuple(attempt.attempt for attempt in self.attempts) != tuple(
            range(1, len(self.attempts) + 1)
        ):
            raise ValueError("supervised attempts must be contiguous")
        if any(
            attempt.worker_id != self.worker_id
            or attempt.worker_kind is not self.worker_kind
            or attempt.job_id != self.job_id
            for attempt in self.attempts
        ):
            raise ValueError("supervised attempt identity differs from its run")
        if self.restart_count != len(self.attempts) - 1:
            raise ValueError("restart count differs from attempt history")
        if self.final_outcome is not self.attempts[-1].outcome:
            raise ValueError("final outcome differs from final attempt")
        if self.degraded != (self.final_outcome is not ProcessOutcome.COMPLETED):
            raise ValueError("degraded state differs from final outcome")
        if any(attempt.outcome is ProcessOutcome.COMPLETED for attempt in self.attempts[:-1]):
            raise ValueError("completed worker cannot be restarted")
        return self


def run_supervised_worker(
    spec: ProcessWorkerSpec,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> SupervisedWorkerRun:
    """Run a command with hard process timeout and bounded automatic restart."""

    attempts: list[SupervisedAttemptResult] = []
    for attempt in range(1, spec.maximum_attempts + 1):
        result = _run_attempt(spec, attempt=attempt, clock=clock)
        attempts.append(result)
        if result.outcome is ProcessOutcome.COMPLETED:
            break
    final = attempts[-1]
    return SupervisedWorkerRun(
        worker_id=spec.worker_id,
        worker_kind=spec.worker_kind,
        job_id=spec.job_id,
        maximum_attempts=spec.maximum_attempts,
        attempts=tuple(attempts),
        restart_count=len(attempts) - 1,
        final_outcome=final.outcome,
        degraded=final.outcome is not ProcessOutcome.COMPLETED,
    )


def _run_attempt(
    spec: ProcessWorkerSpec,
    *,
    attempt: int,
    clock: Callable[[], float],
) -> SupervisedAttemptResult:
    started = _clock_value(clock)
    process = subprocess.Popen(  # noqa: S603 - argv is an explicit project-owned boundary
        spec.command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    terminate_sent = False
    kill_sent = False
    try:
        stdout, stderr = process.communicate(timeout=spec.hard_timeout_seconds)
        finished = _clock_value(clock)
        if process.returncode == 0:
            outcome = ProcessOutcome.COMPLETED
            error_message = None
        else:
            outcome = ProcessOutcome.FAILED
            error_message = f"worker process exited with code {process.returncode}"
    except subprocess.TimeoutExpired:
        terminate_sent = True
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=spec.termination_grace_seconds)
        except subprocess.TimeoutExpired:
            kill_sent = True
            process.kill()
            stdout, stderr = process.communicate()
        finished = _clock_value(clock)
        outcome = ProcessOutcome.TIMED_OUT
        error_message = f"worker exceeded hard timeout of {spec.hard_timeout_seconds:.6g} seconds"
    return SupervisedAttemptResult(
        worker_id=spec.worker_id,
        worker_kind=spec.worker_kind,
        job_id=spec.job_id,
        attempt=attempt,
        outcome=outcome,
        processing_started_seconds=started,
        processing_finished_seconds=finished,
        exit_code=process.returncode,
        hard_timeout_seconds=spec.hard_timeout_seconds,
        terminate_sent=terminate_sent,
        kill_sent=kill_sent,
        stdout=stdout,
        stderr=stderr,
        error_message=error_message,
    )


def _clock_value(clock: Callable[[], float]) -> float:
    value = float(clock())
    if not math.isfinite(value) or value < 0:
        raise ValueError("supervisor clock must return finite non-negative seconds")
    return value
