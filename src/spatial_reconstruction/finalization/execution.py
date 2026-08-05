"""Measured retained-output assembly contracts for S07."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal, Self

from pydantic import field_validator, model_validator

from spatial_reconstruction.contracts import (
    ContractModel,
    PositiveFloat,
    Sha256Digest,
)


class FinalRunStepName(StrEnum):
    """Measured steps in the reproducible S07 retained-output assembly."""

    ENTRY_VERIFICATION = "entry_verification"
    RERUN_EXPORT = "rerun_export"


class MeasuredFinalRunStep(ContractModel):
    """One successful wall-clock-measured final assembly step."""

    name: FinalRunStepName
    command: tuple[str, ...]
    wall_seconds: PositiveFloat
    return_code: Literal[0] = 0
    stdout_ref: str
    stdout_sha256: Sha256Digest
    stderr_ref: str
    stderr_sha256: Sha256Digest

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not part or part.strip() != part for part in value):
            raise ValueError("measured command arguments must be non-empty and trimmed")
        return value

    @field_validator("stdout_ref", "stderr_ref")
    @classmethod
    def validate_ref(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("measured log reference must be non-empty and trimmed")
        return value


class Stage07FinalRunExecution(ContractModel):
    """Measured S07 assembly of retained evidence into the final Rerun file."""

    schema_version: Literal[1] = 1
    stage: Literal["S07"] = "S07"
    work_package: Literal[2] = 2
    status: Literal["completed"] = "completed"
    source_final_run_manifest_id: Sha256Digest
    steps: tuple[MeasuredFinalRunStep, MeasuredFinalRunStep]
    total_wall_seconds: PositiveFloat
    capture_duration_seconds: PositiveFloat
    assembly_realtime_factor: PositiveFloat
    capture_seconds_per_assembly_second: PositiveFloat
    recording_ref: str
    recording_sha256: Sha256Digest
    recording_bytes: int
    export_summary_ref: str
    export_summary_sha256: Sha256Digest
    model_inference_performed: Literal[False] = False
    evidence_kind: Literal["measured_retained_output_assembly"] = (
        "measured_retained_output_assembly"
    )
    demonstrated_live_capacity: Literal[False] = False

    @model_validator(mode="after")
    def validate_execution(self) -> Self:
        if tuple(step.name for step in self.steps) != (
            FinalRunStepName.ENTRY_VERIFICATION,
            FinalRunStepName.RERUN_EXPORT,
        ):
            raise ValueError("final-run steps must retain the required execution order")
        expected_total = sum(step.wall_seconds for step in self.steps)
        if not math.isclose(self.total_wall_seconds, expected_total, abs_tol=1e-9):
            raise ValueError("total wall time differs from measured step sum")
        expected_factor = self.total_wall_seconds / self.capture_duration_seconds
        if not math.isclose(self.assembly_realtime_factor, expected_factor, rel_tol=1e-9):
            raise ValueError("assembly realtime factor differs from measured time")
        expected_throughput = self.capture_duration_seconds / self.total_wall_seconds
        if not math.isclose(
            self.capture_seconds_per_assembly_second,
            expected_throughput,
            rel_tol=1e-9,
        ):
            raise ValueError("assembly throughput differs from measured time")
        if self.recording_bytes <= 0:
            raise ValueError("final Rerun recording must be non-empty")
        for value in (self.recording_ref, self.export_summary_ref):
            if not value or value.strip() != value:
                raise ValueError("final output references must be non-empty and trimmed")
        return self
