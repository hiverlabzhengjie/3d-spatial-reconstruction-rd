from __future__ import annotations

import pytest
from pydantic import ValidationError

from spatial_reconstruction.finalization import (
    FinalArtifactRole,
    FinalRunArtifact,
    FinalRunStepName,
    MeasuredFinalRunStep,
    Stage07FinalRunExecution,
    Stage07FinalRunManifest,
)
from spatial_reconstruction.orchestration import SourceVideo


def _manifest() -> Stage07FinalRunManifest:
    videos = (
        SourceVideo(
            camera_id="camera_a",
            source_ref="artifacts/s01/action_take_01/camera_a.mp4",
            source_sha256="1" * 64,
            decoded_frame_count=1047,
            duration_seconds=34.922667,
            nominal_frame_rate_fps=30.0,
        ),
        SourceVideo(
            camera_id="camera_b",
            source_ref="artifacts/s01/action_take_01/camera_b.mp4",
            source_sha256="2" * 64,
            decoded_frame_count=1047,
            duration_seconds=34.922667,
            nominal_frame_rate_fps=30.0,
        ),
    )
    artifacts = tuple(
        FinalRunArtifact(
            role=role,
            source_ref=f"artifacts/s06/{role.value}",
            source_sha256=f"{index + 3:x}" * 64,
        )
        for index, role in enumerate(FinalArtifactRole)
    )
    return Stage07FinalRunManifest.create(
        source_stage06_manifest_id="a" * 64,
        source_videos=videos,
        artifacts=artifacts,
    )


def test_final_run_manifest_is_stable_and_hash_bound() -> None:
    first = _manifest()
    second = _manifest()

    assert first == second
    assert first.manifest_id == second.manifest_id
    assert first.recording.recording_name == "action_take_01"
    assert first.recording.recapture_required is False
    assert first.recording.recalibration_required is False
    assert first.policy.model_inference_required_for_final_assembly is False
    assert first.policy.qwen_has_spatial_authority is False
    assert first.policy.demonstrated_live_capacity is False

    tampered = first.model_dump(mode="json")
    tampered["artifacts"][0]["source_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="manifest ID"):
        Stage07FinalRunManifest.model_validate(tampered)


def test_final_run_manifest_requires_every_unique_artifact_role() -> None:
    manifest = _manifest()
    missing = manifest.model_dump(mode="json")
    missing["artifacts"] = missing["artifacts"][:-1]
    with pytest.raises(ValidationError, match="every required artifact role"):
        Stage07FinalRunManifest.model_validate(missing)

    duplicated = manifest.model_dump(mode="json")
    duplicated["artifacts"][-1] = duplicated["artifacts"][0]
    with pytest.raises(ValidationError, match="roles must be unique"):
        Stage07FinalRunManifest.model_validate(duplicated)


def test_final_run_rejects_wrong_recording_or_frame_count() -> None:
    manifest = _manifest()
    wrong_recording = manifest.model_dump(mode="json")
    wrong_recording["source_videos"][0]["source_ref"] = "artifacts/action_take_02/a.mp4"
    with pytest.raises(ValidationError, match="action_take_01"):
        Stage07FinalRunManifest.model_validate(wrong_recording)

    wrong_count = manifest.model_dump(mode="json")
    wrong_count["source_videos"][1]["decoded_frame_count"] = 1046
    with pytest.raises(ValidationError, match="1,047"):
        Stage07FinalRunManifest.model_validate(wrong_count)


def _step(name: FinalRunStepName, wall_seconds: float) -> MeasuredFinalRunStep:
    return MeasuredFinalRunStep(
        name=name,
        command=(".venv/bin/python", f"scripts/{name.value}.py"),
        wall_seconds=wall_seconds,
        stdout_ref=f"artifacts/{name.value}_stdout.log",
        stdout_sha256="3" * 64,
        stderr_ref=f"artifacts/{name.value}_stderr.log",
        stderr_sha256="4" * 64,
    )


def test_measured_final_run_preserves_real_measurement_boundaries() -> None:
    steps = (
        _step(FinalRunStepName.ENTRY_VERIFICATION, 0.5),
        _step(FinalRunStepName.RERUN_EXPORT, 3.0),
    )
    execution = Stage07FinalRunExecution(
        source_final_run_manifest_id="a" * 64,
        steps=steps,
        total_wall_seconds=3.5,
        capture_duration_seconds=35.0,
        assembly_realtime_factor=0.1,
        capture_seconds_per_assembly_second=10.0,
        recording_ref="artifacts/final.rrd",
        recording_sha256="5" * 64,
        recording_bytes=1024,
        export_summary_ref="artifacts/final_export_summary.json",
        export_summary_sha256="6" * 64,
    )

    assert execution.model_inference_performed is False
    assert execution.evidence_kind == "measured_retained_output_assembly"
    assert execution.demonstrated_live_capacity is False


def test_measured_final_run_rejects_reordering_and_false_throughput() -> None:
    steps = (
        _step(FinalRunStepName.RERUN_EXPORT, 3.0),
        _step(FinalRunStepName.ENTRY_VERIFICATION, 0.5),
    )
    with pytest.raises(ValidationError, match="required execution order"):
        Stage07FinalRunExecution(
            source_final_run_manifest_id="a" * 64,
            steps=steps,
            total_wall_seconds=3.5,
            capture_duration_seconds=35.0,
            assembly_realtime_factor=0.1,
            capture_seconds_per_assembly_second=10.0,
            recording_ref="artifacts/final.rrd",
            recording_sha256="5" * 64,
            recording_bytes=1024,
            export_summary_ref="artifacts/final_export_summary.json",
            export_summary_sha256="6" * 64,
        )

    ordered = tuple(reversed(steps))
    with pytest.raises(ValidationError, match="throughput differs"):
        Stage07FinalRunExecution(
            source_final_run_manifest_id="a" * 64,
            steps=ordered,
            total_wall_seconds=3.5,
            capture_duration_seconds=35.0,
            assembly_realtime_factor=0.1,
            capture_seconds_per_assembly_second=9.0,
            recording_ref="artifacts/final.rrd",
            recording_sha256="5" * 64,
            recording_bytes=1024,
            export_summary_ref="artifacts/final_export_summary.json",
            export_summary_sha256="6" * 64,
        )
