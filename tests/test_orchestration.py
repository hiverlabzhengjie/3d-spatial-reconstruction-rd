from __future__ import annotations

import signal
import sys

import pytest
from pydantic import ValidationError

from spatial_reconstruction.interaction import (
    InteractionEventKind,
    PhaseAuthority,
    SemanticEventCandidate,
)
from spatial_reconstruction.orchestration import (
    ArtifactRole,
    OrchestrationArtifact,
    ProcessOutcome,
    ProcessWorkerSpec,
    RerunEventMarker,
    SourceVideo,
    Stage06EventExportRecord,
    Stage06OrchestrationManifest,
    WorkerKind,
    build_event_markers,
    point_style,
    run_integrated_replay,
    run_supervised_worker,
)

DIGEST = "a" * 64
JOB_ID = "b" * 64


def _manifest() -> Stage06OrchestrationManifest:
    videos = (
        SourceVideo(
            camera_id="camera_a",
            source_ref="artifacts/camera_a.mp4",
            source_sha256="1" * 64,
            decoded_frame_count=1047,
            duration_seconds=34.9,
            nominal_frame_rate_fps=30.0,
        ),
        SourceVideo(
            camera_id="camera_b",
            source_ref="artifacts/camera_b.mp4",
            source_sha256="2" * 64,
            decoded_frame_count=1047,
            duration_seconds=34.9,
            nominal_frame_rate_fps=30.0,
        ),
    )
    artifacts = tuple(
        OrchestrationArtifact(
            role=role,
            source_ref=f"artifacts/{role.value}/summary.json",
            source_sha256=f"{index + 3:x}" * 64,
        )
        for index, role in enumerate(ArtifactRole)
    )
    return Stage06OrchestrationManifest.create(
        capture_session_id="s01_capture_20260729",
        synchronization_manifest_ref="artifacts/s01/action/sync.json",
        synchronization_manifest_sha256=DIGEST,
        source_videos=videos,
        artifacts=artifacts,
    )


def test_offline_orchestration_manifest_is_stable_and_hash_bound() -> None:
    first = _manifest()
    second = _manifest()

    assert first == second
    assert first.manifest_id == second.manifest_id
    assert first.policy.heavy_mps_permit_count == 1
    assert first.policy.worker_completion_order_is_authoritative is False
    assert first.policy.qwen_failure_blocks_geometry is False

    tampered = first.model_dump(mode="json")
    tampered["source_videos"][0]["decoded_frame_count"] = 1048
    with pytest.raises(ValidationError, match="manifest ID"):
        Stage06OrchestrationManifest.model_validate(tampered)


def test_orchestration_manifest_requires_every_unique_input_role() -> None:
    manifest = _manifest()
    invalid = manifest.model_dump(mode="json")
    invalid["artifacts"] = invalid["artifacts"][:-1]

    with pytest.raises(ValidationError, match="every required artifact role"):
        Stage06OrchestrationManifest.model_validate(invalid)


def _spec(
    code: str,
    *,
    maximum_attempts: int = 2,
    hard_timeout_seconds: float = 1.0,
    termination_grace_seconds: float = 0.1,
    worker_kind: WorkerKind = WorkerKind.QWEN,
) -> ProcessWorkerSpec:
    return ProcessWorkerSpec(
        worker_id=f"{worker_kind.value}_worker",
        worker_kind=worker_kind,
        job_id=JOB_ID,
        command=(sys.executable, "-c", code),
        hard_timeout_seconds=hard_timeout_seconds,
        termination_grace_seconds=termination_grace_seconds,
        maximum_attempts=maximum_attempts,
    )


def test_supervisor_records_success_without_restart() -> None:
    run = run_supervised_worker(_spec("print('valid semantic result')"))

    assert run.final_outcome is ProcessOutcome.COMPLETED
    assert run.restart_count == 0
    assert run.degraded is False
    assert run.attempts[0].stdout.strip() == "valid semantic result"


def test_supervisor_restarts_failure_only_to_the_configured_bound() -> None:
    run = run_supervised_worker(_spec("import sys; sys.exit(7)", maximum_attempts=2))

    assert run.final_outcome is ProcessOutcome.FAILED
    assert run.restart_count == 1
    assert run.degraded is True
    assert tuple(attempt.attempt for attempt in run.attempts) == (1, 2)
    assert all(attempt.exit_code == 7 for attempt in run.attempts)


def test_hard_timeout_kills_stuck_qwen_while_geometry_remains_runnable() -> None:
    stuck = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(10)"
    qwen_run = run_supervised_worker(
        _spec(
            stuck,
            maximum_attempts=1,
            hard_timeout_seconds=0.2,
            termination_grace_seconds=0.05,
        )
    )

    assert qwen_run.final_outcome is ProcessOutcome.TIMED_OUT
    assert qwen_run.attempts[0].terminate_sent is True
    assert qwen_run.attempts[0].kill_sent is True
    assert qwen_run.attempts[0].exit_code == -signal.SIGKILL

    geometry_run = run_supervised_worker(
        _spec(
            "print('geometry continued')",
            maximum_attempts=1,
            worker_kind=WorkerKind.GEOMETRY,
        )
    )
    assert geometry_run.final_outcome is ProcessOutcome.COMPLETED
    assert geometry_run.attempts[0].stdout.strip() == "geometry continued"


def test_rerun_styles_keep_measured_stale_occluded_and_anchor_kinds_distinct() -> None:
    footpoint = point_style(target="person", state="measured", anchor_kind="person_footpoint")
    upper_body = point_style(
        target="person", state="measured", anchor_kind="person_upper_body_surface"
    )
    stale = point_style(target="backpack", state="stale", anchor_kind=None)
    occluded = point_style(target="backpack", state="occluded", anchor_kind=None)

    assert footpoint.color != upper_body.color
    assert stale.show_position is True
    assert stale.label_prefix == "stale display-only"
    assert occluded.show_position is False
    assert occluded.radius_m == 0.0


def test_qwen_review_time_cannot_move_the_carry_transition() -> None:
    jobs = []
    results = []
    for index, (kind, transition, review) in enumerate(
        (("pickup", 10, 10), ("carry", 20, 30), ("place", 40, 40))
    ):
        job_id = f"{index + 1:x}" * 64
        jobs.append(
            {
                "job_id": job_id,
                "event_kind": kind,
                "source_frame_index": transition,
                "capture_timestamp_seconds": transition / 30,
                "review_frame_index": review,
                "review_timestamp_seconds": review / 30,
            }
        )
        results.append(
            {
                "job": {"job_id": job_id},
                "interpretation": {
                    "event_label": kind,
                    "summary": f"visible {kind}",
                    "matches_candidate": True,
                },
            }
        )

    markers = build_event_markers(jobs, results)

    assert markers[1].transition_frame_index == 20
    assert markers[1].review_frame_index == 30
    assert markers[1].transition_timestamp_seconds != markers[1].review_timestamp_seconds


def test_integrated_replay_is_capture_ordered_despite_reordered_completions() -> None:
    manifest = _manifest()

    report = run_integrated_replay(
        manifest_id=manifest.manifest_id,
        policy=manifest.policy,
    )

    assert report.completion_order_a != report.completion_order_b
    assert report.completion_order_a != report.capture_output_order
    assert tuple(result.job_id for result in report.results) == (report.capture_output_order)
    assert report.maximum_observed_accelerator_occupancy == 1
    assert all(
        current.finished_seconds <= following.started_seconds
        for current, following in zip(
            report.accelerator_intervals,
            report.accelerator_intervals[1:],
            strict=False,
        )
    )


def test_integrated_replay_exercises_bounded_failure_restart_and_shutdown() -> None:
    manifest = _manifest()
    report = run_integrated_replay(
        manifest_id=manifest.manifest_id,
        policy=manifest.policy,
    )

    diagnostics = {item.queue_id: item for item in report.queue_diagnostics}
    assert diagnostics["perception_camera_a"].capacity == 8
    assert diagnostics["perception_camera_a"].throttled_count == 2
    assert diagnostics["da3"].capacity == 2
    assert diagnostics["da3"].throttled_count == 2
    assert diagnostics["qwen"].coalesced_count == 1
    assert all(item.dropped_count == 0 for item in diagnostics.values())
    assert all(item.peak_depth <= item.capacity for item in diagnostics.values())
    assert report.qwen_retry_count == 1
    assert report.duplicate_results_suppressed == 1
    assert sum(result.attempt_count == 2 for result in report.results) == 1
    assert report.degraded_result_count == 2
    assert report.qwen_failure_blocked_geometry is False
    assert report.shutdown.cancelled_pending_count == 4
    assert report.shutdown.final_queue_depth == 0
    assert report.shutdown.accelerator_permit_released is True


def test_integrated_replay_rejects_capture_reordering_and_permit_overlap() -> None:
    manifest = _manifest()
    report = run_integrated_replay(
        manifest_id=manifest.manifest_id,
        policy=manifest.policy,
    )

    reordered = report.model_dump(mode="json")
    reordered["results"][0], reordered["results"][1] = (
        reordered["results"][1],
        reordered["results"][0],
    )
    with pytest.raises(ValidationError, match="capture output order"):
        type(report).model_validate(reordered)

    overlapped = report.model_dump(mode="json")
    overlapped["accelerator_intervals"][1]["started_seconds"] = 0.0
    with pytest.raises(ValidationError, match="intervals overlap"):
        type(report).model_validate(overlapped)


def test_event_export_preserves_carry_review_and_spatial_authority_boundary() -> None:
    candidate = SemanticEventCandidate(
        candidate_id="0ddf29960353f221c8f0320d2f1f246ae1b8c6736be9c05a5972e658e64dac19",
        policy_id="s05_semantic_interaction_v2",
        event_kind=InteractionEventKind.CARRY,
        source_state_record_id=(
            "f45a7106c27b49ccb713597329215a544ac3e9e6ffb18eac105db41e5c1b004c"
        ),
        source_frame_index=468,
        capture_timestamp_seconds=15.606666666666667,
        clip_start_timestamp_seconds=13.606666666666667,
        clip_end_timestamp_seconds=17.60666666666667,
        phase_authority=PhaseAuthority.SEQUENCE_CONTINUITY,
        spatial_transition_authority=False,
    )
    marker = RerunEventMarker(
        event_kind="carry",
        transition_frame_index=468,
        transition_timestamp_seconds=15.606666666666667,
        review_frame_index=567,
        review_timestamp_seconds=18.9,
        qwen_event_label="carry",
        qwen_summary="person carries the backpack",
        qwen_matches_candidate=True,
    )
    exported = Stage06EventExportRecord.create(
        candidate=candidate,
        marker=marker,
        qwen_job_id=JOB_ID,
        qwen_outcome="completed",
    )

    assert exported.transition_frame_index == 468
    assert exported.review_frame_index == 567
    assert exported.spatial_transition_authority is False
    assert exported.qwen_changed_spatial_facts is False

    tampered = exported.model_dump(mode="json")
    tampered["qwen_changed_spatial_facts"] = True
    with pytest.raises(ValidationError):
        Stage06EventExportRecord.model_validate(tampered)
