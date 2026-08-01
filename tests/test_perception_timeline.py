from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from spatial_reconstruction.contracts import (
    FrameIdentity,
    FrameSourceKind,
    PerceptionCandidate,
    PerceptionTarget,
    PixelBox,
    SegmentationDetection,
    SourceFingerprintKind,
)
from spatial_reconstruction.perception import (
    ImagePlaneVisibility,
    PerceptionFrameResult,
    PerceptionJob,
    PerceptionPresenceState,
    PerceptionResultOutcome,
    build_target_frame_states,
)

MODEL_REVISION = "a" * 64


def make_job() -> PerceptionJob:
    identity = FrameIdentity.create(
        capture_session_id="session_01",
        camera_id="camera_a",
        source_kind=FrameSourceKind.FILE,
        source_frame_index=0,
        source_timestamp_seconds=0.0,
        capture_timestamp_seconds=0.0,
        source_ref="captures/camera_a.mp4",
        source_fingerprint="b" * 64,
        source_fingerprint_kind=SourceFingerprintKind.CONTENT_SHA256,
        synchronization_manifest_ref="captures/sync.json",
        synchronization_manifest_sha256="c" * 64,
        pose_version_id="session_01:pose:v1",
        image_width=4,
        image_height=3,
    )
    return PerceptionJob.create(
        frame_identity=identity,
        model_id="yolov8n-seg.pt",
        model_revision=MODEL_REVISION,
        policy_id="d028_guarded_backpack_handbag_v1",
        created_processing_seconds=1.0,
    )


def make_candidate(
    *,
    index: int,
    target: PerceptionTarget,
    track_id: str | None,
    edge_box: bool = False,
) -> PerceptionCandidate:
    identity = make_job().frame_identity
    class_id = 0 if target is PerceptionTarget.PERSON else 26
    class_name = "person" if target is PerceptionTarget.PERSON else "handbag"
    return PerceptionCandidate(
        detection_index=index,
        target=target,
        source_detection=SegmentationDetection(
            frame=identity.as_frame_ref(),
            class_id=class_id,
            class_name=class_name,
            confidence=0.8,
            box=(
                PixelBox(x_min=0.0, y_min=0.0, x_max=2.0, y_max=2.0)
                if edge_box
                else PixelBox(x_min=1.0, y_min=1.0, x_max=3.0, y_max=2.0)
            ),
            mask_ref=f"artifacts/s03/raw.npz#mask_{index:04d}",
            camera_local_track_id=track_id,
        ),
        policy_id="d028_guarded_backpack_handbag_v1",
    )


def make_completed(
    candidates: tuple[PerceptionCandidate, ...],
) -> PerceptionFrameResult:
    return PerceptionFrameResult(
        job=make_job(),
        outcome=PerceptionResultOutcome.COMPLETED,
        candidates=candidates,
        raw_artifact_refs=("artifacts/s03/raw.npz",),
        processing_started_seconds=2.0,
        processing_finished_seconds=2.1,
    )


def test_timeline_derives_observed_mask_area_and_missing_target() -> None:
    person = make_candidate(
        index=0,
        target=PerceptionTarget.PERSON,
        track_id="camera_a:1",
    )
    masks = np.zeros((1, 3, 4), dtype=np.uint8)
    masks[0, 1, 1:3] = 1

    person_state, backpack_state = build_target_frame_states(
        make_completed((person,)), masks
    )

    assert person_state.state is PerceptionPresenceState.OBSERVED
    assert person_state.candidate_metrics[0].mask_area_pixels == 2
    assert person_state.candidate_metrics[0].mask_area_fraction == pytest.approx(2 / 12)
    assert (
        person_state.candidate_metrics[0].visibility
        is ImagePlaneVisibility.FULLY_IN_FRAME
    )
    assert backpack_state.state is PerceptionPresenceState.MISSING
    assert backpack_state.candidate_metrics == ()


def test_timeline_marks_edge_visibility_and_untracked_candidate() -> None:
    backpack = make_candidate(
        index=0,
        target=PerceptionTarget.BACKPACK,
        track_id=None,
        edge_box=True,
    )
    masks = np.zeros((1, 3, 4), dtype=np.uint8)
    masks[0, 0, 0] = 1

    _, backpack_state = build_target_frame_states(make_completed((backpack,)), masks)

    assert backpack_state.state is PerceptionPresenceState.UNTRACKED
    metric = backpack_state.candidate_metrics[0]
    assert metric.touches_frame_border
    assert metric.visibility is ImagePlaneVisibility.FRAME_EDGE_TRUNCATED


def test_timeline_marks_multiple_candidates_ambiguous() -> None:
    candidates = (
        make_candidate(
            index=0,
            target=PerceptionTarget.BACKPACK,
            track_id="camera_a:2",
        ),
        make_candidate(
            index=1,
            target=PerceptionTarget.BACKPACK,
            track_id=None,
        ),
    )
    masks = np.zeros((2, 3, 4), dtype=np.uint8)

    _, backpack_state = build_target_frame_states(make_completed(candidates), masks)

    assert backpack_state.state is PerceptionPresenceState.AMBIGUOUS
    assert len(backpack_state.candidate_metrics) == 2


def test_timeline_propagates_worker_failure_to_both_targets() -> None:
    result = PerceptionFrameResult(
        job=make_job(),
        outcome=PerceptionResultOutcome.FAILED,
        processing_started_seconds=2.0,
        processing_finished_seconds=2.1,
        error_type="RuntimeError",
        error_message="inference unavailable",
    )

    states = build_target_frame_states(result, None)

    assert {state.target for state in states} == set(PerceptionTarget)
    assert all(state.state is PerceptionPresenceState.FAILED for state in states)
    assert all(state.error_type == "RuntimeError" for state in states)


@pytest.mark.parametrize(
    "masks,match",
    [
        (np.zeros((1, 2, 4), dtype=np.uint8), "N-by-height-by-width"),
        (np.zeros((1, 3, 4), dtype=np.float32), "N-by-height-by-width"),
        (np.zeros((0, 3, 4), dtype=np.uint8), "outside retained masks"),
    ],
)
def test_timeline_rejects_invalid_or_missing_candidate_masks(
    masks: NDArray[np.generic], match: str
) -> None:
    candidate = make_candidate(
        index=0,
        target=PerceptionTarget.PERSON,
        track_id="camera_a:1",
    )

    with pytest.raises(ValueError, match=match):
        build_target_frame_states(make_completed((candidate,)), masks)

    with pytest.raises(ValueError, match="requires retained"):
        build_target_frame_states(make_completed((candidate,)), None)
