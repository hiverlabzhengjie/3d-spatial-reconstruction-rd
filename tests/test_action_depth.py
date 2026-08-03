from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from spatial_reconstruction.contracts import (
    CameraIntrinsics,
    FrameIdentity,
    FrameSourceKind,
    PerceptionCandidate,
    PerceptionTarget,
    PixelBox,
    SegmentationDetection,
    SourceFingerprintKind,
    SynchronizedFrameBundle,
)
from spatial_reconstruction.localization import (
    ActionDepthJob,
    ActionDepthKeyframeSpec,
    ActionDepthRunSummary,
    align_source_mask_to_da3_grid,
    build_da3_upper_bound_resize_transform,
    resize_intrinsics_for_da3_grid,
    select_action_depth_jobs,
    transform_undistorted_rgb_to_da3_grid,
)
from spatial_reconstruction.perception import (
    CandidateMaskMetrics,
    ImagePlaneVisibility,
    PerceptionPresenceState,
    PerceptionTargetFrameState,
)

SOURCE_HASH = "a" * 64
MANIFEST_HASH = "b" * 64
MODEL_REVISION = "c" * 40
MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


def make_frame(camera_id: str, index: int, timestamp: float) -> FrameIdentity:
    return FrameIdentity.create(
        capture_session_id="session",
        camera_id=camera_id,
        source_kind=FrameSourceKind.FILE,
        source_frame_index=index,
        source_timestamp_seconds=timestamp,
        capture_timestamp_seconds=timestamp,
        source_ref=f"{camera_id}.mp4",
        source_fingerprint=SOURCE_HASH,
        source_fingerprint_kind=SourceFingerprintKind.CONTENT_SHA256,
        synchronization_manifest_ref="artifacts/s01/action_sync.json",
        synchronization_manifest_sha256=MANIFEST_HASH,
        pose_version_id="session:action:v1",
        image_width=20,
        image_height=10,
    )


def make_bundle(index: int, timestamp: float) -> SynchronizedFrameBundle:
    return SynchronizedFrameBundle.create(
        bundle_index=index,
        capture_session_id="session",
        capture_timestamp_seconds=timestamp,
        reference_camera_id="camera_a",
        expected_camera_ids=("camera_a", "camera_b"),
        frames=(
            make_frame("camera_a", index, timestamp),
            make_frame("camera_b", index, timestamp + 0.001),
        ),
        pairing_tolerance_seconds=0.01,
        synchronization_manifest_ref="artifacts/s01/action_sync.json",
        synchronization_manifest_sha256=MANIFEST_HASH,
    )


def make_state(
    frame: FrameIdentity,
    target: PerceptionTarget,
    *,
    state: PerceptionPresenceState = PerceptionPresenceState.OBSERVED,
) -> PerceptionTargetFrameState:
    if state is PerceptionPresenceState.MISSING:
        return PerceptionTargetFrameState(
            job_id=("d" if frame.camera_id == "camera_a" else "e") * 64,
            frame_identity=frame,
            target=target,
            state=state,
        )
    class_name = "person" if target is PerceptionTarget.PERSON else "backpack"
    class_id = 0 if target is PerceptionTarget.PERSON else 24
    candidate = PerceptionCandidate(
        detection_index=0,
        target=target,
        source_detection=SegmentationDetection(
            frame=frame.as_frame_ref(),
            class_id=class_id,
            class_name=class_name,
            confidence=0.8,
            box=PixelBox(x_min=1, y_min=1, x_max=5, y_max=5),
            mask_ref=f"raw/{frame.frame_id}.npz#mask_0000",
            camera_local_track_id=f"{frame.camera_id}:1",
        ),
        policy_id="d028_guarded_backpack_handbag_v1",
    )
    return PerceptionTargetFrameState(
        job_id=("d" if frame.camera_id == "camera_a" else "e") * 64,
        frame_identity=frame,
        target=target,
        state=state,
        candidate_metrics=(
            CandidateMaskMetrics(
                candidate=candidate,
                mask_area_pixels=16,
                mask_area_fraction=0.08,
                touches_frame_border=False,
                visibility=ImagePlaneVisibility.FULLY_IN_FRAME,
            ),
        ),
    )


def make_states(bundle: SynchronizedFrameBundle) -> tuple[PerceptionTargetFrameState, ...]:
    return tuple(
        make_state(frame, target)
        for frame in bundle.frames
        for target in PerceptionTarget
    )


def make_summary_payload(job: ActionDepthJob) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "completed_pending_mask_depth_qa",
        "stage": "S04",
        "created_at_utc": "2026-08-01T00:00:00+00:00",
        "capture_session_id": "session",
        "pose_version_id": "session:action:v1",
        "input_provenance": {"test": True},
        "model": {"model_id": MODEL_ID, "is_metric": True},
        "selection_config_ref": "configs/s04_action_keyframes.json",
        "selection_config_sha256": "f" * 64,
        "processing": {
            "process_resolution": 504,
            "source_frames_undistorted_before_inference": True,
            "raw_da3_metric_depth_preserved": True,
            "s02_marker_scale_applied": False,
            "s02_static_confidence_policy_applied": False,
            "s02_door_supplement_applied": False,
            "mask_resampling_or_localization_performed": False,
        },
        "predictions": [
            {
                "job": job.model_dump(mode="json"),
                "raw_prediction_ref": "artifacts/s04/raw.npz",
                "raw_prediction_sha256": "1" * 64,
                "depth_confidence_preview_ref": "artifacts/s04/preview.png",
                "depth_confidence_preview_sha256": "2" * 64,
                "cameras": {"camera_a": {}, "camera_b": {}},
            }
        ],
        "runtime": {"device": "mps"},
        "limitations": ["raw depth only"],
    }


def test_action_depth_selection_is_stable_and_exact_frame_bound() -> None:
    bundle = make_bundle(204, 6.8)
    states = make_states(bundle)
    spec = ActionDepthKeyframeSpec(
        source_frame_index=204,
        phase_id="pickup_side_stationary",
        selection_reason="Observed person and backpack evidence.",
        required_backpack_camera_id="camera_a",
    )

    first = select_action_depth_jobs(
        bundles=(bundle,),
        states=states,
        specs=(spec,),
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        process_resolution=504,
    )
    second = select_action_depth_jobs(
        bundles=(bundle,),
        states=tuple(reversed(states)),
        specs=(spec,),
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        process_resolution=504,
    )

    assert first == second
    assert first[0].job_id == second[0].job_id
    assert first[0].bundle.bundle_id == bundle.bundle_id
    assert {item.target for item in first[0].mask_evidence} == set(PerceptionTarget)


def test_action_depth_selection_rejects_missing_required_backpack() -> None:
    bundle = make_bundle(204, 6.8)
    states = list(make_states(bundle))
    states = [
        make_state(item.frame_identity, item.target, state=PerceptionPresenceState.MISSING)
        if item.frame_identity.camera_id == "camera_a"
        and item.target is PerceptionTarget.BACKPACK
        else item
        for item in states
    ]
    spec = ActionDepthKeyframeSpec(
        source_frame_index=204,
        phase_id="pickup",
        selection_reason="Expected Camera A backpack evidence.",
        required_backpack_camera_id="camera_a",
    )

    with pytest.raises(ValueError, match="required observed backpack"):
        select_action_depth_jobs(
            bundles=(bundle,),
            states=states,
            specs=(spec,),
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            process_resolution=504,
        )


def test_action_depth_job_rejects_tampered_frame_and_non_da3_resolution() -> None:
    bundle = make_bundle(204, 6.8)
    states = make_states(bundle)
    job = ActionDepthJob.create(
        bundle=bundle,
        phase_id="pickup",
        selection_reason="Observed masks.",
        mask_evidence=states,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        process_resolution=504,
    )
    payload = job.model_dump(mode="json")
    payload["mask_evidence"][0]["frame_identity"] = make_frame(
        "camera_a", 205, 7.0
    ).model_dump(mode="json")
    with pytest.raises(ValidationError, match="different source frame"):
        ActionDepthJob.model_validate(payload)

    payload = job.model_dump(mode="json")
    payload["process_resolution"] = 500
    with pytest.raises(ValidationError, match="multiple of 14"):
        ActionDepthJob.model_validate(payload)


def test_action_depth_summary_forbids_s02_policy_leakage() -> None:
    bundle = make_bundle(204, 6.8)
    job = ActionDepthJob.create(
        bundle=bundle,
        phase_id="pickup",
        selection_reason="Observed masks.",
        mask_evidence=make_states(bundle),
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        process_resolution=504,
    )
    payload = make_summary_payload(job)
    summary = ActionDepthRunSummary.model_validate(payload)
    assert ActionDepthRunSummary.model_validate_json(summary.model_dump_json()) == summary

    invalid = deepcopy(payload)
    invalid["processing"]["s02_marker_scale_applied"] = True
    with pytest.raises(ValidationError):
        ActionDepthRunSummary.model_validate(invalid)


def test_da3_grid_transform_matches_real_1920_by_1080_preprocessing() -> None:
    transform = build_da3_upper_bound_resize_transform(
        source_width=1920,
        source_height=1080,
        process_resolution=504,
    )

    assert (transform.boundary_width, transform.boundary_height) == (504, 284)
    assert (transform.processed_width, transform.processed_height) == (504, 280)
    assert transform.batch_center_crop_left == 0
    assert transform.batch_center_crop_top == 0
    assert transform.image_boundary_interpolation == "area"
    assert transform.image_patch_interpolation == "area"

    resized = resize_intrinsics_for_da3_grid(
        CameraIntrinsics(
            camera_id="camera_a",
            fx=865.3748029338908,
            fy=864.2529801932611,
            cx=944.105718862249,
            cy=543.5316875373612,
            image_width=1920,
            image_height=1080,
        ),
        transform,
    )
    assert resized == pytest.approx(
        np.array(
            [
                [227.16088577014634, 0.0, 247.82775120134037],
                [0.0, 224.06558745751215, 140.91562269561217],
                [0.0, 0.0, 1.0],
            ]
        ),
        abs=1e-9,
    )


def test_mask_alignment_preserves_binary_values_and_expected_grid() -> None:
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[2:8, 4:12] = 1
    intrinsics = CameraIntrinsics(
        camera_id="camera_a",
        fx=10,
        fy=10,
        cx=10,
        cy=5,
        image_width=20,
        image_height=10,
        distortion_coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),
    )

    aligned = align_source_mask_to_da3_grid(
        mask,
        intrinsics=intrinsics,
        process_resolution=28,
    )

    assert aligned.undistorted_source_mask.shape == (10, 20)
    assert aligned.processed_mask.shape == (14, 28)
    assert set(np.unique(aligned.processed_mask)) == {0, 1}
    assert np.count_nonzero(aligned.processed_mask) > 0
    assert not aligned.undistorted_source_mask.flags.writeable
    assert not aligned.processed_mask.flags.writeable


def test_rgb_reproduction_uses_same_two_stage_shape_and_rejects_bad_masks() -> None:
    image = np.arange(10 * 20 * 3, dtype=np.uint8).reshape(10, 20, 3)
    processed, transform = transform_undistorted_rgb_to_da3_grid(
        image,
        process_resolution=28,
    )
    assert processed.shape == (14, 28, 3)
    assert (transform.boundary_width, transform.boundary_height) == (28, 14)

    intrinsics = CameraIntrinsics(
        camera_id="camera_a",
        fx=10,
        fy=10,
        cx=10,
        cy=5,
        image_width=20,
        image_height=10,
    )
    with pytest.raises(ValueError, match="foreground"):
        align_source_mask_to_da3_grid(
            np.zeros((10, 20), dtype=np.uint8),
            intrinsics=intrinsics,
            process_resolution=28,
        )
    with pytest.raises(ValueError, match="uint8 or bool"):
        align_source_mask_to_da3_grid(
            np.ones((10, 20), dtype=np.float32),
            intrinsics=intrinsics,
            process_resolution=28,
        )
