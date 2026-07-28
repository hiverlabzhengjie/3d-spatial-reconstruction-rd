from typing import Any

import pytest
from pydantic import ValidationError

from spatial_reconstruction.contracts import (
    CameraIntrinsics,
    CameraPose,
    DepthPrediction,
    FrameRef,
    MemoryObservation,
    ModelRunObservation,
    ObservationState,
    PixelBox,
    RunOutcome,
    SegmentationDetection,
    SpatialObservation,
    TimingObservation,
)

IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def make_frame() -> FrameRef:
    return FrameRef(
        camera_id="camera_01",
        frame_index=12,
        timestamp_seconds=0.4,
        source_ref="captures/session_01/camera_01.mp4",
        image_width=1920,
        image_height=1080,
    )


def test_frame_ref_round_trip_and_immutability() -> None:
    frame = make_frame()

    assert FrameRef.model_validate_json(frame.model_dump_json()) == frame
    with pytest.raises(ValidationError):
        frame.frame_index = 13


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("camera_id", ""),
        ("frame_index", -1),
        ("timestamp_seconds", -0.1),
        ("image_width", 0),
        ("image_height", 0),
    ],
)
def test_frame_ref_rejects_invalid_values(field: str, value: Any) -> None:
    payload = make_frame().model_dump()
    payload[field] = value

    with pytest.raises(ValidationError):
        FrameRef.model_validate(payload)


def test_missing_frame_fails_with_typed_validation_error() -> None:
    with pytest.raises(ValidationError):
        FrameRef.model_validate(None)


def test_camera_intrinsics_reject_non_finite_or_non_positive_values() -> None:
    with pytest.raises(ValidationError):
        CameraIntrinsics(
            camera_id="camera_01",
            fx=float("nan"),
            fy=1000.0,
            cx=960.0,
            cy=540.0,
            image_width=1920,
            image_height=1080,
        )

    with pytest.raises(ValidationError):
        CameraIntrinsics(
            camera_id="camera_01",
            fx=0.0,
            fy=1000.0,
            cx=960.0,
            cy=540.0,
            image_width=1920,
            image_height=1080,
        )


def test_camera_pose_accepts_explicit_inverse_transforms() -> None:
    world_from_camera = (
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 2.0),
        (0.0, 0.0, 1.0, 3.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    camera_from_world = (
        (1.0, 0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0, -2.0),
        (0.0, 0.0, 1.0, -3.0),
        (0.0, 0.0, 0.0, 1.0),
    )

    pose = CameraPose(
        camera_id="camera_01",
        T_world_from_camera=world_from_camera,
        T_camera_from_world=camera_from_world,
    )

    restored = CameraPose.model_validate_json(pose.model_dump_json())
    assert restored == pose
    assert "extrinsics" not in pose.model_dump()


def test_camera_pose_rejects_wrong_shape_non_rigid_and_non_inverse() -> None:
    with pytest.raises(ValidationError):
        CameraPose.model_validate(
            {
                "camera_id": "camera_01",
                "T_world_from_camera": IDENTITY[:3],
                "T_camera_from_world": IDENTITY,
            }
        )

    non_rigid = [list(row) for row in IDENTITY]
    non_rigid[0][0] = 2.0
    with pytest.raises(ValidationError, match="orthonormal"):
        CameraPose(
            camera_id="camera_01",
            T_world_from_camera=non_rigid,  # type: ignore[arg-type]
            T_camera_from_world=IDENTITY,
        )

    translated = [list(row) for row in IDENTITY]
    translated[0][3] = 1.0
    with pytest.raises(ValidationError, match="mutual inverses"):
        CameraPose(
            camera_id="camera_01",
            T_world_from_camera=translated,  # type: ignore[arg-type]
            T_camera_from_world=IDENTITY,
        )


def test_ambiguous_extrinsics_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="extrinsics"):
        CameraPose.model_validate(
            {
                "camera_id": "camera_01",
                "T_world_from_camera": IDENTITY,
                "T_camera_from_world": IDENTITY,
                "extrinsics": IDENTITY,
            }
        )


def test_depth_prediction_round_trip() -> None:
    prediction = DepthPrediction(
        frame=make_frame(),
        depth_ref="artifacts/s00/da3/depth.npz",
        confidence_ref="artifacts/s00/da3/confidence.npz",
        raw_output_ref="artifacts/s00/da3/raw.npz",
        model_id="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        process_width=504,
        process_height=336,
    )

    assert DepthPrediction.model_validate_json(prediction.model_dump_json()) == prediction


def test_segmentation_detection_validates_box_bounds() -> None:
    detection = SegmentationDetection(
        frame=make_frame(),
        class_id=0,
        class_name="person",
        confidence=0.9,
        box=PixelBox(x_min=10.0, y_min=20.0, x_max=200.0, y_max=400.0),
        mask_ref="artifacts/s00/yolo/mask.npz",
    )
    assert detection.box.x_min == 10.0

    with pytest.raises(ValidationError, match="within the source frame"):
        SegmentationDetection(
            frame=make_frame(),
            class_id=0,
            class_name="person",
            confidence=0.9,
            box=PixelBox(x_min=10.0, y_min=20.0, x_max=2000.0, y_max=400.0),
            mask_ref="artifacts/s00/yolo/mask.npz",
        )


def test_observed_spatial_observation_requires_raw_xyz_source_and_confidence() -> None:
    observation = SpatialObservation(
        entity_type="backpack",
        entity_id="backpack_01",
        timestamp_seconds=1.0,
        state=ObservationState.OBSERVED,
        raw_world_xyz_m=(1.0, 2.0, 0.4),
        source_camera_ids=("camera_01",),
        confidence=0.8,
        provenance="DA3 depth back-projection",
    )

    assert SpatialObservation.model_validate_json(observation.model_dump_json()) == observation

    payload = observation.model_dump()
    payload["raw_world_xyz_m"] = None
    with pytest.raises(ValidationError, match="requires raw_world_xyz_m"):
        SpatialObservation.model_validate(payload)


@pytest.mark.parametrize("state", [ObservationState.MISSING, ObservationState.OCCLUDED])
def test_unavailable_observation_cannot_fabricate_xyz(state: ObservationState) -> None:
    with pytest.raises(ValidationError, match="must not carry raw_world_xyz_m"):
        SpatialObservation(
            entity_type="backpack",
            entity_id="backpack_01",
            timestamp_seconds=2.0,
            state=state,
            raw_world_xyz_m=(1.0, 2.0, 0.4),
            provenance="detector unavailable",
        )


def test_stale_observation_has_timestamp_but_no_raw_xyz() -> None:
    stale = SpatialObservation(
        entity_type="backpack",
        entity_id="backpack_01",
        timestamp_seconds=3.0,
        state=ObservationState.STALE,
        last_observed_timestamp_seconds=2.5,
        provenance="last observation retained for presentation",
    )
    assert stale.raw_world_xyz_m is None

    with pytest.raises(ValidationError, match="cannot be in the future"):
        SpatialObservation(
            entity_type="backpack",
            entity_id="backpack_01",
            timestamp_seconds=3.0,
            state=ObservationState.STALE,
            last_observed_timestamp_seconds=3.5,
            provenance="invalid stale state",
        )


def test_model_run_observation_round_trip_and_unique_phases() -> None:
    run = ModelRunObservation(
        model_id="yolov8n-seg.pt",
        requested_device="mps",
        device="mps",
        precision="float32",
        input_description="one representative image",
        timings=(TimingObservation(phase="cold_inference", seconds=1.2),),
        memory=(MemoryObservation(phase="after_inference", process_rss_bytes=1024),),
        outcome=RunOutcome.PASSED,
        artifact_refs=("artifacts/s00/yolo/summary.json",),
    )
    assert ModelRunObservation.model_validate_json(run.model_dump_json()) == run

    with pytest.raises(ValidationError, match="timing phases must be unique"):
        ModelRunObservation(
            model_id="yolov8n-seg.pt",
            requested_device="mps",
            device="mps",
            precision="float32",
            input_description="one representative image",
            timings=(
                TimingObservation(phase="warm", seconds=0.8),
                TimingObservation(phase="warm", seconds=0.7),
            ),
            memory=(),
            outcome=RunOutcome.PASSED,
        )
