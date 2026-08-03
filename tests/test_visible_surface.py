from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from spatial_reconstruction.contracts import CameraIntrinsics, CameraPose, PerceptionTarget
from spatial_reconstruction.localization import (
    ExactFrameDepthJoin,
    MaskDepthDiagnosticConfig,
    MaskDepthStrategy,
    TargetVisibleSurfaceRule,
    VisibleSurfaceAvailability,
    VisibleSurfaceUnavailableReason,
    localize_visible_surface,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def make_join() -> ExactFrameDepthJoin:
    return ExactFrameDepthJoin(
        action_depth_job_id_from_mask=HASH_A,
        action_depth_job_id_from_depth=HASH_A,
        bundle_id_from_mask=HASH_B,
        bundle_id_from_depth=HASH_B,
        frame_id_from_mask=HASH_C,
        frame_id_from_depth=HASH_C,
        camera_id_from_mask="camera_a",
        camera_id_from_depth="camera_a",
        capture_timestamp_seconds_from_mask=2.0,
        capture_timestamp_seconds_from_depth=2.0,
        timestamp_difference_seconds=0.0,
    )


def make_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        camera_id="camera_a",
        fx=20.0,
        fy=20.0,
        cx=15.0,
        cy=15.0,
        image_width=30,
        image_height=30,
    )


def make_pose() -> CameraPose:
    world_from_camera = (
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 2.0),
        (0.0, 0.0, 1.0, 0.5),
        (0.0, 0.0, 0.0, 1.0),
    )
    camera_from_world = (
        (1.0, 0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0, -2.0),
        (0.0, 0.0, 1.0, -0.5),
        (0.0, 0.0, 0.0, 1.0),
    )
    return CameraPose(
        camera_id="camera_a",
        T_world_from_camera=world_from_camera,
        T_camera_from_world=camera_from_world,
    )


def make_person_rule(*, minimum: int = 10) -> TargetVisibleSurfaceRule:
    return TargetVisibleSurfaceRule(
        target=PerceptionTarget.PERSON,
        candidate_strategy=MaskDepthStrategy.PERSON_LOWER_BODY,
        confidence_threshold_basis="candidate_valid_sample_percentile",
        confidence_percentile=20,
        minimum_retained_sample_count=minimum,
        depth_aggregate="median_ray_depth",
        insufficient_data_state="unavailable",
        coordinate_semantics="Visible lower-body surface.",
    )


def make_arrays() -> tuple[
    NDArray[np.uint8], NDArray[np.float64], NDArray[np.float64]
]:
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:25, 5:25] = 1
    depth = np.full((30, 30), 4.0, dtype=np.float64)
    depth[mask == 1] = 2.0
    confidence = np.linspace(1.0, 10.0, 900, dtype=np.float64).reshape(30, 30)
    return mask, depth, confidence


def test_visible_surface_backprojects_and_round_trips_in_explicit_frames() -> None:
    mask, depth, confidence = make_arrays()
    result = localize_visible_surface(
        source_mask=mask,
        depth_m=depth,
        confidence=confidence,
        target=PerceptionTarget.PERSON,
        config=MaskDepthDiagnosticConfig(
            erosion_radius_pixels=1,
            person_lower_body_fraction=0.4,
        ),
        rule=make_person_rule(),
        intrinsics=make_intrinsics(),
        pose=make_pose(),
        join=make_join(),
    )

    assert result.availability is VisibleSurfaceAvailability.OBSERVED
    assert result.aggregate_camera_xyz_m is not None
    assert result.aggregate_world_xyz_m is not None
    assert result.aggregate_camera_xyz_m[2] == pytest.approx(2.0)
    np.testing.assert_allclose(
        result.aggregate_world_xyz_m,
        np.asarray(result.aggregate_camera_xyz_m) + np.array([1.0, 2.0, 0.5]),
        atol=1e-12,
    )
    assert result.sample_reprojection_max_error_px == pytest.approx(0.0, abs=1e-12)
    assert result.world_camera_round_trip_max_error_m == pytest.approx(0.0, abs=1e-12)
    assert result.retained_sample_count >= 10
    assert not result.points_world_m.flags.writeable


@pytest.mark.parametrize("failure", ["invalid", "undersampled"])
def test_visible_surface_unavailable_never_carries_placeholder_xyz(failure: str) -> None:
    mask, depth, confidence = make_arrays()
    rule = make_person_rule(minimum=1000 if failure == "undersampled" else 10)
    if failure == "invalid":
        depth[mask == 1] = np.nan
    result = localize_visible_surface(
        source_mask=mask,
        depth_m=depth,
        confidence=confidence,
        target=PerceptionTarget.PERSON,
        config=MaskDepthDiagnosticConfig(
            erosion_radius_pixels=1,
            person_lower_body_fraction=0.4,
        ),
        rule=rule,
        intrinsics=make_intrinsics(),
        pose=make_pose(),
        join=make_join(),
    )

    assert result.availability is VisibleSurfaceAvailability.UNAVAILABLE
    assert result.unavailable_reason is (
        VisibleSurfaceUnavailableReason.INVALID_OR_INSUFFICIENT_SAMPLES
    )
    assert result.retained_sample_count == 0
    assert result.aggregate_camera_xyz_m is None
    assert result.aggregate_world_xyz_m is None
    assert result.points_world_m.shape == (0, 3)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("frame_id_from_depth", "d" * 64, "frame identity"),
        ("camera_id_from_depth", "camera_b", "camera identity"),
        ("capture_timestamp_seconds_from_depth", 2.1, "identical timestamps"),
        ("worker_completion_order_used", True, "Input should be False"),
    ],
)
def test_exact_join_rejects_mismatch_staleness_and_completion_order(
    field: str, value: object, message: str
) -> None:
    payload = make_join().model_dump(mode="json")
    payload[field] = value
    if field == "capture_timestamp_seconds_from_depth":
        payload["timestamp_difference_seconds"] = 0.1
    with pytest.raises(ValidationError, match=message):
        ExactFrameDepthJoin.model_validate(payload)


def test_visible_surface_rejects_camera_contract_mismatch() -> None:
    mask, depth, confidence = make_arrays()
    intrinsics_payload = make_intrinsics().model_dump(mode="json")
    intrinsics_payload["camera_id"] = "camera_b"
    with pytest.raises(ValueError, match="camera IDs must match"):
        localize_visible_surface(
            source_mask=mask,
            depth_m=depth,
            confidence=confidence,
            target=PerceptionTarget.PERSON,
            config=MaskDepthDiagnosticConfig(
                erosion_radius_pixels=1,
                person_lower_body_fraction=0.4,
            ),
            rule=make_person_rule(),
            intrinsics=CameraIntrinsics.model_validate(intrinsics_payload),
            pose=make_pose(),
            join=make_join(),
        )


def test_foreground_action_depth_cannot_be_relabelled_as_static_frame() -> None:
    payload = deepcopy(make_join().model_dump(mode="json"))
    payload["frame_id_from_depth"] = "e" * 64
    with pytest.raises(ValidationError, match="frame identity"):
        ExactFrameDepthJoin.model_validate(payload)
