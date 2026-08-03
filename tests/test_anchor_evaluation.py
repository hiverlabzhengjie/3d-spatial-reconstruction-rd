from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from spatial_reconstruction.contracts import CameraIntrinsics, CameraPose, PerceptionTarget
from spatial_reconstruction.localization import (
    AnchorAvailability,
    AnchorCandidateMethod,
    AnchorCandidateRecord,
    AnchorEvaluationConfig,
    AnchorUnavailableReason,
    CrossCameraAnchorComparison,
    CrossCameraAnchorState,
    SelectedAnchorStateRecord,
    evaluate_anchor_candidates,
    intersect_pixels_with_world_floor,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64


def make_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        camera_id="camera_a",
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        image_width=100,
        image_height=100,
    )


def make_downward_pose() -> CameraPose:
    return CameraPose(
        camera_id="camera_a",
        T_world_from_camera=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0, 2.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        T_camera_from_world=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0, 2.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )


def make_person_arrays() -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    pixels = np.column_stack((np.linspace(40, 60, 100), np.linspace(55, 95, 100)))
    points = np.column_stack(
        (
            np.linspace(0.8, 1.2, 100),
            np.linspace(1.8, 2.2, 100),
            np.linspace(0.10, 0.80, 100),
        )
    )
    confidence = np.linspace(1.0, 2.0, 100)
    return pixels, points, confidence


def test_floor_intersection_uses_declared_world_plane() -> None:
    points, valid = intersect_pixels_with_world_floor(
        [[50.0, 50.0], [60.0, 50.0]],
        intrinsics=make_intrinsics(),
        pose=make_downward_pose(),
        world_floor_z_m=0.0,
    )

    assert valid.tolist() == [True, True]
    np.testing.assert_allclose(points, [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])


def test_floor_intersection_rejects_rays_pointing_away() -> None:
    identity_pose = CameraPose(
        camera_id="camera_a",
        T_world_from_camera=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 2.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        T_camera_from_world=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, -2.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    points, valid = intersect_pixels_with_world_floor(
        [[50.0, 50.0]],
        intrinsics=make_intrinsics(),
        pose=identity_pose,
        world_floor_z_m=0.0,
    )

    assert valid.tolist() == [False]
    assert np.isnan(points).all()


def test_person_candidates_separate_measured_lower_body_and_ground_contact() -> None:
    pixels, points, confidence = make_person_arrays()
    candidates = evaluate_anchor_candidates(
        target=PerceptionTarget.PERSON,
        pixels_uv=pixels,
        points_world_m=points,
        confidence=confidence,
        intrinsics=make_intrinsics(),
        pose=make_downward_pose(),
        raw_visible_surface_world_xyz_m=[1.0, 2.0, 0.5],
        config=AnchorEvaluationConfig(minimum_candidate_support_count=10),
    )
    by_method = {candidate.method: candidate for candidate in candidates}
    lower = by_method[AnchorCandidateMethod.PERSON_LOWEST_WORLD_Z_QUINTILE]
    ground = by_method[AnchorCandidateMethod.PERSON_VALIDATED_GROUND_CONTACT]

    assert lower.availability is AnchorAvailability.OBSERVED
    assert lower.world_xyz_m is not None and lower.world_xyz_m[2] > 0
    assert ground.availability is AnchorAvailability.OBSERVED
    assert ground.world_xyz_m is not None and ground.world_xyz_m[2] == 0
    assert ground.measured_support_world_z_m == pytest.approx(lower.world_xyz_m[2])


def test_hidden_feet_make_ground_contact_unavailable_without_placeholder() -> None:
    pixels, points, confidence = make_person_arrays()
    points[:, 2] += 1.0
    candidates = evaluate_anchor_candidates(
        target=PerceptionTarget.PERSON,
        pixels_uv=pixels,
        points_world_m=points,
        confidence=confidence,
        intrinsics=make_intrinsics(),
        pose=make_downward_pose(),
        raw_visible_surface_world_xyz_m=[1.0, 2.0, 1.5],
        config=AnchorEvaluationConfig(minimum_candidate_support_count=10),
    )
    ground = next(
        item
        for item in candidates
        if item.method is AnchorCandidateMethod.PERSON_VALIDATED_GROUND_CONTACT
    )

    assert ground.availability is AnchorAvailability.UNAVAILABLE
    assert ground.unavailable_reason is (
        AnchorUnavailableReason.INSUFFICIENT_FLOOR_PROXIMITY
    )
    assert ground.world_xyz_m is None


def test_backpack_world_median_resists_one_extreme_surface_outlier() -> None:
    pixels = np.column_stack((np.arange(101), np.arange(101))).astype(float)
    points = np.ones((101, 3), dtype=float)
    points[-1] = [100.0, 100.0, 100.0]
    candidates = evaluate_anchor_candidates(
        target=PerceptionTarget.BACKPACK,
        pixels_uv=pixels,
        points_world_m=points,
        confidence=np.ones(101),
        intrinsics=make_intrinsics(),
        pose=make_downward_pose(),
        raw_visible_surface_world_xyz_m=[1.0, 1.0, 1.0],
        config=AnchorEvaluationConfig(minimum_candidate_support_count=10),
    )
    median = next(
        item
        for item in candidates
        if item.method is AnchorCandidateMethod.BACKPACK_WORLD_COMPONENT_MEDIAN
    )

    assert median.world_xyz_m == (1.0, 1.0, 1.0)


def test_anchor_candidate_record_validates_stable_identity_and_schema() -> None:
    method = AnchorCandidateMethod.BACKPACK_WORLD_COMPONENT_MEDIAN
    candidate_id = AnchorCandidateRecord.create_candidate_id(
        source_observation_id=HASH_A,
        method=method,
    )
    record = AnchorCandidateRecord(
        candidate_id=candidate_id,
        source_observation_id=HASH_A,
        action_depth_job_id=HASH_B,
        bundle_id=HASH_C,
        frame_id=HASH_D,
        source_frame_index=1,
        capture_timestamp_seconds=1.0,
        phase_id="phase",
        camera_id="camera_a",
        target=PerceptionTarget.BACKPACK,
        method=method,
        availability=AnchorAvailability.OBSERVED,
        unavailable_reason=None,
        source_sample_count=100,
        support_sample_count=100,
        support_fraction=1.0,
        world_xyz_m=(1.0, 2.0, 0.5),
        measured_support_world_z_m=0.5,
        inside_room_bounds=True,
        coordinate_semantics="Visible cluster centre.",
        selected_for_tracking=True,
        selected_for_ground_contact=False,
        source_sample_cloud_ref="artifacts/sample.npz",
        source_sample_cloud_sha256=HASH_E,
        source_raw_aggregate_world_xyz_m=(1.0, 2.0, 0.5),
    )

    assert AnchorCandidateRecord.model_validate_json(record.model_dump_json()) == record
    payload = record.model_dump(mode="json")
    payload["candidate_id"] = "f" * 64
    with pytest.raises(ValidationError, match="candidate ID"):
        AnchorCandidateRecord.model_validate(payload)


def test_missing_selected_anchor_state_has_no_xyz() -> None:
    state = SelectedAnchorStateRecord(
        action_depth_job_id=HASH_A,
        bundle_id=HASH_B,
        frame_id=HASH_C,
        source_frame_index=3,
        capture_timestamp_seconds=1.0,
        phase_id="missing",
        camera_id="camera_a",
        target=PerceptionTarget.PERSON,
        selected_method=AnchorCandidateMethod.PERSON_LOWEST_WORLD_Z_QUINTILE,
        availability=AnchorAvailability.UNAVAILABLE,
        unavailable_reason=AnchorUnavailableReason.SOURCE_OBSERVATION_UNAVAILABLE,
        source_observation_id=None,
        source_candidate_id=None,
        anchor_world_xyz_m=None,
        coordinate_semantics="No exact current-frame source observation.",
    )

    assert state.anchor_world_xyz_m is None


@pytest.mark.parametrize(
    ("availability_a", "availability_b", "distance", "state", "eligible"),
    [
        (
            AnchorAvailability.OBSERVED,
            AnchorAvailability.OBSERVED,
            0.2,
            CrossCameraAnchorState.PAIRED_ELIGIBLE,
            True,
        ),
        (
            AnchorAvailability.OBSERVED,
            AnchorAvailability.OBSERVED,
            0.6,
            CrossCameraAnchorState.PAIRED_DISAGREEMENT,
            False,
        ),
        (
            AnchorAvailability.OBSERVED,
            AnchorAvailability.UNAVAILABLE,
            None,
            CrossCameraAnchorState.SINGLE_CAMERA,
            False,
        ),
        (
            AnchorAvailability.UNAVAILABLE,
            AnchorAvailability.UNAVAILABLE,
            None,
            CrossCameraAnchorState.UNAVAILABLE,
            False,
        ),
    ],
)
def test_cross_camera_comparison_handles_eligible_disagreement_and_missing(
    availability_a: AnchorAvailability,
    availability_b: AnchorAvailability,
    distance: float | None,
    state: CrossCameraAnchorState,
    eligible: bool,
) -> None:
    comparison = CrossCameraAnchorComparison(
        action_depth_job_id=HASH_A,
        bundle_id=HASH_B,
        source_frame_index=1,
        phase_id="phase",
        target=PerceptionTarget.PERSON,
        selected_method=AnchorCandidateMethod.PERSON_LOWEST_WORLD_Z_QUINTILE,
        camera_a_availability=availability_a,
        camera_b_availability=availability_b,
        state=state,
        disagreement_distance_m=distance,
        maximum_eligible_disagreement_m=0.35,
        eligible_for_fusion=eligible,
    )

    assert comparison.camera_fusion_performed is False
