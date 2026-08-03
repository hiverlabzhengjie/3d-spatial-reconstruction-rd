from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError

from spatial_reconstruction.contracts import PerceptionTarget
from spatial_reconstruction.localization import (
    AnchorAvailability,
    AnchorCandidateMethod,
    AnchorUnavailableReason,
    CrossCameraCombinationMethod,
    CrossCameraObservationRecord,
    CrossCameraObservationState,
    FusionSourceEvidence,
    FusionSourceMeasurement,
    reliability_score,
    resolve_cross_camera_observation,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64
HASH_F = "f" * 64


def observed_source(
    camera_id: str,
    *,
    xyz: tuple[float, float, float],
    support: int = 100,
    confidence: float = 4.0,
    depth_median: float = 2.0,
    depth_mad: float = 0.2,
) -> FusionSourceMeasurement:
    return FusionSourceMeasurement(
        camera_id=camera_id,  # type: ignore[arg-type]
        availability=AnchorAvailability.OBSERVED,
        unavailable_reason=None,
        source_observation_id=HASH_A if camera_id == "camera_a" else HASH_B,
        source_candidate_id=HASH_C if camera_id == "camera_a" else HASH_D,
        anchor_world_xyz_m=xyz,
        support_sample_count=support,
        retained_confidence_median=confidence,
        retained_depth_median_m=depth_median,
        retained_depth_mad_m=depth_mad,
    )


def unavailable_source(camera_id: str) -> FusionSourceMeasurement:
    return FusionSourceMeasurement(
        camera_id=camera_id,  # type: ignore[arg-type]
        availability=AnchorAvailability.UNAVAILABLE,
        unavailable_reason=AnchorUnavailableReason.SOURCE_OBSERVATION_UNAVAILABLE,
        source_observation_id=None,
        source_candidate_id=None,
        anchor_world_xyz_m=None,
        support_sample_count=None,
        retained_confidence_median=None,
        retained_depth_median_m=None,
        retained_depth_mad_m=None,
    )


def test_reliability_score_is_transparent_formula() -> None:
    score = reliability_score(
        support_sample_count=100,
        retained_confidence_median=4.0,
        retained_depth_median_m=2.0,
        retained_depth_mad_m=0.2,
    )

    assert score == pytest.approx(10.0 * 4.0 / 1.1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("support_sample_count", 0),
        ("retained_confidence_median", 0.0),
        ("retained_depth_median_m", 0.0),
        ("retained_depth_mad_m", -0.1),
    ],
)
def test_invalid_reliability_input_is_rejected(field: str, value: float) -> None:
    payload = {
        "support_sample_count": 100,
        "retained_confidence_median": 4.0,
        "retained_depth_median_m": 2.0,
        "retained_depth_mad_m": 0.2,
    }
    payload[field] = value
    with pytest.raises(ValueError, match="invalid"):
        reliability_score(**payload)  # type: ignore[arg-type]


def test_eligible_pair_uses_normalized_reliability_weighted_mean() -> None:
    camera_a = observed_source("camera_a", xyz=(0.0, 0.0, 0.0), confidence=4.0)
    camera_b = observed_source("camera_b", xyz=(0.2, 0.0, 0.0), confidence=2.0)
    result = resolve_cross_camera_observation(
        sources=(camera_a, camera_b), maximum_disagreement_m=0.35
    )

    assert result.state is CrossCameraObservationState.FUSED
    assert result.camera_fusion_performed is True
    assert result.contribution_weights == pytest.approx((2 / 3, 1 / 3))
    assert result.world_xyz_m == pytest.approx((0.2 / 3, 0.0, 0.0))


def test_source_order_cannot_change_fused_result() -> None:
    camera_a = observed_source("camera_a", xyz=(0.0, 0.0, 0.0), confidence=4.0)
    camera_b = observed_source("camera_b", xyz=(0.2, 0.0, 0.0), confidence=2.0)

    forward = resolve_cross_camera_observation(
        sources=(camera_a, camera_b), maximum_disagreement_m=0.35
    )
    reversed_result = resolve_cross_camera_observation(
        sources=(camera_b, camera_a), maximum_disagreement_m=0.35
    )

    assert reversed_result == forward


def test_disagreement_preserves_scores_but_emits_no_weights_or_xyz() -> None:
    result = resolve_cross_camera_observation(
        sources=(
            observed_source("camera_a", xyz=(0.0, 0.0, 0.0)),
            observed_source("camera_b", xyz=(1.0, 0.0, 0.0)),
        ),
        maximum_disagreement_m=0.35,
    )

    assert result.state is CrossCameraObservationState.DISAGREEMENT
    assert all(score is not None for score in result.reliability_scores)
    assert result.contribution_weights == (None, None)
    assert result.world_xyz_m is None
    assert result.camera_fusion_performed is False


def test_single_camera_is_passthrough_not_fusion() -> None:
    result = resolve_cross_camera_observation(
        sources=(
            unavailable_source("camera_a"),
            observed_source("camera_b", xyz=(1.0, 2.0, 0.5)),
        ),
        maximum_disagreement_m=0.35,
    )

    assert result.state is CrossCameraObservationState.SINGLE_CAMERA
    assert result.combination_method is (
        CrossCameraCombinationMethod.SINGLE_CAMERA_PASSTHROUGH
    )
    assert result.contribution_weights == (None, 1.0)
    assert result.world_xyz_m == (1.0, 2.0, 0.5)
    assert result.camera_fusion_performed is False


def test_both_unavailable_emit_no_xyz() -> None:
    result = resolve_cross_camera_observation(
        sources=(unavailable_source("camera_a"), unavailable_source("camera_b")),
        maximum_disagreement_m=0.35,
    )

    assert result.state is CrossCameraObservationState.UNAVAILABLE
    assert result.world_xyz_m is None
    assert result.reliability_scores == (None, None)


def test_duplicate_camera_sources_are_rejected() -> None:
    camera_a = observed_source("camera_a", xyz=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="unique source"):
        resolve_cross_camera_observation(
            sources=(camera_a, replace(camera_a, anchor_world_xyz_m=(0.1, 0.0, 0.0))),
            maximum_disagreement_m=0.35,
        )


def test_persistent_fused_record_validates_weighted_xyz_and_round_trip() -> None:
    source_a = FusionSourceEvidence(
        camera_id="camera_a",
        availability=AnchorAvailability.OBSERVED,
        unavailable_reason=None,
        source_observation_id=HASH_A,
        source_candidate_id=HASH_B,
        anchor_world_xyz_m=(0.0, 0.0, 0.0),
        support_sample_count=100,
        retained_confidence_median=4.0,
        retained_depth_median_m=2.0,
        retained_depth_mad_m=0.2,
        retained_depth_relative_mad=0.1,
        reliability_score=40.0,
        contribution_weight=0.8,
    )
    source_b = FusionSourceEvidence(
        camera_id="camera_b",
        availability=AnchorAvailability.OBSERVED,
        unavailable_reason=None,
        source_observation_id=HASH_C,
        source_candidate_id=HASH_D,
        anchor_world_xyz_m=(0.2, 0.0, 0.0),
        support_sample_count=100,
        retained_confidence_median=1.0,
        retained_depth_median_m=2.0,
        retained_depth_mad_m=0.2,
        retained_depth_relative_mad=0.1,
        reliability_score=10.0,
        contribution_weight=0.2,
    )
    observation_id = CrossCameraObservationRecord.create_observation_id(
        action_depth_job_id=HASH_E,
        bundle_id=HASH_F,
        target=PerceptionTarget.PERSON,
        policy_id="s04_cross_camera_observation_v1",
    )
    record = CrossCameraObservationRecord(
        observation_id=observation_id,
        policy_id="s04_cross_camera_observation_v1",
        anchor_policy_id="s04_target_anchor_v1",
        action_depth_job_id=HASH_E,
        bundle_id=HASH_F,
        camera_a_frame_id=HASH_A,
        camera_b_frame_id=HASH_C,
        source_frame_index=1,
        capture_timestamp_seconds=1.0,
        maximum_source_time_difference_seconds=0.001,
        phase_id="paired",
        target=PerceptionTarget.PERSON,
        selected_anchor_method=AnchorCandidateMethod.PERSON_LOWEST_WORLD_Z_QUINTILE,
        state=CrossCameraObservationState.FUSED,
        combination_method=CrossCameraCombinationMethod.RELIABILITY_WEIGHTED_MEAN,
        sources=(source_a, source_b),
        disagreement_distance_m=0.2,
        maximum_eligible_disagreement_m=0.35,
        world_xyz_m=(0.04, 0.0, 0.0),
        inside_room_bounds=True,
        coordinate_semantics="Reliability-weighted selected person anchors.",
        camera_fusion_performed=True,
        single_source_passthrough=False,
    )

    assert CrossCameraObservationRecord.model_validate_json(record.model_dump_json()) == record
    payload = record.model_dump(mode="json")
    payload["world_xyz_m"] = [0.1, 0.0, 0.0]
    with pytest.raises(ValidationError, match="weighted anchors"):
        CrossCameraObservationRecord.model_validate(payload)
