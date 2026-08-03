from __future__ import annotations

import pytest

from spatial_reconstruction.contracts import PerceptionTarget
from spatial_reconstruction.localization import (
    CorrectedAnchorKind,
    CorrectedPairObservationRecord,
    CorrectedPairSource,
    CorrectedPairState,
    TemporalCoordinateProvenance,
    TemporalPresentationPolicy,
    TemporalPresentationReason,
    TemporalPresentationResolution,
    TemporalPresentationState,
    build_measured_trajectory_segments,
    make_temporal_record,
    resolve_temporal_presentation,
)
from spatial_reconstruction.perception import PerceptionPresenceState

SHA_B = "b" * 64
SHA_C = "c" * 64


def _observation(
    *,
    frame: int,
    timestamp: float,
    target: PerceptionTarget = PerceptionTarget.PERSON,
    kind: CorrectedAnchorKind = CorrectedAnchorKind.PERSON_FOOTPOINT,
    xyz: tuple[float, float, float] = (0.1, 0.2, 0.0),
) -> CorrectedPairObservationRecord:
    action_depth_job_id = f"{frame:064x}"
    observation_id = CorrectedPairObservationRecord.create_observation_id(
        action_depth_job_id=action_depth_job_id,
        target=target,
        policy_id="s04_corrected_margin_aware_tracking_v1",
    )
    return CorrectedPairObservationRecord(
        observation_id=observation_id,
        policy_id="s04_corrected_margin_aware_tracking_v1",
        action_depth_job_id=action_depth_job_id,
        bundle_id=SHA_B,
        source_frame_index=frame,
        capture_timestamp_seconds=timestamp,
        phase_id="test",
        target=target,
        state=CorrectedPairState.SINGLE_CAMERA,
        selected_kind=kind,
        world_xyz_m=xyz,
        sources=(
            CorrectedPairSource(
                camera_id="camera_a",
                source_anchor_id=SHA_C,
                kind=kind,
                world_xyz_m=xyz,
                reliability_score=10.0,
                contribution_weight=1.0,
                selected_for_output=True,
            ),
            CorrectedPairSource(
                camera_id="camera_b",
                source_anchor_id=None,
                kind=None,
                world_xyz_m=None,
                reliability_score=None,
                contribution_weight=None,
                selected_for_output=False,
            ),
        ),
        selected_camera_ids=("camera_a",),
        disagreement_distance_m=None,
        maximum_cross_camera_disagreement_m=0.35,
        fallback_surface_used=kind is not CorrectedAnchorKind.PERSON_FOOTPOINT,
        selection_reason="test",
    )


def _resolve(
    *,
    frame: int,
    timestamp: float,
    current: CorrectedPairObservationRecord | None = None,
    last: CorrectedPairObservationRecord | None = None,
    confirmed_occluded: bool = False,
    camera_a_state: PerceptionPresenceState = PerceptionPresenceState.OBSERVED,
    camera_b_state: PerceptionPresenceState = PerceptionPresenceState.MISSING,
) -> tuple[TemporalPresentationPolicy, TemporalPresentationResolution]:
    policy = TemporalPresentationPolicy()
    result = resolve_temporal_presentation(
        source_frame_index=frame,
        capture_timestamp_seconds=timestamp,
        target=PerceptionTarget.PERSON,
        camera_a_perception_state=camera_a_state,
        camera_b_perception_state=camera_b_state,
        current_observation=current,
        last_measurement=last,
        confirmed_occluded=confirmed_occluded,
        policy=policy,
    )
    return policy, result


def test_current_measurement_is_authoritative_at_exact_tick() -> None:
    measurement = _observation(frame=204, timestamp=6.8)
    policy, resolution = _resolve(
        frame=204, timestamp=6.8, current=measurement, last=None
    )

    record = make_temporal_record(
        source_frame_index=204,
        capture_timestamp_seconds=6.8,
        target=PerceptionTarget.PERSON,
        camera_a_perception_state=PerceptionPresenceState.OBSERVED,
        camera_b_perception_state=PerceptionPresenceState.MISSING,
        resolution=resolution,
        policy=policy,
    )

    assert record.state is TemporalPresentationState.MEASURED
    assert record.raw_world_xyz_m == record.presentation_world_xyz_m
    assert record.coordinate_provenance is TemporalCoordinateProvenance.CURRENT_MEASUREMENT
    assert record.may_update_zone_membership
    assert record.may_extend_trajectory


def test_recent_measurement_is_visible_only_as_stale_hold() -> None:
    measurement = _observation(
        frame=330,
        timestamp=11.0,
        kind=CorrectedAnchorKind.PERSON_UPPER_BODY_SURFACE,
        xyz=(1.0, 2.0, 1.3),
    )
    _, resolution = _resolve(frame=348, timestamp=11.6, last=measurement)

    assert resolution.state is TemporalPresentationState.STALE
    assert resolution.raw_world_xyz_m is None
    assert resolution.presentation_world_xyz_m == (1.0, 2.0, 1.3)
    assert resolution.anchor_kind is CorrectedAnchorKind.PERSON_UPPER_BODY_SURFACE
    assert not resolution.may_update_zone_membership
    assert not resolution.may_extend_trajectory


def test_expired_measurement_becomes_missing_without_xyz() -> None:
    measurement = _observation(frame=204, timestamp=6.8)
    _, resolution = _resolve(frame=240, timestamp=8.0, last=measurement)

    assert resolution.state is TemporalPresentationState.MISSING
    assert resolution.raw_world_xyz_m is None
    assert resolution.presentation_world_xyz_m is None
    assert resolution.reason is TemporalPresentationReason.NO_CURRENT_DEPTH


def test_confirmed_occlusion_has_no_coordinate_and_overrides_stale_hold() -> None:
    measurement = _observation(frame=204, timestamp=6.8)
    _, resolution = _resolve(
        frame=210,
        timestamp=7.0,
        last=measurement,
        confirmed_occluded=True,
    )

    assert resolution.state is TemporalPresentationState.OCCLUDED
    assert resolution.presentation_world_xyz_m is None
    assert resolution.reason is TemporalPresentationReason.CONFIRMED_OCCLUSION


def test_missing_before_first_measurement_has_no_prior_reason() -> None:
    _, resolution = _resolve(
        frame=30,
        timestamp=1.0,
        camera_a_state=PerceptionPresenceState.MISSING,
        camera_b_state=PerceptionPresenceState.MISSING,
    )

    assert resolution.state is TemporalPresentationState.MISSING
    assert resolution.reason is TemporalPresentationReason.NO_PRIOR_MEASUREMENT
    assert resolution.presentation_world_xyz_m is None


def test_perception_failure_is_exposed_without_xyz() -> None:
    _, resolution = _resolve(
        frame=240,
        timestamp=8.0,
        last=_observation(frame=204, timestamp=6.8),
        camera_a_state=PerceptionPresenceState.FAILED,
        camera_b_state=PerceptionPresenceState.MISSING,
    )

    assert resolution.state is TemporalPresentationState.MISSING
    assert resolution.reason is TemporalPresentationReason.SOURCE_FAILURE
    assert resolution.raw_world_xyz_m is None


def test_current_observation_must_match_authoritative_tick() -> None:
    measurement = _observation(frame=204, timestamp=6.8)

    with pytest.raises(ValueError, match="frame differs"):
        _resolve(frame=210, timestamp=7.0, current=measurement)


def test_policy_rejects_enabling_inferred_positions() -> None:
    with pytest.raises(ValueError):
        TemporalPresentationPolicy(inferred_positions_allowed=True)  # type: ignore[arg-type]


def test_close_adjacent_matching_measurements_form_segment() -> None:
    first = _observation(frame=666, timestamp=22.2, xyz=(0.0, 0.0, 0.0))
    second = _observation(frame=708, timestamp=23.6, xyz=(0.2, 0.1, 0.0))

    segments = build_measured_trajectory_segments(
        (first, second), policy=TemporalPresentationPolicy()
    )

    assert len(segments) == 1
    assert segments[0].elapsed_seconds == pytest.approx(1.4)
    assert not segments[0].interpolation_performed
    assert not segments[0].stale_points_used


def test_long_gap_does_not_form_trajectory_segment() -> None:
    first = _observation(frame=204, timestamp=6.8)
    second = _observation(frame=330, timestamp=11.0)

    assert not build_measured_trajectory_segments(
        (first, second), policy=TemporalPresentationPolicy()
    )


def test_mixed_person_anchor_semantics_do_not_form_segment() -> None:
    footpoint = _observation(frame=204, timestamp=6.8)
    upper = _observation(
        frame=330,
        timestamp=8.0,
        kind=CorrectedAnchorKind.PERSON_UPPER_BODY_SURFACE,
        xyz=(1.0, 2.0, 1.3),
    )

    assert not build_measured_trajectory_segments(
        (footpoint, upper), policy=TemporalPresentationPolicy()
    )


def test_segment_builder_does_not_skip_incompatible_intermediate_measurement() -> None:
    first = _observation(frame=204, timestamp=6.8)
    upper = _observation(
        frame=240,
        timestamp=7.6,
        kind=CorrectedAnchorKind.PERSON_UPPER_BODY_SURFACE,
        xyz=(1.0, 2.0, 1.3),
    )
    later_footpoint = _observation(frame=264, timestamp=8.4, xyz=(0.3, 0.1, 0.0))

    assert not build_measured_trajectory_segments(
        (first, upper, later_footpoint), policy=TemporalPresentationPolicy()
    )
