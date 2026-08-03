from __future__ import annotations

import numpy as np
import pytest

from spatial_reconstruction.localization import (
    ActionDepthScaleUnavailableError,
    ActionMarkerScaleObservation,
    ActionPairScalePolicy,
    estimate_action_pair_scale,
    sample_floor_marker_scale,
)


def _observation(
    camera_id: str, marker_id: int, ratio: float = 1.1
) -> ActionMarkerScaleObservation:
    return ActionMarkerScaleObservation(
        camera_id=camera_id,  # type: ignore[arg-type]
        marker_id=marker_id,
        detected_center_uv=(10.0, 10.0),
        projected_center_uv=(10.2, 10.1),
        reprojection_error_px=0.3,
        valid_sample_count=25,
        expected_camera_depth_median=2.2,
        raw_da3_depth_median=2.0,
        expected_over_raw_ratio=ratio,
        ratio_mad=0.01,
    )


def test_estimate_action_pair_scale_accepts_one_shared_robust_scale() -> None:
    observations = [
        _observation("camera_a", 40, 1.10),
        _observation("camera_a", 41, 1.11),
        _observation("camera_a", 42, 1.09),
        _observation("camera_b", 41, 1.105),
        _observation("camera_b", 42, 1.095),
    ]

    estimate = estimate_action_pair_scale(observations, policy=ActionPairScalePolicy())

    assert estimate.scale == pytest.approx(1.10)
    assert estimate.marker_count_by_camera == {"camera_a": 3, "camera_b": 2}
    assert estimate.maximum_relative_deviation < 0.01


@pytest.mark.parametrize(
    "observations",
    [
        [_observation("camera_a", 40), _observation("camera_a", 41)],
        [
            _observation("camera_a", 40),
            _observation("camera_a", 41),
            _observation("camera_a", 42),
            _observation("camera_b", 41),
        ],
    ],
)
def test_estimate_action_pair_scale_rejects_missing_pair_evidence(
    observations: list[ActionMarkerScaleObservation],
) -> None:
    with pytest.raises(ActionDepthScaleUnavailableError):
        estimate_action_pair_scale(observations, policy=ActionPairScalePolicy())


def test_estimate_action_pair_scale_rejects_cross_marker_disagreement() -> None:
    observations = [
        _observation("camera_a", 40),
        _observation("camera_a", 41),
        _observation("camera_a", 42),
        _observation("camera_b", 41),
        _observation("camera_b", 42, 1.25),
    ]
    with pytest.raises(ActionDepthScaleUnavailableError, match="disagree"):
        estimate_action_pair_scale(observations, policy=ActionPairScalePolicy())


def test_estimate_action_pair_scale_rejects_bad_reprojection() -> None:
    observations = [
        _observation("camera_a", 40),
        _observation("camera_a", 41),
        _observation("camera_a", 42),
        _observation("camera_b", 41),
        _observation("camera_b", 42),
    ]
    observations[-1] = observations[-1].model_copy(update={"reprojection_error_px": 5.1})
    with pytest.raises(ActionDepthScaleUnavailableError, match="reprojection"):
        estimate_action_pair_scale(observations, policy=ActionPairScalePolicy())


def test_sample_floor_marker_scale_recovers_known_scalar() -> None:
    intrinsics = np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]])
    world_from_camera = np.array(
        [[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, 2.0], [0, 0, 0, 1.0]]
    )
    depth = np.full((100, 100), 2.0 / 1.2, dtype=np.float64)

    count, expected, raw, ratio, mad, mask = sample_floor_marker_scale(
        depth_m=depth,
        processed_intrinsics=intrinsics,
        T_world_from_camera=world_from_camera,
        marker_center_world_m=(0.0, 0.0, 0.0),
        marker_length_m=0.5,
        protected_inner_fraction=0.6,
    )

    assert count >= 16
    assert expected == pytest.approx(2.0)
    assert raw == pytest.approx(2.0 / 1.2)
    assert ratio == pytest.approx(1.2)
    assert mad == pytest.approx(0.0)
    assert int(mask.sum()) == count


def test_scale_policy_rejects_camera_specific_fallback_configuration() -> None:
    with pytest.raises(ValueError):
        ActionPairScalePolicy(camera_specific_fallback_allowed=True)  # type: ignore[arg-type]
