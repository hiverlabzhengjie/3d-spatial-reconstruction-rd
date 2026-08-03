from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from spatial_reconstruction.contracts import PerceptionTarget
from spatial_reconstruction.localization import (
    MaskDepthDiagnosticConfig,
    MaskDepthSamplingPolicy,
    MaskDepthStrategy,
    TargetVisibleSurfaceRule,
    build_mask_depth_candidates,
    compute_mask_depth_diagnostics,
    select_candidate_relative_confidence,
    summarize_distribution,
)


def make_fixture() -> tuple[
    NDArray[np.uint8], NDArray[np.float64], NDArray[np.float64]
]:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[3:18, 3:17] = 1
    depth = np.full((20, 20), 9.0, dtype=np.float64)
    depth[mask == 1] = 2.0
    depth[5:8, 12:17] = 5.0
    confidence = np.arange(400, dtype=np.float64).reshape(20, 20) / 20 + 1
    return mask, depth, confidence


def test_candidate_masks_are_deterministic_and_reject_far_leakage() -> None:
    mask, depth, _ = make_fixture()
    candidates = build_mask_depth_candidates(
        mask,
        depth,
        target=PerceptionTarget.PERSON,
        config=MaskDepthDiagnosticConfig(),
    )
    by_strategy = {candidate.strategy: candidate for candidate in candidates}

    assert set(by_strategy) == set(MaskDepthStrategy)
    assert np.count_nonzero(by_strategy[MaskDepthStrategy.ERODED_INTERIOR].mask) < (
        np.count_nonzero(mask)
    )
    cluster = by_strategy[MaskDepthStrategy.CONNECTED_DEPTH_CLUSTER]
    assert np.all(depth[cluster.mask] == 2.0)
    assert cluster.cluster_seed_median_m == pytest.approx(2.0)
    assert cluster.cluster_half_width_m == pytest.approx(0.15)
    lower = by_strategy[MaskDepthStrategy.PERSON_LOWER_BODY].mask
    assert np.min(np.nonzero(lower)[0]) >= 12
    assert all(not candidate.mask.flags.writeable for candidate in candidates)


def test_backpack_candidates_do_not_create_person_lower_body_region() -> None:
    mask, depth, _ = make_fixture()
    strategies = {
        candidate.strategy
        for candidate in build_mask_depth_candidates(
            mask,
            depth,
            target=PerceptionTarget.BACKPACK,
            config=MaskDepthDiagnosticConfig(),
        )
    }
    assert strategies == {
        MaskDepthStrategy.WHOLE_MASK,
        MaskDepthStrategy.ERODED_INTERIOR,
        MaskDepthStrategy.CONNECTED_DEPTH_CLUSTER,
    }


def test_diagnostics_sweep_uses_full_frame_confidence_and_never_localizes() -> None:
    mask, depth, confidence = make_fixture()
    records = compute_mask_depth_diagnostics(
        mask=mask,
        depth_m=depth,
        confidence=confidence,
        target=PerceptionTarget.BACKPACK,
        config=MaskDepthDiagnosticConfig(),
    )

    assert len(records) == 3
    whole = records[0]
    assert whole["strategy"] == "whole_mask"
    assert whole["depth_m"]["median"] == pytest.approx(2.0)
    assert whole["localization_performed"] is False
    thresholds = [item["threshold"] for item in whole["confidence_sweep"]]
    retained = [item["retained_count"] for item in whole["confidence_sweep"]]
    assert thresholds == sorted(thresholds)
    assert retained == sorted(retained, reverse=True)


def test_distribution_and_inputs_reject_invalid_diagnostic_data() -> None:
    with pytest.raises(ValueError, match="no finite"):
        summarize_distribution(np.array([np.nan]))
    with pytest.raises(ValueError, match="non-negative"):
        summarize_distribution(np.array([-1.0]))

    mask, depth, confidence = make_fixture()
    with pytest.raises(ValueError, match="uint8 or bool"):
        compute_mask_depth_diagnostics(
            mask=mask.astype(np.float32),
            depth_m=depth,
            confidence=confidence,
            target=PerceptionTarget.PERSON,
            config=MaskDepthDiagnosticConfig(),
        )


def test_candidate_relative_confidence_filters_and_enforces_minimum_count() -> None:
    mask, depth, confidence = make_fixture()
    candidate = build_mask_depth_candidates(
        mask,
        depth,
        target=PerceptionTarget.BACKPACK,
        config=MaskDepthDiagnosticConfig(),
    )[2]
    selection = select_candidate_relative_confidence(
        candidate_mask=candidate.mask,
        depth_m=depth,
        confidence=confidence,
        percentile=20,
        minimum_retained_sample_count=10,
    )

    assert selection.retained_count >= 0.8 * selection.valid_candidate_count
    assert np.all(confidence[selection.mask] >= selection.confidence_threshold)
    assert not selection.mask.flags.writeable

    with pytest.raises(ValueError, match="too few"):
        select_candidate_relative_confidence(
            candidate_mask=candidate.mask,
            depth_m=depth,
            confidence=confidence,
            percentile=20,
            minimum_retained_sample_count=selection.retained_count + 1,
        )
    with pytest.raises(ValueError, match="confidence shape"):
        compute_mask_depth_diagnostics(
            mask=mask,
            depth_m=depth,
            confidence=confidence[:-1],
            target=PerceptionTarget.PERSON,
            config=MaskDepthDiagnosticConfig(),
        )


def test_sampling_policy_forbids_wrong_target_strategy() -> None:
    person = TargetVisibleSurfaceRule(
        target=PerceptionTarget.PERSON,
        candidate_strategy=MaskDepthStrategy.PERSON_LOWER_BODY,
        confidence_threshold_basis="candidate_valid_sample_percentile",
        confidence_percentile=20,
        minimum_retained_sample_count=256,
        depth_aggregate="median_ray_depth",
        insufficient_data_state="unavailable",
        coordinate_semantics="Visible lower-body surface.",
    )
    backpack = TargetVisibleSurfaceRule(
        target=PerceptionTarget.BACKPACK,
        candidate_strategy=MaskDepthStrategy.CONNECTED_DEPTH_CLUSTER,
        confidence_threshold_basis="candidate_valid_sample_percentile",
        confidence_percentile=20,
        minimum_retained_sample_count=128,
        depth_aggregate="median_ray_depth",
        insufficient_data_state="unavailable",
        coordinate_semantics="Visible backpack surface.",
    )
    policy = MaskDepthSamplingPolicy(rules=(person, backpack))
    assert MaskDepthSamplingPolicy.model_validate_json(policy.model_dump_json()) == policy

    with pytest.raises(ValidationError, match="person rule"):
        TargetVisibleSurfaceRule(
            target=PerceptionTarget.PERSON,
            candidate_strategy=MaskDepthStrategy.ERODED_INTERIOR,
            confidence_threshold_basis="candidate_valid_sample_percentile",
            confidence_percentile=20,
            minimum_retained_sample_count=256,
            depth_aggregate="median_ray_depth",
            insufficient_data_state="unavailable",
            coordinate_semantics="Invalid generic interior.",
        )
