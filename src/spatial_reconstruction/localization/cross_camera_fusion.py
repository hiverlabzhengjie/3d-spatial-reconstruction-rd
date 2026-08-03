"""D032-gated cross-camera observations with transparent reliability weights."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self, cast

import numpy as np
from pydantic import field_validator, model_validator

from spatial_reconstruction.contracts import (
    ContractModel,
    NonNegativeFloat,
    NonNegativeInt,
    PerceptionTarget,
    PositiveFloat,
    PositiveInt,
    Sha256Digest,
    Vector3,
)
from spatial_reconstruction.localization.anchor_evaluation import (
    AnchorAvailability,
    AnchorCandidateMethod,
    AnchorUnavailableReason,
)


class CrossCameraObservationState(StrEnum):
    """Result state after applying D032 eligibility and missing-data rules."""

    FUSED = "fused"
    SINGLE_CAMERA = "single_camera"
    DISAGREEMENT = "disagreement"
    UNAVAILABLE = "unavailable"


class CrossCameraCombinationMethod(StrEnum):
    """How the output XYZ was or was not produced."""

    RELIABILITY_WEIGHTED_MEAN = "reliability_weighted_mean"
    SINGLE_CAMERA_PASSTHROUGH = "single_camera_passthrough"
    NONE_DISAGREEMENT = "none_disagreement"
    NONE_UNAVAILABLE = "none_unavailable"


class FusionReliabilityConfig(ContractModel):
    """Explicit bounded reliability and D032 fusion policy."""

    policy_id: Literal["s04_cross_camera_observation_v1"] = (
        "s04_cross_camera_observation_v1"
    )
    anchor_policy_id: Literal["s04_target_anchor_v1"] = "s04_target_anchor_v1"
    support_count_exponent: Literal["one_half"] = "one_half"
    confidence_statistic: Literal["retained_median"] = "retained_median"
    dispersion_statistic: Literal["relative_depth_mad"] = "relative_depth_mad"
    dispersion_penalty: Literal["one_plus_relative_mad"] = "one_plus_relative_mad"
    maximum_cross_camera_disagreement_m: PositiveFloat = 0.35
    paired_disagreement_behavior: Literal["no_xyz"] = "no_xyz"
    single_camera_behavior: Literal["passthrough_with_provenance"] = (
        "passthrough_with_provenance"
    )
    both_unavailable_behavior: Literal["unavailable_without_xyz"] = (
        "unavailable_without_xyz"
    )
    temporal_filling_performed: Literal[False] = False
    presentation_smoothing_performed: Literal[False] = False


@dataclass(frozen=True, slots=True)
class FusionSourceMeasurement:
    """Runtime selected-anchor input and reliability evidence for one camera."""

    camera_id: Literal["camera_a", "camera_b"]
    availability: AnchorAvailability
    unavailable_reason: AnchorUnavailableReason | None
    source_observation_id: str | None
    source_candidate_id: str | None
    anchor_world_xyz_m: tuple[float, float, float] | None
    support_sample_count: int | None
    retained_confidence_median: float | None
    retained_depth_median_m: float | None
    retained_depth_mad_m: float | None

    def __post_init__(self) -> None:
        observed = self.availability is AnchorAvailability.OBSERVED
        evidence = (
            self.source_observation_id,
            self.source_candidate_id,
            self.anchor_world_xyz_m,
            self.support_sample_count,
            self.retained_confidence_median,
            self.retained_depth_median_m,
            self.retained_depth_mad_m,
        )
        if observed:
            if self.unavailable_reason is not None or any(item is None for item in evidence):
                raise ValueError("observed fusion source lacks anchor reliability evidence")
            assert self.anchor_world_xyz_m is not None
            assert self.support_sample_count is not None
            assert self.retained_confidence_median is not None
            assert self.retained_depth_median_m is not None
            assert self.retained_depth_mad_m is not None
            if (
                not np.isfinite(self.anchor_world_xyz_m).all()
                or self.support_sample_count <= 0
                or not math.isfinite(self.retained_confidence_median)
                or self.retained_confidence_median <= 0
                or not math.isfinite(self.retained_depth_median_m)
                or self.retained_depth_median_m <= 0
                or not math.isfinite(self.retained_depth_mad_m)
                or self.retained_depth_mad_m < 0
            ):
                raise ValueError("observed fusion source has invalid reliability evidence")
        elif self.unavailable_reason is None or any(item is not None for item in evidence):
            raise ValueError("unavailable fusion source must not carry anchor evidence")


@dataclass(frozen=True, slots=True)
class CrossCameraFusionResult:
    """Runtime cross-camera result before persistent job provenance is assigned."""

    state: CrossCameraObservationState
    combination_method: CrossCameraCombinationMethod
    sources: tuple[FusionSourceMeasurement, FusionSourceMeasurement]
    reliability_scores: tuple[float | None, float | None]
    contribution_weights: tuple[float | None, float | None]
    disagreement_distance_m: float | None
    world_xyz_m: tuple[float, float, float] | None
    camera_fusion_performed: bool


class FusionSourceEvidence(ContractModel):
    """Persistent input reliability and actual contribution for one camera."""

    camera_id: Literal["camera_a", "camera_b"]
    availability: AnchorAvailability
    unavailable_reason: AnchorUnavailableReason | None
    source_observation_id: Sha256Digest | None
    source_candidate_id: Sha256Digest | None
    anchor_world_xyz_m: Vector3 | None
    support_sample_count: PositiveInt | None
    retained_confidence_median: PositiveFloat | None
    retained_depth_median_m: PositiveFloat | None
    retained_depth_mad_m: NonNegativeFloat | None
    retained_depth_relative_mad: NonNegativeFloat | None
    reliability_score: PositiveFloat | None
    contribution_weight: NonNegativeFloat | None

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        observed = self.availability is AnchorAvailability.OBSERVED
        required = (
            self.source_observation_id,
            self.source_candidate_id,
            self.anchor_world_xyz_m,
            self.support_sample_count,
            self.retained_confidence_median,
            self.retained_depth_median_m,
            self.retained_depth_mad_m,
            self.retained_depth_relative_mad,
            self.reliability_score,
        )
        if observed:
            if self.unavailable_reason is not None or any(item is None for item in required):
                raise ValueError("observed fusion evidence lacks source values")
            assert self.retained_depth_median_m is not None
            assert self.retained_depth_mad_m is not None
            assert self.retained_depth_relative_mad is not None
            expected_relative = self.retained_depth_mad_m / self.retained_depth_median_m
            if not np.isclose(
                self.retained_depth_relative_mad, expected_relative, atol=1e-12
            ):
                raise ValueError("fusion relative depth MAD differs")
        elif self.unavailable_reason is None or any(item is not None for item in required):
            raise ValueError("unavailable fusion evidence carries source values")
        if self.contribution_weight is not None and not 0 <= self.contribution_weight <= 1:
            raise ValueError("fusion contribution weight must be within zero and one")
        return self


class CrossCameraObservationRecord(ContractModel):
    """One same-job target observation after D032-gated camera combination."""

    schema_version: Literal[1] = 1
    observation_id: Sha256Digest
    policy_id: Literal["s04_cross_camera_observation_v1"]
    anchor_policy_id: Literal["s04_target_anchor_v1"]
    action_depth_job_id: Sha256Digest
    bundle_id: Sha256Digest
    camera_a_frame_id: Sha256Digest
    camera_b_frame_id: Sha256Digest
    source_frame_index: NonNegativeInt
    capture_timestamp_seconds: NonNegativeFloat
    maximum_source_time_difference_seconds: NonNegativeFloat
    phase_id: str
    target: PerceptionTarget
    selected_anchor_method: AnchorCandidateMethod
    state: CrossCameraObservationState
    combination_method: CrossCameraCombinationMethod
    sources: tuple[FusionSourceEvidence, FusionSourceEvidence]
    disagreement_distance_m: NonNegativeFloat | None
    maximum_eligible_disagreement_m: PositiveFloat
    world_xyz_m: Vector3 | None
    inside_room_bounds: bool | None
    coordinate_semantics: str
    camera_fusion_performed: bool
    single_source_passthrough: bool
    worker_completion_order_used: Literal[False] = False
    temporal_filling_performed: Literal[False] = False
    presentation_smoothing_performed: Literal[False] = False

    @field_validator("phase_id", "coordinate_semantics")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("cross-camera observation text must be non-empty and trimmed")
        return value

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        if tuple(source.camera_id for source in self.sources) != (
            "camera_a",
            "camera_b",
        ):
            raise ValueError("fusion sources must be ordered camera_a then camera_b")
        observed = [
            source
            for source in self.sources
            if source.availability is AnchorAvailability.OBSERVED
        ]
        if self.state is CrossCameraObservationState.FUSED:
            if (
                len(observed) != 2
                or self.combination_method
                is not CrossCameraCombinationMethod.RELIABILITY_WEIGHTED_MEAN
                or self.disagreement_distance_m is None
                or self.disagreement_distance_m > self.maximum_eligible_disagreement_m
                or self.world_xyz_m is None
                or self.inside_room_bounds is None
                or not self.camera_fusion_performed
                or self.single_source_passthrough
            ):
                raise ValueError("fused cross-camera observation fields are invalid")
            weights = tuple(source.contribution_weight for source in self.sources)
            if any(weight is None for weight in weights) or not np.isclose(
                sum(cast(float, weight) for weight in weights), 1.0, atol=1e-12
            ):
                raise ValueError("fused contribution weights must sum to one")
            expected = sum(
                cast(float, source.contribution_weight)
                * np.asarray(source.anchor_world_xyz_m)
                for source in self.sources
            )
            if not np.allclose(self.world_xyz_m, expected, atol=1e-12):
                raise ValueError("fused XYZ differs from weighted anchors")
        elif self.state is CrossCameraObservationState.SINGLE_CAMERA:
            if (
                len(observed) != 1
                or self.combination_method
                is not CrossCameraCombinationMethod.SINGLE_CAMERA_PASSTHROUGH
                or self.disagreement_distance_m is not None
                or self.world_xyz_m != observed[0].anchor_world_xyz_m
                or self.inside_room_bounds is None
                or self.camera_fusion_performed
                or not self.single_source_passthrough
                or observed[0].contribution_weight != 1.0
            ):
                raise ValueError("single-camera observation fields are invalid")
        elif self.state is CrossCameraObservationState.DISAGREEMENT:
            if (
                len(observed) != 2
                or self.combination_method
                is not CrossCameraCombinationMethod.NONE_DISAGREEMENT
                or self.disagreement_distance_m is None
                or self.disagreement_distance_m <= self.maximum_eligible_disagreement_m
                or self.world_xyz_m is not None
                or self.inside_room_bounds is not None
                or self.camera_fusion_performed
                or self.single_source_passthrough
                or any(source.contribution_weight is not None for source in observed)
            ):
                raise ValueError("disagreement observation fields are invalid")
        elif (
            observed
            or self.combination_method
            is not CrossCameraCombinationMethod.NONE_UNAVAILABLE
            or self.disagreement_distance_m is not None
            or self.world_xyz_m is not None
            or self.inside_room_bounds is not None
            or self.camera_fusion_performed
            or self.single_source_passthrough
        ):
            raise ValueError("unavailable observation fields are invalid")
        expected_id = self.create_observation_id(
            action_depth_job_id=self.action_depth_job_id,
            bundle_id=self.bundle_id,
            target=self.target,
            policy_id=self.policy_id,
        )
        if self.observation_id != expected_id:
            raise ValueError("cross-camera observation ID differs from identity")
        return self

    @classmethod
    def create_observation_id(
        cls,
        *,
        action_depth_job_id: str,
        bundle_id: str,
        target: PerceptionTarget,
        policy_id: str,
    ) -> str:
        return _stable_digest(
            {
                "schema_version": 1,
                "action_depth_job_id": action_depth_job_id,
                "bundle_id": bundle_id,
                "target": target.value,
                "policy_id": policy_id,
            }
        )


class CrossCameraFusionRunSummary(ContractModel):
    """Persistent S04 selected-anchor cross-camera observation run."""

    schema_version: Literal[1]
    status: Literal["completed_pending_visual_qa"]
    stage: Literal["S04"]
    created_at_utc: datetime
    source_anchor_evaluation_summary_ref: str
    source_anchor_evaluation_summary_sha256: Sha256Digest
    source_anchor_evaluation_verification_ref: str
    source_anchor_evaluation_verification_sha256: Sha256Digest
    source_visible_surface_summary_ref: str
    source_visible_surface_summary_sha256: Sha256Digest
    configuration: FusionReliabilityConfig
    observations: tuple[CrossCameraObservationRecord, ...]
    observation_csv_ref: str
    observation_csv_sha256: Sha256Digest
    reliability_diagnostic_ref: str
    reliability_diagnostic_sha256: Sha256Digest
    world_preview_ref: str
    world_preview_sha256: Sha256Digest
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        keys = {
            (item.action_depth_job_id, item.target) for item in self.observations
        }
        if not self.observations or len(keys) != len(self.observations):
            raise ValueError("cross-camera observations must be non-empty and unique")
        frame_order = [item.source_frame_index for item in self.observations]
        if frame_order != sorted(frame_order):
            raise ValueError("cross-camera observations must follow capture order")
        return self


def reliability_score(
    *,
    support_sample_count: int,
    retained_confidence_median: float,
    retained_depth_median_m: float,
    retained_depth_mad_m: float,
) -> float:
    """Compute the D033 inspectable reliability score for one selected anchor."""

    values = (
        retained_confidence_median,
        retained_depth_median_m,
        retained_depth_mad_m,
    )
    if (
        support_sample_count <= 0
        or not all(math.isfinite(value) for value in values)
        or retained_confidence_median <= 0
        or retained_depth_median_m <= 0
        or retained_depth_mad_m < 0
    ):
        raise ValueError("fusion reliability inputs are invalid")
    relative_mad = retained_depth_mad_m / retained_depth_median_m
    return float(
        math.sqrt(support_sample_count)
        * retained_confidence_median
        / (1.0 + relative_mad)
    )


def resolve_cross_camera_observation(
    *,
    sources: tuple[FusionSourceMeasurement, FusionSourceMeasurement],
    maximum_disagreement_m: float,
) -> CrossCameraFusionResult:
    """Apply D032 state rules and D033 weights without time propagation."""

    sorted_sources = sorted(sources, key=lambda source: source.camera_id)
    ordered = (sorted_sources[0], sorted_sources[1])
    if tuple(source.camera_id for source in ordered) != ("camera_a", "camera_b"):
        raise ValueError("fusion requires one unique source per camera")
    observed = [
        source
        for source in ordered
        if source.availability is AnchorAvailability.OBSERVED
    ]
    scores: list[float | None] = []
    for source in ordered:
        if source.availability is AnchorAvailability.UNAVAILABLE:
            scores.append(None)
            continue
        assert source.support_sample_count is not None
        assert source.retained_confidence_median is not None
        assert source.retained_depth_median_m is not None
        assert source.retained_depth_mad_m is not None
        scores.append(
            reliability_score(
                support_sample_count=source.support_sample_count,
                retained_confidence_median=source.retained_confidence_median,
                retained_depth_median_m=source.retained_depth_median_m,
                retained_depth_mad_m=source.retained_depth_mad_m,
            )
        )
    if len(observed) == 2:
        assert ordered[0].anchor_world_xyz_m is not None
        assert ordered[1].anchor_world_xyz_m is not None
        distance = float(
            np.linalg.norm(
                np.asarray(ordered[0].anchor_world_xyz_m)
                - np.asarray(ordered[1].anchor_world_xyz_m)
            )
        )
        if distance > maximum_disagreement_m:
            return CrossCameraFusionResult(
                state=CrossCameraObservationState.DISAGREEMENT,
                combination_method=CrossCameraCombinationMethod.NONE_DISAGREEMENT,
                sources=ordered,
                reliability_scores=(scores[0], scores[1]),
                contribution_weights=(None, None),
                disagreement_distance_m=distance,
                world_xyz_m=None,
                camera_fusion_performed=False,
            )
        assert scores[0] is not None and scores[1] is not None
        score_sum = scores[0] + scores[1]
        pair_weights = (scores[0] / score_sum, scores[1] / score_sum)
        xyz = (
            pair_weights[0] * np.asarray(ordered[0].anchor_world_xyz_m)
            + pair_weights[1] * np.asarray(ordered[1].anchor_world_xyz_m)
        )
        return CrossCameraFusionResult(
            state=CrossCameraObservationState.FUSED,
            combination_method=CrossCameraCombinationMethod.RELIABILITY_WEIGHTED_MEAN,
            sources=ordered,
            reliability_scores=(scores[0], scores[1]),
            contribution_weights=pair_weights,
            disagreement_distance_m=distance,
            world_xyz_m=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            camera_fusion_performed=True,
        )
    if len(observed) == 1:
        index = 0 if ordered[0].availability is AnchorAvailability.OBSERVED else 1
        assert ordered[index].anchor_world_xyz_m is not None
        single_weights: tuple[float | None, float | None] = (
            (1.0, None) if index == 0 else (None, 1.0)
        )
        return CrossCameraFusionResult(
            state=CrossCameraObservationState.SINGLE_CAMERA,
            combination_method=CrossCameraCombinationMethod.SINGLE_CAMERA_PASSTHROUGH,
            sources=ordered,
            reliability_scores=(scores[0], scores[1]),
            contribution_weights=single_weights,
            disagreement_distance_m=None,
            world_xyz_m=ordered[index].anchor_world_xyz_m,
            camera_fusion_performed=False,
        )
    return CrossCameraFusionResult(
        state=CrossCameraObservationState.UNAVAILABLE,
        combination_method=CrossCameraCombinationMethod.NONE_UNAVAILABLE,
        sources=ordered,
        reliability_scores=(None, None),
        contribution_weights=(None, None),
        disagreement_distance_m=None,
        world_xyz_m=None,
        camera_fusion_performed=False,
    )


def _stable_digest(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
