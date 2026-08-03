"""In-mask DA3 depth/confidence diagnostics before any XYZ localization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import Field, field_validator, model_validator

from spatial_reconstruction.contracts import (
    ContractModel,
    NonNegativeFloat,
    NonNegativeInt,
    PerceptionTarget,
    PositiveFloat,
    PositiveInt,
    Sha256Digest,
)

BoolArray = NDArray[np.bool_]
Float64Array = NDArray[np.float64]


class InsufficientCandidateSamplesError(ValueError):
    """Raised when an honest visible-surface candidate cannot meet its rule."""


class MaskDepthStrategy(StrEnum):
    """Candidate image regions compared before choosing a localization rule."""

    WHOLE_MASK = "whole_mask"
    ERODED_INTERIOR = "eroded_interior"
    CONNECTED_DEPTH_CLUSTER = "connected_depth_cluster"
    PERSON_LOWER_BODY = "person_lower_body"


class MaskDepthDiagnosticConfig(ContractModel):
    """Explicit parameters for the non-localizing diagnostic comparison."""

    erosion_radius_pixels: PositiveInt = 2
    person_lower_body_fraction: float = Field(default=0.35, gt=0.0, le=1.0)
    cluster_minimum_half_width_m: PositiveFloat = 0.15
    cluster_mad_scale: PositiveFloat = 2.5
    confidence_percentiles: tuple[float, ...] = (20.0, 40.0, 60.0, 80.0)
    backprojection_performed: Literal[False] = False
    xyz_generated: Literal[False] = False
    s02_confidence_policy_applied: Literal[False] = False

    @field_validator("confidence_percentiles")
    @classmethod
    def validate_percentiles(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value:
            raise ValueError("confidence percentile sweep must not be empty")
        if any(not math.isfinite(item) or not 0 <= item <= 100 for item in value):
            raise ValueError("confidence percentiles must be finite and within 0..100")
        if tuple(sorted(set(value))) != value:
            raise ValueError("confidence percentiles must be strictly increasing")
        return value


class SampleDistribution(ContractModel):
    """Finite distribution summary for one candidate sample."""

    count: PositiveInt
    minimum: NonNegativeFloat
    p05: NonNegativeFloat
    p20: NonNegativeFloat
    p40: NonNegativeFloat
    median: NonNegativeFloat
    p60: NonNegativeFloat
    p80: NonNegativeFloat
    p95: NonNegativeFloat
    maximum: NonNegativeFloat
    mean: NonNegativeFloat
    standard_deviation: NonNegativeFloat
    median_absolute_deviation: NonNegativeFloat
    interquartile_range: NonNegativeFloat

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        ordered = (
            self.minimum,
            self.p05,
            self.p20,
            self.p40,
            self.median,
            self.p60,
            self.p80,
            self.p95,
            self.maximum,
        )
        if any(
            left > right
            for left, right in zip(ordered[:-1], ordered[1:], strict=True)
        ):
            raise ValueError("sample distribution percentiles must be ordered")
        return self


class ConfidenceRetention(ContractModel):
    """Candidate retention under one same-action-frame confidence percentile."""

    full_frame_percentile: NonNegativeFloat
    threshold: NonNegativeFloat
    retained_count: NonNegativeInt
    retained_fraction: NonNegativeFloat
    retained_depth_m: SampleDistribution | None

    @model_validator(mode="after")
    def validate_retention(self) -> Self:
        if self.full_frame_percentile > 100:
            raise ValueError("confidence percentile must not exceed 100")
        if self.retained_fraction > 1:
            raise ValueError("confidence retained fraction must not exceed one")
        if (self.retained_count == 0) != (self.retained_depth_m is None):
            raise ValueError("retained depth summary must match retained count")
        if self.retained_depth_m is not None:
            if self.retained_depth_m.count != self.retained_count:
                raise ValueError("retained depth count differs from retention count")
        return self


class MaskDepthDiagnosticRecord(ContractModel):
    """Persistent non-XYZ diagnostic for one aligned mask and strategy."""

    action_depth_job_id: Sha256Digest
    bundle_id: Sha256Digest
    camera_id: Literal["camera_a", "camera_b"]
    frame_id: Sha256Digest
    source_frame_index: NonNegativeInt
    phase_id: str
    target: PerceptionTarget
    perception_job_id: Sha256Digest
    camera_local_track_id: str
    aligned_mask_artifact_ref: str
    aligned_mask_artifact_sha256: Sha256Digest
    aligned_mask_index: NonNegativeInt
    raw_prediction_ref: str
    raw_prediction_sha256: Sha256Digest
    strategy: MaskDepthStrategy
    source_mask_pixel_count: PositiveInt
    candidate_pixel_count: PositiveInt
    candidate_fraction_of_source_mask: PositiveFloat
    finite_positive_depth_count: PositiveInt
    finite_confidence_count: PositiveInt
    valid_depth_confidence_pair_count: PositiveInt
    depth_m: SampleDistribution
    confidence: SampleDistribution
    confidence_sweep: tuple[ConfidenceRetention, ...]
    cluster_seed_median_m: NonNegativeFloat | None = None
    cluster_half_width_m: PositiveFloat | None = None
    connected_component_count: NonNegativeInt | None = None
    localization_performed: Literal[False] = False

    @field_validator(
        "phase_id",
        "camera_local_track_id",
        "aligned_mask_artifact_ref",
        "raw_prediction_ref",
    )
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("mask-depth diagnostic text must be non-empty and trimmed")
        return value

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        expected_fraction = self.candidate_pixel_count / self.source_mask_pixel_count
        if not np.isclose(
            self.candidate_fraction_of_source_mask, expected_fraction, atol=1e-12
        ):
            raise ValueError("candidate fraction differs from its pixel counts")
        if self.candidate_fraction_of_source_mask > 1:
            raise ValueError("candidate mask cannot exceed source mask")
        if self.valid_depth_confidence_pair_count > self.candidate_pixel_count:
            raise ValueError("valid pair count cannot exceed candidate pixels")
        if self.depth_m.count != self.valid_depth_confidence_pair_count:
            raise ValueError("depth summary count differs from valid pair count")
        if self.confidence.count != self.valid_depth_confidence_pair_count:
            raise ValueError("confidence summary count differs from valid pair count")
        cluster_fields = (
            self.cluster_seed_median_m,
            self.cluster_half_width_m,
            self.connected_component_count,
        )
        if self.strategy is MaskDepthStrategy.CONNECTED_DEPTH_CLUSTER:
            if any(value is None for value in cluster_fields):
                raise ValueError("connected cluster diagnostics require cluster metadata")
        elif any(value is not None for value in cluster_fields):
            raise ValueError("cluster metadata belongs only to connected strategy")
        return self


class MaskDepthDiagnosticRunSummary(ContractModel):
    """Strict summary of raw action-mask depth/confidence comparisons."""

    schema_version: Literal[1]
    status: Literal["completed_pending_policy_selection"]
    stage: Literal["S04"]
    created_at_utc: datetime
    source_mask_alignment_summary_ref: str
    source_mask_alignment_summary_sha256: Sha256Digest
    source_action_depth_summary_ref: str
    source_action_depth_summary_sha256: Sha256Digest
    configuration: MaskDepthDiagnosticConfig
    records: tuple[MaskDepthDiagnosticRecord, ...]
    comparison_csv_ref: str
    comparison_csv_sha256: Sha256Digest
    strategy_comparison_ref: str
    strategy_comparison_sha256: Sha256Digest
    contact_sheet_ref: str
    contact_sheet_sha256: Sha256Digest
    per_mask_diagnostics: tuple[dict[str, Any], ...]
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        if not self.records:
            raise ValueError("mask-depth diagnostic run must contain records")
        keys = {
            (
                record.action_depth_job_id,
                record.camera_id,
                record.target,
                record.strategy,
            )
            for record in self.records
        }
        if len(keys) != len(self.records):
            raise ValueError("mask-depth diagnostics cannot duplicate identity/strategy")
        if any(record.localization_performed for record in self.records):
            raise ValueError("diagnostic records cannot perform localization")
        return self


class TargetVisibleSurfaceRule(ContractModel):
    """Selected non-XYZ visible-surface sampling rule for one target."""

    target: PerceptionTarget
    candidate_strategy: MaskDepthStrategy
    confidence_threshold_basis: Literal["candidate_valid_sample_percentile"]
    confidence_percentile: NonNegativeFloat
    minimum_retained_sample_count: PositiveInt
    depth_aggregate: Literal["median_ray_depth"]
    insufficient_data_state: Literal["unavailable"]
    coordinate_semantics: str

    @model_validator(mode="after")
    def validate_rule(self) -> Self:
        if self.confidence_percentile > 100:
            raise ValueError("confidence percentile must not exceed 100")
        if self.target is PerceptionTarget.BACKPACK:
            if self.candidate_strategy is not MaskDepthStrategy.CONNECTED_DEPTH_CLUSTER:
                raise ValueError("backpack rule must use the connected depth cluster")
        elif self.candidate_strategy is not MaskDepthStrategy.PERSON_LOWER_BODY:
            raise ValueError("person rule must use the lower-body candidate")
        if not self.coordinate_semantics or self.coordinate_semantics.strip() != (
            self.coordinate_semantics
        ):
            raise ValueError("coordinate semantics must be non-empty and trimmed")
        return self


class MaskDepthSamplingPolicy(ContractModel):
    """Selected S04 sample-validity and ray-depth aggregation policy."""

    schema_version: Literal[1] = 1
    policy_id: Literal["s04_dynamic_visible_surface_v1"] = (
        "s04_dynamic_visible_surface_v1"
    )
    rules: tuple[TargetVisibleSurfaceRule, TargetVisibleSurfaceRule]
    require_finite_positive_depth: Literal[True] = True
    require_finite_confidence: Literal[True] = True
    full_frame_confidence_threshold_rejected: Literal[True] = True
    s02_confidence_policy_applied: Literal[False] = False
    backprojection_performed: Literal[False] = False
    xyz_generated: Literal[False] = False

    @model_validator(mode="after")
    def validate_rules(self) -> Self:
        if {rule.target for rule in self.rules} != set(PerceptionTarget):
            raise ValueError("sampling policy requires one rule per target")
        if any(rule.confidence_percentile != 20 for rule in self.rules):
            raise ValueError("selected dynamic candidate confidence percentile is 20")
        return self


class MaskDepthPolicySelectionSummary(ContractModel):
    """Persistent evidence and outcome for the pre-localization rule choice."""

    schema_version: Literal[1]
    status: Literal["selected"]
    stage: Literal["S04"]
    created_at_utc: datetime
    source_diagnostics_summary_ref: str
    source_diagnostics_summary_sha256: Sha256Digest
    policy: MaskDepthSamplingPolicy
    evidence_by_target: dict[str, dict[str, Any]]
    rejected_alternatives: tuple[dict[str, Any], ...]
    limitations: tuple[str, ...]
    localization_performed: Literal[False]

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if set(self.evidence_by_target) != {target.value for target in PerceptionTarget}:
            raise ValueError("policy evidence must cover person and backpack")
        return self


@dataclass(frozen=True, slots=True)
class CandidateMask:
    """One immutable runtime candidate mask and optional cluster metadata."""

    strategy: MaskDepthStrategy
    mask: BoolArray
    cluster_seed_median_m: float | None = None
    cluster_half_width_m: float | None = None
    connected_component_count: int | None = None

    def __post_init__(self) -> None:
        value = np.asarray(self.mask)
        if value.dtype != np.bool_ or value.ndim != 2:
            raise ValueError("candidate mask must be a two-dimensional bool array")
        if not np.any(value):
            raise ValueError("candidate mask must contain foreground pixels")
        value.setflags(write=False)
        object.__setattr__(self, "mask", value)


@dataclass(frozen=True, slots=True)
class ConfidentCandidateSelection:
    """Immutable candidate-relative confidence selection before back-projection."""

    mask: BoolArray
    confidence_threshold: float
    valid_candidate_count: int
    retained_count: int

    def __post_init__(self) -> None:
        value = np.asarray(self.mask)
        if value.dtype != np.bool_ or value.ndim != 2:
            raise ValueError("confidence selection mask must be two-dimensional bool")
        if not math.isfinite(self.confidence_threshold):
            raise ValueError("confidence selection threshold must be finite")
        if self.valid_candidate_count <= 0 or self.retained_count <= 0:
            raise ValueError("confidence selection counts must be positive")
        if self.retained_count > self.valid_candidate_count:
            raise ValueError("retained count cannot exceed valid candidate count")
        if int(np.count_nonzero(value)) != self.retained_count:
            raise ValueError("confidence selection mask differs from retained count")
        value.setflags(write=False)
        object.__setattr__(self, "mask", value)


def build_mask_depth_candidates(
    mask: ArrayLike,
    depth_m: ArrayLike,
    *,
    target: PerceptionTarget,
    config: MaskDepthDiagnosticConfig,
) -> tuple[CandidateMask, ...]:
    """Build deterministic mask strategies without back-projecting any sample."""

    source, depth = _validated_inputs(mask, depth_m)
    kernel_size = config.erosion_radius_pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(source.astype(np.uint8), kernel, iterations=1).astype(bool)
    if not np.any(eroded):
        raise ValueError("erosion removed every source-mask pixel")

    candidates = [
        CandidateMask(MaskDepthStrategy.WHOLE_MASK, source.copy()),
        CandidateMask(MaskDepthStrategy.ERODED_INTERIOR, eroded.copy()),
        _connected_depth_cluster(eroded, depth, config=config),
    ]
    if target is PerceptionTarget.PERSON:
        y_pixels = np.flatnonzero(np.any(source, axis=1))
        y_min = int(y_pixels[0])
        y_max = int(y_pixels[-1])
        height = y_max - y_min + 1
        cutoff = y_min + int(math.floor((1 - config.person_lower_body_fraction) * height))
        rows = np.arange(source.shape[0])[:, np.newaxis]
        lower_body = source & (rows >= cutoff)
        if not np.any(lower_body):
            raise ValueError("person lower-body selection contains no pixels")
        candidates.append(
            CandidateMask(MaskDepthStrategy.PERSON_LOWER_BODY, lower_body)
        )
    return tuple(candidates)


def summarize_distribution(values: ArrayLike) -> SampleDistribution:
    """Summarize finite non-negative values with fixed robust percentiles."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("distribution contains no finite values")
    if np.any(finite < 0):
        raise ValueError("diagnostic distributions must be non-negative")
    percentiles = np.percentile(finite, [5, 20, 25, 40, 50, 60, 75, 80, 95])
    median = float(percentiles[4])
    return SampleDistribution(
        count=int(finite.size),
        minimum=float(np.min(finite)),
        p05=float(percentiles[0]),
        p20=float(percentiles[1]),
        p40=float(percentiles[3]),
        median=median,
        p60=float(percentiles[5]),
        p80=float(percentiles[7]),
        p95=float(percentiles[8]),
        maximum=float(np.max(finite)),
        mean=float(np.mean(finite)),
        standard_deviation=float(np.std(finite)),
        median_absolute_deviation=float(np.median(np.abs(finite - median))),
        interquartile_range=float(percentiles[6] - percentiles[2]),
    )


def compute_mask_depth_diagnostics(
    *,
    mask: ArrayLike,
    depth_m: ArrayLike,
    confidence: ArrayLike,
    target: PerceptionTarget,
    config: MaskDepthDiagnosticConfig,
) -> tuple[dict[str, Any], ...]:
    """Compute raw diagnostic payloads for every applicable strategy."""

    source, depth = _validated_inputs(mask, depth_m)
    scores = np.asarray(confidence, dtype=np.float64)
    if scores.shape != depth.shape:
        raise ValueError("confidence shape must match mask and depth")
    finite_full_frame_confidence = scores[np.isfinite(scores)]
    if finite_full_frame_confidence.size == 0:
        raise ValueError("full action frame contains no finite confidence")
    thresholds = {
        percentile: float(np.percentile(finite_full_frame_confidence, percentile))
        for percentile in config.confidence_percentiles
    }
    source_count = int(np.count_nonzero(source))
    payloads: list[dict[str, Any]] = []
    for candidate in build_mask_depth_candidates(
        source,
        depth,
        target=target,
        config=config,
    ):
        finite_depth = candidate.mask & np.isfinite(depth) & (depth > 0)
        finite_confidence = candidate.mask & np.isfinite(scores)
        valid = finite_depth & finite_confidence
        valid_count = int(np.count_nonzero(valid))
        if valid_count == 0:
            raise ValueError(f"{candidate.strategy} contains no valid depth/confidence pair")
        sweeps: list[dict[str, Any]] = []
        for percentile, threshold in thresholds.items():
            retained = valid & (scores >= threshold)
            retained_depth = depth[retained]
            count = int(retained_depth.size)
            sweeps.append(
                {
                    "full_frame_percentile": percentile,
                    "threshold": threshold,
                    "retained_count": count,
                    "retained_fraction": count / valid_count,
                    "retained_depth_m": (
                        None
                        if count == 0
                        else summarize_distribution(retained_depth).model_dump(mode="json")
                    ),
                }
            )
        payloads.append(
            {
                "strategy": candidate.strategy.value,
                "source_mask_pixel_count": source_count,
                "candidate_pixel_count": int(np.count_nonzero(candidate.mask)),
                "candidate_fraction_of_source_mask": (
                    int(np.count_nonzero(candidate.mask)) / source_count
                ),
                "finite_positive_depth_count": int(np.count_nonzero(finite_depth)),
                "finite_confidence_count": int(np.count_nonzero(finite_confidence)),
                "valid_depth_confidence_pair_count": valid_count,
                "depth_m": summarize_distribution(depth[valid]).model_dump(mode="json"),
                "confidence": summarize_distribution(scores[valid]).model_dump(mode="json"),
                "confidence_sweep": sweeps,
                "cluster_seed_median_m": candidate.cluster_seed_median_m,
                "cluster_half_width_m": candidate.cluster_half_width_m,
                "connected_component_count": candidate.connected_component_count,
                "localization_performed": False,
            }
        )
    return tuple(payloads)


def select_candidate_relative_confidence(
    *,
    candidate_mask: ArrayLike,
    depth_m: ArrayLike,
    confidence: ArrayLike,
    percentile: float,
    minimum_retained_sample_count: int,
) -> ConfidentCandidateSelection:
    """Select the higher-confidence portion of one valid object candidate."""

    candidate_array = np.asarray(candidate_mask)
    if candidate_array.dtype != np.bool_ or candidate_array.ndim != 2:
        raise ValueError("candidate mask must be a two-dimensional bool array")
    if not np.any(candidate_array):
        raise ValueError("candidate mask contains no pixels")
    if not math.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError("candidate confidence percentile must be within 0..100")
    if minimum_retained_sample_count <= 0:
        raise ValueError("minimum retained sample count must be positive")
    depth = np.asarray(depth_m, dtype=np.float64)
    scores = np.asarray(confidence, dtype=np.float64)
    if depth.shape != candidate_array.shape or scores.shape != candidate_array.shape:
        raise ValueError("candidate mask, depth, and confidence shapes must match")
    valid = (
        candidate_array
        & np.isfinite(depth)
        & (depth > 0)
        & np.isfinite(scores)
    )
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        raise InsufficientCandidateSamplesError(
            "candidate contains no finite positive depth/confidence pairs"
        )
    threshold = float(np.percentile(scores[valid], percentile))
    retained = valid & (scores >= threshold)
    retained_count = int(np.count_nonzero(retained))
    if retained_count < minimum_retained_sample_count:
        raise InsufficientCandidateSamplesError(
            "candidate-relative confidence selection retained too few samples: "
            f"{retained_count} < {minimum_retained_sample_count}"
        )
    return ConfidentCandidateSelection(
        mask=np.asarray(retained, dtype=bool),
        confidence_threshold=threshold,
        valid_candidate_count=valid_count,
        retained_count=retained_count,
    )


def _validated_inputs(mask: ArrayLike, depth_m: ArrayLike) -> tuple[BoolArray, Float64Array]:
    source_array = np.asarray(mask)
    if source_array.dtype not in (np.uint8, np.bool_):
        raise ValueError("source mask must be uint8 or bool")
    if source_array.ndim != 2:
        raise ValueError("source mask must be two-dimensional")
    unique = set(np.unique(source_array).tolist())
    if not unique.issubset({0, 1}) or 1 not in unique:
        raise ValueError("source mask must be binary with foreground")
    source = np.asarray(source_array, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.shape != source.shape:
        raise ValueError("depth shape must match source mask")
    return source, depth


def _connected_depth_cluster(
    eroded: BoolArray,
    depth: Float64Array,
    *,
    config: MaskDepthDiagnosticConfig,
) -> CandidateMask:
    valid = eroded & np.isfinite(depth) & (depth > 0)
    values = depth[valid]
    if values.size == 0:
        raise InsufficientCandidateSamplesError(
            "eroded mask contains no finite positive depth"
        )
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    half_width = max(
        config.cluster_minimum_half_width_m,
        config.cluster_mad_scale * 1.4826 * mad,
    )
    interval = valid & (np.abs(depth - median) <= half_width)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        interval.astype(np.uint8),
        connectivity=8,
    )
    if component_count <= 1:
        raise InsufficientCandidateSamplesError(
            "connected-depth interval contains no foreground component"
        )
    foreground_areas = stats[1:, cv2.CC_STAT_AREA]
    selected_label = int(np.argmax(foreground_areas)) + 1
    selected = labels == selected_label
    return CandidateMask(
        strategy=MaskDepthStrategy.CONNECTED_DEPTH_CLUSTER,
        mask=np.asarray(selected, dtype=bool),
        cluster_seed_median_m=median,
        cluster_half_width_m=half_width,
        connected_component_count=component_count - 1,
    )
