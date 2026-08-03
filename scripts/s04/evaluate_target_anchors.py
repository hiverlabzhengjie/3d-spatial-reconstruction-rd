"""Evaluate S04 person/backpack anchor candidates before camera fusion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from spatial_reconstruction.contracts import PerceptionTarget
from spatial_reconstruction.localization import (
    ActionDepthRunSummary,
    AnchorAvailability,
    AnchorCandidateMethod,
    AnchorCandidateRecord,
    AnchorEvaluationConfig,
    AnchorEvaluationRunSummary,
    AnchorUnavailableReason,
    CrossCameraAnchorComparison,
    CrossCameraAnchorState,
    SelectedAnchorStateRecord,
    VisibleSurfaceObservationRecord,
    VisibleSurfaceRunSummary,
    evaluate_anchor_candidates,
)

Float64Array = NDArray[np.float64]
CameraId = Literal["camera_a", "camera_b"]
CAMERA_IDS: tuple[CameraId, ...] = ("camera_a", "camera_b")
TARGET_COLOR = {
    PerceptionTarget.PERSON: "#00a6c7",
    PerceptionTarget.BACKPACK: "#e62e91",
}
TRACKING_METHOD = {
    PerceptionTarget.PERSON: AnchorCandidateMethod.PERSON_LOWEST_WORLD_Z_QUINTILE,
    PerceptionTarget.BACKPACK: AnchorCandidateMethod.BACKPACK_WORLD_COMPONENT_MEDIAN,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--visible-surface-summary",
        type=Path,
        default=Path("artifacts/s04/visible_surfaces_20260802/summary.json"),
    )
    parser.add_argument(
        "--visible-surface-verification",
        type=Path,
        default=Path("artifacts/s04/visible_surfaces_20260802/verification.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    surface_path = _resolve(project_root, args.visible_surface_summary)
    verification_path = _resolve(project_root, args.visible_surface_verification)
    output_dir = _resolve(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    surface = VisibleSurfaceRunSummary.model_validate_json(
        surface_path.read_text(encoding="utf-8")
    )
    verification = _read_object(verification_path)
    _verify_prerequisite(
        verification=verification,
        surface_path=surface_path,
        expected_observation_count=len(surface.observations),
    )
    action_path = _resolve(project_root, Path(surface.source_action_depth_summary_ref))
    _require_hash(action_path, surface.source_action_depth_summary_sha256)
    action = ActionDepthRunSummary.model_validate_json(
        action_path.read_text(encoding="utf-8")
    )
    scene_path = _resolve(project_root, Path(surface.scene_metadata_ref))
    _require_hash(scene_path, surface.scene_metadata_sha256)
    scene = _read_object(scene_path)
    room_bounds = cast(dict[str, Any], scene["room_bounds"])
    bounds_min = np.asarray(room_bounds["minimum_world_xyz_m"], dtype=np.float64)
    bounds_max = np.asarray(room_bounds["maximum_world_xyz_m"], dtype=np.float64)
    config = AnchorEvaluationConfig()

    candidate_records: list[AnchorCandidateRecord] = []
    candidate_lookup: dict[
        tuple[str, str, PerceptionTarget, AnchorCandidateMethod],
        AnchorCandidateRecord,
    ] = {}
    observation_lookup = {
        (record.action_depth_job_id, record.camera_id, record.target): record
        for record in surface.observations
    }
    for observation in surface.observations:
        sample_path = _resolve(project_root, Path(observation.sample_cloud_ref))
        _require_hash(sample_path, observation.sample_cloud_sha256)
        with np.load(sample_path, allow_pickle=False) as arrays:
            _verify_sample_identity(arrays, observation)
            pixels = np.asarray(arrays["pixels_uv"], dtype=np.float64)
            points_world = np.asarray(arrays["points_world_m"], dtype=np.float64)
            confidence = np.asarray(arrays["confidence"], dtype=np.float64)
        candidates = evaluate_anchor_candidates(
            target=observation.target,
            pixels_uv=pixels,
            points_world_m=points_world,
            confidence=confidence,
            intrinsics=observation.processed_intrinsics,
            pose=observation.camera_pose,
            raw_visible_surface_world_xyz_m=observation.aggregate_world_xyz_m,
            config=config,
        )
        for candidate in candidates:
            world_xyz = candidate.world_xyz_m
            inside = None
            if world_xyz is not None:
                point = np.asarray(world_xyz, dtype=np.float64)
                inside = bool(np.all((point >= bounds_min) & (point <= bounds_max)))
            record = AnchorCandidateRecord(
                candidate_id=AnchorCandidateRecord.create_candidate_id(
                    source_observation_id=observation.observation_id,
                    method=candidate.method,
                ),
                source_observation_id=observation.observation_id,
                action_depth_job_id=observation.action_depth_job_id,
                bundle_id=observation.bundle_id,
                frame_id=observation.frame_id,
                source_frame_index=observation.source_frame_index,
                capture_timestamp_seconds=observation.capture_timestamp_seconds,
                phase_id=observation.phase_id,
                camera_id=cast(CameraId, observation.camera_id),
                target=observation.target,
                method=candidate.method,
                availability=candidate.availability,
                unavailable_reason=candidate.unavailable_reason,
                source_sample_count=candidate.source_sample_count,
                support_sample_count=candidate.support_sample_count,
                support_fraction=(
                    candidate.support_sample_count / candidate.source_sample_count
                ),
                world_xyz_m=world_xyz,
                measured_support_world_z_m=candidate.measured_support_world_z_m,
                inside_room_bounds=inside,
                coordinate_semantics=candidate.coordinate_semantics,
                selected_for_tracking=(candidate.method is TRACKING_METHOD[observation.target]),
                selected_for_ground_contact=(
                    candidate.method
                    is AnchorCandidateMethod.PERSON_VALIDATED_GROUND_CONTACT
                ),
                source_sample_cloud_ref=observation.sample_cloud_ref,
                source_sample_cloud_sha256=observation.sample_cloud_sha256,
                source_raw_aggregate_world_xyz_m=(
                    observation.aggregate_world_xyz_m
                ),
            )
            candidate_records.append(record)
            candidate_lookup[
                (
                    observation.action_depth_job_id,
                    observation.camera_id,
                    observation.target,
                    candidate.method,
                )
            ] = record

    selected_states = _build_selected_states(
        action=action,
        observation_lookup=observation_lookup,
        candidate_lookup=candidate_lookup,
    )
    comparisons = _build_cross_camera_comparisons(
        action=action,
        states=selected_states,
        maximum_disagreement_m=config.maximum_cross_camera_disagreement_m,
    )
    metrics = _candidate_metrics(candidate_records)
    selected_policy = _selected_policy(config=config, metrics=metrics)

    csv_path = output_dir / "anchor_candidate_comparison.csv"
    _save_candidate_csv(candidate_records, csv_path)
    comparison_path = output_dir / "anchor_candidate_comparison.png"
    _save_candidate_comparison(
        records=candidate_records,
        metrics=metrics,
        path=comparison_path,
    )
    world_path = output_dir / "selected_anchor_world_preview.png"
    _save_world_preview(
        states=selected_states,
        candidates=candidate_records,
        raw_observations=surface.observations,
        room_bounds=room_bounds,
        zones=cast(dict[str, Any], scene["zones"]),
        path=world_path,
    )
    summary = AnchorEvaluationRunSummary(
        schema_version=1,
        status="completed_pending_visual_qa",
        stage="S04",
        created_at_utc=datetime.now(UTC),
        source_visible_surface_summary_ref=_relative(surface_path, project_root),
        source_visible_surface_summary_sha256=_sha256(surface_path),
        source_visible_surface_verification_ref=_relative(
            verification_path, project_root
        ),
        source_visible_surface_verification_sha256=_sha256(verification_path),
        source_action_depth_summary_ref=_relative(action_path, project_root),
        source_action_depth_summary_sha256=_sha256(action_path),
        configuration=config,
        selected_policy=selected_policy,
        candidate_metrics=metrics,
        candidate_records=tuple(candidate_records),
        selected_anchor_states=tuple(selected_states),
        cross_camera_comparisons=tuple(comparisons),
        comparison_csv_ref=_relative(csv_path, project_root),
        comparison_csv_sha256=_sha256(csv_path),
        candidate_comparison_ref=_relative(comparison_path, project_root),
        candidate_comparison_sha256=_sha256(comparison_path),
        selected_anchor_world_preview_ref=_relative(world_path, project_root),
        selected_anchor_world_preview_sha256=_sha256(world_path),
        limitations=(
            "Person tracking anchors remain measured lower-body surfaces, not anatomical centres.",
            "Validated person ground contact is unavailable when near-floor support is absent.",
            (
                "Backpack anchors describe the visible depth cluster, not the hidden "
                "physical centroid."
            ),
            "The retained subset contains no same-frame two-camera backpack observation.",
            "No camera fusion, temporal filling, or presentation smoothing occurs in this action.",
        ),
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary.status,
                "candidate_record_count": len(candidate_records),
                "selected_anchor_state_count": len(selected_states),
                "selected_observed_count": sum(
                    item.availability is AnchorAvailability.OBSERVED
                    for item in selected_states
                ),
                "ground_contact_observed_count": sum(
                    item.method
                    is AnchorCandidateMethod.PERSON_VALIDATED_GROUND_CONTACT
                    and item.availability is AnchorAvailability.OBSERVED
                    for item in candidate_records
                ),
                "cross_camera_state_counts": _state_counts(comparisons),
                "selected_person_pair_distances_m": [
                    item.disagreement_distance_m
                    for item in comparisons
                    if item.target is PerceptionTarget.PERSON
                    and item.disagreement_distance_m is not None
                ],
                "summary": _relative(summary_path, project_root),
            },
            indent=2,
        )
    )
    return 0


def _build_selected_states(
    *,
    action: ActionDepthRunSummary,
    observation_lookup: dict[
        tuple[str, str, PerceptionTarget], VisibleSurfaceObservationRecord
    ],
    candidate_lookup: dict[
        tuple[str, str, PerceptionTarget, AnchorCandidateMethod],
        AnchorCandidateRecord,
    ],
) -> list[SelectedAnchorStateRecord]:
    states: list[SelectedAnchorStateRecord] = []
    for prediction in action.predictions:
        frames = {frame.camera_id: frame for frame in prediction.job.bundle.frames}
        for camera_id in CAMERA_IDS:
            frame = frames[camera_id]
            for target in PerceptionTarget:
                method = TRACKING_METHOD[target]
                observation = observation_lookup.get(
                    (prediction.job.job_id, camera_id, target)
                )
                if observation is None:
                    states.append(
                        SelectedAnchorStateRecord(
                            action_depth_job_id=prediction.job.job_id,
                            bundle_id=prediction.job.bundle.bundle_id,
                            frame_id=frame.frame_id,
                            source_frame_index=frame.source_frame_index,
                            capture_timestamp_seconds=frame.capture_timestamp_seconds,
                            phase_id=prediction.job.phase_id,
                            camera_id=camera_id,
                            target=target,
                            selected_method=method,
                            availability=AnchorAvailability.UNAVAILABLE,
                            unavailable_reason=(
                                AnchorUnavailableReason.SOURCE_OBSERVATION_UNAVAILABLE
                            ),
                            source_observation_id=None,
                            source_candidate_id=None,
                            anchor_world_xyz_m=None,
                            coordinate_semantics=(
                                "No exact current-frame raw visible-surface observation."
                            ),
                        )
                    )
                    continue
                candidate = candidate_lookup[
                    (prediction.job.job_id, camera_id, target, method)
                ]
                states.append(
                    SelectedAnchorStateRecord(
                        action_depth_job_id=prediction.job.job_id,
                        bundle_id=prediction.job.bundle.bundle_id,
                        frame_id=frame.frame_id,
                        source_frame_index=frame.source_frame_index,
                        capture_timestamp_seconds=frame.capture_timestamp_seconds,
                        phase_id=prediction.job.phase_id,
                        camera_id=camera_id,
                        target=target,
                        selected_method=method,
                        availability=candidate.availability,
                        unavailable_reason=candidate.unavailable_reason,
                        source_observation_id=observation.observation_id,
                        source_candidate_id=(
                            candidate.candidate_id
                            if candidate.availability is AnchorAvailability.OBSERVED
                            else None
                        ),
                        anchor_world_xyz_m=candidate.world_xyz_m,
                        coordinate_semantics=candidate.coordinate_semantics,
                    )
                )
    return states


def _build_cross_camera_comparisons(
    *,
    action: ActionDepthRunSummary,
    states: list[SelectedAnchorStateRecord],
    maximum_disagreement_m: float,
) -> list[CrossCameraAnchorComparison]:
    lookup = {
        (item.action_depth_job_id, item.camera_id, item.target): item
        for item in states
    }
    comparisons: list[CrossCameraAnchorComparison] = []
    for prediction in action.predictions:
        for target in PerceptionTarget:
            camera_a = lookup[(prediction.job.job_id, "camera_a", target)]
            camera_b = lookup[(prediction.job.job_id, "camera_b", target)]
            observed = [
                item
                for item in (camera_a, camera_b)
                if item.availability is AnchorAvailability.OBSERVED
            ]
            distance: float | None = None
            eligible = False
            if len(observed) == 2:
                assert camera_a.anchor_world_xyz_m is not None
                assert camera_b.anchor_world_xyz_m is not None
                distance = float(
                    np.linalg.norm(
                        np.asarray(camera_a.anchor_world_xyz_m)
                        - np.asarray(camera_b.anchor_world_xyz_m)
                    )
                )
                eligible = distance <= maximum_disagreement_m
                state = (
                    CrossCameraAnchorState.PAIRED_ELIGIBLE
                    if eligible
                    else CrossCameraAnchorState.PAIRED_DISAGREEMENT
                )
            elif len(observed) == 1:
                state = CrossCameraAnchorState.SINGLE_CAMERA
            else:
                state = CrossCameraAnchorState.UNAVAILABLE
            comparisons.append(
                CrossCameraAnchorComparison(
                    action_depth_job_id=prediction.job.job_id,
                    bundle_id=prediction.job.bundle.bundle_id,
                    source_frame_index=(
                        prediction.job.bundle.frames[0].source_frame_index
                    ),
                    phase_id=prediction.job.phase_id,
                    target=target,
                    selected_method=TRACKING_METHOD[target],
                    camera_a_availability=camera_a.availability,
                    camera_b_availability=camera_b.availability,
                    state=state,
                    disagreement_distance_m=distance,
                    maximum_eligible_disagreement_m=maximum_disagreement_m,
                    eligible_for_fusion=eligible,
                )
            )
    return comparisons


def _candidate_metrics(records: list[AnchorCandidateRecord]) -> dict[str, Any]:
    by_method: dict[AnchorCandidateMethod, list[AnchorCandidateRecord]] = defaultdict(list)
    for record in records:
        by_method[record.method].append(record)
    metrics: dict[str, Any] = {}
    for method, method_records in by_method.items():
        observed = [
            item for item in method_records if item.availability is AnchorAvailability.OBSERVED
        ]
        heights = [
            item.measured_support_world_z_m
            for item in observed
            if item.measured_support_world_z_m is not None
        ]
        paired_distances = _paired_candidate_distances(observed)
        method_metrics: dict[str, Any] = {
            "target": method_records[0].target.value,
            "record_count": len(method_records),
            "observed_count": len(observed),
            "unavailable_count": len(method_records) - len(observed),
            "inside_room_count": sum(item.inside_room_bounds is True for item in observed),
            "outside_room_count": sum(item.inside_room_bounds is False for item in observed),
            "support_count_minimum": min(
                item.support_sample_count for item in method_records
            ),
            "paired_distance_m": _distribution(paired_distances),
            "paired_distance_by_source_frame_m": (
                _paired_candidate_distance_by_frame(observed)
            ),
            "measured_support_world_z_m": _distribution(heights),
        }
        if method_records[0].target is PerceptionTarget.BACKPACK:
            method_metrics["pickup_stationary_maximum_pair_distance_m"] = (
                _maximum_phase_distance(observed, {204, 330, 408})
            )
            method_metrics["placed_pair_distance_m"] = _maximum_phase_distance(
                observed, {780, 858}
            )
        metrics[method.value] = method_metrics
    return metrics


def _selected_policy(
    *,
    config: AnchorEvaluationConfig,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    ground_metrics = cast(
        dict[str, Any],
        metrics[AnchorCandidateMethod.PERSON_VALIDATED_GROUND_CONTACT.value],
    )
    floor_ray_metrics = cast(
        dict[str, Any],
        metrics[
            AnchorCandidateMethod.PERSON_BOTTOM_IMAGE_QUINTILE_FLOOR_INTERSECTION.value
        ],
    )
    backpack_metrics = cast(
        dict[str, Any],
        metrics[AnchorCandidateMethod.BACKPACK_WORLD_COMPONENT_MEDIAN.value],
    )
    person_metrics = cast(
        dict[str, Any],
        metrics[AnchorCandidateMethod.PERSON_LOWEST_WORLD_Z_QUINTILE.value],
    )
    return {
        "policy_id": config.policy_id,
        "person_tracking_anchor": {
            "method": AnchorCandidateMethod.PERSON_LOWEST_WORLD_Z_QUINTILE.value,
            "semantics": "Measured robust lower-body surface anchor; not ground contact.",
            "minimum_support_count": config.minimum_candidate_support_count,
        },
        "person_ground_contact": {
            "method": AnchorCandidateMethod.PERSON_VALIDATED_GROUND_CONTACT.value,
            "maximum_measured_support_height_m": (
                config.maximum_ground_support_height_m
            ),
            "observed_count": ground_metrics["observed_count"],
            "unavailable_without_placeholder_count": ground_metrics[
                "unavailable_count"
            ],
            "semantics": "Separate derived floor contact only with near-floor evidence.",
        },
        "person_floor_ray_candidate": {
            "method": (
                AnchorCandidateMethod.PERSON_BOTTOM_IMAGE_QUINTILE_FLOOR_INTERSECTION.value
            ),
            "selected": False,
            "outside_room_count": floor_ray_metrics["outside_room_count"],
            "reason": (
                "Direct bottom-ray floor intersections leave room bounds in one "
                "retained view and do not improve paired consistency reliably."
            ),
        },
        "backpack_tracking_anchor": {
            "method": AnchorCandidateMethod.BACKPACK_WORLD_COMPONENT_MEDIAN.value,
            "semantics": "Robust centre of the visible backpack cluster, not hidden centroid.",
            "pickup_stationary_maximum_pair_distance_m": backpack_metrics[
                "pickup_stationary_maximum_pair_distance_m"
            ],
            "placed_pair_distance_m": backpack_metrics["placed_pair_distance_m"],
        },
        "cross_camera_disagreement": {
            "maximum_eligible_distance_m": (
                config.maximum_cross_camera_disagreement_m
            ),
            "selected_person_evidence_by_source_frame_m": person_metrics[
                "paired_distance_by_source_frame_m"
            ],
            "prototype_threshold_rationale": (
                "The bounded evidence has one close pair at 0.231 m and the next "
                "closest pair at 0.474 m; 0.35 m separates them without claiming a "
                "production-calibrated tolerance."
            ),
            "paired_above_threshold_behavior": "disagreement_no_fusion",
            "single_camera_behavior": "retain_single_camera_anchor_with_provenance",
            "both_unavailable_behavior": "unavailable_without_xyz",
        },
        "camera_fusion_performed": False,
    }


def _paired_candidate_distances(records: list[AnchorCandidateRecord]) -> list[float]:
    grouped: dict[
        tuple[str, PerceptionTarget], list[AnchorCandidateRecord]
    ] = defaultdict(list)
    for record in records:
        grouped[(record.action_depth_job_id, record.target)].append(record)
    distances: list[float] = []
    for paired in grouped.values():
        if len(paired) != 2:
            continue
        assert paired[0].world_xyz_m is not None and paired[1].world_xyz_m is not None
        distances.append(
            float(
                np.linalg.norm(
                    np.asarray(paired[0].world_xyz_m)
                    - np.asarray(paired[1].world_xyz_m)
                )
            )
        )
    return distances


def _paired_candidate_distance_by_frame(
    records: list[AnchorCandidateRecord],
) -> dict[str, float]:
    grouped: dict[str, list[AnchorCandidateRecord]] = defaultdict(list)
    for record in records:
        grouped[record.action_depth_job_id].append(record)
    distances: dict[str, float] = {}
    for paired in grouped.values():
        if len(paired) != 2:
            continue
        assert paired[0].world_xyz_m is not None and paired[1].world_xyz_m is not None
        distances[str(paired[0].source_frame_index)] = float(
            np.linalg.norm(
                np.asarray(paired[0].world_xyz_m)
                - np.asarray(paired[1].world_xyz_m)
            )
        )
    return dict(sorted(distances.items(), key=lambda item: int(item[0])))


def _maximum_phase_distance(
    records: list[AnchorCandidateRecord], frame_indices: set[int]
) -> float | None:
    points = [
        np.asarray(item.world_xyz_m)
        for item in records
        if item.source_frame_index in frame_indices and item.world_xyz_m is not None
    ]
    if len(points) < 2:
        return None
    return float(
        max(
            np.linalg.norm(left - right)
            for index, left in enumerate(points)
            for right in points[index + 1 :]
        )
    )


def _distribution(values: Sequence[float | None]) -> dict[str, Any] | None:
    finite = np.asarray([item for item in values if item is not None], dtype=np.float64)
    if len(finite) == 0:
        return None
    return {
        "count": len(finite),
        "minimum": float(np.min(finite)),
        "median": float(np.median(finite)),
        "maximum": float(np.max(finite)),
    }


def _save_candidate_csv(records: list[AnchorCandidateRecord], path: Path) -> None:
    fields = (
        "candidate_id",
        "source_observation_id",
        "source_frame_index",
        "phase_id",
        "camera_id",
        "target",
        "method",
        "availability",
        "unavailable_reason",
        "support_sample_count",
        "support_fraction",
        "world_x_m",
        "world_y_m",
        "world_z_m",
        "measured_support_world_z_m",
        "inside_room_bounds",
        "selected_for_tracking",
        "selected_for_ground_contact",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            xyz = record.world_xyz_m or (None, None, None)
            writer.writerow(
                {
                    "candidate_id": record.candidate_id,
                    "source_observation_id": record.source_observation_id,
                    "source_frame_index": record.source_frame_index,
                    "phase_id": record.phase_id,
                    "camera_id": record.camera_id,
                    "target": record.target.value,
                    "method": record.method.value,
                    "availability": record.availability.value,
                    "unavailable_reason": (
                        record.unavailable_reason.value
                        if record.unavailable_reason is not None
                        else None
                    ),
                    "support_sample_count": record.support_sample_count,
                    "support_fraction": record.support_fraction,
                    "world_x_m": xyz[0],
                    "world_y_m": xyz[1],
                    "world_z_m": xyz[2],
                    "measured_support_world_z_m": (
                        record.measured_support_world_z_m
                    ),
                    "inside_room_bounds": record.inside_room_bounds,
                    "selected_for_tracking": record.selected_for_tracking,
                    "selected_for_ground_contact": (
                        record.selected_for_ground_contact
                    ),
                }
            )


def _save_candidate_comparison(
    *,
    records: list[AnchorCandidateRecord],
    metrics: dict[str, Any],
    path: Path,
) -> None:
    person_methods = [
        method for method in AnchorCandidateMethod if method.value.startswith("person_")
    ]
    backpack_methods = [
        method for method in AnchorCandidateMethod if method.value.startswith("backpack_")
    ]
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    labels = [_short_label(method) for method in person_methods]
    observed = [int(metrics[method.value]["observed_count"]) for method in person_methods]
    inside = [int(metrics[method.value]["inside_room_count"]) for method in person_methods]
    x = np.arange(len(person_methods))
    axes[0, 0].bar(x - 0.18, observed, width=0.36, label="available", color="#00a6c7")
    axes[0, 0].bar(x + 0.18, inside, width=0.36, label="inside bounds", color="#7ad7e8")
    axes[0, 0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0, 0].set_ylabel("observation count")
    axes[0, 0].set_title("Person candidate availability and room bounds")
    axes[0, 0].legend()

    for method in person_methods:
        distances = _paired_candidate_distances(
            [
                record
                for record in records
                if record.method is method
                and record.availability is AnchorAvailability.OBSERVED
            ]
        )
        if distances:
            axes[0, 1].plot(
                range(1, len(distances) + 1),
                distances,
                marker="o",
                label=_short_label(method),
            )
    axes[0, 1].axhline(0.35, color="red", linestyle="--", label="0.35 m eligibility")
    axes[0, 1].set_xlabel("paired person observation")
    axes[0, 1].set_ylabel("camera disagreement (m)")
    axes[0, 1].set_title("Paired-camera disagreement by person candidate")
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(alpha=0.25)

    pickup = [
        metrics[method.value]["pickup_stationary_maximum_pair_distance_m"]
        for method in backpack_methods
    ]
    placed = [metrics[method.value]["placed_pair_distance_m"] for method in backpack_methods]
    x = np.arange(len(backpack_methods))
    axes[1, 0].bar(x - 0.18, pickup, width=0.36, label="pickup stationary max")
    axes[1, 0].bar(x + 0.18, placed, width=0.36, label="placed pair")
    axes[1, 0].set_xticks(
        x, [_short_label(method) for method in backpack_methods], rotation=25, ha="right"
    )
    axes[1, 0].set_ylabel("distance (m)")
    axes[1, 0].set_title("Backpack repeatability evidence")
    axes[1, 0].legend(fontsize=8)

    person_selected = [
        record
        for record in records
        if record.method is AnchorCandidateMethod.PERSON_LOWEST_WORLD_Z_QUINTILE
    ]
    ground = [
        record
        for record in records
        if record.method is AnchorCandidateMethod.PERSON_VALIDATED_GROUND_CONTACT
    ]
    axes[1, 1].scatter(
        [record.source_frame_index for record in person_selected],
        [record.measured_support_world_z_m for record in person_selected],
        c=[
            "#377eb8" if record.camera_id == "camera_a" else "#4daf4a"
            for record in person_selected
        ],
        label="measured lower-quintile height",
    )
    axes[1, 1].axhline(0.35, color="red", linestyle="--", label="ground evidence limit")
    unavailable_frames = [
        record.source_frame_index
        for record in ground
        if record.availability is AnchorAvailability.UNAVAILABLE
    ]
    axes[1, 1].scatter(
        unavailable_frames,
        [0.35] * len(unavailable_frames),
        marker="x",
        c="red",
        label="ground contact unavailable",
    )
    axes[1, 1].set_xlabel("source frame index")
    axes[1, 1].set_ylabel("measured support world Z (m)")
    axes[1, 1].set_title("Person ground-contact evidence gate")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.25)
    figure.suptitle("S04 target-anchor candidate evaluation — no camera fusion")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_world_preview(
    *,
    states: list[SelectedAnchorStateRecord],
    candidates: list[AnchorCandidateRecord],
    raw_observations: tuple[VisibleSurfaceObservationRecord, ...],
    room_bounds: dict[str, Any],
    zones: dict[str, Any],
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
    for target in PerceptionTarget:
        raw = np.asarray(
            [
                record.aggregate_world_xyz_m
                for record in raw_observations
                if record.target is target
            ]
        )
        axis.scatter(
            raw[:, 0],
            raw[:, 1],
            s=22,
            c=TARGET_COLOR[target],
            alpha=0.18,
            marker="x",
            label=f"{target.value} D031 raw surface",
        )
        selected = [
            state
            for state in states
            if state.target is target
            and state.availability is AnchorAvailability.OBSERVED
        ]
        for camera_id, marker in (("camera_a", "o"), ("camera_b", "^")):
            camera_states = [item for item in selected if item.camera_id == camera_id]
            xyz = np.asarray([item.anchor_world_xyz_m for item in camera_states])
            axis.scatter(
                xyz[:, 0],
                xyz[:, 1],
                s=55,
                c=TARGET_COLOR[target],
                marker=marker,
                edgecolors="black",
                linewidths=0.5,
                label=f"{target.value} selected · {camera_id}",
            )
            for item, point in zip(camera_states, xyz, strict=True):
                axis.annotate(
                    str(item.source_frame_index),
                    (point[0], point[1]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                )
    ground = [
        record
        for record in candidates
        if record.method is AnchorCandidateMethod.PERSON_VALIDATED_GROUND_CONTACT
        and record.availability is AnchorAvailability.OBSERVED
    ]
    ground_xyz = np.asarray([record.world_xyz_m for record in ground])
    axis.scatter(
        ground_xyz[:, 0],
        ground_xyz[:, 1],
        s=85,
        facecolors="none",
        edgecolors="#ff8c00",
        marker="*",
        linewidths=1.3,
        label="validated person ground contact",
    )
    for label, zone, color in (
        ("pickup", cast(dict[str, Any], zones["pickup_blue_bed"]), "#1677ff"),
        ("drop-off", cast(dict[str, Any], zones["dropoff_white_floor"]), "#777777"),
    ):
        centre = np.asarray(zone["center_world_m"], dtype=np.float64)
        axis.add_patch(
            Circle(
                (centre[0], centre[1]),
                float(zone["radius_m"]),
                fill=False,
                color=color,
                linewidth=1.5,
            )
        )
        axis.scatter(centre[0], centre[1], marker="s", c=color, s=60, label=label)
    minimum = np.asarray(room_bounds["minimum_world_xyz_m"], dtype=np.float64)
    maximum = np.asarray(room_bounds["maximum_world_xyz_m"], dtype=np.float64)
    axis.set_xlim(minimum[0], maximum[0])
    axis.set_ylim(minimum[1], maximum[1])
    axis.set_aspect("equal")
    axis.set_xlabel("world X (m)")
    axis.set_ylabel("world Y (m)")
    axis.set_title(
        "Selected measured anchors and validated ground contacts\n"
        "frame labels shown · no fusion or temporal filling"
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, loc="upper right")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _short_label(method: AnchorCandidateMethod) -> str:
    replacements = {
        AnchorCandidateMethod.PERSON_RAW_VISIBLE_SURFACE_REFERENCE: "raw reference",
        AnchorCandidateMethod.PERSON_LOWEST_WORLD_Z_DECILE: "low-Z 10%",
        AnchorCandidateMethod.PERSON_LOWEST_WORLD_Z_QUINTILE: "low-Z 20%",
        AnchorCandidateMethod.PERSON_BOTTOM_IMAGE_QUINTILE: "bottom-image 20%",
        AnchorCandidateMethod.PERSON_BOTTOM_IMAGE_QUINTILE_FLOOR_INTERSECTION: "floor ray",
        AnchorCandidateMethod.PERSON_VALIDATED_GROUND_CONTACT: "validated ground",
        AnchorCandidateMethod.BACKPACK_RAW_VISIBLE_SURFACE_REFERENCE: "raw reference",
        AnchorCandidateMethod.BACKPACK_WORLD_COMPONENT_MEDIAN: "world median",
        AnchorCandidateMethod.BACKPACK_TRIMMED_BOUNDS_CENTER: "trimmed-box centre",
        AnchorCandidateMethod.BACKPACK_TRIMMED_MEAN: "trimmed mean",
    }
    return replacements[method]


def _state_counts(comparisons: list[CrossCameraAnchorComparison]) -> dict[str, int]:
    return {
        state.value: sum(item.state is state for item in comparisons)
        for state in CrossCameraAnchorState
    }


def _verify_prerequisite(
    *,
    verification: dict[str, Any],
    surface_path: Path,
    expected_observation_count: int,
) -> None:
    required = (
        verification.get("stage") == "S04",
        verification.get("status") == "passed",
        verification.get("purpose")
        == "exact_frame_per_camera_raw_visible_surface_verification",
        verification.get("source_summary_sha256") == _sha256(surface_path),
        verification.get("observation_count") == expected_observation_count == 20,
        verification.get("complete_aligned_mask_coverage") is True,
        verification.get("all_samples_regenerated_from_d030") is True,
        verification.get("track_anchor_derived") is False,
        verification.get("cross_camera_fusion_performed") is False,
    )
    if not all(required):
        raise ValueError("verified raw visible-surface prerequisite did not pass")


def _verify_sample_identity(
    arrays: Any, observation: VisibleSurfaceObservationRecord
) -> None:
    expected = {
        "observation_id": observation.observation_id,
        "action_depth_job_id": observation.action_depth_job_id,
        "bundle_id": observation.bundle_id,
        "frame_id": observation.frame_id,
        "camera_id": observation.camera_id,
        "target": observation.target.value,
    }
    for name, value in expected.items():
        if str(arrays[name].item()) != value:
            raise ValueError(f"sample-cloud {name} differs from source summary")
    if bool(arrays["camera_fusion_performed"].item()):
        raise ValueError("source sample cloud unexpectedly claims camera fusion")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return cast(dict[str, Any], value)


def _require_hash(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"artifact hash changed for {path}: {actual} != {expected}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _relative(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
