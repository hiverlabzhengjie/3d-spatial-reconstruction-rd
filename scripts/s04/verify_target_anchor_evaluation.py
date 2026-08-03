"""Verify S04 target-anchor evaluation and recorded visual QA."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np

from spatial_reconstruction.contracts import PerceptionTarget
from spatial_reconstruction.localization import (
    ActionDepthRunSummary,
    AnchorAvailability,
    AnchorCandidateMethod,
    AnchorCandidateRecord,
    AnchorEvaluationRunSummary,
    AnchorUnavailableReason,
    CrossCameraAnchorComparison,
    CrossCameraAnchorState,
    SelectedAnchorStateRecord,
    VisibleSurfaceObservationRecord,
    VisibleSurfaceRunSummary,
    evaluate_anchor_candidates,
)

TRACKING_METHOD = {
    PerceptionTarget.PERSON: AnchorCandidateMethod.PERSON_LOWEST_WORLD_Z_QUINTILE,
    PerceptionTarget.BACKPACK: AnchorCandidateMethod.BACKPACK_WORLD_COMPONENT_MEDIAN,
}
METHODS_BY_TARGET = {
    PerceptionTarget.PERSON: {
        method for method in AnchorCandidateMethod if method.value.startswith("person_")
    },
    PerceptionTarget.BACKPACK: {
        method
        for method in AnchorCandidateMethod
        if method.value.startswith("backpack_")
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/s04/anchor_evaluation_20260802_v2/summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/s04/anchor_evaluation_20260802_v2/verification.json"
        ),
    )
    parser.add_argument("--visual-qa-passed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.visual_qa_passed:
        raise ValueError("target-anchor verification requires explicit visual QA")
    project_root = args.project_root.resolve()
    summary_path = _resolve(project_root, args.summary)
    output_path = _resolve(project_root, args.output)
    if output_path.exists():
        raise FileExistsError(f"verification output already exists: {output_path}")

    summary = AnchorEvaluationRunSummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    surface_path = _resolve(
        project_root, Path(summary.source_visible_surface_summary_ref)
    )
    surface_verification_path = _resolve(
        project_root, Path(summary.source_visible_surface_verification_ref)
    )
    action_path = _resolve(project_root, Path(summary.source_action_depth_summary_ref))
    _require_hash(surface_path, summary.source_visible_surface_summary_sha256)
    _require_hash(
        surface_verification_path,
        summary.source_visible_surface_verification_sha256,
    )
    _require_hash(action_path, summary.source_action_depth_summary_sha256)
    _require_hash(
        _resolve(project_root, Path(summary.comparison_csv_ref)),
        summary.comparison_csv_sha256,
    )
    _require_hash(
        _resolve(project_root, Path(summary.candidate_comparison_ref)),
        summary.candidate_comparison_sha256,
    )
    _require_hash(
        _resolve(project_root, Path(summary.selected_anchor_world_preview_ref)),
        summary.selected_anchor_world_preview_sha256,
    )

    surface = VisibleSurfaceRunSummary.model_validate_json(
        surface_path.read_text(encoding="utf-8")
    )
    surface_verification = _read_object(surface_verification_path)
    action = ActionDepthRunSummary.model_validate_json(
        action_path.read_text(encoding="utf-8")
    )
    _verify_surface_prerequisite(
        verification=surface_verification,
        surface_path=surface_path,
    )
    bounds_min = np.asarray(
        surface.room_bounds_world_m["minimum_world_xyz_m"], dtype=np.float64
    )
    bounds_max = np.asarray(
        surface.room_bounds_world_m["maximum_world_xyz_m"], dtype=np.float64
    )

    candidates_by_observation: dict[str, list[AnchorCandidateRecord]] = {}
    for record in summary.candidate_records:
        candidates_by_observation.setdefault(record.source_observation_id, []).append(
            record
        )
    if set(candidates_by_observation) != {
        item.observation_id for item in surface.observations
    }:
        raise ValueError("anchor candidates do not cover raw observations")

    for observation in surface.observations:
        records = candidates_by_observation[observation.observation_id]
        if {item.method for item in records} != METHODS_BY_TARGET[observation.target]:
            raise ValueError("anchor candidate methods differ from target comparison")
        sample_path = _resolve(project_root, Path(observation.sample_cloud_ref))
        _require_hash(sample_path, observation.sample_cloud_sha256)
        with np.load(sample_path, allow_pickle=False) as arrays:
            pixels = np.asarray(arrays["pixels_uv"], dtype=np.float64)
            points_world = np.asarray(arrays["points_world_m"], dtype=np.float64)
            confidence = np.asarray(arrays["confidence"], dtype=np.float64)
        regenerated = evaluate_anchor_candidates(
            target=observation.target,
            pixels_uv=pixels,
            points_world_m=points_world,
            confidence=confidence,
            intrinsics=observation.processed_intrinsics,
            pose=observation.camera_pose,
            raw_visible_surface_world_xyz_m=observation.aggregate_world_xyz_m,
            config=summary.configuration,
        )
        stored_by_method = {item.method: item for item in records}
        for candidate in regenerated:
            stored = stored_by_method[candidate.method]
            _verify_candidate_record(
                stored=stored,
                regenerated=candidate,
                observation=observation,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
            )

    observation_lookup: dict[
        tuple[str, str, PerceptionTarget], VisibleSurfaceObservationRecord
    ] = {
        (item.action_depth_job_id, item.camera_id, item.target): item
        for item in surface.observations
    }
    candidate_lookup: dict[
        tuple[str, str, PerceptionTarget, AnchorCandidateMethod],
        AnchorCandidateRecord,
    ] = {
        (item.action_depth_job_id, item.camera_id, item.target, item.method): item
        for item in summary.candidate_records
    }
    state_lookup: dict[
        tuple[str, str, PerceptionTarget], SelectedAnchorStateRecord
    ] = {
        (item.action_depth_job_id, item.camera_id, item.target): item
        for item in summary.selected_anchor_states
    }
    expected_state_keys: set[tuple[str, str, PerceptionTarget]] = set()
    for prediction in action.predictions:
        for camera_id in ("camera_a", "camera_b"):
            for target in PerceptionTarget:
                key = (prediction.job.job_id, camera_id, target)
                expected_state_keys.add(key)
                state = state_lookup.get(key)
                if state is None:
                    raise ValueError("selected anchor state is missing")
                source_observation = observation_lookup.get(key)
                if source_observation is None:
                    if (
                        state.availability is not AnchorAvailability.UNAVAILABLE
                        or state.unavailable_reason
                        is not AnchorUnavailableReason.SOURCE_OBSERVATION_UNAVAILABLE
                        or state.anchor_world_xyz_m is not None
                    ):
                        raise ValueError("missing raw observation produced an anchor")
                else:
                    chosen_record = candidate_lookup[
                        (*key, TRACKING_METHOD[target])
                    ]
                    if (
                        state.source_observation_id
                        != source_observation.observation_id
                        or state.source_candidate_id != chosen_record.candidate_id
                        or state.anchor_world_xyz_m != chosen_record.world_xyz_m
                        or state.availability is not chosen_record.availability
                    ):
                        raise ValueError("selected anchor differs from chosen candidate")
    if set(state_lookup) != expected_state_keys:
        raise ValueError("selected anchor states differ from complete job/camera/target grid")

    comparison_lookup: dict[
        tuple[str, PerceptionTarget], CrossCameraAnchorComparison
    ] = {
        (item.action_depth_job_id, item.target): item
        for item in summary.cross_camera_comparisons
    }
    expected_comparison_keys: set[tuple[str, PerceptionTarget]] = set()
    for prediction in action.predictions:
        for target in PerceptionTarget:
            comparison_key = (prediction.job.job_id, target)
            expected_comparison_keys.add(comparison_key)
            comparison = comparison_lookup.get(comparison_key)
            if comparison is None:
                raise ValueError("cross-camera comparison is missing")
            camera_a = state_lookup[(prediction.job.job_id, "camera_a", target)]
            camera_b = state_lookup[(prediction.job.job_id, "camera_b", target)]
            _verify_comparison(
                comparison=comparison,
                camera_a=camera_a,
                camera_b=camera_b,
            )
    if set(comparison_lookup) != expected_comparison_keys:
        raise ValueError("cross-camera comparisons differ from action jobs")

    candidate_count_by_method = Counter(
        item.method for item in summary.candidate_records
    )
    ground = [
        item
        for item in summary.candidate_records
        if item.method is AnchorCandidateMethod.PERSON_VALIDATED_GROUND_CONTACT
    ]
    floor_rays = [
        item
        for item in summary.candidate_records
        if item.method
        is AnchorCandidateMethod.PERSON_BOTTOM_IMAGE_QUINTILE_FLOOR_INTERSECTION
    ]
    comparisons_by_state = Counter(
        item.state for item in summary.cross_camera_comparisons
    )
    selected_person_distances = sorted(
        item.disagreement_distance_m
        for item in summary.cross_camera_comparisons
        if item.target is PerceptionTarget.PERSON
        and item.disagreement_distance_m is not None
    )
    policy = summary.selected_policy
    required_policy = (
        policy.get("policy_id") == "s04_target_anchor_v1",
        cast(dict[str, Any], policy.get("person_tracking_anchor", {})).get("method")
        == AnchorCandidateMethod.PERSON_LOWEST_WORLD_Z_QUINTILE.value,
        cast(dict[str, Any], policy.get("backpack_tracking_anchor", {})).get(
            "method"
        )
        == AnchorCandidateMethod.BACKPACK_WORLD_COMPONENT_MEDIAN.value,
        cast(dict[str, Any], policy.get("person_floor_ray_candidate", {})).get(
            "selected"
        )
        is False,
        cast(dict[str, Any], policy.get("cross_camera_disagreement", {})).get(
            "maximum_eligible_distance_m"
        )
        == 0.35,
        policy.get("camera_fusion_performed") is False,
    )
    if not all(required_policy):
        raise ValueError("selected anchor policy differs from verified choice")

    verification = {
        "schema_version": 1,
        "stage": "S04",
        "status": "passed",
        "purpose": "target_anchor_candidate_and_disagreement_policy_verification",
        "source_summary_ref": _relative(summary_path, project_root),
        "source_summary_sha256": _sha256(summary_path),
        "schema_round_trip_passed": (
            AnchorEvaluationRunSummary.model_validate_json(summary.model_dump_json())
            == summary
        ),
        "source_visible_surface_observation_count": len(surface.observations),
        "candidate_record_count": len(summary.candidate_records),
        "candidate_count_by_method": {
            method.value: candidate_count_by_method[method]
            for method in AnchorCandidateMethod
        },
        "all_candidates_regenerated_from_source_clouds": True,
        "selected_anchor_state_count": len(summary.selected_anchor_states),
        "selected_observed_count": sum(
            item.availability is AnchorAvailability.OBSERVED
            for item in summary.selected_anchor_states
        ),
        "selected_missing_without_xyz_count": sum(
            item.availability is AnchorAvailability.UNAVAILABLE
            and item.anchor_world_xyz_m is None
            for item in summary.selected_anchor_states
        ),
        "ground_contact_observed_count": sum(
            item.availability is AnchorAvailability.OBSERVED for item in ground
        ),
        "ground_contact_unavailable_without_xyz_count": sum(
            item.availability is AnchorAvailability.UNAVAILABLE
            and item.world_xyz_m is None
            for item in ground
        ),
        "floor_ray_outside_room_count": sum(
            item.inside_room_bounds is False for item in floor_rays
        ),
        "cross_camera_comparison_count": len(summary.cross_camera_comparisons),
        "cross_camera_state_counts": {
            state.value: comparisons_by_state[state]
            for state in CrossCameraAnchorState
        },
        "selected_person_pair_distances_m": selected_person_distances,
        "maximum_eligible_disagreement_m": (
            summary.configuration.maximum_cross_camera_disagreement_m
        ),
        "visual_qa": {
            "status": "passed",
            "candidate_comparison_ref": summary.candidate_comparison_ref,
            "selected_anchor_world_preview_ref": (
                summary.selected_anchor_world_preview_ref
            ),
            "finding": (
                "The selected lower-body and backpack anchors preserve plausible "
                "pickup-to-drop-off motion; the comparison exposes elevated/hidden "
                "person ground evidence and camera disagreement without filling it."
            ),
        },
        "camera_fusion_performed": False,
        "temporal_filling_performed": False,
        "presentation_smoothing_performed": False,
    }
    required = (
        verification["schema_round_trip_passed"],
        verification["candidate_record_count"] == 104,
        verification["selected_anchor_state_count"] == 32,
        verification["selected_observed_count"] == 20,
        verification["selected_missing_without_xyz_count"] == 12,
        verification["ground_contact_observed_count"] == 6,
        verification["ground_contact_unavailable_without_xyz_count"] == 6,
        verification["floor_ray_outside_room_count"] == 1,
        verification["cross_camera_comparison_count"] == 16,
        comparisons_by_state
        == Counter(
            {
                CrossCameraAnchorState.PAIRED_ELIGIBLE: 1,
                CrossCameraAnchorState.PAIRED_DISAGREEMENT: 3,
                CrossCameraAnchorState.SINGLE_CAMERA: 12,
            }
        ),
        selected_person_distances
        == sorted(
            [
                0.23085401540156403,
                0.7586156952278071,
                0.5100211097356167,
                0.4739457577397271,
            ]
        ),
        not verification["camera_fusion_performed"],
        not verification["temporal_filling_performed"],
        not verification["presentation_smoothing_performed"],
    )
    if not all(required):
        raise RuntimeError("S04 target-anchor evaluation verification did not pass")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))
    return 0


def _verify_candidate_record(
    *,
    stored: AnchorCandidateRecord,
    regenerated: Any,
    observation: Any,
    bounds_min: np.ndarray[Any, Any],
    bounds_max: np.ndarray[Any, Any],
) -> None:
    if (
        stored.action_depth_job_id != observation.action_depth_job_id
        or stored.bundle_id != observation.bundle_id
        or stored.frame_id != observation.frame_id
        or stored.source_frame_index != observation.source_frame_index
        or stored.camera_id != observation.camera_id
        or stored.target is not observation.target
        or stored.source_sample_cloud_ref != observation.sample_cloud_ref
        or stored.source_sample_cloud_sha256 != observation.sample_cloud_sha256
        or stored.source_raw_aggregate_world_xyz_m
        != observation.aggregate_world_xyz_m
    ):
        raise ValueError("anchor candidate source provenance differs")
    if (
        stored.availability is not regenerated.availability
        or stored.unavailable_reason is not regenerated.unavailable_reason
        or stored.source_sample_count != regenerated.source_sample_count
        or stored.support_sample_count != regenerated.support_sample_count
        or stored.measured_support_world_z_m
        != regenerated.measured_support_world_z_m
        or stored.coordinate_semantics != regenerated.coordinate_semantics
    ):
        raise ValueError("stored anchor candidate differs from regeneration")
    if regenerated.world_xyz_m is None:
        if stored.world_xyz_m is not None or stored.inside_room_bounds is not None:
            raise ValueError("unavailable anchor candidate carries XYZ")
    else:
        if stored.world_xyz_m is None:
            raise ValueError("observed anchor candidate lacks XYZ")
        if not np.allclose(stored.world_xyz_m, regenerated.world_xyz_m, atol=1e-12):
            raise ValueError("stored anchor candidate XYZ differs")
        point = np.asarray(stored.world_xyz_m, dtype=np.float64)
        inside = bool(np.all((point >= bounds_min) & (point <= bounds_max)))
        if stored.inside_room_bounds != inside:
            raise ValueError("stored anchor room-bounds state differs")
    expected_tracking = stored.method is TRACKING_METHOD[stored.target]
    expected_ground = (
        stored.method is AnchorCandidateMethod.PERSON_VALIDATED_GROUND_CONTACT
    )
    if (
        stored.selected_for_tracking != expected_tracking
        or stored.selected_for_ground_contact != expected_ground
        or stored.camera_fusion_performed
        or stored.presentation_smoothing_performed
    ):
        raise ValueError("anchor candidate selection/provenance flags differ")


def _verify_comparison(*, comparison: Any, camera_a: Any, camera_b: Any) -> None:
    observed = [
        item
        for item in (camera_a, camera_b)
        if item.availability is AnchorAvailability.OBSERVED
    ]
    if len(observed) == 2:
        distance = float(
            np.linalg.norm(
                np.asarray(camera_a.anchor_world_xyz_m)
                - np.asarray(camera_b.anchor_world_xyz_m)
            )
        )
        if not np.isclose(comparison.disagreement_distance_m, distance, atol=1e-12):
            raise ValueError("cross-camera disagreement distance differs")
        eligible = distance <= comparison.maximum_eligible_disagreement_m
        expected = (
            CrossCameraAnchorState.PAIRED_ELIGIBLE
            if eligible
            else CrossCameraAnchorState.PAIRED_DISAGREEMENT
        )
        if comparison.state is not expected or comparison.eligible_for_fusion != eligible:
            raise ValueError("paired cross-camera state differs")
    elif len(observed) == 1:
        if comparison.state is not CrossCameraAnchorState.SINGLE_CAMERA:
            raise ValueError("single-camera anchor state differs")
    elif comparison.state is not CrossCameraAnchorState.UNAVAILABLE:
        raise ValueError("both-unavailable anchor state differs")
    if comparison.camera_fusion_performed:
        raise ValueError("anchor comparison unexpectedly performed fusion")


def _verify_surface_prerequisite(
    *, verification: dict[str, Any], surface_path: Path
) -> None:
    if not all(
        (
            verification.get("status") == "passed",
            verification.get("source_summary_sha256") == _sha256(surface_path),
            verification.get("observation_count") == 20,
            verification.get("all_samples_regenerated_from_d030") is True,
            verification.get("track_anchor_derived") is False,
            verification.get("cross_camera_fusion_performed") is False,
        )
    ):
        raise ValueError("raw visible-surface prerequisite differs")


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
