"""Verify S04 D032-gated cross-camera observations and visual QA."""

from __future__ import annotations

import argparse
import csv
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
    AnchorCandidateRecord,
    AnchorEvaluationRunSummary,
    CrossCameraAnchorState,
    CrossCameraFusionRunSummary,
    CrossCameraObservationRecord,
    CrossCameraObservationState,
    FusionSourceMeasurement,
    SelectedAnchorStateRecord,
    VisibleSurfaceObservationRecord,
    VisibleSurfaceRunSummary,
    reliability_score,
    resolve_cross_camera_observation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/s04/cross_camera_observations_20260802/summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/s04/cross_camera_observations_20260802/verification.json"
        ),
    )
    parser.add_argument("--visual-qa-passed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.visual_qa_passed:
        raise ValueError("cross-camera verification requires explicit visual QA")
    project_root = args.project_root.resolve()
    summary_path = _resolve(project_root, args.summary)
    output_path = _resolve(project_root, args.output)
    if output_path.exists():
        raise FileExistsError(f"verification output already exists: {output_path}")

    summary = CrossCameraFusionRunSummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    anchor_path = _resolve(
        project_root, Path(summary.source_anchor_evaluation_summary_ref)
    )
    anchor_verification_path = _resolve(
        project_root, Path(summary.source_anchor_evaluation_verification_ref)
    )
    surface_path = _resolve(
        project_root, Path(summary.source_visible_surface_summary_ref)
    )
    _require_hash(anchor_path, summary.source_anchor_evaluation_summary_sha256)
    _require_hash(
        anchor_verification_path,
        summary.source_anchor_evaluation_verification_sha256,
    )
    _require_hash(surface_path, summary.source_visible_surface_summary_sha256)
    csv_path = _resolve(project_root, Path(summary.observation_csv_ref))
    _require_hash(csv_path, summary.observation_csv_sha256)
    _require_hash(
        _resolve(project_root, Path(summary.reliability_diagnostic_ref)),
        summary.reliability_diagnostic_sha256,
    )
    _require_hash(
        _resolve(project_root, Path(summary.world_preview_ref)),
        summary.world_preview_sha256,
    )

    anchor = AnchorEvaluationRunSummary.model_validate_json(
        anchor_path.read_text(encoding="utf-8")
    )
    anchor_verification = _read_object(anchor_verification_path)
    surface = VisibleSurfaceRunSummary.model_validate_json(
        surface_path.read_text(encoding="utf-8")
    )
    action_path = _resolve(project_root, Path(anchor.source_action_depth_summary_ref))
    _require_hash(action_path, anchor.source_action_depth_summary_sha256)
    action = ActionDepthRunSummary.model_validate_json(
        action_path.read_text(encoding="utf-8")
    )
    _verify_anchor_prerequisite(
        verification=anchor_verification,
        anchor_path=anchor_path,
    )
    if (
        summary.configuration.anchor_policy_id
        != anchor.configuration.policy_id
        or summary.configuration.maximum_cross_camera_disagreement_m
        != anchor.configuration.maximum_cross_camera_disagreement_m
    ):
        raise ValueError("cross-camera configuration differs from D032")

    state_lookup: dict[
        tuple[str, str, PerceptionTarget], SelectedAnchorStateRecord
    ] = {
        (item.action_depth_job_id, item.camera_id, item.target): item
        for item in anchor.selected_anchor_states
    }
    candidate_lookup = {item.candidate_id: item for item in anchor.candidate_records}
    surface_lookup = {item.observation_id: item for item in surface.observations}
    comparison_lookup = {
        (item.action_depth_job_id, item.target): item
        for item in anchor.cross_camera_comparisons
    }
    prediction_lookup = {
        item.job.job_id: item for item in action.predictions
    }
    record_lookup = {
        (item.action_depth_job_id, item.target): item for item in summary.observations
    }
    if set(record_lookup) != set(comparison_lookup):
        raise ValueError("cross-camera outputs do not cover D032 comparisons")

    for key, comparison in comparison_lookup.items():
        record = record_lookup[key]
        prediction = prediction_lookup[record.action_depth_job_id]
        frame_lookup = {
            frame.camera_id: frame for frame in prediction.job.bundle.frames
        }
        states = (
            state_lookup[(record.action_depth_job_id, "camera_a", record.target)],
            state_lookup[(record.action_depth_job_id, "camera_b", record.target)],
        )
        measurements = tuple(
            _measurement_from_state(
                state=state,
                candidate_lookup=candidate_lookup,
                surface_lookup=surface_lookup,
            )
            for state in states
        )
        regenerated = resolve_cross_camera_observation(
            sources=(measurements[0], measurements[1]),
            maximum_disagreement_m=(
                summary.configuration.maximum_cross_camera_disagreement_m
            ),
        )
        _verify_record(
            record=record,
            regenerated=regenerated,
            comparison_state=comparison.state,
            measurements=(measurements[0], measurements[1]),
        )
        if (
            record.bundle_id != prediction.job.bundle.bundle_id
            or record.camera_a_frame_id != frame_lookup["camera_a"].frame_id
            or record.camera_b_frame_id != frame_lookup["camera_b"].frame_id
            or record.source_frame_index
            != prediction.job.bundle.frames[0].source_frame_index
            or record.phase_id != prediction.job.phase_id
        ):
            raise ValueError("cross-camera job/frame provenance differs")
        times = [
            frame.capture_timestamp_seconds for frame in prediction.job.bundle.frames
        ]
        if not np.isclose(
            record.maximum_source_time_difference_seconds,
            max(times) - min(times),
            atol=1e-12,
        ):
            raise ValueError("cross-camera source time difference differs")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != len(summary.observations) or {
        row["observation_id"] for row in csv_rows
    } != {item.observation_id for item in summary.observations}:
        raise ValueError("cross-camera CSV coverage differs from summary")

    state_counts = Counter(item.state for item in summary.observations)
    fused = [
        item
        for item in summary.observations
        if item.state is CrossCameraObservationState.FUSED
    ]
    single = [
        item
        for item in summary.observations
        if item.state is CrossCameraObservationState.SINGLE_CAMERA
    ]
    disagreement = [
        item
        for item in summary.observations
        if item.state is CrossCameraObservationState.DISAGREEMENT
    ]
    unavailable = [
        item
        for item in summary.observations
        if item.state is CrossCameraObservationState.UNAVAILABLE
    ]
    (fused_record,) = fused
    verification = {
        "schema_version": 1,
        "stage": "S04",
        "status": "passed",
        "purpose": "selected_anchor_cross_camera_observation_verification",
        "source_summary_ref": _relative(summary_path, project_root),
        "source_summary_sha256": _sha256(summary_path),
        "schema_round_trip_passed": (
            CrossCameraFusionRunSummary.model_validate_json(summary.model_dump_json())
            == summary
        ),
        "observation_count": len(summary.observations),
        "state_counts": {
            state.value: state_counts[state] for state in CrossCameraObservationState
        },
        "world_xyz_observation_count": sum(
            item.world_xyz_m is not None for item in summary.observations
        ),
        "all_outputs_regenerated_from_selected_anchors": True,
        "capture_order_passed": [
            item.source_frame_index for item in summary.observations
        ]
        == sorted(item.source_frame_index for item in summary.observations),
        "maximum_source_time_difference_seconds": max(
            item.maximum_source_time_difference_seconds
            for item in summary.observations
        ),
        "fused_source_frame_index": fused_record.source_frame_index,
        "fused_target": fused_record.target.value,
        "fused_disagreement_distance_m": fused_record.disagreement_distance_m,
        "fused_weights": {
            source.camera_id: source.contribution_weight
            for source in fused_record.sources
        },
        "fused_world_xyz_m": fused_record.world_xyz_m,
        "single_camera_passthrough_count": len(single),
        "disagreement_without_xyz_count": sum(
            item.world_xyz_m is None for item in disagreement
        ),
        "unavailable_without_xyz_count": sum(
            item.world_xyz_m is None for item in unavailable
        ),
        "all_emitted_xyz_inside_room_bounds": all(
            item.inside_room_bounds is True
            for item in summary.observations
            if item.world_xyz_m is not None
        ),
        "reliability_formula": (
            "sqrt(anchor_support_count) * retained_confidence_median / "
            "(1 + retained_depth_relative_mad)"
        ),
        "visual_qa": {
            "status": "passed",
            "reliability_diagnostic_ref": summary.reliability_diagnostic_ref,
            "world_preview_ref": summary.world_preview_ref,
            "finding": (
                "The single eligible pair is fused inside its two anchors, "
                "single-camera observations preserve pickup-to-drop-off motion, and "
                "the three disagreement pairs remain visible without combined XYZ."
            ),
        },
        "temporal_filling_performed": False,
        "presentation_smoothing_performed": False,
    }
    expected_weights = {
        "camera_a": 0.652162061644496,
        "camera_b": 0.3478379383555041,
    }
    required = (
        verification["schema_round_trip_passed"],
        verification["observation_count"] == 16,
        state_counts
        == Counter(
            {
                CrossCameraObservationState.FUSED: 1,
                CrossCameraObservationState.SINGLE_CAMERA: 12,
                CrossCameraObservationState.DISAGREEMENT: 3,
            }
        ),
        verification["world_xyz_observation_count"] == 13,
        verification["capture_order_passed"],
        verification["fused_source_frame_index"] == 204,
        verification["fused_target"] == "person",
        all(
            np.isclose(
                cast(dict[str, float], verification["fused_weights"])[camera_id],
                weight,
                atol=1e-12,
            )
            for camera_id, weight in expected_weights.items()
        ),
        verification["single_camera_passthrough_count"] == 12,
        verification["disagreement_without_xyz_count"] == 3,
        verification["unavailable_without_xyz_count"] == 0,
        verification["all_emitted_xyz_inside_room_bounds"],
        not verification["temporal_filling_performed"],
        not verification["presentation_smoothing_performed"],
    )
    if not all(required):
        raise RuntimeError("S04 cross-camera observation verification did not pass")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))
    return 0


def _measurement_from_state(
    *,
    state: SelectedAnchorStateRecord,
    candidate_lookup: dict[str, AnchorCandidateRecord],
    surface_lookup: dict[str, VisibleSurfaceObservationRecord],
) -> FusionSourceMeasurement:
    if state.availability is AnchorAvailability.UNAVAILABLE:
        return FusionSourceMeasurement(
            camera_id=state.camera_id,
            availability=state.availability,
            unavailable_reason=state.unavailable_reason,
            source_observation_id=None,
            source_candidate_id=None,
            anchor_world_xyz_m=None,
            support_sample_count=None,
            retained_confidence_median=None,
            retained_depth_median_m=None,
            retained_depth_mad_m=None,
        )
    assert state.source_candidate_id is not None
    assert state.source_observation_id is not None
    assert state.anchor_world_xyz_m is not None
    candidate = candidate_lookup[state.source_candidate_id]
    surface = surface_lookup[state.source_observation_id]
    return FusionSourceMeasurement(
        camera_id=state.camera_id,
        availability=state.availability,
        unavailable_reason=None,
        source_observation_id=state.source_observation_id,
        source_candidate_id=state.source_candidate_id,
        anchor_world_xyz_m=state.anchor_world_xyz_m,
        support_sample_count=candidate.support_sample_count,
        retained_confidence_median=surface.retained_confidence.median,
        retained_depth_median_m=surface.retained_depth_m.median,
        retained_depth_mad_m=surface.retained_depth_m.median_absolute_deviation,
    )


def _verify_record(
    *,
    record: CrossCameraObservationRecord,
    regenerated: Any,
    comparison_state: CrossCameraAnchorState,
    measurements: tuple[FusionSourceMeasurement, FusionSourceMeasurement],
) -> None:
    expected_state = {
        CrossCameraAnchorState.PAIRED_ELIGIBLE: CrossCameraObservationState.FUSED,
        CrossCameraAnchorState.PAIRED_DISAGREEMENT: (
            CrossCameraObservationState.DISAGREEMENT
        ),
        CrossCameraAnchorState.SINGLE_CAMERA: (
            CrossCameraObservationState.SINGLE_CAMERA
        ),
        CrossCameraAnchorState.UNAVAILABLE: CrossCameraObservationState.UNAVAILABLE,
    }[comparison_state]
    if (
        regenerated.state is not expected_state
        or record.state is not regenerated.state
        or record.combination_method is not regenerated.combination_method
        or record.disagreement_distance_m != regenerated.disagreement_distance_m
        or record.world_xyz_m != regenerated.world_xyz_m
        or record.camera_fusion_performed != regenerated.camera_fusion_performed
    ):
        raise ValueError("stored cross-camera result differs from regeneration")
    for index, source in enumerate(record.sources):
        measurement = measurements[index]
        if (
            source.camera_id != measurement.camera_id
            or source.availability is not measurement.availability
            or source.source_observation_id != measurement.source_observation_id
            or source.source_candidate_id != measurement.source_candidate_id
            or source.anchor_world_xyz_m != measurement.anchor_world_xyz_m
            or source.support_sample_count != measurement.support_sample_count
            or source.retained_confidence_median
            != measurement.retained_confidence_median
            or source.retained_depth_median_m != measurement.retained_depth_median_m
            or source.retained_depth_mad_m != measurement.retained_depth_mad_m
            or source.contribution_weight
            != regenerated.contribution_weights[index]
        ):
            raise ValueError("stored fusion source evidence differs")
        if measurement.availability is AnchorAvailability.OBSERVED:
            assert measurement.support_sample_count is not None
            assert measurement.retained_confidence_median is not None
            assert measurement.retained_depth_median_m is not None
            assert measurement.retained_depth_mad_m is not None
            expected_score = reliability_score(
                support_sample_count=measurement.support_sample_count,
                retained_confidence_median=measurement.retained_confidence_median,
                retained_depth_median_m=measurement.retained_depth_median_m,
                retained_depth_mad_m=measurement.retained_depth_mad_m,
            )
            if not np.isclose(
                cast(float, source.reliability_score), expected_score, atol=1e-12
            ):
                raise ValueError("stored reliability score differs")
        elif source.reliability_score is not None:
            raise ValueError("unavailable source carries reliability score")


def _verify_anchor_prerequisite(
    *, verification: dict[str, Any], anchor_path: Path
) -> None:
    if not all(
        (
            verification.get("status") == "passed",
            verification.get("source_summary_sha256") == _sha256(anchor_path),
            verification.get("candidate_record_count") == 104,
            verification.get("selected_anchor_state_count") == 32,
            verification.get("maximum_eligible_disagreement_m") == 0.35,
            verification.get("camera_fusion_performed") is False,
        )
    ):
        raise ValueError("D032 anchor prerequisite differs")


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
