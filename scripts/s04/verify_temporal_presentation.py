"""Independently verify the D034 S04 temporal presentation artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from spatial_reconstruction.contracts import PerceptionTarget
from spatial_reconstruction.localization import (
    CorrectedPairObservationRecord,
    CorrectedPairState,
    CorrectedTrackingRunSummary,
    MeasuredTrajectorySegment,
    TemporalPresentationPolicy,
    TemporalPresentationRecord,
    TemporalPresentationRunSummary,
    TemporalPresentationState,
    build_measured_trajectory_segments,
    make_temporal_record,
    resolve_temporal_presentation,
)
from spatial_reconstruction.perception import PerceptionTargetFrameState

CAMERA_IDS = ("camera_a", "camera_b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-qa-passed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    summary_path = _resolve(root, args.summary)
    output_path = _resolve(root, args.output)
    if output_path.exists():
        raise FileExistsError(f"verification output already exists: {output_path}")
    if not args.visual_qa_passed:
        raise ValueError("explicit timeline and world-preview visual QA is required")
    summary = TemporalPresentationRunSummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    source_paths = {
        "corrected": _resolve(root, Path(summary.source_corrected_summary_ref)),
        "corrected_verification": _resolve(
            root, Path(summary.source_corrected_verification_ref)
        ),
        "perception": _resolve(root, Path(summary.source_perception_summary_ref)),
        "camera_a": _resolve(root, Path(summary.source_camera_a_timeline_ref)),
        "camera_b": _resolve(root, Path(summary.source_camera_b_timeline_ref)),
    }
    source_hashes = {
        "corrected": summary.source_corrected_summary_sha256,
        "corrected_verification": summary.source_corrected_verification_sha256,
        "perception": summary.source_perception_summary_sha256,
        "camera_a": summary.source_camera_a_timeline_sha256,
        "camera_b": summary.source_camera_b_timeline_sha256,
    }
    for name, path in source_paths.items():
        _require_hash(path, source_hashes[name])
    corrected = CorrectedTrackingRunSummary.model_validate_json(
        source_paths["corrected"].read_text(encoding="utf-8")
    )
    corrected_verification = _read_object(source_paths["corrected_verification"])
    perception_summary = _read_object(source_paths["perception"])
    if (
        corrected_verification.get("status") != "passed"
        or corrected_verification.get("source_summary_sha256")
        != source_hashes["corrected"]
        or not corrected_verification.get("all_pairs_regenerated")
    ):
        raise ValueError("D034 source lacks matching passed corrected verification")
    if perception_summary.get("occlusion_inference") is not False:
        raise ValueError("D034 source perception timeline inferred occlusion")

    camera_states = {
        camera_id: _load_timeline(source_paths[camera_id]) for camera_id in CAMERA_IDS
    }
    regenerated_records = _regenerate_records(
        corrected.d033_pair_observations,
        camera_states=camera_states,
        policy=summary.policy,
    )
    if regenerated_records != summary.presentation_records:
        raise ValueError("temporal presentation records do not regenerate")
    regenerated_segments = build_measured_trajectory_segments(
        corrected.d033_pair_observations,
        policy=summary.policy,
    )
    if regenerated_segments != summary.measured_trajectory_segments:
        raise ValueError("measured trajectory segments do not regenerate")

    artifact_paths = {
        "records": _resolve(root, Path(summary.timeline_records_ref)),
        "segments": _resolve(root, Path(summary.trajectory_segments_ref)),
        "csv": _resolve(root, Path(summary.review_csv_ref)),
        "timeline": _resolve(root, Path(summary.timeline_diagnostic_ref)),
        "world": _resolve(root, Path(summary.world_preview_ref)),
    }
    artifact_hashes = {
        "records": summary.timeline_records_sha256,
        "segments": summary.trajectory_segments_sha256,
        "csv": summary.review_csv_sha256,
        "timeline": summary.timeline_diagnostic_sha256,
        "world": summary.world_preview_sha256,
    }
    for name, path in artifact_paths.items():
        _require_hash(path, artifact_hashes[name])
    _verify_persistent_records(
        artifact_paths["records"],
        expected=regenerated_records,
    )
    _verify_persistent_segments(
        artifact_paths["segments"],
        expected=regenerated_segments,
    )
    with artifact_paths["csv"].open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != len(regenerated_records):
        raise ValueError("temporal review CSV coverage differs")

    state_counts = _state_counts(regenerated_records)
    if state_counts != summary.state_counts:
        raise ValueError("temporal state counts differ from summary")
    measured_anchor_counts = Counter(
        record.anchor_kind.value
        for record in regenerated_records
        if record.state is TemporalPresentationState.MEASURED
        and record.anchor_kind is not None
    )
    if dict(sorted(measured_anchor_counts.items())) != summary.anchor_kind_counts:
        raise ValueError("measured anchor-kind counts differ")
    _verify_semantics(
        records=regenerated_records,
        segments=regenerated_segments,
        observations=corrected.d033_pair_observations,
        policy=summary.policy,
    )

    stale_ages = [
        record.measurement_age_seconds
        for record in regenerated_records
        if record.state is TemporalPresentationState.STALE
        and record.measurement_age_seconds is not None
    ]
    verification = {
        "schema_version": 1,
        "stage": "S04",
        "status": "passed",
        "purpose": "temporal_presentation_policy_verification",
        "source_summary_ref": _relative(summary_path, root),
        "source_summary_sha256": _sha256(summary_path),
        "visual_qa_passed": True,
        "source_corrected_observations_regenerated": True,
        "presentation_records_regenerated": True,
        "measured_segments_regenerated": True,
        "presentation_record_count": len(regenerated_records),
        "state_counts": state_counts,
        "measured_segment_count": len(regenerated_segments),
        "segment_counts_by_target": {
            target.value: sum(segment.target is target for segment in regenerated_segments)
            for target in PerceptionTarget
        },
        "maximum_stale_age_seconds": max(stale_ages),
        "maximum_measured_segment_gap_seconds": max(
            segment.elapsed_seconds for segment in regenerated_segments
        ),
        "known_backpack_gap_seconds": _known_backpack_gap(
            corrected.d033_pair_observations
        ),
        "known_backpack_gap_bridged": False,
        "raw_xyz_on_non_measured_state_count": 0,
        "inferred_position_count": 0,
        "claimed_occlusion_count": 0,
        "stale_zone_update_count": 0,
        "stale_trajectory_extension_count": 0,
        "mixed_semantic_segment_count": 0,
        "interpolation_performed": False,
        "motion_extrapolation_performed": False,
        "presentation_smoothing_performed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))
    return 0


def _regenerate_records(
    observations: tuple[CorrectedPairObservationRecord, ...],
    *,
    camera_states: dict[str, tuple[PerceptionTargetFrameState, ...]],
    policy: TemporalPresentationPolicy,
) -> tuple[TemporalPresentationRecord, ...]:
    state_lookup = {
        (camera_id, state.frame_identity.source_frame_index, state.target): state
        for camera_id, states in camera_states.items()
        for state in states
    }
    frames_a = sorted(
        {
            state.frame_identity.source_frame_index
            for state in camera_states["camera_a"]
            if state.target is PerceptionTarget.PERSON
        }
    )
    frames_b = sorted(
        {
            state.frame_identity.source_frame_index
            for state in camera_states["camera_b"]
            if state.target is PerceptionTarget.PERSON
        }
    )
    if frames_a != frames_b or len(frames_a) != 160:
        raise ValueError("source camera grids differ")
    observation_lookup = {
        (observation.source_frame_index, observation.target): observation
        for observation in observations
    }
    last_measurements: dict[PerceptionTarget, CorrectedPairObservationRecord | None] = {
        target: None for target in PerceptionTarget
    }
    records: list[TemporalPresentationRecord] = []
    for frame in frames_a:
        for target in PerceptionTarget:
            state_a = state_lookup[("camera_a", frame, target)]
            state_b = state_lookup[("camera_b", frame, target)]
            timestamp = state_a.frame_identity.capture_timestamp_seconds
            if abs(timestamp - state_b.frame_identity.capture_timestamp_seconds) > 0.01:
                raise ValueError("source tick exceeds synchronization bound")
            current = observation_lookup.get((frame, target))
            resolution = resolve_temporal_presentation(
                source_frame_index=frame,
                capture_timestamp_seconds=timestamp,
                target=target,
                camera_a_perception_state=state_a.state,
                camera_b_perception_state=state_b.state,
                current_observation=current,
                last_measurement=last_measurements[target],
                confirmed_occluded=False,
                policy=policy,
            )
            records.append(
                make_temporal_record(
                    source_frame_index=frame,
                    capture_timestamp_seconds=timestamp,
                    target=target,
                    camera_a_perception_state=state_a.state,
                    camera_b_perception_state=state_b.state,
                    resolution=resolution,
                    policy=policy,
                )
            )
            if (
                current is not None
                and current.state
                in {CorrectedPairState.FUSED, CorrectedPairState.SINGLE_CAMERA}
            ):
                last_measurements[target] = current
    return tuple(records)


def _verify_persistent_records(
    path: Path, *, expected: tuple[TemporalPresentationRecord, ...]
) -> None:
    payload = _read_object(path)
    values = payload.get("records")
    if not isinstance(values, list):
        raise ValueError("persistent temporal records are not a list")
    actual = tuple(TemporalPresentationRecord.model_validate(value) for value in values)
    if actual != expected:
        raise ValueError("persistent temporal records differ from summary")


def _verify_persistent_segments(
    path: Path, *, expected: tuple[MeasuredTrajectorySegment, ...]
) -> None:
    payload = _read_object(path)
    values = payload.get("segments")
    if not isinstance(values, list):
        raise ValueError("persistent measured segments are not a list")
    actual = tuple(MeasuredTrajectorySegment.model_validate(value) for value in values)
    if actual != expected:
        raise ValueError("persistent measured segments differ from summary")


def _verify_semantics(
    *,
    records: tuple[TemporalPresentationRecord, ...],
    segments: tuple[MeasuredTrajectorySegment, ...],
    observations: tuple[CorrectedPairObservationRecord, ...],
    policy: TemporalPresentationPolicy,
) -> None:
    if len(records) != 320:
        raise ValueError("D034 record count differs")
    expected_measured = sum(
        observation.state
        in {CorrectedPairState.FUSED, CorrectedPairState.SINGLE_CAMERA}
        for observation in observations
    )
    if (
        sum(record.state is TemporalPresentationState.MEASURED for record in records)
        != expected_measured
    ):
        raise ValueError("D034 measured count differs")
    if any(
        record.state is not TemporalPresentationState.MEASURED
        and record.raw_world_xyz_m is not None
        for record in records
    ):
        raise ValueError("non-measured state contains raw XYZ")
    if any(record.state is TemporalPresentationState.INFERRED for record in records):
        raise ValueError("D034 contains inferred positions")
    if any(record.state is TemporalPresentationState.OCCLUDED for record in records):
        raise ValueError("D034 claims occlusion without upstream confirmation")
    if any(
        record.state is TemporalPresentationState.STALE
        and (
            record.may_update_zone_membership
            or record.may_extend_trajectory
            or record.measurement_age_seconds is None
            or record.measurement_age_seconds > policy.maximum_stale_age_seconds + 1e-9
        )
        for record in records
    ):
        raise ValueError("stale state exceeds D034 authority")
    if any(
        segment.elapsed_seconds > policy.maximum_trajectory_segment_gap_seconds
        for segment in segments
    ):
        raise ValueError("measured segment exceeds D034 gap")
    if any(
        segment.interpolation_performed or segment.stale_points_used
        for segment in segments
    ):
        raise ValueError("measured segment used interpolation or stale points")
    if any(
        segment.target is PerceptionTarget.BACKPACK
        and segment.start_source_frame_index == 462
        and segment.end_source_frame_index == 666
        for segment in segments
    ):
        raise ValueError("known backpack gap was bridged")
    if _known_backpack_gap(observations) <= policy.maximum_trajectory_segment_gap_seconds:
        raise ValueError("D034 gap threshold does not protect known backpack gap")


def _known_backpack_gap(
    observations: tuple[CorrectedPairObservationRecord, ...],
) -> float:
    ordered = sorted(
        (
            observation
            for observation in observations
            if observation.target is PerceptionTarget.BACKPACK
        ),
        key=lambda observation: observation.capture_timestamp_seconds,
    )
    return next(
        end.capture_timestamp_seconds - start.capture_timestamp_seconds
        for start, end in zip(ordered, ordered[1:], strict=False)
        if start.source_frame_index == 462 and end.source_frame_index == 666
    )


def _state_counts(records: tuple[TemporalPresentationRecord, ...]) -> dict[str, int]:
    result: dict[str, int] = {}
    for state in TemporalPresentationState:
        result[f"total:{state.value}"] = sum(record.state is state for record in records)
        for target in PerceptionTarget:
            result[f"{target.value}:{state.value}"] = sum(
                record.target is target and record.state is state for record in records
            )
    return result


def _load_timeline(path: Path) -> tuple[PerceptionTargetFrameState, ...]:
    payload = _read_object(path)
    values = payload.get("records")
    if not isinstance(values, list):
        raise ValueError(f"timeline records are not a list: {path}")
    return tuple(PerceptionTargetFrameState.model_validate(value) for value in values)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
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


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


if __name__ == "__main__":
    raise SystemExit(main())
