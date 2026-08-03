"""Build D032-gated S04 cross-camera observations from selected anchors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from spatial_reconstruction.contracts import PerceptionTarget
from spatial_reconstruction.localization import (
    ActionDepthRunSummary,
    AnchorAvailability,
    AnchorEvaluationRunSummary,
    CrossCameraAnchorState,
    CrossCameraFusionRunSummary,
    CrossCameraObservationRecord,
    CrossCameraObservationState,
    FusionReliabilityConfig,
    FusionSourceEvidence,
    FusionSourceMeasurement,
    SelectedAnchorStateRecord,
    VisibleSurfaceObservationRecord,
    VisibleSurfaceRunSummary,
    resolve_cross_camera_observation,
)

TARGET_COLOR = {
    PerceptionTarget.PERSON: "#00a6c7",
    PerceptionTarget.BACKPACK: "#e62e91",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--anchor-summary",
        type=Path,
        default=Path("artifacts/s04/anchor_evaluation_20260802_v2/summary.json"),
    )
    parser.add_argument(
        "--anchor-verification",
        type=Path,
        default=Path(
            "artifacts/s04/anchor_evaluation_20260802_v2/verification.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    anchor_path = _resolve(project_root, args.anchor_summary)
    anchor_verification_path = _resolve(project_root, args.anchor_verification)
    output_dir = _resolve(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    anchor = AnchorEvaluationRunSummary.model_validate_json(
        anchor_path.read_text(encoding="utf-8")
    )
    anchor_verification = _read_object(anchor_verification_path)
    _verify_anchor_prerequisite(
        verification=anchor_verification,
        anchor_path=anchor_path,
    )
    surface_path = _resolve(
        project_root, Path(anchor.source_visible_surface_summary_ref)
    )
    _require_hash(surface_path, anchor.source_visible_surface_summary_sha256)
    surface = VisibleSurfaceRunSummary.model_validate_json(
        surface_path.read_text(encoding="utf-8")
    )
    action_path = _resolve(project_root, Path(anchor.source_action_depth_summary_ref))
    _require_hash(action_path, anchor.source_action_depth_summary_sha256)
    action = ActionDepthRunSummary.model_validate_json(
        action_path.read_text(encoding="utf-8")
    )
    scene_path = _resolve(project_root, Path(surface.scene_metadata_ref))
    _require_hash(scene_path, surface.scene_metadata_sha256)
    scene = _read_object(scene_path)
    room_bounds = cast(dict[str, Any], scene["room_bounds"])
    bounds_min = np.asarray(room_bounds["minimum_world_xyz_m"], dtype=np.float64)
    bounds_max = np.asarray(room_bounds["maximum_world_xyz_m"], dtype=np.float64)
    config = FusionReliabilityConfig(
        maximum_cross_camera_disagreement_m=(
            anchor.configuration.maximum_cross_camera_disagreement_m
        )
    )

    state_lookup = {
        (item.action_depth_job_id, item.camera_id, item.target): item
        for item in anchor.selected_anchor_states
    }
    candidate_lookup = {item.candidate_id: item for item in anchor.candidate_records}
    surface_lookup = {item.observation_id: item for item in surface.observations}
    prediction_lookup = {
        prediction.job.job_id: prediction for prediction in action.predictions
    }
    records: list[CrossCameraObservationRecord] = []
    for comparison in anchor.cross_camera_comparisons:
        prediction = prediction_lookup[comparison.action_depth_job_id]
        frame_lookup = {
            frame.camera_id: frame for frame in prediction.job.bundle.frames
        }
        states = (
            state_lookup[
                (comparison.action_depth_job_id, "camera_a", comparison.target)
            ],
            state_lookup[
                (comparison.action_depth_job_id, "camera_b", comparison.target)
            ],
        )
        measurements = tuple(
            _measurement_from_state(
                state=state,
                candidate_lookup=candidate_lookup,
                surface_lookup=surface_lookup,
            )
            for state in states
        )
        result = resolve_cross_camera_observation(
            sources=(measurements[0], measurements[1]),
            maximum_disagreement_m=config.maximum_cross_camera_disagreement_m,
        )
        _require_state_alignment(comparison.state, result.state)
        sources = tuple(
            _source_evidence(
                measurement=measurement,
                reliability_score_value=result.reliability_scores[index],
                contribution_weight=result.contribution_weights[index],
            )
            for index, measurement in enumerate(result.sources)
        )
        world_xyz = result.world_xyz_m
        inside = None
        if world_xyz is not None:
            point = np.asarray(world_xyz, dtype=np.float64)
            inside = bool(np.all((point >= bounds_min) & (point <= bounds_max)))
        capture_times = [
            frame.capture_timestamp_seconds for frame in prediction.job.bundle.frames
        ]
        record = CrossCameraObservationRecord(
            observation_id=CrossCameraObservationRecord.create_observation_id(
                action_depth_job_id=comparison.action_depth_job_id,
                bundle_id=comparison.bundle_id,
                target=comparison.target,
                policy_id=config.policy_id,
            ),
            policy_id=config.policy_id,
            anchor_policy_id=config.anchor_policy_id,
            action_depth_job_id=comparison.action_depth_job_id,
            bundle_id=comparison.bundle_id,
            camera_a_frame_id=frame_lookup["camera_a"].frame_id,
            camera_b_frame_id=frame_lookup["camera_b"].frame_id,
            source_frame_index=comparison.source_frame_index,
            capture_timestamp_seconds=prediction.job.bundle.capture_timestamp_seconds,
            maximum_source_time_difference_seconds=max(capture_times)
            - min(capture_times),
            phase_id=comparison.phase_id,
            target=comparison.target,
            selected_anchor_method=comparison.selected_method,
            state=result.state,
            combination_method=result.combination_method,
            sources=(sources[0], sources[1]),
            disagreement_distance_m=result.disagreement_distance_m,
            maximum_eligible_disagreement_m=(
                config.maximum_cross_camera_disagreement_m
            ),
            world_xyz_m=world_xyz,
            inside_room_bounds=inside,
            coordinate_semantics=_coordinate_semantics(
                state=result.state,
                target=comparison.target,
                sources=(sources[0], sources[1]),
            ),
            camera_fusion_performed=result.camera_fusion_performed,
            single_source_passthrough=(
                result.state is CrossCameraObservationState.SINGLE_CAMERA
            ),
        )
        records.append(record)

    csv_path = output_dir / "cross_camera_observations.csv"
    _save_csv(records, csv_path)
    reliability_path = output_dir / "reliability_and_state_diagnostic.png"
    _save_reliability_diagnostic(records, reliability_path)
    world_path = output_dir / "cross_camera_world_preview.png"
    _save_world_preview(
        records=records,
        selected_states=anchor.selected_anchor_states,
        room_bounds=room_bounds,
        zones=cast(dict[str, Any], scene["zones"]),
        path=world_path,
    )
    summary = CrossCameraFusionRunSummary(
        schema_version=1,
        status="completed_pending_visual_qa",
        stage="S04",
        created_at_utc=datetime.now(UTC),
        source_anchor_evaluation_summary_ref=_relative(anchor_path, project_root),
        source_anchor_evaluation_summary_sha256=_sha256(anchor_path),
        source_anchor_evaluation_verification_ref=_relative(
            anchor_verification_path, project_root
        ),
        source_anchor_evaluation_verification_sha256=_sha256(
            anchor_verification_path
        ),
        source_visible_surface_summary_ref=_relative(surface_path, project_root),
        source_visible_surface_summary_sha256=_sha256(surface_path),
        configuration=config,
        observations=tuple(records),
        observation_csv_ref=_relative(csv_path, project_root),
        observation_csv_sha256=_sha256(csv_path),
        reliability_diagnostic_ref=_relative(reliability_path, project_root),
        reliability_diagnostic_sha256=_sha256(reliability_path),
        world_preview_ref=_relative(world_path, project_root),
        world_preview_sha256=_sha256(world_path),
        limitations=(
            "Only one retained person pair passes D032 and is actually fused.",
            "All retained backpack observations are explicit single-camera passthroughs.",
            "Reliability scores are bounded prototype evidence, not calibrated probabilities.",
            "Disagreement states emit no combined XYZ even when both source anchors exist.",
            "No temporal interpolation, stale carry-forward, or presentation smoothing occurs.",
        ),
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    fused = next(
        item for item in records if item.state is CrossCameraObservationState.FUSED
    )
    print(
        json.dumps(
            {
                "status": summary.status,
                "observation_count": len(records),
                "state_counts": _state_counts(records),
                "world_xyz_observation_count": sum(
                    item.world_xyz_m is not None for item in records
                ),
                "fused_source_frame_index": fused.source_frame_index,
                "fused_weights": {
                    source.camera_id: source.contribution_weight
                    for source in fused.sources
                },
                "fused_world_xyz_m": fused.world_xyz_m,
                "summary": _relative(summary_path, project_root),
            },
            indent=2,
        )
    )
    return 0


def _measurement_from_state(
    *,
    state: SelectedAnchorStateRecord,
    candidate_lookup: dict[str, Any],
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
    assert state.source_observation_id is not None
    assert state.source_candidate_id is not None
    assert state.anchor_world_xyz_m is not None
    candidate = candidate_lookup[state.source_candidate_id]
    surface = surface_lookup[state.source_observation_id]
    if candidate.source_observation_id != surface.observation_id:
        raise ValueError("selected anchor candidate and raw observation differ")
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


def _source_evidence(
    *,
    measurement: FusionSourceMeasurement,
    reliability_score_value: float | None,
    contribution_weight: float | None,
) -> FusionSourceEvidence:
    relative_mad = None
    if measurement.availability is AnchorAvailability.OBSERVED:
        assert measurement.retained_depth_mad_m is not None
        assert measurement.retained_depth_median_m is not None
        relative_mad = (
            measurement.retained_depth_mad_m / measurement.retained_depth_median_m
        )
    return FusionSourceEvidence(
        camera_id=measurement.camera_id,
        availability=measurement.availability,
        unavailable_reason=measurement.unavailable_reason,
        source_observation_id=measurement.source_observation_id,
        source_candidate_id=measurement.source_candidate_id,
        anchor_world_xyz_m=measurement.anchor_world_xyz_m,
        support_sample_count=measurement.support_sample_count,
        retained_confidence_median=measurement.retained_confidence_median,
        retained_depth_median_m=measurement.retained_depth_median_m,
        retained_depth_mad_m=measurement.retained_depth_mad_m,
        retained_depth_relative_mad=relative_mad,
        reliability_score=reliability_score_value,
        contribution_weight=contribution_weight,
    )


def _require_state_alignment(
    anchor_state: CrossCameraAnchorState,
    output_state: CrossCameraObservationState,
) -> None:
    expected = {
        CrossCameraAnchorState.PAIRED_ELIGIBLE: CrossCameraObservationState.FUSED,
        CrossCameraAnchorState.PAIRED_DISAGREEMENT: (
            CrossCameraObservationState.DISAGREEMENT
        ),
        CrossCameraAnchorState.SINGLE_CAMERA: (
            CrossCameraObservationState.SINGLE_CAMERA
        ),
        CrossCameraAnchorState.UNAVAILABLE: CrossCameraObservationState.UNAVAILABLE,
    }
    if output_state is not expected[anchor_state]:
        raise ValueError("D032 comparison and cross-camera output state differ")


def _coordinate_semantics(
    *,
    state: CrossCameraObservationState,
    target: PerceptionTarget,
    sources: tuple[FusionSourceEvidence, FusionSourceEvidence],
) -> str:
    target_name = target.value
    if state is CrossCameraObservationState.FUSED:
        return (
            f"Reliability-weighted exact-frame {target_name} selected anchors from "
            "camera_a and camera_b."
        )
    if state is CrossCameraObservationState.SINGLE_CAMERA:
        source = next(
            item for item in sources if item.availability is AnchorAvailability.OBSERVED
        )
        return (
            f"Exact-frame {target_name} selected anchor passed through from "
            f"{source.camera_id}; not camera-fused."
        )
    if state is CrossCameraObservationState.DISAGREEMENT:
        return (
            f"Exact-frame {target_name} anchors exceed D032 disagreement gate; no "
            "combined XYZ."
        )
    return f"No exact-frame {target_name} selected anchor is available; no XYZ."


def _save_csv(records: list[CrossCameraObservationRecord], path: Path) -> None:
    fields = (
        "observation_id",
        "source_frame_index",
        "phase_id",
        "target",
        "state",
        "combination_method",
        "disagreement_distance_m",
        "world_x_m",
        "world_y_m",
        "world_z_m",
        "camera_a_reliability_score",
        "camera_a_contribution_weight",
        "camera_b_reliability_score",
        "camera_b_contribution_weight",
        "inside_room_bounds",
        "camera_fusion_performed",
        "single_source_passthrough",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            xyz = record.world_xyz_m or (None, None, None)
            writer.writerow(
                {
                    "observation_id": record.observation_id,
                    "source_frame_index": record.source_frame_index,
                    "phase_id": record.phase_id,
                    "target": record.target.value,
                    "state": record.state.value,
                    "combination_method": record.combination_method.value,
                    "disagreement_distance_m": record.disagreement_distance_m,
                    "world_x_m": xyz[0],
                    "world_y_m": xyz[1],
                    "world_z_m": xyz[2],
                    "camera_a_reliability_score": record.sources[0].reliability_score,
                    "camera_a_contribution_weight": (
                        record.sources[0].contribution_weight
                    ),
                    "camera_b_reliability_score": record.sources[1].reliability_score,
                    "camera_b_contribution_weight": (
                        record.sources[1].contribution_weight
                    ),
                    "inside_room_bounds": record.inside_room_bounds,
                    "camera_fusion_performed": record.camera_fusion_performed,
                    "single_source_passthrough": record.single_source_passthrough,
                }
            )


def _save_reliability_diagnostic(
    records: list[CrossCameraObservationRecord], path: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    state_counts = Counter(record.state for record in records)
    states = tuple(CrossCameraObservationState)
    axes[0, 0].bar(
        [state.value for state in states],
        [state_counts[state] for state in states],
        color=["#4daf4a", "#377eb8", "#e41a1c", "#999999"],
    )
    axes[0, 0].set_ylabel("job/target count")
    axes[0, 0].set_title("Cross-camera observation states")
    axes[0, 0].tick_params(axis="x", rotation=20)

    fused = next(
        record for record in records if record.state is CrossCameraObservationState.FUSED
    )
    axes[0, 1].bar(
        [source.camera_id for source in fused.sources],
        [cast(float, source.reliability_score) for source in fused.sources],
        color=["#377eb8", "#4daf4a"],
        alpha=0.65,
        label="reliability score",
    )
    twin = axes[0, 1].twinx()
    twin.plot(
        [source.camera_id for source in fused.sources],
        [cast(float, source.contribution_weight) for source in fused.sources],
        color="black",
        marker="o",
        label="normalized weight",
    )
    axes[0, 1].set_ylabel("reliability score")
    twin.set_ylabel("contribution weight")
    twin.set_ylim(0, 1)
    axes[0, 1].set_title(
        f"Only fused pair · frame {fused.source_frame_index} · {fused.target.value}"
    )

    person_pairs = [
        record
        for record in records
        if record.target is PerceptionTarget.PERSON
        and record.disagreement_distance_m is not None
    ]
    axes[1, 0].bar(
        [str(record.source_frame_index) for record in person_pairs],
        [cast(float, record.disagreement_distance_m) for record in person_pairs],
        color=[
            "#4daf4a"
            if record.state is CrossCameraObservationState.FUSED
            else "#e41a1c"
            for record in person_pairs
        ],
    )
    axes[1, 0].axhline(0.35, color="black", linestyle="--", label="D032 gate")
    axes[1, 0].set_xlabel("source frame index")
    axes[1, 0].set_ylabel("anchor disagreement (m)")
    axes[1, 0].set_title("Paired-person eligibility")
    axes[1, 0].legend()

    observed_sources = [
        (record, source)
        for record in records
        for source in record.sources
        if source.availability is AnchorAvailability.OBSERVED
    ]
    for target, marker in (
        (PerceptionTarget.PERSON, "o"),
        (PerceptionTarget.BACKPACK, "s"),
    ):
        matching = [item for item in observed_sources if item[0].target is target]
        axes[1, 1].scatter(
            [record.source_frame_index for record, _ in matching],
            [cast(float, source.reliability_score) for _, source in matching],
            c=[
                "#377eb8" if source.camera_id == "camera_a" else "#4daf4a"
                for _, source in matching
            ],
            marker=marker,
            label=target.value,
        )
    axes[1, 1].set_xlabel("source frame index")
    axes[1, 1].set_ylabel("inspectable reliability score")
    axes[1, 1].set_title("Source reliability retained for every observed anchor")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)
    figure.suptitle("S04 D033 reliability and cross-camera state diagnostic")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_world_preview(
    *,
    records: list[CrossCameraObservationRecord],
    selected_states: tuple[SelectedAnchorStateRecord, ...],
    room_bounds: dict[str, Any],
    zones: dict[str, Any],
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
    for target in PerceptionTarget:
        selected = [
            state
            for state in selected_states
            if state.target is target
            and state.availability is AnchorAvailability.OBSERVED
        ]
        anchors = np.asarray([state.anchor_world_xyz_m for state in selected])
        axis.scatter(
            anchors[:, 0],
            anchors[:, 1],
            s=20,
            c=TARGET_COLOR[target],
            alpha=0.18,
            marker="x",
            label=f"{target.value} selected camera anchors",
        )
        world_records = [
            record
            for record in records
            if record.target is target and record.world_xyz_m is not None
        ]
        world = np.asarray([record.world_xyz_m for record in world_records])
        axis.scatter(
            world[:, 0],
            world[:, 1],
            s=65,
            c=TARGET_COLOR[target],
            marker="o",
            edgecolors="black",
            linewidths=0.5,
            label=f"{target.value} combined observation",
        )
        for record, point in zip(world_records, world, strict=True):
            marker = "F" if record.state is CrossCameraObservationState.FUSED else "S"
            axis.annotate(
                f"{record.source_frame_index}{marker}",
                (point[0], point[1]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
    state_lookup = {
        (state.action_depth_job_id, state.camera_id, state.target): state
        for state in selected_states
    }
    for record in records:
        if record.state is not CrossCameraObservationState.DISAGREEMENT:
            continue
        camera_a = state_lookup[
            (record.action_depth_job_id, "camera_a", record.target)
        ]
        camera_b = state_lookup[
            (record.action_depth_job_id, "camera_b", record.target)
        ]
        assert camera_a.anchor_world_xyz_m is not None
        assert camera_b.anchor_world_xyz_m is not None
        pair = np.asarray([camera_a.anchor_world_xyz_m, camera_b.anchor_world_xyz_m])
        axis.plot(
            pair[:, 0], pair[:, 1], color="#e41a1c", linestyle="--", linewidth=1.4
        )
        midpoint = np.mean(pair, axis=0)
        axis.annotate(
            f"{record.source_frame_index} disagreement\n(no XYZ)",
            (midpoint[0], midpoint[1]),
            fontsize=7,
            color="#b00000",
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
        "D032-gated cross-camera observations\n"
        "F=fused · S=single source · disagreement lines emit no XYZ"
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, loc="upper right")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _state_counts(records: list[CrossCameraObservationRecord]) -> dict[str, int]:
    counts = Counter(record.state for record in records)
    return {state.value: counts[state] for state in CrossCameraObservationState}


def _verify_anchor_prerequisite(
    *, verification: dict[str, Any], anchor_path: Path
) -> None:
    required = (
        verification.get("stage") == "S04",
        verification.get("status") == "passed",
        verification.get("purpose")
        == "target_anchor_candidate_and_disagreement_policy_verification",
        verification.get("source_summary_sha256") == _sha256(anchor_path),
        verification.get("candidate_record_count") == 104,
        verification.get("selected_anchor_state_count") == 32,
        verification.get("maximum_eligible_disagreement_m") == 0.35,
        verification.get("camera_fusion_performed") is False,
    )
    if not all(required):
        raise ValueError("verified D032 anchor prerequisite did not pass")


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
