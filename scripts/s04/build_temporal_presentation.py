"""Build D034 temporal presentation states from corrected S04 observations."""

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
from matplotlib.lines import Line2D  # noqa: E402

from spatial_reconstruction.contracts import PerceptionTarget
from spatial_reconstruction.localization import (
    CorrectedAnchorKind,
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
from spatial_reconstruction.perception import (
    BackpackVisibilityRecord,
    BackpackVisibilityRunSummary,
    PerceptionTargetFrameState,
)

CAMERA_IDS = ("camera_a", "camera_b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--corrected-summary",
        type=Path,
        default=Path("artifacts/s04/corrected_tracking_20260803/summary.json"),
    )
    parser.add_argument(
        "--corrected-verification",
        type=Path,
        default=Path("artifacts/s04/corrected_tracking_20260803/verification.json"),
    )
    parser.add_argument(
        "--perception-summary",
        type=Path,
        default=Path("artifacts/s03/target_timeline_5fps_20260801/summary.json"),
    )
    parser.add_argument(
        "--camera-a-timeline",
        type=Path,
        default=Path(
            "artifacts/s03/target_timeline_5fps_20260801/"
            "camera_a_target_timeline.json"
        ),
    )
    parser.add_argument(
        "--camera-b-timeline",
        type=Path,
        default=Path(
            "artifacts/s03/target_timeline_5fps_20260801/"
            "camera_b_target_timeline.json"
        ),
    )
    parser.add_argument(
        "--visibility-summary",
        type=Path,
        help=(
            "Optional explicit backpack-visibility overlay; detector absence alone "
            "is insufficient."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    paths = {
        "corrected": _resolve(root, args.corrected_summary),
        "corrected_verification": _resolve(root, args.corrected_verification),
        "perception": _resolve(root, args.perception_summary),
        "camera_a": _resolve(root, args.camera_a_timeline),
        "camera_b": _resolve(root, args.camera_b_timeline),
    }
    if args.visibility_summary is not None:
        paths["visibility"] = _resolve(root, args.visibility_summary)
    output_dir = _resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    corrected = CorrectedTrackingRunSummary.model_validate_json(
        paths["corrected"].read_text(encoding="utf-8")
    )
    corrected_verification = _read_object(paths["corrected_verification"])
    perception_summary = _read_object(paths["perception"])
    camera_states = {
        camera_id: _load_timeline(paths[camera_id]) for camera_id in CAMERA_IDS
    }
    visibility_summary = (
        BackpackVisibilityRunSummary.model_validate_json(
            paths["visibility"].read_text(encoding="utf-8")
        )
        if "visibility" in paths
        else None
    )
    policy = TemporalPresentationPolicy()
    _verify_prerequisites(
        corrected=corrected,
        corrected_path=paths["corrected"],
        corrected_verification=corrected_verification,
        perception_summary=perception_summary,
        camera_states=camera_states,
        visibility_summary=visibility_summary,
        policy=policy,
    )

    presentation_records = _build_timeline(
        corrected.d033_pair_observations,
        camera_states=camera_states,
        visibility_records=(
            visibility_summary.records if visibility_summary is not None else ()
        ),
        policy=policy,
    )
    trajectory_segments = build_measured_trajectory_segments(
        corrected.d033_pair_observations,
        policy=policy,
    )
    _validate_real_result(
        records=presentation_records,
        segments=trajectory_segments,
        corrected_observations=corrected.d033_pair_observations,
        policy=policy,
    )

    records_path = output_dir / "temporal_presentation_records.json"
    records_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "S04",
                "policy_id": policy.policy_id,
                "records": [record.model_dump(mode="json") for record in presentation_records],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    segments_path = output_dir / "measured_trajectory_segments.json"
    segments_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "S04",
                "policy_id": policy.policy_id,
                "segments": [segment.model_dump(mode="json") for segment in trajectory_segments],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "temporal_presentation_review.csv"
    _write_review_csv(presentation_records, csv_path)
    timeline_path = output_dir / "temporal_state_timeline.png"
    _save_timeline_diagnostic(
        presentation_records,
        corrected.d033_pair_observations,
        policy=policy,
        path=timeline_path,
    )
    world_path = output_dir / "measured_segment_world_preview.png"
    _save_world_preview(
        corrected.d033_pair_observations,
        trajectory_segments,
        path=world_path,
    )

    state_counts = _state_counts(presentation_records)
    anchor_counts = Counter(
        record.anchor_kind.value
        for record in presentation_records
        if record.state is TemporalPresentationState.MEASURED
        and record.anchor_kind is not None
    )
    summary = TemporalPresentationRunSummary(
        status="completed_pending_visual_qa",
        created_at_utc=datetime.now(UTC),
        policy=policy,
        source_corrected_summary_ref=_relative(paths["corrected"], root),
        source_corrected_summary_sha256=_sha256(paths["corrected"]),
        source_corrected_verification_ref=_relative(
            paths["corrected_verification"], root
        ),
        source_corrected_verification_sha256=_sha256(
            paths["corrected_verification"]
        ),
        source_perception_summary_ref=_relative(paths["perception"], root),
        source_perception_summary_sha256=_sha256(paths["perception"]),
        source_camera_a_timeline_ref=_relative(paths["camera_a"], root),
        source_camera_a_timeline_sha256=_sha256(paths["camera_a"]),
        source_camera_b_timeline_ref=_relative(paths["camera_b"], root),
        source_camera_b_timeline_sha256=_sha256(paths["camera_b"]),
        source_visibility_summary_ref=(
            _relative(paths["visibility"], root) if "visibility" in paths else None
        ),
        source_visibility_summary_sha256=(
            _sha256(paths["visibility"]) if "visibility" in paths else None
        ),
        presentation_records=presentation_records,
        measured_trajectory_segments=trajectory_segments,
        state_counts=state_counts,
        anchor_kind_counts=dict(sorted(anchor_counts.items())),
        timeline_records_ref=_relative(records_path, root),
        timeline_records_sha256=_sha256(records_path),
        trajectory_segments_ref=_relative(segments_path, root),
        trajectory_segments_sha256=_sha256(segments_path),
        review_csv_ref=_relative(csv_path, root),
        review_csv_sha256=_sha256(csv_path),
        timeline_diagnostic_ref=_relative(timeline_path, root),
        timeline_diagnostic_sha256=_sha256(timeline_path),
        world_preview_ref=_relative(world_path, root),
        world_preview_sha256=_sha256(world_path),
        limitations=(
            "The one-second stale hold is presentation-only and is not a new measurement.",
            "No interpolation, motion extrapolation, inferred position, or smoothing is used.",
            "Missing S03 detections are not automatically labelled occluded; only "
            "the versioned explicit visibility overlay can do so.",
            "Body-surface fallbacks retain their measured semantics and never become footpoints.",
            "Measured segment lines connect exact endpoints only and are not sampled paths.",
        ),
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "presentation_record_count": len(presentation_records),
                "state_counts": state_counts,
                "measured_segment_count": len(trajectory_segments),
                "segment_counts_by_target": {
                    target.value: sum(
                        segment.target is target for segment in trajectory_segments
                    )
                    for target in PerceptionTarget
                },
                "inferred_position_count": 0,
                "occluded_count": sum(
                    record.state is TemporalPresentationState.OCCLUDED
                    for record in presentation_records
                ),
            },
            indent=2,
        )
    )
    return 0


def _load_timeline(path: Path) -> tuple[PerceptionTargetFrameState, ...]:
    payload = _read_object(path)
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"timeline records are not a list: {path}")
    return tuple(PerceptionTargetFrameState.model_validate(item) for item in records)


def _verify_prerequisites(
    *,
    corrected: CorrectedTrackingRunSummary,
    corrected_path: Path,
    corrected_verification: dict[str, Any],
    perception_summary: dict[str, Any],
    camera_states: dict[str, tuple[PerceptionTargetFrameState, ...]],
    visibility_summary: BackpackVisibilityRunSummary | None,
    policy: TemporalPresentationPolicy,
) -> None:
    if (
        corrected_verification.get("status") != "passed"
        or corrected_verification.get("source_summary_sha256")
        != _sha256(corrected_path)
        or not corrected_verification.get("all_pairs_regenerated")
    ):
        raise ValueError("corrected D030-D033 prerequisite is not verified")
    if corrected.policy.policy_id != policy.source_observation_policy_id:
        raise ValueError("D034 source observation policy differs")
    if not corrected.d033_pair_observations:
        raise ValueError("D034 requires corrected pair observations")
    sampling = cast(dict[str, Any], perception_summary["source"])["sampling"]
    if (
        perception_summary.get("stage") != "S03"
        or perception_summary.get("occlusion_inference") is not False
        or int(sampling["frame_stride"]) != policy.timeline_frame_stride
        or float(sampling["nominal_fps_per_camera"])
        != policy.nominal_timeline_rate_fps
        or int(sampling["selected_bundle_count"]) != 160
    ):
        raise ValueError("S03 timeline policy differs from D034 assumptions")
    for camera_id in CAMERA_IDS:
        states = camera_states[camera_id]
        if len(states) != 320:
            raise ValueError(f"{camera_id} timeline does not contain 320 target states")
        if any(state.frame_identity.camera_id != camera_id for state in states):
            raise ValueError(f"{camera_id} timeline contains another camera")
    if visibility_summary is not None:
        if len(visibility_summary.records) != 160:
            raise ValueError("visibility overlay does not cover all 160 ticks")
        expected = {
            state.frame_identity.source_frame_index
            for state in camera_states["camera_a"]
            if state.target is PerceptionTarget.BACKPACK
        }
        actual = {record.source_frame_index for record in visibility_summary.records}
        if actual != expected:
            raise ValueError("visibility overlay frame grid differs from S03")


def _build_timeline(
    observations: tuple[CorrectedPairObservationRecord, ...],
    *,
    camera_states: dict[str, tuple[PerceptionTargetFrameState, ...]],
    visibility_records: tuple[BackpackVisibilityRecord, ...] = (),
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
        raise ValueError("camera timeline frame grids differ")
    if any(
        next_frame - frame != policy.timeline_frame_stride
        for frame, next_frame in zip(frames_a, frames_a[1:], strict=False)
    ):
        raise ValueError("timeline frame stride differs from D034 policy")
    observation_lookup = {
        (record.source_frame_index, record.target): record for record in observations
    }
    visibility_lookup = {record.source_frame_index: record for record in visibility_records}
    if len(visibility_lookup) != len(visibility_records):
        raise ValueError("visibility records are not unique by frame")
    if len(observation_lookup) != len(observations):
        raise ValueError("corrected observations are not unique by frame and target")
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
                raise ValueError("cross-camera timeline tick exceeds synchronization bound")
            current = observation_lookup.get((frame, target))
            visibility = visibility_lookup.get(frame)
            if target is PerceptionTarget.BACKPACK and visibility is not None and (
                abs(visibility.capture_timestamp_seconds - timestamp) > 0.01
                or visibility.camera_a_detection_state is not state_a.state
                or visibility.camera_b_detection_state is not state_b.state
            ):
                raise ValueError("visibility evidence differs from the paired S03 tick")
            resolution = resolve_temporal_presentation(
                source_frame_index=frame,
                capture_timestamp_seconds=timestamp,
                target=target,
                camera_a_perception_state=state_a.state,
                camera_b_perception_state=state_b.state,
                current_observation=current,
                last_measurement=last_measurements[target],
                confirmed_occluded=(
                    target is PerceptionTarget.BACKPACK
                    and visibility is not None
                    and visibility.confirmed_occluded_for_localization
                ),
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


def _validate_real_result(
    *,
    records: tuple[TemporalPresentationRecord, ...],
    segments: tuple[MeasuredTrajectorySegment, ...],
    corrected_observations: tuple[CorrectedPairObservationRecord, ...],
    policy: TemporalPresentationPolicy,
) -> None:
    if len(records) != 320:
        raise ValueError("D034 timeline does not cover 160 ticks and two targets")
    expected_measured = sum(
        observation.state
        in {CorrectedPairState.FUSED, CorrectedPairState.SINGLE_CAMERA}
        for observation in corrected_observations
    )
    if (
        sum(record.state is TemporalPresentationState.MEASURED for record in records)
        != expected_measured
    ):
        raise ValueError("every usable corrected pair observation must appear as measured")
    if any(record.state is TemporalPresentationState.INFERRED for record in records):
        raise ValueError("D034 real result cannot contain inferred positions")
    if any(
        record.state is TemporalPresentationState.STALE
        and (
            record.measurement_age_seconds is None
            or record.measurement_age_seconds > policy.maximum_stale_age_seconds + 1e-9
        )
        for record in records
    ):
        raise ValueError("D034 stale record exceeds its horizon")
    if any(
        record.state is not TemporalPresentationState.MEASURED
        and (record.may_update_zone_membership or record.may_extend_trajectory)
        for record in records
    ):
        raise ValueError("non-measured state gained spatial authority")
    backpack_observations = sorted(
        (
            observation
            for observation in corrected_observations
            if observation.target is PerceptionTarget.BACKPACK
        ),
        key=lambda observation: observation.capture_timestamp_seconds,
    )
    known_gap = next(
        end.capture_timestamp_seconds - start.capture_timestamp_seconds
        for start, end in zip(
            backpack_observations, backpack_observations[1:], strict=False
        )
        if start.source_frame_index == 462 and end.source_frame_index == 666
    )
    if known_gap <= policy.maximum_trajectory_segment_gap_seconds:
        raise ValueError("D034 gap gate would bridge the known backpack hole")
    if any(
        segment.target is PerceptionTarget.BACKPACK
        and segment.start_source_frame_index == 462
        and segment.end_source_frame_index == 666
        for segment in segments
    ):
        raise ValueError("known backpack hole was bridged")


def _state_counts(records: tuple[TemporalPresentationRecord, ...]) -> dict[str, int]:
    result: dict[str, int] = {}
    for state in TemporalPresentationState:
        result[f"total:{state.value}"] = sum(record.state is state for record in records)
        for target in PerceptionTarget:
            result[f"{target.value}:{state.value}"] = sum(
                record.target is target and record.state is state for record in records
            )
    return result


def _write_review_csv(
    records: tuple[TemporalPresentationRecord, ...], path: Path
) -> None:
    rows = [
        {
            "source_frame_index": record.source_frame_index,
            "capture_timestamp_seconds": record.capture_timestamp_seconds,
            "target": record.target.value,
            "camera_a_perception_state": record.camera_a_perception_state.value,
            "camera_b_perception_state": record.camera_b_perception_state.value,
            "presentation_state": record.state.value,
            "coordinate_provenance": record.coordinate_provenance.value,
            "reason": record.reason.value,
            "anchor_kind": record.anchor_kind.value if record.anchor_kind else "",
            "raw_x_m": record.raw_world_xyz_m[0] if record.raw_world_xyz_m else "",
            "raw_y_m": record.raw_world_xyz_m[1] if record.raw_world_xyz_m else "",
            "raw_z_m": record.raw_world_xyz_m[2] if record.raw_world_xyz_m else "",
            "presentation_x_m": (
                record.presentation_world_xyz_m[0]
                if record.presentation_world_xyz_m
                else ""
            ),
            "presentation_y_m": (
                record.presentation_world_xyz_m[1]
                if record.presentation_world_xyz_m
                else ""
            ),
            "presentation_z_m": (
                record.presentation_world_xyz_m[2]
                if record.presentation_world_xyz_m
                else ""
            ),
            "measurement_age_seconds": (
                record.measurement_age_seconds
                if record.measurement_age_seconds is not None
                else ""
            ),
            "may_update_zone_membership": record.may_update_zone_membership,
            "may_extend_trajectory": record.may_extend_trajectory,
            "visual_style_id": record.visual_style_id,
        }
        for record in records
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_timeline_diagnostic(
    records: tuple[TemporalPresentationRecord, ...],
    observations: tuple[CorrectedPairObservationRecord, ...],
    *,
    policy: TemporalPresentationPolicy,
    path: Path,
) -> None:
    figure, (timeline, gaps) = plt.subplots(
        2, 1, figsize=(14, 7), constrained_layout=True, height_ratios=(2.0, 1.0)
    )
    state_colors = {
        TemporalPresentationState.MEASURED: "#00a86b",
        TemporalPresentationState.STALE: "#f2a900",
        TemporalPresentationState.MISSING: "#c6c6c6",
        TemporalPresentationState.OCCLUDED: "#7e57c2",
        TemporalPresentationState.INFERRED: "#00acc1",
    }
    y_values = {PerceptionTarget.PERSON: 1.0, PerceptionTarget.BACKPACK: 0.0}
    for record in records:
        timeline.scatter(
            record.capture_timestamp_seconds,
            y_values[record.target],
            color=state_colors[record.state],
            marker="s",
            s=25 if record.state is TemporalPresentationState.MEASURED else 13,
            alpha=1.0 if record.state is TemporalPresentationState.MEASURED else 0.72,
        )
    anchor_markers = {
        CorrectedAnchorKind.PERSON_FOOTPOINT: "o",
        CorrectedAnchorKind.PERSON_LOWER_BODY_SURFACE: "s",
        CorrectedAnchorKind.PERSON_UPPER_BODY_SURFACE: "^",
        CorrectedAnchorKind.BACKPACK_VISIBLE_CLUSTER: "D",
    }
    label_levels = Counter[PerceptionTarget]()
    for observation in observations:
        row_y = y_values[observation.target]
        label_y = row_y + 0.12 + 0.10 * (label_levels[observation.target] % 3)
        label_levels[observation.target] += 1
        if observation.world_xyz_m is not None and observation.selected_kind is not None:
            timeline.scatter(
                observation.capture_timestamp_seconds,
                row_y,
                color=state_colors[TemporalPresentationState.MEASURED],
                edgecolor="black",
                linewidth=0.45,
                marker=anchor_markers[observation.selected_kind],
                s=42,
                zorder=4,
            )
            label = str(observation.source_frame_index)
            label_color = "#202020"
        else:
            timeline.scatter(
                observation.capture_timestamp_seconds,
                row_y,
                color="#d32f2f",
                marker="x",
                linewidth=1.5,
                s=48,
                zorder=4,
            )
            label = f"{observation.source_frame_index} disagree"
            label_color = "#b71c1c"
        timeline.annotate(
            label,
            (observation.capture_timestamp_seconds, label_y),
            fontsize=6.5,
            rotation=65,
            ha="left",
            va="bottom",
            color=label_color,
        )
    timeline.set_yticks([0.0, 1.0], ["backpack", "person"])
    timeline.set_ylim(-0.35, 1.5)
    timeline.set_title(
        "D034 capture-time states: measured (green), stale hold (amber), missing (grey)"
    )
    timeline.set_xlabel("capture time (s)")
    timeline.grid(axis="x", alpha=0.2)
    timeline.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=state_colors[TemporalPresentationState.MEASURED],
                markeredgecolor="black",
                markersize=6,
                label=label,
            )
            for marker, label in (
                ("o", "person footpoint"),
                ("s", "person lower body"),
                ("^", "person upper body"),
                ("D", "backpack cluster"),
            )
        ]
        + [
            Line2D(
                [0],
                [0],
                marker="x",
                color="#d32f2f",
                linestyle="none",
                markersize=7,
                label="rejected disagreement",
            )
        ],
        loc="upper right",
        fontsize=7,
        ncol=3,
    )

    target_offset = {PerceptionTarget.PERSON: 0.08, PerceptionTarget.BACKPACK: -0.08}
    for target in PerceptionTarget:
        ordered = sorted(
            (record for record in observations if record.target is target),
            key=lambda record: record.capture_timestamp_seconds,
        )
        for start, end in zip(ordered, ordered[1:], strict=False):
            gap = end.capture_timestamp_seconds - start.capture_timestamp_seconds
            compatible = (
                start.selected_kind is not None
                and start.selected_kind is end.selected_kind
                and gap <= policy.maximum_trajectory_segment_gap_seconds
            )
            gaps.scatter(
                (start.capture_timestamp_seconds + end.capture_timestamp_seconds) / 2,
                gap + target_offset[target],
                color="#00a86b" if compatible else "#d32f2f",
                marker="o" if target is PerceptionTarget.PERSON else "D",
                s=45,
            )
    gaps.axhline(
        policy.maximum_trajectory_segment_gap_seconds,
        color="black",
        linestyle="--",
        linewidth=1,
        label="3.0 s measured-segment gate",
    )
    gaps.set_title("Adjacent measurement gaps: green connects exact compatible endpoints")
    gaps.set_xlabel("midpoint capture time (s)")
    gaps.set_ylabel("gap (s)")
    gaps.grid(alpha=0.2)
    gaps.legend(loc="upper right")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_world_preview(
    observations: tuple[CorrectedPairObservationRecord, ...],
    segments: tuple[MeasuredTrajectorySegment, ...],
    *,
    path: Path,
) -> None:
    figure, (person_world, backpack_world, height) = plt.subplots(
        1, 3, figsize=(17, 5.5), constrained_layout=True
    )
    styles = {
        CorrectedAnchorKind.PERSON_FOOTPOINT: ("#00a86b", "o"),
        CorrectedAnchorKind.PERSON_LOWER_BODY_SURFACE: ("#ff8c00", "s"),
        CorrectedAnchorKind.PERSON_UPPER_BODY_SURFACE: ("#e53935", "^"),
        CorrectedAnchorKind.BACKPACK_VISIBLE_CLUSTER: ("#9c27b0", "D"),
    }
    original_boundary_frames = {204, 330, 408, 462, 666, 708, 780, 858}
    for segment in segments:
        start = np.asarray(segment.start_world_xyz_m)
        end = np.asarray(segment.end_world_xyz_m)
        color, _ = styles[segment.anchor_kind]
        axis = (
            person_world
            if segment.target is PerceptionTarget.PERSON
            else backpack_world
        )
        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=2,
            alpha=0.75,
        )
    for observation in observations:
        if observation.world_xyz_m is None or observation.selected_kind is None:
            continue
        point = np.asarray(observation.world_xyz_m)
        color, marker = styles[observation.selected_kind]
        axis = (
            person_world
            if observation.target is PerceptionTarget.PERSON
            else backpack_world
        )
        is_boundary = observation.source_frame_index in original_boundary_frames
        axis.scatter(
            point[0],
            point[1],
            color=color,
            edgecolor="black" if is_boundary else "white",
            linewidth=0.7,
            marker=marker,
            s=72 if is_boundary else 48,
            zorder=3,
        )
        height.scatter(
            observation.capture_timestamp_seconds,
            point[2],
            color=color,
            edgecolor="black" if is_boundary else "white",
            linewidth=0.6,
            marker=marker,
            s=62 if is_boundary else 44,
        )

    for observation in observations:
        if observation.state is not CorrectedPairState.DISAGREEMENT:
            continue
        candidates = [
            np.asarray(source.world_xyz_m)
            for source in observation.sources
            if source.world_xyz_m is not None
        ]
        if len(candidates) == 2:
            person_world.plot(
                [candidates[0][0], candidates[1][0]],
                [candidates[0][1], candidates[1][1]],
                color="#d32f2f",
                linestyle="--",
                linewidth=1.2,
            )
            for candidate in candidates:
                person_world.scatter(
                    candidate[0], candidate[1], color="#d32f2f", marker="x", s=52
                )
            midpoint = (candidates[0] + candidates[1]) / 2.0
            person_world.annotate(
                f"{observation.source_frame_index} rejected\n"
                f"{observation.disagreement_distance_m:.3f} m",
                (midpoint[0], midpoint[1]),
                xytext=(6, 8),
                textcoords="offset points",
                fontsize=7,
                color="#b71c1c",
            )

    for axis, target, title in (
        (person_world, PerceptionTarget.PERSON, "Person measured endpoints"),
        (backpack_world, PerceptionTarget.BACKPACK, "Backpack measured endpoints"),
    ):
        measured_count = sum(
            observation.target is target and observation.world_xyz_m is not None
            for observation in observations
        )
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_title(f"{title} ({measured_count})")
        axis.set_xlabel("world X (m)")
        axis.set_ylabel("world Y (m)")
        axis.grid(alpha=0.25)

    height.axhline(0.0, color="black", linewidth=1)
    height.set_title("Measured anchor semantics remain height-distinct")
    height.set_xlabel("capture time (s)")
    height.set_ylabel("world Z (m)")
    height.grid(alpha=0.25)
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=marker,
                color=color,
                markerfacecolor=color,
                linestyle="none",
                markersize=7,
                label=label,
            )
            for (color, marker), label in (
                (styles[CorrectedAnchorKind.PERSON_FOOTPOINT], "person footpoint"),
                (
                    styles[CorrectedAnchorKind.PERSON_LOWER_BODY_SURFACE],
                    "person lower body",
                ),
                (
                    styles[CorrectedAnchorKind.PERSON_UPPER_BODY_SURFACE],
                    "person upper body",
                ),
                (
                    styles[CorrectedAnchorKind.BACKPACK_VISIBLE_CLUSTER],
                    "backpack cluster",
                ),
            )
        ]
        + [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#777777",
                markeredgecolor="black",
                markersize=8,
                label="original action boundary",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#777777",
                markeredgecolor="white",
                markersize=7,
                label="dense added keyframe",
            ),
            Line2D(
                [0],
                [0],
                marker="x",
                color="#d32f2f",
                linestyle="--",
                markersize=7,
                label="rejected disagreement",
            ),
        ],
        loc="outside lower center",
        ncol=4,
        fontsize=8,
    )
    figure.suptitle(
        "Dense D035 measured-segment preview: exact endpoints only; no interpolation",
        fontsize=14,
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


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
