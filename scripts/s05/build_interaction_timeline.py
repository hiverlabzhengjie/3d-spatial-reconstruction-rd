"""Build the measured-only S05 pickup-carry-place interaction timeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import cv2
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from spatial_reconstruction.interaction import (
    BackpackInteractionState,
    InteractionEventCandidate,
    InteractionPolicy,
    InteractionStateRecord,
    InteractionTimelineRunSummary,
    InteractionZone,
    build_event_candidates,
    build_interaction_timeline,
)
from spatial_reconstruction.localization import TemporalPresentationRunSummary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--temporal-summary",
        type=Path,
        default=Path(
            "artifacts/s04/temporal_presentation_dense_final_20260803_v2/summary.json"
        ),
    )
    parser.add_argument(
        "--temporal-verification",
        type=Path,
        default=Path(
            "artifacts/s04/temporal_presentation_dense_final_20260803_v2/verification.json"
        ),
    )
    parser.add_argument(
        "--zone-metadata",
        type=Path,
        default=Path("artifacts/s01/zones/estimated_zones.json"),
    )
    parser.add_argument(
        "--synchronization-manifest",
        type=Path,
        default=Path(
            "artifacts/s01/action_take_01/synchronized/synchronization_manifest.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    paths = {
        "temporal": _resolve(root, args.temporal_summary),
        "temporal_verification": _resolve(root, args.temporal_verification),
        "zones": _resolve(root, args.zone_metadata),
        "synchronization": _resolve(root, args.synchronization_manifest),
    }
    output_dir = _resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    temporal = TemporalPresentationRunSummary.model_validate_json(
        paths["temporal"].read_text(encoding="utf-8")
    )
    temporal_verification = _read_object(paths["temporal_verification"])
    zone_payload = _read_object(paths["zones"])
    synchronization = _read_object(paths["synchronization"])
    _verify_sources(
        temporal_path=paths["temporal"],
        temporal_verification=temporal_verification,
        zone_payload=zone_payload,
    )

    pickup_zone, dropoff_zone = _load_zones(zone_payload)
    video_paths, video_hashes, video_duration = _load_video_sources(
        root=root,
        synchronization=synchronization,
    )
    policy = InteractionPolicy()
    state_records = build_interaction_timeline(
        presentation_records=temporal.presentation_records,
        pickup_zone=pickup_zone,
        dropoff_zone=dropoff_zone,
        policy=policy,
    )
    candidates = build_event_candidates(
        state_records=state_records,
        video_duration_seconds=video_duration,
        policy=policy,
    )
    _validate_real_result(state_records=state_records, candidates=candidates)

    records_path = output_dir / "interaction_state_records.json"
    records_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "S05",
                "policy_id": policy.policy_id,
                "records": [record.model_dump(mode="json") for record in state_records],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    candidates_path = output_dir / "interaction_event_candidates.json"
    candidates_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "S05",
                "policy_id": policy.policy_id,
                "candidates": [
                    candidate.model_dump(mode="json") for candidate in candidates
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "interaction_state_review.csv"
    _write_review_csv(state_records, csv_path)
    timeline_path = output_dir / "interaction_state_timeline.png"
    _save_timeline(state_records, candidates=candidates, path=timeline_path)
    contact_sheet_path = output_dir / "candidate_event_contact_sheet.jpg"
    _save_contact_sheet(
        candidates=candidates,
        video_paths=video_paths,
        path=contact_sheet_path,
    )

    state_counts = dict(sorted(Counter(record.state.value for record in state_records).items()))
    transition_counts = dict(
        sorted(Counter(candidate.event_kind.value for candidate in candidates).items())
    )
    summary = InteractionTimelineRunSummary(
        status="completed_pending_visual_qa",
        created_at_utc=datetime.now(UTC),
        policy=policy,
        pickup_zone=pickup_zone,
        dropoff_zone=dropoff_zone,
        source_temporal_summary_ref=_relative(paths["temporal"], root),
        source_temporal_summary_sha256=_sha256(paths["temporal"]),
        source_temporal_verification_ref=_relative(
            paths["temporal_verification"], root
        ),
        source_temporal_verification_sha256=_sha256(
            paths["temporal_verification"]
        ),
        source_zone_metadata_ref=_relative(paths["zones"], root),
        source_zone_metadata_sha256=_sha256(paths["zones"]),
        source_synchronization_manifest_ref=_relative(paths["synchronization"], root),
        source_synchronization_manifest_sha256=_sha256(paths["synchronization"]),
        source_camera_a_video_ref=_relative(video_paths["camera_a"], root),
        source_camera_a_video_sha256=video_hashes["camera_a"],
        source_camera_b_video_ref=_relative(video_paths["camera_b"], root),
        source_camera_b_video_sha256=video_hashes["camera_b"],
        state_records=state_records,
        event_candidates=candidates,
        state_counts=state_counts,
        transition_counts=transition_counts,
        records_ref=_relative(records_path, root),
        records_sha256=_sha256(records_path),
        candidates_ref=_relative(candidates_path, root),
        candidates_sha256=_sha256(candidates_path),
        review_csv_ref=_relative(csv_path, root),
        review_csv_sha256=_sha256(csv_path),
        timeline_diagnostic_ref=_relative(timeline_path, root),
        timeline_diagnostic_sha256=_sha256(timeline_path),
        candidate_contact_sheet_ref=_relative(contact_sheet_path, root),
        candidate_contact_sheet_sha256=_sha256(contact_sheet_path),
        limitations=(
            "The S04 evidence gap remains unknown and carries no interpolated position.",
            "The retained measured samples establish pickup and place but no separate "
            "current measured carry tick.",
            "Person/backpack XY proximity preserves each source anchor kind and is "
            "not an anatomical-centre distance.",
            "Candidate windows are deterministic review inputs; Qwen has not been "
            "run and cannot change spatial facts.",
        ),
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "state_record_count": len(state_records),
                "state_counts": state_counts,
                "transition_counts": transition_counts,
                "candidates": [
                    {
                        "event_kind": candidate.event_kind.value,
                        "source_frame_index": candidate.source_frame_index,
                        "capture_timestamp_seconds": candidate.capture_timestamp_seconds,
                    }
                    for candidate in candidates
                ],
            },
            indent=2,
        )
    )
    return 0


def _verify_sources(
    *,
    temporal_path: Path,
    temporal_verification: dict[str, Any],
    zone_payload: dict[str, Any],
) -> None:
    if temporal_verification.get("status") != "passed":
        raise ValueError("S05 requires passed D034 temporal verification")
    if temporal_verification.get("source_summary_sha256") != _sha256(temporal_path):
        raise ValueError("D034 verification does not match the supplied summary")
    user_validation = zone_payload.get("user_validation")
    if not isinstance(user_validation, dict) or user_validation.get("accepted") is not True:
        raise ValueError("S05 requires user-accepted zone metadata")
    if zone_payload.get("automated_checks_passed") is not True:
        raise ValueError("S05 requires zones that passed their automated checks")


def _load_zones(
    payload: dict[str, Any],
) -> tuple[InteractionZone, InteractionZone]:
    zones = payload.get("zones")
    if not isinstance(zones, dict):
        raise ValueError("zone metadata lacks a zones object")
    pickup = _load_zone(zones, "pickup_blue_bed", "pickup")
    dropoff = _load_zone(zones, "dropoff_white_floor", "dropoff")
    return pickup, dropoff


def _load_zone(
    zones: dict[str, Any], zone_id: str, role: str
) -> InteractionZone:
    value = zones.get(zone_id)
    if not isinstance(value, dict) or value.get("role") != role:
        raise ValueError(f"zone metadata lacks accepted {role} zone")
    return InteractionZone(
        zone_id=zone_id,
        role=role,  # type: ignore[arg-type]
        center_world_m=cast(tuple[float, float, float], tuple(value["center_world_m"])),
        radius_m=float(value["radius_m"]),
        coordinate_source="video_estimated_and_user_validated",
    )


def _load_video_sources(
    *, root: Path, synchronization: dict[str, Any]
) -> tuple[dict[str, Path], dict[str, str], float]:
    outputs = synchronization.get("derived_outputs")
    if not isinstance(outputs, dict):
        raise ValueError("synchronization manifest lacks derived outputs")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    durations: list[float] = []
    for camera_id in ("camera_a", "camera_b"):
        value = outputs.get(camera_id)
        if not isinstance(value, dict):
            raise ValueError(f"synchronization manifest lacks {camera_id} output")
        path = _resolve(root, Path(str(value["path"])))
        expected_hash = str(value["sha256"])
        if _sha256(path) != expected_hash:
            raise ValueError(f"{camera_id} synchronized video hash differs")
        paths[camera_id] = path
        hashes[camera_id] = expected_hash
        durations.append(float(value["duration_seconds"]))
    if abs(durations[0] - durations[1]) > 1e-6:
        raise ValueError("synchronized camera durations differ")
    return paths, hashes, min(durations)


def _validate_real_result(
    *,
    state_records: tuple[InteractionStateRecord, ...],
    candidates: tuple[InteractionEventCandidate, ...],
) -> None:
    if len(state_records) != 160:
        raise ValueError("real S05 timeline must contain 160 records")
    candidate_kinds = [candidate.event_kind.value for candidate in candidates]
    if candidate_kinds != ["pickup", "place"]:
        raise ValueError(
            "retained measured evidence must produce exactly pickup then place; "
            "carry remains unmeasured"
        )
    pickup = candidates[0]
    place = candidates[1]
    if pickup.source_frame_index != 462 or place.source_frame_index != 666:
        raise ValueError("retained S05 event boundaries differ from verified evidence")
    if any(record.state is BackpackInteractionState.OCCLUDED for record in state_records):
        raise ValueError("retained D034 input contains no explicit occlusion evidence")
    if any(record.invented_xyz for record in state_records):
        raise ValueError("S05 state timeline contains invented XYZ")


def _write_review_csv(records: tuple[InteractionStateRecord, ...], path: Path) -> None:
    fields = [
        "source_frame_index",
        "capture_timestamp_seconds",
        "state",
        "previous_state",
        "last_authoritative_state",
        "pickup_confirmed",
        "reason",
        "backpack_zone_membership",
        "backpack_pickup_center_distance_xy_m",
        "person_backpack_distance_xy_m",
        "backpack_anchor_kind",
        "person_anchor_kind",
        "spatial_transition_authority",
        "invented_xyz",
        "qwen_influenced_spatial_state",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            payload = record.model_dump(mode="json")
            writer.writerow({field: payload.get(field) for field in fields})


def _save_timeline(
    records: tuple[InteractionStateRecord, ...],
    *,
    candidates: tuple[InteractionEventCandidate, ...],
    path: Path,
) -> None:
    state_order = list(BackpackInteractionState)
    state_y = {state: index for index, state in enumerate(state_order)}
    figure, (state_axis, distance_axis) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True, constrained_layout=True
    )
    times = [record.capture_timestamp_seconds for record in records]
    values = [state_y[record.state] for record in records]
    colors = [
        "#1565c0" if record.spatial_transition_authority else "#9e9e9e"
        for record in records
    ]
    state_axis.scatter(times, values, c=colors, s=18)
    state_axis.set_yticks(range(len(state_order)), [state.value for state in state_order])
    state_axis.set_title("S05 measured-only interaction states (grey = no current authority)")
    state_axis.grid(alpha=0.25)

    measured_times = [
        record.capture_timestamp_seconds
        for record in records
        if record.backpack_pickup_center_distance_xy_m is not None
    ]
    pickup_distances = [
        record.backpack_pickup_center_distance_xy_m
        for record in records
        if record.backpack_pickup_center_distance_xy_m is not None
    ]
    proximity_times = [
        record.capture_timestamp_seconds
        for record in records
        if record.person_backpack_distance_xy_m is not None
    ]
    proximity_distances = [
        record.person_backpack_distance_xy_m
        for record in records
        if record.person_backpack_distance_xy_m is not None
    ]
    distance_axis.plot(
        measured_times, pickup_distances, "o-", label="backpack to pickup centre XY"
    )
    distance_axis.plot(
        proximity_times, proximity_distances, "s-", label="person/backpack XY"
    )
    distance_axis.axhline(0.30, color="#2e7d32", linestyle="--", label="pickup radius")
    distance_axis.axhline(1.0, color="#ef6c00", linestyle="--", label="proximity gate")
    distance_axis.set_xlabel("capture timestamp (s)")
    distance_axis.set_ylabel("distance (m)")
    distance_axis.grid(alpha=0.25)
    distance_axis.legend(loc="upper left", ncol=2)
    for candidate in candidates:
        for axis in (state_axis, distance_axis):
            axis.axvline(candidate.capture_timestamp_seconds, color="#c62828", alpha=0.6)
        state_axis.annotate(
            f"{candidate.event_kind.value}\nframe {candidate.source_frame_index}",
            (
                candidate.capture_timestamp_seconds,
                state_y[BackpackInteractionState(candidate.event_kind.value)],
            ),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=8,
        )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_contact_sheet(
    *,
    candidates: tuple[InteractionEventCandidate, ...],
    video_paths: dict[str, Path],
    path: Path,
) -> None:
    review_points: list[tuple[str, float]] = []
    for candidate in candidates:
        review_points.extend(
            (
                (
                    f"{candidate.event_kind.value} window start",
                    candidate.clip_start_timestamp_seconds,
                ),
                (
                    f"{candidate.event_kind.value} transition",
                    candidate.capture_timestamp_seconds,
                ),
                (
                    f"{candidate.event_kind.value} window end",
                    candidate.clip_end_timestamp_seconds,
                ),
            )
        )
    if len(candidates) == 2:
        midpoint = (
            candidates[0].capture_timestamp_seconds
            + candidates[1].capture_timestamp_seconds
        ) / 2.0
        review_points.insert(
            3, ("carry interval review (state remains unknown)", midpoint)
        )
    figure, axes = plt.subplots(
        len(review_points), 2, figsize=(14, 4.2 * len(review_points)), constrained_layout=True
    )
    for row, (label, timestamp) in enumerate(review_points):
        for column, camera_id in enumerate(("camera_a", "camera_b")):
            image = _read_video_frame(video_paths[camera_id], timestamp)
            axes[row, column].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[row, column].set_title(
                f"{label} | {camera_id} | t={timestamp:.3f}s"
            )
            axes[row, column].axis("off")
    figure.suptitle(
        "S05 candidate-event review: synchronized source frames; no Qwen inference",
        fontsize=15,
    )
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _read_video_frame(path: Path, timestamp_seconds: float) -> Any:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open synchronized video: {path}")
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"cannot decode video frame at {timestamp_seconds:.3f}s")
        return frame
    finally:
        capture.release()


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
