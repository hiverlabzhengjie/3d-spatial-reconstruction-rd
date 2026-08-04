"""Build an explicit backpack-visibility overlay over retained S03 timelines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from spatial_reconstruction.contracts import PerceptionTarget
from spatial_reconstruction.perception import (
    BackpackVisibilityPolicy,
    BackpackVisibilityRecord,
    BackpackVisibilityRunSummary,
    BackpackVisibilityState,
    PerceptionPresenceState,
    PerceptionTargetFrameState,
    VisibilityEvidenceSource,
)

DETECTED_STATES = {
    PerceptionPresenceState.OBSERVED,
    PerceptionPresenceState.UNTRACKED,
    PerceptionPresenceState.AMBIGUOUS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
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
        "--policy",
        type=Path,
        default=Path("configs/s05_backpack_visibility_evidence.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    paths = {
        "perception": _resolve(root, args.perception_summary),
        "camera_a": _resolve(root, args.camera_a_timeline),
        "camera_b": _resolve(root, args.camera_b_timeline),
        "policy": _resolve(root, args.policy),
    }
    output_dir = _resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    perception_summary = _read_object(paths["perception"])
    policy = BackpackVisibilityPolicy.model_validate_json(
        paths["policy"].read_text(encoding="utf-8")
    )
    camera_records = {
        camera_id: _load_backpack_timeline(paths[camera_id])
        for camera_id in ("camera_a", "camera_b")
    }
    _verify_prerequisites(
        root=root,
        perception_summary=perception_summary,
        camera_records=camera_records,
        policy=policy,
    )
    records = _build_records(camera_records=camera_records, policy=policy)

    records_path = output_dir / "backpack_visibility_records.json"
    records_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "S05",
                "policy_id": policy.policy_id,
                "records": [record.model_dump(mode="json") for record in records],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "backpack_visibility_review.csv"
    _write_csv(records, csv_path)
    counts = Counter(record.visibility_state.value for record in records)
    summary = BackpackVisibilityRunSummary(
        status="completed_pending_visual_qa",
        created_at_utc=datetime.now(UTC),
        policy=policy,
        source_perception_summary_ref=_relative(paths["perception"], root),
        source_perception_summary_sha256=_sha256(paths["perception"]),
        source_camera_a_timeline_ref=_relative(paths["camera_a"], root),
        source_camera_a_timeline_sha256=_sha256(paths["camera_a"]),
        source_camera_b_timeline_ref=_relative(paths["camera_b"], root),
        source_camera_b_timeline_sha256=_sha256(paths["camera_b"]),
        records=records,
        state_counts=dict(sorted(counts.items())),
        records_ref=_relative(records_path, root),
        records_sha256=_sha256(records_path),
        review_csv_ref=_relative(csv_path, root),
        review_csv_sha256=_sha256(csv_path),
        limitations=(
            "This overlay does not modify the immutable S03 detector timeline.",
            "Detector absence alone never establishes occlusion.",
            "Partial occlusion is a synchronized-video-review label, not a detector output.",
            "Visibility evidence supplies no XYZ and cannot repair localization.",
        ),
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "record_count": len(records),
                "state_counts": summary.state_counts,
                "confirmed_occluded_for_localization_count": sum(
                    record.confirmed_occluded_for_localization for record in records
                ),
                "supplied_xyz_count": 0,
            },
            indent=2,
        )
    )
    return 0


def _load_backpack_timeline(path: Path) -> tuple[PerceptionTargetFrameState, ...]:
    values = _read_object(path).get("records")
    if not isinstance(values, list):
        raise ValueError(f"timeline records are not a list: {path}")
    records = tuple(PerceptionTargetFrameState.model_validate(value) for value in values)
    return tuple(record for record in records if record.target is PerceptionTarget.BACKPACK)


def _verify_prerequisites(
    *,
    root: Path,
    perception_summary: dict[str, Any],
    camera_records: dict[str, tuple[PerceptionTargetFrameState, ...]],
    policy: BackpackVisibilityPolicy,
) -> None:
    sampling = cast(dict[str, Any], perception_summary["source"])["sampling"]
    if (
        perception_summary.get("stage") != "S03"
        or perception_summary.get("occlusion_inference") is not False
        or int(sampling["frame_stride"]) != policy.timeline_frame_stride
        or int(sampling["selected_bundle_count"]) != 160
    ):
        raise ValueError("retained S03 timeline differs from visibility assumptions")
    grids: list[list[int]] = []
    for camera_id, records in camera_records.items():
        if len(records) != 160:
            raise ValueError(f"{camera_id} lacks 160 backpack detector states")
        if any(record.frame_identity.camera_id != camera_id for record in records):
            raise ValueError(f"{camera_id} timeline contains another camera")
        grids.append([record.frame_identity.source_frame_index for record in records])
    if grids[0] != grids[1]:
        raise ValueError("S03 camera frame grids differ")
    grid = set(grids[0])
    for interval in policy.review_intervals:
        if (
            interval.start_source_frame_index not in grid
            or interval.end_source_frame_index not in grid
        ):
            raise ValueError("visibility review interval is outside the S03 frame grid")
        for evidence_ref in interval.evidence_refs:
            if not _resolve(root, Path(evidence_ref)).is_file():
                raise ValueError(f"visibility evidence does not exist: {evidence_ref}")


def _build_records(
    *,
    camera_records: dict[str, tuple[PerceptionTargetFrameState, ...]],
    policy: BackpackVisibilityPolicy,
) -> tuple[BackpackVisibilityRecord, ...]:
    lookups = {
        camera_id: {
            record.frame_identity.source_frame_index: record for record in records
        }
        for camera_id, records in camera_records.items()
    }
    output: list[BackpackVisibilityRecord] = []
    for frame in sorted(lookups["camera_a"]):
        record_a = lookups["camera_a"][frame]
        record_b = lookups["camera_b"][frame]
        timestamp = record_a.frame_identity.capture_timestamp_seconds
        if abs(timestamp - record_b.frame_identity.capture_timestamp_seconds) > 0.01:
            raise ValueError("visibility source tick exceeds synchronization bound")
        review = next(
            (
                interval
                for interval in policy.review_intervals
                if interval.start_source_frame_index <= frame <= interval.end_source_frame_index
            ),
            None,
        )
        visibility_state: BackpackVisibilityState
        evidence_source: VisibilityEvidenceSource
        rationale: str
        evidence_refs: tuple[str, ...]
        if review is not None:
            visibility_state = review.visibility_state
            evidence_source = review.evidence_source
            rationale = review.rationale
            evidence_refs = review.evidence_refs
        elif record_a.state in DETECTED_STATES or record_b.state in DETECTED_STATES:
            visibility_state = BackpackVisibilityState.VISIBLE
            evidence_source = VisibilityEvidenceSource.DETECTOR_OBSERVATION
            rationale = "At least one retained S03 camera timeline contains a backpack candidate."
            evidence_refs = ()
        else:
            visibility_state = BackpackVisibilityState.UNKNOWN
            evidence_source = VisibilityEvidenceSource.NONE
            rationale = (
                "Neither detector presence nor explicit synchronized-video review "
                "establishes visibility."
            )
            evidence_refs = ()
        output.append(
            BackpackVisibilityRecord(
                record_id=BackpackVisibilityRecord.create_record_id(
                    policy_id=policy.policy_id,
                    source_frame_index=frame,
                    capture_timestamp_seconds=timestamp,
                ),
                policy_id=policy.policy_id,
                source_frame_index=frame,
                capture_timestamp_seconds=timestamp,
                camera_a_detection_state=record_a.state,
                camera_b_detection_state=record_b.state,
                visibility_state=visibility_state,
                evidence_source=evidence_source,
                confirmed_occluded_for_localization=visibility_state
                in {
                    BackpackVisibilityState.PARTIALLY_OCCLUDED,
                    BackpackVisibilityState.FULLY_OCCLUDED,
                },
                rationale=rationale,
                evidence_refs=evidence_refs,
            )
        )
    return tuple(output)


def _write_csv(records: tuple[BackpackVisibilityRecord, ...], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "source_frame_index",
                "capture_timestamp_seconds",
                "camera_a_detection_state",
                "camera_b_detection_state",
                "visibility_state",
                "evidence_source",
                "confirmed_occluded_for_localization",
                "supplies_xyz",
                "rationale",
            )
        )
        for record in records:
            writer.writerow(
                (
                    record.source_frame_index,
                    f"{record.capture_timestamp_seconds:.9f}",
                    record.camera_a_detection_state.value,
                    record.camera_b_detection_state.value,
                    record.visibility_state.value,
                    record.evidence_source.value,
                    str(record.confirmed_occluded_for_localization).lower(),
                    "false",
                    record.rationale,
                )
            )


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
