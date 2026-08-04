"""Independently verify the S05 backpack-visibility overlay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from spatial_reconstruction.contracts import PerceptionTarget
from spatial_reconstruction.perception import (
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
        raise ValueError("explicit synchronized-video visual QA is required")
    summary = BackpackVisibilityRunSummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    source_paths = {
        "perception": _resolve(root, Path(summary.source_perception_summary_ref)),
        "camera_a": _resolve(root, Path(summary.source_camera_a_timeline_ref)),
        "camera_b": _resolve(root, Path(summary.source_camera_b_timeline_ref)),
    }
    expected_hashes = {
        "perception": summary.source_perception_summary_sha256,
        "camera_a": summary.source_camera_a_timeline_sha256,
        "camera_b": summary.source_camera_b_timeline_sha256,
    }
    for key, path in source_paths.items():
        _require_hash(path, expected_hashes[key])
    perception = _read_object(source_paths["perception"])
    if (
        perception.get("stage") != "S03"
        or perception.get("occlusion_inference") is not False
    ):
        raise ValueError(
            "visibility overlay source is not the retained non-inference S03 timeline"
        )

    camera_records = {
        camera_id: _load_backpack_timeline(source_paths[camera_id])
        for camera_id in ("camera_a", "camera_b")
    }
    regenerated = _regenerate(camera_records=camera_records, summary=summary)
    if regenerated != summary.records:
        raise ValueError("visibility overlay records do not regenerate")
    counts = dict(sorted(Counter(record.visibility_state.value for record in regenerated).items()))
    if counts != summary.state_counts:
        raise ValueError("visibility state counts differ")

    records_path = _resolve(root, Path(summary.records_ref))
    csv_path = _resolve(root, Path(summary.review_csv_ref))
    _require_hash(records_path, summary.records_sha256)
    _require_hash(csv_path, summary.review_csv_sha256)
    persistent = _read_object(records_path).get("records")
    if not isinstance(persistent, list):
        raise ValueError("persistent visibility records are not a list")
    if tuple(BackpackVisibilityRecord.model_validate(item) for item in persistent) != regenerated:
        raise ValueError("persistent visibility records differ")
    with csv_path.open(encoding="utf-8", newline="") as handle:
        if len(list(csv.DictReader(handle))) != len(regenerated):
            raise ValueError("visibility review CSV coverage differs")

    reviewed = tuple(
        record
        for record in regenerated
        if record.evidence_source is VisibilityEvidenceSource.SYNCHRONIZED_VIDEO_REVIEW
    )
    if not reviewed or any(not record.evidence_refs for record in reviewed):
        raise ValueError("no review-backed visibility interval was retained")
    if any(record.supplies_xyz for record in regenerated):
        raise ValueError("visibility overlay supplied XYZ")
    verification = {
        "schema_version": 1,
        "stage": "S05",
        "status": "passed",
        "purpose": "backpack_visibility_overlay_verification",
        "source_summary_ref": _relative(summary_path, root),
        "source_summary_sha256": _sha256(summary_path),
        "visual_qa_passed": True,
        "records_regenerated": True,
        "record_count": len(regenerated),
        "state_counts": counts,
        "reviewed_record_count": len(reviewed),
        "confirmed_occluded_for_localization_count": sum(
            record.confirmed_occluded_for_localization for record in regenerated
        ),
        "detector_missing_automatically_labelled_occluded_count": 0,
        "supplied_xyz_count": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))
    return 0


def _load_backpack_timeline(path: Path) -> tuple[PerceptionTargetFrameState, ...]:
    values = _read_object(path).get("records")
    if not isinstance(values, list):
        raise ValueError(f"timeline records are not a list: {path}")
    return tuple(
        record
        for value in values
        if (record := PerceptionTargetFrameState.model_validate(value)).target
        is PerceptionTarget.BACKPACK
    )


def _regenerate(
    *,
    camera_records: dict[str, tuple[PerceptionTargetFrameState, ...]],
    summary: BackpackVisibilityRunSummary,
) -> tuple[BackpackVisibilityRecord, ...]:
    lookups = {
        camera_id: {record.frame_identity.source_frame_index: record for record in records}
        for camera_id, records in camera_records.items()
    }
    if set(lookups["camera_a"]) != set(lookups["camera_b"]) or len(lookups["camera_a"]) != 160:
        raise ValueError("visibility source camera grids differ")
    output: list[BackpackVisibilityRecord] = []
    for frame in sorted(lookups["camera_a"]):
        state_a = lookups["camera_a"][frame]
        state_b = lookups["camera_b"][frame]
        timestamp = state_a.frame_identity.capture_timestamp_seconds
        review = next(
            (
                interval
                for interval in summary.policy.review_intervals
                if interval.start_source_frame_index <= frame <= interval.end_source_frame_index
            ),
            None,
        )
        visibility: BackpackVisibilityState
        source: VisibilityEvidenceSource
        rationale: str
        refs: tuple[str, ...]
        if review is not None:
            visibility = review.visibility_state
            source = review.evidence_source
            rationale = review.rationale
            refs = review.evidence_refs
        elif state_a.state in DETECTED_STATES or state_b.state in DETECTED_STATES:
            visibility = BackpackVisibilityState.VISIBLE
            source = VisibilityEvidenceSource.DETECTOR_OBSERVATION
            rationale = "At least one retained S03 camera timeline contains a backpack candidate."
            refs = ()
        else:
            visibility = BackpackVisibilityState.UNKNOWN
            source = VisibilityEvidenceSource.NONE
            rationale = (
                "Neither detector presence nor explicit synchronized-video review "
                "establishes visibility."
            )
            refs = ()
        output.append(
            BackpackVisibilityRecord(
                record_id=BackpackVisibilityRecord.create_record_id(
                    policy_id=summary.policy.policy_id,
                    source_frame_index=frame,
                    capture_timestamp_seconds=timestamp,
                ),
                policy_id=summary.policy.policy_id,
                source_frame_index=frame,
                capture_timestamp_seconds=timestamp,
                camera_a_detection_state=state_a.state,
                camera_b_detection_state=state_b.state,
                visibility_state=visibility,
                evidence_source=source,
                confirmed_occluded_for_localization=visibility
                in {
                    BackpackVisibilityState.PARTIALLY_OCCLUDED,
                    BackpackVisibilityState.FULLY_OCCLUDED,
                },
                rationale=rationale,
                evidence_refs=refs,
            )
        )
    return tuple(output)


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
