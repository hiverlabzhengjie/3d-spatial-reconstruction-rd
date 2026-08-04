"""Independently verify the measured-only S05 interaction timeline artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from spatial_reconstruction.interaction import (
    BackpackInteractionState,
    InteractionEventCandidate,
    InteractionStateRecord,
    InteractionTimelineRunSummary,
    build_event_candidates,
    build_interaction_timeline,
)
from spatial_reconstruction.localization import TemporalPresentationRunSummary


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
        raise ValueError("explicit timeline and candidate-frame visual QA is required")

    summary = InteractionTimelineRunSummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    source_paths = {
        "temporal": _resolve(root, Path(summary.source_temporal_summary_ref)),
        "temporal_verification": _resolve(
            root, Path(summary.source_temporal_verification_ref)
        ),
        "zones": _resolve(root, Path(summary.source_zone_metadata_ref)),
        "synchronization": _resolve(
            root, Path(summary.source_synchronization_manifest_ref)
        ),
        "camera_a": _resolve(root, Path(summary.source_camera_a_video_ref)),
        "camera_b": _resolve(root, Path(summary.source_camera_b_video_ref)),
    }
    source_hashes = {
        "temporal": summary.source_temporal_summary_sha256,
        "temporal_verification": summary.source_temporal_verification_sha256,
        "zones": summary.source_zone_metadata_sha256,
        "synchronization": summary.source_synchronization_manifest_sha256,
        "camera_a": summary.source_camera_a_video_sha256,
        "camera_b": summary.source_camera_b_video_sha256,
    }
    for name, path in source_paths.items():
        _require_hash(path, source_hashes[name])

    temporal_verification = _read_object(source_paths["temporal_verification"])
    if (
        temporal_verification.get("status") != "passed"
        or temporal_verification.get("source_summary_sha256")
        != source_hashes["temporal"]
    ):
        raise ValueError("S05 source lacks matching passed D034 verification")
    temporal = TemporalPresentationRunSummary.model_validate_json(
        source_paths["temporal"].read_text(encoding="utf-8")
    )
    regenerated_records = build_interaction_timeline(
        presentation_records=temporal.presentation_records,
        pickup_zone=summary.pickup_zone,
        dropoff_zone=summary.dropoff_zone,
        policy=summary.policy,
    )
    if regenerated_records != summary.state_records:
        raise ValueError("interaction state records do not regenerate")

    synchronization = _read_object(source_paths["synchronization"])
    duration = _video_duration(synchronization)
    regenerated_candidates = build_event_candidates(
        state_records=regenerated_records,
        video_duration_seconds=duration,
        policy=summary.policy,
    )
    if regenerated_candidates != summary.event_candidates:
        raise ValueError("interaction event candidates do not regenerate")

    artifact_paths = {
        "records": _resolve(root, Path(summary.records_ref)),
        "candidates": _resolve(root, Path(summary.candidates_ref)),
        "csv": _resolve(root, Path(summary.review_csv_ref)),
        "timeline": _resolve(root, Path(summary.timeline_diagnostic_ref)),
        "contact_sheet": _resolve(root, Path(summary.candidate_contact_sheet_ref)),
    }
    artifact_hashes = {
        "records": summary.records_sha256,
        "candidates": summary.candidates_sha256,
        "csv": summary.review_csv_sha256,
        "timeline": summary.timeline_diagnostic_sha256,
        "contact_sheet": summary.candidate_contact_sheet_sha256,
    }
    for name, path in artifact_paths.items():
        _require_hash(path, artifact_hashes[name])
    _verify_records(artifact_paths["records"], regenerated_records)
    _verify_candidates(artifact_paths["candidates"], regenerated_candidates)
    with artifact_paths["csv"].open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != len(regenerated_records):
        raise ValueError("interaction review CSV coverage differs")

    state_counts = dict(
        sorted(Counter(record.state.value for record in regenerated_records).items())
    )
    transition_counts = dict(
        sorted(
            Counter(candidate.event_kind.value for candidate in regenerated_candidates).items()
        )
    )
    if state_counts != summary.state_counts:
        raise ValueError("interaction state counts differ")
    if transition_counts != summary.transition_counts:
        raise ValueError("interaction transition counts differ")
    _verify_semantics(regenerated_records, regenerated_candidates)

    unknown_records = [
        record
        for record in regenerated_records
        if record.state is BackpackInteractionState.UNKNOWN
    ]
    verification = {
        "schema_version": 1,
        "stage": "S05",
        "status": "passed",
        "purpose": "measured_only_interaction_state_verification",
        "source_summary_ref": _relative(summary_path, root),
        "source_summary_sha256": _sha256(summary_path),
        "visual_qa_passed": True,
        "state_records_regenerated": True,
        "event_candidates_regenerated": True,
        "source_video_hashes_verified": True,
        "state_record_count": len(regenerated_records),
        "state_counts": state_counts,
        "transition_counts": transition_counts,
        "candidate_boundaries": [
            {
                "event_kind": candidate.event_kind.value,
                "source_frame_index": candidate.source_frame_index,
                "capture_timestamp_seconds": candidate.capture_timestamp_seconds,
            }
            for candidate in regenerated_candidates
        ],
        "unknown_record_count": len(unknown_records),
        "unknown_records_with_spatial_authority": sum(
            record.spatial_transition_authority for record in unknown_records
        ),
        "occluded_record_count": sum(
            record.state is BackpackInteractionState.OCCLUDED
            for record in regenerated_records
        ),
        "invented_xyz_count": sum(record.invented_xyz for record in regenerated_records),
        "qwen_influenced_spatial_state_count": sum(
            record.qwen_influenced_spatial_state for record in regenerated_records
        ),
        "stale_or_missing_zone_fact_count": sum(
            record.backpack_zone_membership.value != "unknown"
            and not record.spatial_transition_authority
            and record.reason.value == "spatial_evidence_unavailable"
            for record in regenerated_records
        ),
        "measured_carry_transition_present": any(
            record.state is BackpackInteractionState.CARRY
            for record in regenerated_records
        ),
        "known_backpack_gap_filled": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))
    return 0


def _verify_semantics(
    records: tuple[InteractionStateRecord, ...],
    candidates: tuple[InteractionEventCandidate, ...],
) -> None:
    if len(records) != 160:
        raise ValueError("S05 retained timeline must contain 160 records")
    if [(item.event_kind.value, item.source_frame_index) for item in candidates] != [
        ("pickup", 462),
        ("place", 666),
    ]:
        raise ValueError("S05 retained event boundaries differ")
    if any(
        record.state in {BackpackInteractionState.UNKNOWN, BackpackInteractionState.OCCLUDED}
        and record.spatial_transition_authority
        for record in records
    ):
        raise ValueError("unavailable state claims spatial transition authority")
    if any(record.invented_xyz or record.qwen_influenced_spatial_state for record in records):
        raise ValueError("S05 state contains invented or Qwen-controlled spatial facts")


def _verify_records(path: Path, expected: tuple[InteractionStateRecord, ...]) -> None:
    payload = _read_object(path)
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("interaction records artifact lacks records")
    parsed = tuple(InteractionStateRecord.model_validate(item) for item in records)
    if parsed != expected:
        raise ValueError("persistent interaction records differ")


def _verify_candidates(
    path: Path, expected: tuple[InteractionEventCandidate, ...]
) -> None:
    payload = _read_object(path)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("interaction candidates artifact lacks candidates")
    parsed = tuple(InteractionEventCandidate.model_validate(item) for item in candidates)
    if parsed != expected:
        raise ValueError("persistent interaction candidates differ")


def _video_duration(synchronization: dict[str, Any]) -> float:
    outputs = synchronization.get("derived_outputs")
    if not isinstance(outputs, dict):
        raise ValueError("synchronization manifest lacks derived outputs")
    durations: list[float] = []
    for camera_id in ("camera_a", "camera_b"):
        value = outputs.get(camera_id)
        if not isinstance(value, dict):
            raise ValueError(f"manifest lacks {camera_id}")
        durations.append(float(value["duration_seconds"]))
    if abs(durations[0] - durations[1]) > 1e-6:
        raise ValueError("synchronized video durations differ")
    return min(durations)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _require_hash(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"artifact hash differs: {path}")


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
