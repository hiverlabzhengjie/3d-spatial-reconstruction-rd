"""Verify the orthogonal S05 v2 semantic interaction artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from spatial_reconstruction.interaction import (
    InteractionPhase,
    LocalizationAvailability,
    PhaseAuthority,
    SemanticEventCandidate,
    SemanticInteractionRecord,
    SemanticInteractionRunSummary,
    build_semantic_event_candidates,
    build_semantic_interaction_timeline,
)
from spatial_reconstruction.localization import TemporalPresentationRunSummary
from spatial_reconstruction.perception import (
    BackpackVisibilityRunSummary,
    BackpackVisibilityState,
)


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
        raise ValueError("explicit three-axis timeline visual QA is required")
    summary = SemanticInteractionRunSummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    source_paths = {
        "temporal": _resolve(root, Path(summary.source_temporal_summary_ref)),
        "temporal_verification": _resolve(
            root, Path(summary.source_temporal_verification_ref)
        ),
        "visibility": _resolve(root, Path(summary.source_visibility_summary_ref)),
        "visibility_verification": _resolve(
            root, Path(summary.source_visibility_verification_ref)
        ),
        "zones": _resolve(root, Path(summary.source_zone_metadata_ref)),
        "synchronization": _resolve(
            root, Path(summary.source_synchronization_manifest_ref)
        ),
    }
    hashes = {
        "temporal": summary.source_temporal_summary_sha256,
        "temporal_verification": summary.source_temporal_verification_sha256,
        "visibility": summary.source_visibility_summary_sha256,
        "visibility_verification": summary.source_visibility_verification_sha256,
        "zones": summary.source_zone_metadata_sha256,
        "synchronization": summary.source_synchronization_manifest_sha256,
    }
    for key, path in source_paths.items():
        _require_hash(path, hashes[key])
    _require_passed_verification(
        source_paths["temporal_verification"], hashes["temporal"]
    )
    _require_passed_verification(
        source_paths["visibility_verification"], hashes["visibility"]
    )
    temporal = TemporalPresentationRunSummary.model_validate_json(
        source_paths["temporal"].read_text(encoding="utf-8")
    )
    visibility = BackpackVisibilityRunSummary.model_validate_json(
        source_paths["visibility"].read_text(encoding="utf-8")
    )
    regenerated = build_semantic_interaction_timeline(
        presentation_records=temporal.presentation_records,
        visibility_records=visibility.records,
        pickup_zone=summary.pickup_zone,
        dropoff_zone=summary.dropoff_zone,
        policy=summary.policy,
    )
    duration = _video_duration(_read_object(source_paths["synchronization"]))
    regenerated_candidates = build_semantic_event_candidates(
        records=regenerated,
        video_duration_seconds=duration,
        policy=summary.policy,
    )
    if regenerated != summary.records:
        raise ValueError("S05 v2 records do not regenerate")
    if regenerated_candidates != summary.event_candidates:
        raise ValueError("S05 v2 candidates do not regenerate")

    artifact_paths = {
        "records": _resolve(root, Path(summary.records_ref)),
        "candidates": _resolve(root, Path(summary.candidates_ref)),
        "csv": _resolve(root, Path(summary.review_csv_ref)),
        "timeline": _resolve(root, Path(summary.timeline_diagnostic_ref)),
    }
    artifact_hashes = {
        "records": summary.records_sha256,
        "candidates": summary.candidates_sha256,
        "csv": summary.review_csv_sha256,
        "timeline": summary.timeline_diagnostic_sha256,
    }
    for key, path in artifact_paths.items():
        _require_hash(path, artifact_hashes[key])
    _verify_persistent(
        artifact_paths["records"], "records", SemanticInteractionRecord, regenerated
    )
    _verify_persistent(
        artifact_paths["candidates"],
        "candidates",
        SemanticEventCandidate,
        regenerated_candidates,
    )
    with artifact_paths["csv"].open(encoding="utf-8", newline="") as handle:
        if len(list(csv.DictReader(handle))) != 160:
            raise ValueError("S05 v2 review CSV coverage differs")

    phase_counts = dict(sorted(Counter(record.phase.value for record in regenerated).items()))
    visibility_counts = dict(
        sorted(Counter(record.visibility_state.value for record in regenerated).items())
    )
    localization_counts = dict(
        sorted(Counter(record.localization_availability.value for record in regenerated).items())
    )
    if (
        phase_counts != summary.phase_counts
        or visibility_counts != summary.visibility_counts
        or localization_counts != summary.localization_counts
    ):
        raise ValueError("S05 v2 axis counts differ")
    carry = tuple(record for record in regenerated if record.phase is InteractionPhase.CARRY)
    if [item.event_kind.value for item in regenerated_candidates] != [
        "pickup",
        "carry",
        "place",
    ]:
        raise ValueError("S05 v2 candidate sequence differs")
    if not carry or any(
        record.phase_authority is not PhaseAuthority.SEQUENCE_CONTINUITY
        or record.visibility_state is not BackpackVisibilityState.PARTIALLY_OCCLUDED
        or record.localization_availability is not LocalizationAvailability.UNAVAILABLE
        or record.backpack_world_xyz_m is not None
        or record.phase_has_current_spatial_authority
        for record in carry
    ):
        raise ValueError("carry does not preserve orthogonal authority semantics")
    verification = {
        "schema_version": 2,
        "stage": "S05",
        "status": "passed",
        "purpose": "orthogonal_semantic_interaction_verification",
        "source_summary_ref": _relative(summary_path, root),
        "source_summary_sha256": _sha256(summary_path),
        "visual_qa_passed": True,
        "records_regenerated": True,
        "candidates_regenerated": True,
        "record_count": len(regenerated),
        "phase_counts": phase_counts,
        "visibility_counts": visibility_counts,
        "localization_counts": localization_counts,
        "candidate_kinds": [item.event_kind.value for item in regenerated_candidates],
        "carry_start_frame": carry[0].source_frame_index,
        "carry_end_frame": carry[-1].source_frame_index,
        "carry_record_count": len(carry),
        "carry_partially_occluded_count": sum(
            item.visibility_state is BackpackVisibilityState.PARTIALLY_OCCLUDED
            for item in carry
        ),
        "carry_localization_unavailable_count": sum(
            item.localization_availability is LocalizationAvailability.UNAVAILABLE
            for item in carry
        ),
        "carry_with_xyz_count": sum(item.backpack_world_xyz_m is not None for item in carry),
        "carry_with_spatial_authority_count": sum(
            item.phase_has_current_spatial_authority for item in carry
        ),
        "invented_xyz_count": sum(item.invented_xyz for item in regenerated),
        "qwen_spatial_influence_count": sum(
            item.qwen_influenced_spatial_facts for item in regenerated
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))
    return 0


def _verify_persistent(
    path: Path,
    key: str,
    model: type[SemanticInteractionRecord] | type[SemanticEventCandidate],
    expected: tuple[SemanticInteractionRecord, ...] | tuple[SemanticEventCandidate, ...],
) -> None:
    values = _read_object(path).get(key)
    if not isinstance(values, list):
        raise ValueError(f"persistent {key} are not a list")
    actual = tuple(model.model_validate(value) for value in values)
    if actual != expected:
        raise ValueError(f"persistent {key} differ")


def _require_passed_verification(path: Path, expected_source_hash: str) -> None:
    payload = _read_object(path)
    if (
        payload.get("status") != "passed"
        or payload.get("source_summary_sha256") != expected_source_hash
    ):
        raise ValueError(f"verification does not match source: {path}")


def _video_duration(payload: dict[str, Any]) -> float:
    outputs = payload.get("derived_outputs")
    if not isinstance(outputs, dict):
        raise ValueError("synchronization manifest lacks derived outputs")
    durations = [
        float(cast(dict[str, Any], outputs[camera_id])["duration_seconds"])
        for camera_id in ("camera_a", "camera_b")
    ]
    return min(durations)


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
