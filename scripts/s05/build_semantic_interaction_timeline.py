"""Build the orthogonal S05 v2 interaction, visibility, and localization timeline."""

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

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from spatial_reconstruction.interaction import (
    InteractionPhase,
    InteractionZone,
    LocalizationAvailability,
    SemanticEventCandidate,
    SemanticInteractionPolicy,
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
    parser.add_argument(
        "--temporal-summary",
        type=Path,
        default=Path(
            "artifacts/s04/temporal_presentation_occlusion_aware_20260803_v2/summary.json"
        ),
    )
    parser.add_argument(
        "--temporal-verification",
        type=Path,
        default=Path(
            "artifacts/s04/temporal_presentation_occlusion_aware_20260803_v2/verification.json"
        ),
    )
    parser.add_argument(
        "--visibility-summary",
        type=Path,
        default=Path("artifacts/s05/backpack_visibility_20260803/summary.json"),
    )
    parser.add_argument(
        "--visibility-verification",
        type=Path,
        default=Path("artifacts/s05/backpack_visibility_20260803/verification.json"),
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
        "visibility": _resolve(root, args.visibility_summary),
        "visibility_verification": _resolve(root, args.visibility_verification),
        "zones": _resolve(root, args.zone_metadata),
        "synchronization": _resolve(root, args.synchronization_manifest),
    }
    output_dir = _resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    temporal = TemporalPresentationRunSummary.model_validate_json(
        paths["temporal"].read_text(encoding="utf-8")
    )
    visibility = BackpackVisibilityRunSummary.model_validate_json(
        paths["visibility"].read_text(encoding="utf-8")
    )
    temporal_verification = _read_object(paths["temporal_verification"])
    visibility_verification = _read_object(paths["visibility_verification"])
    zone_payload = _read_object(paths["zones"])
    synchronization = _read_object(paths["synchronization"])
    _verify_source(
        source_path=paths["temporal"],
        verification=temporal_verification,
        label="temporal presentation",
    )
    _verify_source(
        source_path=paths["visibility"],
        verification=visibility_verification,
        label="visibility overlay",
    )
    if temporal.source_visibility_summary_sha256 != _sha256(paths["visibility"]):
        raise ValueError("S04 temporal artifact does not reference this visibility overlay")
    pickup_zone, dropoff_zone = _load_zones(zone_payload)
    video_duration = _video_duration(synchronization)
    policy = SemanticInteractionPolicy()
    records = build_semantic_interaction_timeline(
        presentation_records=temporal.presentation_records,
        visibility_records=visibility.records,
        pickup_zone=pickup_zone,
        dropoff_zone=dropoff_zone,
        policy=policy,
    )
    candidates = build_semantic_event_candidates(
        records=records,
        video_duration_seconds=video_duration,
        policy=policy,
    )
    _validate_real_result(records=records, candidates=candidates)

    records_path = output_dir / "semantic_interaction_records.json"
    records_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "stage": "S05",
                "policy_id": policy.policy_id,
                "records": [record.model_dump(mode="json") for record in records],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    candidates_path = output_dir / "semantic_event_candidates.json"
    candidates_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "stage": "S05",
                "policy_id": policy.policy_id,
                "candidates": [item.model_dump(mode="json") for item in candidates],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "semantic_interaction_review.csv"
    _write_csv(records, csv_path)
    timeline_path = output_dir / "semantic_visibility_localization_timeline.png"
    _save_timeline(records, candidates=candidates, path=timeline_path)

    phase_counts = dict(sorted(Counter(record.phase.value for record in records).items()))
    visibility_counts = dict(
        sorted(Counter(record.visibility_state.value for record in records).items())
    )
    localization_counts = dict(
        sorted(Counter(record.localization_availability.value for record in records).items())
    )
    summary = SemanticInteractionRunSummary(
        status="completed_pending_visual_qa",
        created_at_utc=datetime.now(UTC),
        policy=policy,
        pickup_zone=pickup_zone,
        dropoff_zone=dropoff_zone,
        source_temporal_summary_ref=_relative(paths["temporal"], root),
        source_temporal_summary_sha256=_sha256(paths["temporal"]),
        source_temporal_verification_ref=_relative(paths["temporal_verification"], root),
        source_temporal_verification_sha256=_sha256(paths["temporal_verification"]),
        source_visibility_summary_ref=_relative(paths["visibility"], root),
        source_visibility_summary_sha256=_sha256(paths["visibility"]),
        source_visibility_verification_ref=_relative(
            paths["visibility_verification"], root
        ),
        source_visibility_verification_sha256=_sha256(
            paths["visibility_verification"]
        ),
        source_zone_metadata_ref=_relative(paths["zones"], root),
        source_zone_metadata_sha256=_sha256(paths["zones"]),
        source_synchronization_manifest_ref=_relative(paths["synchronization"], root),
        source_synchronization_manifest_sha256=_sha256(paths["synchronization"]),
        records=records,
        event_candidates=candidates,
        phase_counts=phase_counts,
        visibility_counts=visibility_counts,
        localization_counts=localization_counts,
        records_ref=_relative(records_path, root),
        records_sha256=_sha256(records_path),
        candidates_ref=_relative(candidates_path, root),
        candidates_sha256=_sha256(candidates_path),
        review_csv_ref=_relative(csv_path, root),
        review_csv_sha256=_sha256(csv_path),
        timeline_diagnostic_ref=_relative(timeline_path, root),
        timeline_diagnostic_sha256=_sha256(timeline_path),
        limitations=(
            "Carry during unavailable localization is a bounded semantic sequence "
            "hypothesis, not measured XYZ.",
            "Visibility and detector presence do not supply coordinates, zones, or "
            "trajectory points.",
            "The reviewed carry interval is partially occluded; the backpack can "
            "still be partly visible while undetected.",
            "Qwen review remains pending and cannot alter spatial facts, identity, "
            "timestamps, or zones.",
        ),
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    carry_without_xyz = [
        record
        for record in records
        if record.phase is InteractionPhase.CARRY
        and record.localization_availability is LocalizationAvailability.UNAVAILABLE
        and record.backpack_world_xyz_m is None
    ]
    print(
        json.dumps(
            {
                "record_count": len(records),
                "phase_counts": phase_counts,
                "visibility_counts": visibility_counts,
                "localization_counts": localization_counts,
                "candidate_kinds": [item.event_kind.value for item in candidates],
                "carry_without_xyz_count": len(carry_without_xyz),
                "invented_xyz_count": 0,
            },
            indent=2,
        )
    )
    return 0


def _verify_source(
    *, source_path: Path, verification: dict[str, Any], label: str
) -> None:
    if (
        verification.get("status") != "passed"
        or verification.get("source_summary_sha256") != _sha256(source_path)
    ):
        raise ValueError(f"S05 v2 requires matching passed {label} verification")


def _load_zones(payload: dict[str, Any]) -> tuple[InteractionZone, InteractionZone]:
    if payload.get("automated_checks_passed") is not True:
        raise ValueError("S05 v2 requires zones that passed automated checks")
    validation = payload.get("user_validation")
    if not isinstance(validation, dict) or validation.get("accepted") is not True:
        raise ValueError("S05 v2 requires user-accepted zones")
    zones = payload.get("zones")
    if not isinstance(zones, dict):
        raise ValueError("zone metadata lacks zones")
    return (
        _load_zone(zones, "pickup_blue_bed", "pickup"),
        _load_zone(zones, "dropoff_white_floor", "dropoff"),
    )


def _load_zone(zones: dict[str, Any], zone_id: str, role: str) -> InteractionZone:
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


def _video_duration(payload: dict[str, Any]) -> float:
    outputs = payload.get("derived_outputs")
    if not isinstance(outputs, dict):
        raise ValueError("synchronization manifest lacks derived outputs")
    durations = []
    for camera_id in ("camera_a", "camera_b"):
        value = outputs.get(camera_id)
        if not isinstance(value, dict):
            raise ValueError(f"synchronization manifest lacks {camera_id}")
        durations.append(float(value["duration_seconds"]))
    if abs(durations[0] - durations[1]) > 1e-6:
        raise ValueError("synchronized video durations differ")
    return min(durations)


def _validate_real_result(
    *,
    records: tuple[SemanticInteractionRecord, ...],
    candidates: tuple[SemanticEventCandidate, ...],
) -> None:
    if [item.event_kind.value for item in candidates] != ["pickup", "carry", "place"]:
        raise ValueError("S05 v2 must produce pickup, carry, then place candidates")
    if [item.source_frame_index for item in candidates] != [462, 468, 666]:
        raise ValueError("S05 v2 event boundaries differ from retained evidence")
    carry = [record for record in records if record.phase is InteractionPhase.CARRY]
    if not carry or carry[0].source_frame_index != 468 or carry[-1].source_frame_index != 660:
        raise ValueError("S05 v2 carry interval differs from the bounded evidence")
    if any(
        record.localization_availability is not LocalizationAvailability.UNAVAILABLE
        or record.backpack_world_xyz_m is not None
        or record.phase_has_current_spatial_authority
        or record.visibility_state is not BackpackVisibilityState.PARTIALLY_OCCLUDED
        for record in carry
    ):
        raise ValueError("unlocalized carry gained XYZ/authority or lost visibility evidence")


def _write_csv(records: tuple[SemanticInteractionRecord, ...], path: Path) -> None:
    fields = (
        "source_frame_index",
        "capture_timestamp_seconds",
        "phase",
        "phase_authority",
        "reason",
        "visibility_state",
        "visibility_evidence_source",
        "localization_availability",
        "backpack_presentation_state",
        "backpack_world_xyz_m",
        "backpack_zone_membership",
        "person_backpack_distance_xy_m",
        "phase_has_current_spatial_authority",
        "invented_xyz",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            payload = record.model_dump(mode="json")
            writer.writerow({field: payload.get(field) for field in fields})


def _save_timeline(
    records: tuple[SemanticInteractionRecord, ...],
    *,
    candidates: tuple[SemanticEventCandidate, ...],
    path: Path,
) -> None:
    axes_values: tuple[tuple[str, tuple[str, ...], list[str]], ...] = (
        (
            "Interaction phase",
            tuple(value.value for value in InteractionPhase),
            [record.phase.value for record in records],
        ),
        (
            "Visibility condition",
            tuple(value.value for value in BackpackVisibilityState),
            [record.visibility_state.value for record in records],
        ),
        (
            "Localization availability",
            tuple(value.value for value in LocalizationAvailability),
            [record.localization_availability.value for record in records],
        ),
    )
    times = [record.capture_timestamp_seconds for record in records]
    figure, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    for axis, (title, order, values) in zip(axes, axes_values, strict=True):
        mapping = {value: index for index, value in enumerate(order)}
        axis.scatter(times, [mapping[value] for value in values], s=18, color="#1565c0")
        axis.set_yticks(range(len(order)), order)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        for candidate in candidates:
            axis.axvline(candidate.capture_timestamp_seconds, color="#ef6c00", alpha=0.7)
    axes[-1].set_xlabel("Capture timestamp (seconds)")
    figure.suptitle(
        "S05 v2: semantic carry can coexist with partial occlusion and unavailable XYZ",
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
