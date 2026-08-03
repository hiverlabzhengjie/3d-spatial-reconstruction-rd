"""Independently verify the corrected, margin-aware S04 D030-D033 rebuild."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from spatial_reconstruction.contracts import PerceptionTarget
from spatial_reconstruction.localization import (
    ActionDepthRunSummary,
    CorrectedAnchor,
    CorrectedAnchorKind,
    CorrectedPairState,
    CorrectedSurfaceRole,
    CorrectedTrackingRunSummary,
    MaskAlignmentRunSummary,
    PairAnchorInput,
    anchor_reliability,
    derive_corrected_anchor,
    localize_corrected_surface,
    resolve_corrected_pair,
    surface_statistics,
)

Float32Array = NDArray[np.float32]
UInt8Array = NDArray[np.uint8]
CameraId = Literal["camera_a", "camera_b"]
CAMERA_IDS: tuple[CameraId, CameraId] = ("camera_a", "camera_b")
CAMERA_INDEX = {"camera_a": 0, "camera_b": 1}


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
        raise ValueError("explicit margin and world diagnostic visual QA is required")
    summary = CorrectedTrackingRunSummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    source_paths = {
        "action": _resolve(root, Path(summary.source_action_depth_summary_ref)),
        "scale": _resolve(root, Path(summary.source_depth_scale_summary_ref)),
        "scale_verification": _resolve(
            root, Path(summary.source_depth_scale_verification_ref)
        ),
        "alignment": _resolve(root, Path(summary.source_mask_alignment_summary_ref)),
        "calibration": _resolve(root, Path(summary.pose_calibration_ref)),
        "scene": _resolve(root, Path(summary.scene_metadata_ref)),
    }
    source_hashes = {
        "action": summary.source_action_depth_summary_sha256,
        "scale": summary.source_depth_scale_summary_sha256,
        "scale_verification": summary.source_depth_scale_verification_sha256,
        "alignment": summary.source_mask_alignment_summary_sha256,
        "calibration": summary.pose_calibration_sha256,
        "scene": summary.scene_metadata_sha256,
    }
    for name, path in source_paths.items():
        _require_hash(path, source_hashes[name])
    action = ActionDepthRunSummary.model_validate_json(
        source_paths["action"].read_text(encoding="utf-8")
    )
    alignment = MaskAlignmentRunSummary.model_validate_json(
        source_paths["alignment"].read_text(encoding="utf-8")
    )
    scale = _read_object(source_paths["scale"])
    scale_verification = _read_object(source_paths["scale_verification"])
    if (
        scale_verification.get("status") != "passed"
        or scale_verification.get("source_summary_sha256") != source_hashes["scale"]
    ):
        raise ValueError("selected D025 summary lacks matching passed verification")

    predictions = {record.job.job_id: record for record in action.predictions}
    scale_jobs = {str(record["job_id"]): record for record in scale["jobs"]}
    aligned_lookup = {
        (record.action_depth_job_id, record.camera_id, record.target): record
        for record in alignment.aligned_masks
    }
    anchor_lookup = {
        (record.action_depth_job_id, record.camera_id, record.target): record
        for record in summary.d032_anchor_records
    }
    raw_cache: dict[str, tuple[Float32Array, Float32Array]] = {}
    corrected_cache: dict[str, Float32Array] = {}
    mask_cache: dict[str, UInt8Array] = {}
    runtime_anchors: dict[tuple[str, str, PerceptionTarget], CorrectedAnchor] = {}
    maximum_reprojection = 0.0
    maximum_round_trip = 0.0
    for record in summary.d030_surface_records:
        aligned_record = aligned_lookup[
            (record.action_depth_job_id, record.camera_id, record.target)
        ]
        if (
            aligned_record.aligned_mask_artifact_ref
            != record.aligned_mask_artifact_ref
            or aligned_record.aligned_mask_index != record.aligned_mask_index
        ):
            raise ValueError("corrected surface aligned-mask identity differs")
        prediction = predictions[record.action_depth_job_id]
        scale_job = scale_jobs[record.action_depth_job_id]
        camera_index = CAMERA_INDEX[record.camera_id]
        raw = raw_cache.get(record.raw_prediction_ref)
        if raw is None:
            raw_path = _resolve(root, Path(record.raw_prediction_ref))
            _require_hash(raw_path, record.raw_prediction_sha256)
            with np.load(raw_path, allow_pickle=False) as arrays:
                raw = (
                    cast(Float32Array, np.asarray(arrays["depth_m"]).copy()),
                    cast(Float32Array, np.asarray(arrays["confidence"]).copy()),
                )
            raw_cache[record.raw_prediction_ref] = raw
        corrected = corrected_cache.get(record.corrected_prediction_ref)
        if corrected is None:
            corrected_path = _resolve(root, Path(record.corrected_prediction_ref))
            _require_hash(corrected_path, record.corrected_prediction_sha256)
            with np.load(corrected_path, allow_pickle=False) as arrays:
                corrected = cast(
                    Float32Array, np.asarray(arrays["corrected_depth_m"]).copy()
                )
                if str(arrays["raw_prediction_sha256"].item()) != record.raw_prediction_sha256:
                    raise ValueError("corrected surface raw source hash differs")
            corrected_cache[record.corrected_prediction_ref] = corrected
        masks = mask_cache.get(record.aligned_mask_artifact_ref)
        if masks is None:
            mask_path = _resolve(root, Path(record.aligned_mask_artifact_ref))
            _require_hash(mask_path, record.aligned_mask_artifact_sha256)
            with np.load(mask_path, allow_pickle=False) as arrays:
                masks = cast(UInt8Array, np.asarray(arrays["masks"]).copy())
            mask_cache[record.aligned_mask_artifact_ref] = masks
        raw_depth, confidence = raw
        expected_corrected = raw_depth * np.float32(record.depth_scale)
        if not np.array_equal(corrected, expected_corrected):
            raise ValueError("D031 corrected input is not exact raw-times-pair-scale")
        if not np.isclose(
            record.depth_scale, scale_job["scale_estimate"]["scale"], atol=1e-12
        ):
            raise ValueError("D031 depth scale differs from D025 pair scale")
        surface = localize_corrected_surface(
            source_mask=masks[record.aligned_mask_index],
            corrected_depth_m=corrected[camera_index],
            confidence=confidence[camera_index],
            target=record.target,
            intrinsics=record.processed_intrinsics,
            pose=record.camera_pose,
            policy=summary.policy,
        )
        depth_stats, confidence_stats = surface_statistics(surface)
        _require_close(surface.aggregate_camera_xyz_m, record.aggregate_camera_xyz_m)
        _require_close(surface.aggregate_world_xyz_m, record.aggregate_world_xyz_m)
        if (
            surface.role is not record.surface_role
            or surface.strategy is not record.candidate_strategy
            or surface.margin_assessment != record.person_margin_assessment
            or surface.candidate_pixel_count != record.candidate_pixel_count
            or surface.valid_candidate_count != record.valid_candidate_count
            or surface.retained_sample_count != record.retained_sample_count
            or depth_stats != record.retained_depth_m
            or confidence_stats != record.retained_confidence
        ):
            raise ValueError("D030/D031 corrected surface regeneration differs")
        sample_path = _resolve(root, Path(record.sample_cloud_ref))
        diagnostic_path = _resolve(root, Path(record.image_diagnostic_ref))
        _require_hash(sample_path, record.sample_cloud_sha256)
        _require_hash(diagnostic_path, record.image_diagnostic_sha256)
        with np.load(sample_path, allow_pickle=False) as arrays:
            for name, expected in (
                ("pixels_uv", surface.pixels_uv),
                ("corrected_depth_m", surface.depth_m),
                ("confidence", surface.confidence),
                ("points_camera_m", surface.points_camera_m),
                ("points_world_m", surface.points_world_m),
            ):
                if not np.array_equal(np.asarray(arrays[name]), expected):
                    raise ValueError(f"stored corrected sample array differs: {name}")
            if str(arrays["observation_id"].item()) != record.observation_id:
                raise ValueError("stored corrected sample observation identity differs")
        anchor = derive_corrected_anchor(
            surface, target=record.target, policy=summary.policy
        )
        anchor_record = anchor_lookup[
            (record.action_depth_job_id, record.camera_id, record.target)
        ]
        score = anchor_reliability(anchor, surface)
        if (
            anchor.kind is not anchor_record.kind
            or anchor.world_xyz_m != anchor_record.world_xyz_m
            or anchor.support_sample_count != anchor_record.support_sample_count
            or anchor.measured_support_world_z_m
            != anchor_record.measured_support_world_z_m
            or anchor.footpoint_available != anchor_record.footpoint_available
            or not np.isclose(score, anchor_record.reliability_score, atol=1e-12)
        ):
            raise ValueError("D032 anchor regeneration differs")
        runtime_anchors[
            (record.action_depth_job_id, record.camera_id, record.target)
        ] = anchor
        maximum_reprojection = max(
            maximum_reprojection, surface.reprojection_max_error_px
        )
        maximum_round_trip = max(maximum_round_trip, surface.round_trip_max_error_m)

    pair_lookup = {
        (record.action_depth_job_id, record.target): record
        for record in summary.d033_pair_observations
    }
    for prediction in action.predictions:
        for target in PerceptionTarget:
            inputs: list[PairAnchorInput] = []
            for camera_id in CAMERA_IDS:
                runtime = runtime_anchors.get((prediction.job.job_id, camera_id, target))
                persistent_anchor = anchor_lookup.get(
                    (prediction.job.job_id, camera_id, target)
                )
                inputs.append(
                    PairAnchorInput(
                        camera_id=camera_id,
                        anchor=runtime,
                        reliability_score=(
                            persistent_anchor.reliability_score
                            if persistent_anchor is not None
                            else None
                        ),
                    )
                )
            rebuilt = resolve_corrected_pair(
                target=target,
                camera_a=inputs[0],
                camera_b=inputs[1],
                maximum_disagreement_m=(
                    summary.policy.maximum_cross_camera_disagreement_m
                ),
            )
            stored = pair_lookup[(prediction.job.job_id, target)]
            if (
                rebuilt.state is not stored.state
                or rebuilt.selected_kind is not stored.selected_kind
                or rebuilt.world_xyz_m != stored.world_xyz_m
                or rebuilt.selected_camera_ids != stored.selected_camera_ids
                or rebuilt.disagreement_distance_m != stored.disagreement_distance_m
                or rebuilt.fallback_surface_used != stored.fallback_surface_used
            ):
                raise ValueError("D033 pair resolution regeneration differs")
            for index, source in enumerate(stored.sources):
                expected_weight = rebuilt.contribution_weights[index]
                if source.contribution_weight != expected_weight:
                    raise ValueError("D033 contribution weight differs")
                should_select = (
                    stored.state
                    in (CorrectedPairState.FUSED, CorrectedPairState.SINGLE_CAMERA)
                    and source.camera_id in stored.selected_camera_ids
                )
                if source.selected_for_output != should_select:
                    raise ValueError("D033 selected source evidence differs")

    artifact_pairs = (
        (summary.d030_sampling_summary_ref, summary.d030_sampling_summary_sha256),
        (
            summary.d031_visible_surface_summary_ref,
            summary.d031_visible_surface_summary_sha256,
        ),
        (summary.d032_anchor_summary_ref, summary.d032_anchor_summary_sha256),
        (
            summary.d033_observation_summary_ref,
            summary.d033_observation_summary_sha256,
        ),
        (summary.observation_csv_ref, summary.observation_csv_sha256),
        (summary.margin_contact_sheet_ref, summary.margin_contact_sheet_sha256),
        (summary.world_preview_ref, summary.world_preview_sha256),
    )
    for ref, expected_hash in artifact_pairs:
        _require_hash(_resolve(root, Path(ref)), expected_hash)

    person_surfaces = [
        record
        for record in summary.d030_surface_records
        if record.target is PerceptionTarget.PERSON
    ]
    person_anchors = [
        record
        for record in summary.d032_anchor_records
        if record.target is PerceptionTarget.PERSON
    ]
    person_pairs = [
        record
        for record in summary.d033_pair_observations
        if record.target is PerceptionTarget.PERSON
    ]
    verification = {
        "schema_version": 1,
        "stage": "S04",
        "status": "passed",
        "purpose": "corrected_margin_aware_d030_d033_verification",
        "source_summary_ref": _relative(summary_path, root),
        "source_summary_sha256": _sha256(summary_path),
        "visual_qa_passed": True,
        "all_surfaces_regenerated_from_d025_corrected_depth": True,
        "all_raw_depth_and_confidence_unchanged": True,
        "all_anchors_regenerated": True,
        "all_pairs_regenerated": True,
        "surface_count": len(summary.d030_surface_records),
        "anchor_count": len(summary.d032_anchor_records),
        "pair_observation_count": len(summary.d033_pair_observations),
        "bottom_truncated_person_view_count": sum(
            record.surface_role is CorrectedSurfaceRole.PERSON_UPPER_BODY
            for record in person_surfaces
        ),
        "per_camera_person_footpoint_count": sum(
            record.kind is CorrectedAnchorKind.PERSON_FOOTPOINT
            for record in person_anchors
        ),
        "person_pair_state_counts": {
            state.value: sum(record.state is state for record in person_pairs)
            for state in CorrectedPairState
        },
        "person_preferred_kind_by_frame": {
            str(record.source_frame_index): (
                record.selected_kind.value if record.selected_kind else None
            )
            for record in person_pairs
        },
        "maximum_reprojection_error_px": maximum_reprojection,
        "maximum_world_camera_round_trip_error_m": maximum_round_trip,
        "temporal_filling_performed": False,
        "presentation_smoothing_performed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))
    return 0


def _require_close(actual: Any, expected: Any) -> None:
    if not np.allclose(np.asarray(actual), np.asarray(expected), atol=1e-12):
        raise ValueError("corrected geometry regeneration differs")


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
    return path.resolve().relative_to(root).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
