"""Rebuild S04 D030-D033 from D025 depth with margin-aware person anchors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib
import numpy as np
from numpy.typing import NDArray
from PIL import Image

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from spatial_reconstruction.contracts import CameraIntrinsics, CameraPose, PerceptionTarget
from spatial_reconstruction.geometry import project_camera_points, world_points_to_camera
from spatial_reconstruction.localization import (
    ActionDepthRunSummary,
    CorrectedAnchor,
    CorrectedAnchorKind,
    CorrectedAnchorRecord,
    CorrectedPairObservationRecord,
    CorrectedPairSource,
    CorrectedPairState,
    CorrectedSurfaceRecord,
    CorrectedSurfaceRole,
    CorrectedTrackingPolicy,
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
    parser.add_argument(
        "--action-depth-summary",
        type=Path,
        default=Path("artifacts/s04/action_depth_preflight_20260801/summary.json"),
    )
    parser.add_argument(
        "--depth-scale-summary",
        type=Path,
        default=Path("artifacts/s04/action_depth_scale_20260803/summary.json"),
    )
    parser.add_argument(
        "--depth-scale-verification",
        type=Path,
        default=Path("artifacts/s04/action_depth_scale_20260803/verification.json"),
    )
    parser.add_argument(
        "--mask-alignment-summary",
        type=Path,
        default=Path("artifacts/s04/mask_alignment_20260801/summary.json"),
    )
    parser.add_argument(
        "--pose-calibration",
        type=Path,
        default=Path(
            "artifacts/s01/calibration/action_take_01_pose/camera_calibration.json"
        ),
    )
    parser.add_argument(
        "--scene-metadata",
        type=Path,
        default=Path("artifacts/s01/scene_metadata.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    paths = {
        "action": _resolve(root, args.action_depth_summary),
        "scale": _resolve(root, args.depth_scale_summary),
        "scale_verification": _resolve(root, args.depth_scale_verification),
        "alignment": _resolve(root, args.mask_alignment_summary),
        "calibration": _resolve(root, args.pose_calibration),
        "scene": _resolve(root, args.scene_metadata),
    }
    output_dir = _resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    samples_dir = output_dir / "sample_clouds"
    diagnostics_dir = output_dir / "image_diagnostics"
    samples_dir.mkdir()
    diagnostics_dir.mkdir()

    action = ActionDepthRunSummary.model_validate_json(
        paths["action"].read_text(encoding="utf-8")
    )
    alignment = MaskAlignmentRunSummary.model_validate_json(
        paths["alignment"].read_text(encoding="utf-8")
    )
    scale = _read_object(paths["scale"])
    scale_verification = _read_object(paths["scale_verification"])
    calibration = _read_object(paths["calibration"])
    scene = _read_object(paths["scene"])
    _verify_prerequisites(
        action=action,
        alignment=alignment,
        scale=scale,
        scale_path=paths["scale"],
        scale_verification=scale_verification,
        action_path=paths["action"],
    )
    policy = CorrectedTrackingPolicy()
    if scale["policy"]["policy_id"] != policy.source_depth_policy_id:
        raise ValueError("corrected tracking source depth policy differs from D025")

    predictions = {record.job.job_id: record for record in action.predictions}
    scale_jobs = {str(record["job_id"]): record for record in scale["jobs"]}
    calibration_cameras = cast(dict[str, dict[str, Any]], calibration["cameras"])
    poses = {
        camera_id: CameraPose.model_validate(calibration_cameras[camera_id]["pose"])
        for camera_id in CAMERA_IDS
    }
    raw_cache: dict[str, tuple[Float32Array, Float32Array, UInt8Array, Float32Array]] = {}
    corrected_cache: dict[str, Float32Array] = {}
    mask_cache: dict[str, UInt8Array] = {}
    surface_records: list[CorrectedSurfaceRecord] = []
    anchor_records: list[CorrectedAnchorRecord] = []
    runtime_anchors: dict[tuple[str, str, PerceptionTarget], CorrectedAnchor] = {}
    diagnostics: list[tuple[CorrectedSurfaceRecord, Path]] = []

    for index, aligned in enumerate(alignment.aligned_masks):
        prediction = predictions[aligned.action_depth_job_id]
        scale_job = scale_jobs[aligned.action_depth_job_id]
        camera_index = CAMERA_INDEX[aligned.camera_id]
        raw = raw_cache.get(prediction.raw_prediction_ref)
        if raw is None:
            raw_path = _resolve(root, Path(prediction.raw_prediction_ref))
            _require_hash(raw_path, prediction.raw_prediction_sha256)
            with np.load(raw_path, allow_pickle=False) as arrays:
                raw = (
                    cast(Float32Array, np.asarray(arrays["depth_m"]).copy()),
                    cast(Float32Array, np.asarray(arrays["confidence"]).copy()),
                    cast(UInt8Array, np.asarray(arrays["processed_images_rgb"]).copy()),
                    cast(Float32Array, np.asarray(arrays["returned_intrinsics"]).copy()),
                )
            raw_cache[prediction.raw_prediction_ref] = raw
        corrected_ref = str(scale_job["corrected_prediction_ref"])
        corrected = corrected_cache.get(corrected_ref)
        if corrected is None:
            corrected_path = _resolve(root, Path(corrected_ref))
            _require_hash(corrected_path, str(scale_job["corrected_prediction_sha256"]))
            with np.load(corrected_path, allow_pickle=False) as arrays:
                if str(arrays["job_id"].item()) != prediction.job.job_id:
                    raise ValueError("corrected depth job identity differs")
                if str(arrays["raw_prediction_sha256"].item()) != prediction.raw_prediction_sha256:
                    raise ValueError("corrected depth raw source identity differs")
                corrected = cast(
                    Float32Array, np.asarray(arrays["corrected_depth_m"]).copy()
                )
            corrected_cache[corrected_ref] = corrected
        masks = mask_cache.get(aligned.aligned_mask_artifact_ref)
        if masks is None:
            mask_path = _resolve(root, Path(aligned.aligned_mask_artifact_ref))
            _require_hash(mask_path, aligned.aligned_mask_artifact_sha256)
            with np.load(mask_path, allow_pickle=False) as arrays:
                masks = cast(UInt8Array, np.asarray(arrays["masks"]).copy())
            mask_cache[aligned.aligned_mask_artifact_ref] = masks
        _, confidence, images, intrinsics_arrays = raw
        matrix = intrinsics_arrays[camera_index]
        intrinsics = CameraIntrinsics(
            camera_id=aligned.camera_id,
            fx=float(matrix[0, 0]),
            fy=float(matrix[1, 1]),
            cx=float(matrix[0, 2]),
            cy=float(matrix[1, 2]),
            image_width=corrected.shape[2],
            image_height=corrected.shape[1],
        )
        pose = poses[aligned.camera_id]
        surface = localize_corrected_surface(
            source_mask=masks[aligned.aligned_mask_index],
            corrected_depth_m=corrected[camera_index],
            confidence=confidence[camera_index],
            target=aligned.target,
            intrinsics=intrinsics,
            pose=pose,
            policy=policy,
        )
        anchor = derive_corrected_anchor(surface, target=aligned.target, policy=policy)
        score = anchor_reliability(anchor, surface)
        observation_id = CorrectedSurfaceRecord.create_observation_id(
            action_depth_job_id=aligned.action_depth_job_id,
            frame_id=aligned.frame_id,
            target=aligned.target,
            policy_id=policy.policy_id,
        )
        anchor_id = CorrectedAnchorRecord.create_anchor_id(
            source_observation_id=observation_id, kind=anchor.kind
        )
        sample_path = samples_dir / (
            f"{index:02d}_{aligned.source_frame_index:04d}_{aligned.camera_id}_"
            f"{aligned.target.value}.npz"
        )
        np.savez_compressed(
            sample_path,
            pixels_uv=surface.pixels_uv,
            corrected_depth_m=surface.depth_m,
            confidence=surface.confidence,
            points_camera_m=surface.points_camera_m,
            points_world_m=surface.points_world_m,
            aggregate_camera_xyz_m=np.asarray(surface.aggregate_camera_xyz_m),
            aggregate_world_xyz_m=np.asarray(surface.aggregate_world_xyz_m),
            observation_id=np.asarray(observation_id),
            anchor_id=np.asarray(anchor_id),
            anchor_kind=np.asarray(anchor.kind.value),
            anchor_world_xyz_m=np.asarray(anchor.world_xyz_m),
            policy_id=np.asarray(policy.policy_id),
            action_depth_job_id=np.asarray(aligned.action_depth_job_id),
            bundle_id=np.asarray(aligned.bundle_id),
            frame_id=np.asarray(aligned.frame_id),
            camera_id=np.asarray(aligned.camera_id),
            target=np.asarray(aligned.target.value),
            raw_prediction_sha256=np.asarray(prediction.raw_prediction_sha256),
            corrected_prediction_sha256=np.asarray(
                scale_job["corrected_prediction_sha256"]
            ),
            depth_scale=np.float64(scale_job["scale_estimate"]["scale"]),
            temporal_filling_performed=np.asarray(False),
            presentation_smoothing_performed=np.asarray(False),
        )
        diagnostic_path = diagnostics_dir / (
            f"{index:02d}_{aligned.source_frame_index:04d}_{aligned.camera_id}_"
            f"{aligned.target.value}.png"
        )
        _save_image_diagnostic(
            image=images[camera_index],
            source_mask=masks[aligned.aligned_mask_index],
            surface=surface,
            anchor=anchor,
            intrinsics=intrinsics,
            pose=pose,
            frame_index=aligned.source_frame_index,
            camera_id=aligned.camera_id,
            target=aligned.target,
            path=diagnostic_path,
        )
        depth_stats, confidence_stats = surface_statistics(surface)
        semantics = _surface_semantics(surface.role)
        record = CorrectedSurfaceRecord(
            observation_id=observation_id,
            policy_id=policy.policy_id,
            action_depth_job_id=aligned.action_depth_job_id,
            bundle_id=aligned.bundle_id,
            frame_id=aligned.frame_id,
            source_frame_index=aligned.source_frame_index,
            capture_timestamp_seconds=next(
                frame.capture_timestamp_seconds
                for frame in prediction.job.bundle.frames
                if frame.camera_id == aligned.camera_id
            ),
            phase_id=prediction.job.phase_id,
            camera_id=cast(Any, aligned.camera_id),
            target=aligned.target,
            perception_job_id=aligned.perception_job_id,
            camera_local_track_id=aligned.camera_local_track_id,
            depth_scale=float(scale_job["scale_estimate"]["scale"]),
            surface_role=surface.role,
            candidate_strategy=surface.strategy,
            person_margin_assessment=surface.margin_assessment,
            source_mask_pixel_count=aligned.processed_mask_area_pixels,
            candidate_pixel_count=surface.candidate_pixel_count,
            valid_candidate_count=surface.valid_candidate_count,
            retained_sample_count=surface.retained_sample_count,
            confidence_percentile=policy.confidence_percentile,
            confidence_threshold=surface.confidence_threshold,
            retained_depth_m=depth_stats,
            retained_confidence=confidence_stats,
            aggregate_camera_xyz_m=surface.aggregate_camera_xyz_m,
            aggregate_world_xyz_m=surface.aggregate_world_xyz_m,
            processed_intrinsics=intrinsics,
            camera_pose=pose,
            raw_prediction_ref=prediction.raw_prediction_ref,
            raw_prediction_sha256=prediction.raw_prediction_sha256,
            corrected_prediction_ref=corrected_ref,
            corrected_prediction_sha256=str(scale_job["corrected_prediction_sha256"]),
            aligned_mask_artifact_ref=aligned.aligned_mask_artifact_ref,
            aligned_mask_artifact_sha256=aligned.aligned_mask_artifact_sha256,
            aligned_mask_index=aligned.aligned_mask_index,
            sample_cloud_ref=_relative(sample_path, root),
            sample_cloud_sha256=_sha256(sample_path),
            image_diagnostic_ref=_relative(diagnostic_path, root),
            image_diagnostic_sha256=_sha256(diagnostic_path),
            reprojection_max_error_px=surface.reprojection_max_error_px,
            round_trip_max_error_m=surface.round_trip_max_error_m,
            coordinate_semantics=semantics,
        )
        anchor_record = CorrectedAnchorRecord(
            anchor_id=anchor_id,
            source_observation_id=observation_id,
            action_depth_job_id=aligned.action_depth_job_id,
            bundle_id=aligned.bundle_id,
            frame_id=aligned.frame_id,
            source_frame_index=aligned.source_frame_index,
            phase_id=prediction.job.phase_id,
            camera_id=cast(Any, aligned.camera_id),
            target=aligned.target,
            kind=anchor.kind,
            world_xyz_m=anchor.world_xyz_m,
            support_sample_count=anchor.support_sample_count,
            measured_support_world_z_m=anchor.measured_support_world_z_m,
            footpoint_available=anchor.footpoint_available,
            selection_reason=anchor.selection_reason,
            retained_confidence_median=confidence_stats.median,
            retained_depth_median_m=depth_stats.median,
            retained_depth_mad_m=depth_stats.median_absolute_deviation,
            reliability_score=score,
            source_sample_cloud_ref=_relative(sample_path, root),
            source_sample_cloud_sha256=_sha256(sample_path),
        )
        surface_records.append(record)
        anchor_records.append(anchor_record)
        runtime_anchors[(aligned.action_depth_job_id, aligned.camera_id, aligned.target)] = anchor
        diagnostics.append((record, diagnostic_path))

    anchor_lookup = {
        (record.action_depth_job_id, record.camera_id, record.target): record
        for record in anchor_records
    }
    pair_records: list[CorrectedPairObservationRecord] = []
    for prediction in action.predictions:
        for target in PerceptionTarget:
            inputs: list[PairAnchorInput] = []
            for camera_id in CAMERA_IDS:
                runtime = runtime_anchors.get((prediction.job.job_id, camera_id, target))
                persistent = anchor_lookup.get((prediction.job.job_id, camera_id, target))
                inputs.append(
                    PairAnchorInput(
                        camera_id=cast(Any, camera_id),
                        anchor=runtime,
                        reliability_score=(
                            persistent.reliability_score if persistent is not None else None
                        ),
                    )
                )
            resolution = resolve_corrected_pair(
                target=target,
                camera_a=inputs[0],
                camera_b=inputs[1],
                maximum_disagreement_m=policy.maximum_cross_camera_disagreement_m,
            )
            sources: list[CorrectedPairSource] = []
            for camera_index, camera_id in enumerate(CAMERA_IDS):
                persistent = anchor_lookup.get((prediction.job.job_id, camera_id, target))
                selected = (
                    resolution.state
                    in (CorrectedPairState.FUSED, CorrectedPairState.SINGLE_CAMERA)
                    and camera_id in resolution.selected_camera_ids
                )
                sources.append(
                    CorrectedPairSource(
                        camera_id=cast(Any, camera_id),
                        source_anchor_id=(persistent.anchor_id if persistent else None),
                        kind=(persistent.kind if persistent else None),
                        world_xyz_m=(persistent.world_xyz_m if persistent else None),
                        reliability_score=(
                            persistent.reliability_score if persistent else None
                        ),
                        contribution_weight=resolution.contribution_weights[camera_index],
                        selected_for_output=selected,
                    )
                )
            pair_records.append(
                CorrectedPairObservationRecord(
                    observation_id=CorrectedPairObservationRecord.create_observation_id(
                        action_depth_job_id=prediction.job.job_id,
                        target=target,
                        policy_id=policy.policy_id,
                    ),
                    policy_id=policy.policy_id,
                    action_depth_job_id=prediction.job.job_id,
                    bundle_id=prediction.job.bundle.bundle_id,
                    source_frame_index=prediction.job.bundle.frames[0].source_frame_index,
                    capture_timestamp_seconds=prediction.job.bundle.capture_timestamp_seconds,
                    phase_id=prediction.job.phase_id,
                    target=target,
                    state=resolution.state,
                    selected_kind=resolution.selected_kind,
                    world_xyz_m=resolution.world_xyz_m,
                    sources=cast(Any, tuple(sources)),
                    selected_camera_ids=resolution.selected_camera_ids,
                    disagreement_distance_m=resolution.disagreement_distance_m,
                    maximum_cross_camera_disagreement_m=(
                        policy.maximum_cross_camera_disagreement_m
                    ),
                    fallback_surface_used=resolution.fallback_surface_used,
                    selection_reason=resolution.selection_reason,
                )
            )

    csv_path = output_dir / "corrected_d030_d033_observations.csv"
    _save_csv(pair_records, csv_path)
    contact_path = output_dir / "person_margin_contact_sheet.png"
    _save_contact_sheet(
        [path for record, path in diagnostics if record.target is PerceptionTarget.PERSON],
        contact_path,
    )
    world_path = output_dir / "corrected_tracking_world_preview.png"
    _save_world_preview(pair_records, scene=scene, path=world_path)

    tier_payloads = {
        "d030_sampling_summary": {
            "schema_version": 1,
            "stage": "S04",
            "policy": policy.model_dump(mode="json"),
            "depth_input": "d025_corrected_action_pair_depth",
            "records": [
                {
                    "observation_id": record.observation_id,
                    "source_frame_index": record.source_frame_index,
                    "camera_id": record.camera_id,
                    "target": record.target.value,
                    "surface_role": record.surface_role.value,
                    "candidate_strategy": record.candidate_strategy.value,
                    "person_margin_assessment": (
                        record.person_margin_assessment.model_dump(mode="json")
                        if record.person_margin_assessment
                        else None
                    ),
                    "candidate_pixel_count": record.candidate_pixel_count,
                    "retained_sample_count": record.retained_sample_count,
                    "confidence_threshold": record.confidence_threshold,
                    "depth_scale": record.depth_scale,
                }
                for record in surface_records
            ],
        },
        "d031_visible_surface_summary": {
            "schema_version": 1,
            "stage": "S04",
            "records": [record.model_dump(mode="json") for record in surface_records],
        },
        "d032_anchor_summary": {
            "schema_version": 1,
            "stage": "S04",
            "records": [record.model_dump(mode="json") for record in anchor_records],
        },
        "d033_observation_summary": {
            "schema_version": 1,
            "stage": "S04",
            "records": [record.model_dump(mode="json") for record in pair_records],
        },
    }
    tier_paths: dict[str, Path] = {}
    for name, payload in tier_payloads.items():
        path = output_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tier_paths[name] = path

    summary = CorrectedTrackingRunSummary(
        schema_version=1,
        status="completed_pending_visual_qa",
        stage="S04",
        created_at_utc=datetime.now(UTC),
        policy=policy,
        source_action_depth_summary_ref=_relative(paths["action"], root),
        source_action_depth_summary_sha256=_sha256(paths["action"]),
        source_depth_scale_summary_ref=_relative(paths["scale"], root),
        source_depth_scale_summary_sha256=_sha256(paths["scale"]),
        source_depth_scale_verification_ref=_relative(
            paths["scale_verification"], root
        ),
        source_depth_scale_verification_sha256=_sha256(paths["scale_verification"]),
        source_mask_alignment_summary_ref=_relative(paths["alignment"], root),
        source_mask_alignment_summary_sha256=_sha256(paths["alignment"]),
        pose_calibration_ref=_relative(paths["calibration"], root),
        pose_calibration_sha256=_sha256(paths["calibration"]),
        scene_metadata_ref=_relative(paths["scene"], root),
        scene_metadata_sha256=_sha256(paths["scene"]),
        d030_sampling_summary_ref=_relative(tier_paths["d030_sampling_summary"], root),
        d030_sampling_summary_sha256=_sha256(tier_paths["d030_sampling_summary"]),
        d031_visible_surface_summary_ref=_relative(
            tier_paths["d031_visible_surface_summary"], root
        ),
        d031_visible_surface_summary_sha256=_sha256(
            tier_paths["d031_visible_surface_summary"]
        ),
        d032_anchor_summary_ref=_relative(tier_paths["d032_anchor_summary"], root),
        d032_anchor_summary_sha256=_sha256(tier_paths["d032_anchor_summary"]),
        d033_observation_summary_ref=_relative(
            tier_paths["d033_observation_summary"], root
        ),
        d033_observation_summary_sha256=_sha256(
            tier_paths["d033_observation_summary"]
        ),
        d030_surface_records=tuple(surface_records),
        d032_anchor_records=tuple(anchor_records),
        d033_pair_observations=tuple(pair_records),
        observation_csv_ref=_relative(csv_path, root),
        observation_csv_sha256=_sha256(csv_path),
        margin_contact_sheet_ref=_relative(contact_path, root),
        margin_contact_sheet_sha256=_sha256(contact_path),
        world_preview_ref=_relative(world_path, root),
        world_preview_sha256=_sha256(world_path),
        limitations=(
            "Footpoints require non-bottom-truncated masks and measured near-floor support.",
            "Upper-body and elevated lower-body surfaces are useful fallback "
            "evidence, not footpoints.",
            "Preferred outputs with different anchor kinds must remain visibly "
            "distinct in later trajectories.",
            "No temporal interpolation, stale carry-forward, or presentation "
            "smoothing is performed.",
        ),
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_result_counts(summary), indent=2))
    return 0


def _verify_prerequisites(
    *,
    action: ActionDepthRunSummary,
    alignment: MaskAlignmentRunSummary,
    scale: dict[str, Any],
    scale_path: Path,
    scale_verification: dict[str, Any],
    action_path: Path,
) -> None:
    if scale_verification.get("status") != "passed":
        raise ValueError("D025 scale verification has not passed")
    if scale_verification.get("source_summary_sha256") != _sha256(scale_path):
        raise ValueError("D025 verification does not cover the selected scale summary")
    if scale.get("raw_action_depth_summary_sha256") != _sha256(action_path):
        raise ValueError("D025 scale and raw action-depth inputs differ")
    if alignment.source_action_depth_summary_sha256 != _sha256(action_path):
        raise ValueError("mask alignment and action-depth inputs differ")
    expected_mask_keys = {
        (prediction.job.job_id, evidence.frame_identity.camera_id, evidence.target)
        for prediction in action.predictions
        for evidence in prediction.job.mask_evidence
    }
    aligned_mask_keys = {
        (record.action_depth_job_id, record.camera_id, record.target)
        for record in alignment.aligned_masks
    }
    if not action.predictions or aligned_mask_keys != expected_mask_keys:
        raise ValueError("corrected rebuild prerequisites have incomplete coverage")


def _surface_semantics(role: CorrectedSurfaceRole) -> str:
    return {
        CorrectedSurfaceRole.PERSON_LOWER_BODY: (
            "Corrected-depth visible lower-body surface; eligible for footpoint "
            "validation but not itself a footpoint."
        ),
        CorrectedSurfaceRole.PERSON_UPPER_BODY: (
            "Corrected-depth upper-body visible surface from a bottom-truncated mask; "
            "never floor-projected."
        ),
        CorrectedSurfaceRole.BACKPACK_VISIBLE_CLUSTER: (
            "Corrected-depth visible backpack cluster; not the hidden physical centroid."
        ),
    }[role]


def _save_image_diagnostic(
    *,
    image: UInt8Array,
    source_mask: UInt8Array,
    surface: Any,
    anchor: CorrectedAnchor,
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    frame_index: int,
    camera_id: str,
    target: PerceptionTarget,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.imshow(image)
    ys, xs = np.nonzero(source_mask)
    axis.plot(xs, ys, ".", color="#00e5ff", markersize=0.4, alpha=0.2)
    axis.plot(
        surface.pixels_uv[:, 0],
        surface.pixels_uv[:, 1],
        ".",
        color="#ffe600",
        markersize=0.8,
        alpha=0.6,
    )
    anchor_camera = np.asarray(
        world_points_to_camera(np.asarray(anchor.world_xyz_m), pose=pose),
        dtype=np.float64,
    )
    if anchor_camera[2] > 0:
        anchor_uv = np.asarray(
            project_camera_points(anchor_camera, intrinsics=intrinsics),
            dtype=np.float64,
        )
        axis.plot(anchor_uv[0], anchor_uv[1], "x", color="#ff3030", markersize=10)
    if surface.margin_assessment is not None:
        margin_y = intrinsics.image_height - 1 - surface.margin_assessment.margin_pixels
        axis.axhline(margin_y, color="#ff7a00", linewidth=1.5, linestyle="--")
        validity = surface.margin_assessment.validity.value
    else:
        validity = "not_applicable"
    axis.set_title(
        f"frame {frame_index} {camera_id} {target.value}\n"
        f"{surface.role.value} -> {anchor.kind.value} | {validity}"
    )
    axis.set_xlim(0, intrinsics.image_width - 1)
    axis.set_ylim(intrinsics.image_height - 1, 0)
    axis.axis("off")
    figure.savefig(path, dpi=130)
    plt.close(figure)


def _save_contact_sheet(paths: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    columns = 2
    thumb_width = 900
    thumb_height = int(images[0].height * thumb_width / images[0].width)
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (thumb_width * columns, thumb_height * rows), "white")
    for index, image in enumerate(images):
        canvas.paste(
            image.resize((thumb_width, thumb_height)),
            ((index % columns) * thumb_width, (index // columns) * thumb_height),
        )
    canvas.save(output)


def _save_world_preview(
    records: list[CorrectedPairObservationRecord], *, scene: dict[str, Any], path: Path
) -> None:
    figure, (top, height) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    styles = {
        CorrectedAnchorKind.PERSON_FOOTPOINT: ("#00a86b", "o"),
        CorrectedAnchorKind.PERSON_LOWER_BODY_SURFACE: ("#ff8c00", "s"),
        CorrectedAnchorKind.PERSON_UPPER_BODY_SURFACE: ("#e53935", "^"),
        CorrectedAnchorKind.BACKPACK_VISIBLE_CLUSTER: ("#d000ff", "D"),
    }
    for record in records:
        if record.world_xyz_m is None or record.selected_kind is None:
            continue
        color, marker = styles[record.selected_kind]
        point = np.asarray(record.world_xyz_m)
        top.scatter(point[0], point[1], color=color, marker=marker, s=55)
        top.annotate(str(record.source_frame_index), (point[0], point[1]), fontsize=8)
        if record.target is PerceptionTarget.PERSON:
            height.scatter(
                record.capture_timestamp_seconds,
                point[2],
                color=color,
                marker=marker,
                s=55,
            )
    bounds = cast(dict[str, list[float]], scene["room_bounds"])
    minimum = bounds["minimum_world_xyz_m"]
    maximum = bounds["maximum_world_xyz_m"]
    top.set_xlim(minimum[0], maximum[0])
    top.set_ylim(minimum[1], maximum[1])
    top.set_aspect("equal", adjustable="box")
    top.set_title("Corrected preferred outputs (top-down)")
    top.set_xlabel("world X (m)")
    top.set_ylabel("world Y (m)")
    top.grid(alpha=0.25)
    height.axhline(0.0, color="black", linewidth=1)
    height.set_title("Person anchor height; green circles are footpoints")
    height.set_xlabel("capture time (s)")
    height.set_ylabel("world Z (m)")
    height.grid(alpha=0.25)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_csv(records: list[CorrectedPairObservationRecord], path: Path) -> None:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.append(
            {
                "source_frame_index": record.source_frame_index,
                "phase_id": record.phase_id,
                "target": record.target.value,
                "state": record.state.value,
                "selected_kind": record.selected_kind.value if record.selected_kind else "",
                "selected_camera_ids": "+".join(record.selected_camera_ids),
                "fallback_surface_used": record.fallback_surface_used,
                "world_x_m": record.world_xyz_m[0] if record.world_xyz_m else "",
                "world_y_m": record.world_xyz_m[1] if record.world_xyz_m else "",
                "world_z_m": record.world_xyz_m[2] if record.world_xyz_m else "",
                "disagreement_distance_m": record.disagreement_distance_m or "",
                "camera_a_kind": record.sources[0].kind.value if record.sources[0].kind else "",
                "camera_b_kind": record.sources[1].kind.value if record.sources[1].kind else "",
                "selection_reason": record.selection_reason,
            }
        )
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _result_counts(summary: CorrectedTrackingRunSummary) -> dict[str, object]:
    person_surfaces = [
        record
        for record in summary.d030_surface_records
        if record.target is PerceptionTarget.PERSON
    ]
    person_pairs = [
        record
        for record in summary.d033_pair_observations
        if record.target is PerceptionTarget.PERSON
    ]
    return {
        "surface_count": len(summary.d030_surface_records),
        "person_bottom_truncated_count": sum(
            record.surface_role is CorrectedSurfaceRole.PERSON_UPPER_BODY
            for record in person_surfaces
        ),
        "per_camera_person_footpoint_count": sum(
            record.kind is CorrectedAnchorKind.PERSON_FOOTPOINT
            for record in summary.d032_anchor_records
        ),
        "person_pair_state_counts": {
            state.value: sum(record.state is state for record in person_pairs)
            for state in CorrectedPairState
        },
        "person_selected_kind_by_frame": {
            str(record.source_frame_index): (
                record.selected_kind.value if record.selected_kind else None
            )
            for record in person_pairs
        },
    }


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
