"""Verify accepted S02 geometry, provenance, overlap, and Rerun evidence."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from plyfile import PlyData  # type: ignore[import-untyped]
from pydantic import Field

from spatial_reconstruction.contracts import ContractModel, Sha256Digest
from spatial_reconstruction.geometry import (
    StaticSceneRerunExportSummary,
    StaticSceneRunSummary,
    radius_overlap_fraction,
)

FloatArray = NDArray[np.float64]
CAMERA_IDS = ("camera_a", "camera_b")


class StaticSceneVerificationReport(ContractModel):
    """Persistent automated gate evidence for the accepted S02 run."""

    schema_version: Literal[1]
    stage: Literal["S02"]
    status: Literal["passed"]
    source_summary_ref: str
    source_summary_sha256: Sha256Digest
    rerun_export_summary_ref: str
    rerun_export_summary_sha256: Sha256Digest
    rerun_recording_ref: str
    rerun_recording_sha256: Sha256Digest
    visual_evidence: dict[str, dict[str, str]]
    verified_artifact_count: int = Field(gt=0)
    point_counts: dict[str, int]
    cross_camera_overlap: dict[str, float]
    marker_scale_diagnostics: dict[str, float]
    world_extent_m: dict[str, tuple[float, float, float]]
    automated_gate_checks: dict[str, bool]
    limitations: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--run-summary",
        type=Path,
        default=Path(
            "artifacts/s02/candidate_three_keyframes_20260731/summary.json"
        ),
    )
    parser.add_argument(
        "--rerun-export-summary",
        type=Path,
        default=Path(
            "artifacts/s02/candidate_three_keyframes_20260731/"
            "static_scene_accepted_v2_export_summary.json"
        ),
    )
    parser.add_argument(
        "--rerun-screenshot",
        type=Path,
        default=Path(
            "artifacts/s02/candidate_three_keyframes_20260731/previews/"
            "rerun_static_scene_accepted.png"
        ),
    )
    parser.add_argument("--overlap-radius-m", type=float, default=0.1)
    parser.add_argument("--minimum-overlap-fraction", type=float, default=0.65)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    summary_path = (project_root / args.run_summary).resolve()
    rerun_summary_path = (project_root / args.rerun_export_summary).resolve()
    screenshot_path = (project_root / args.rerun_screenshot).resolve()
    output_path = (
        (project_root / args.output).resolve()
        if args.output is not None
        else summary_path.parent / "verification.json"
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite verification: {output_path}")

    summary = StaticSceneRunSummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    rerun_summary = StaticSceneRerunExportSummary.model_validate_json(
        rerun_summary_path.read_text(encoding="utf-8")
    )
    if rerun_summary.source_summary_sha256 != _sha256(summary_path):
        raise ValueError("Rerun export does not reference the supplied S02 summary")

    verified_paths = _verify_summary_artifacts(
        project_root=project_root,
        summary=summary,
    )
    recording_path = project_root / rerun_summary.recording_ref
    if _sha256(recording_path) != rerun_summary.recording_sha256:
        raise ValueError("Rerun recording hash does not match its export summary")
    if not screenshot_path.is_file():
        raise ValueError("accepted Rerun screenshot is missing")
    verified_paths.extend((recording_path, screenshot_path))

    camera_points = {
        camera_id: _read_points(
            project_root / summary.point_clouds[camera_id].ply_ref
        )
        for camera_id in CAMERA_IDS
    }
    fused_points = _read_points(
        project_root / summary.point_clouds["fused"].ply_ref
    )
    overlap_radius = float(args.overlap_radius_m)
    a_in_b = radius_overlap_fraction(
        camera_points["camera_a"],
        camera_points["camera_b"],
        radius_m=overlap_radius,
    )
    b_in_a = radius_overlap_fraction(
        camera_points["camera_b"],
        camera_points["camera_a"],
        radius_m=overlap_radius,
    )
    minimum_overlap = float(args.minimum_overlap_fraction)

    processing = summary.processing
    room_bounds = cast(dict[str, Any], processing["room_bounds"])
    bounds_minimum = np.asarray(
        room_bounds["minimum_world_xyz_m"],
        dtype=np.float64,
    )
    bounds_maximum = np.asarray(
        room_bounds["maximum_world_xyz_m"],
        dtype=np.float64,
    )
    actual_minimum = np.min(fused_points, axis=0)
    actual_maximum = np.max(fused_points, axis=0)
    within_bounds = bool(
        np.all(actual_minimum >= bounds_minimum)
        and np.all(actual_maximum <= bounds_maximum)
    )

    scales: list[float] = []
    marker_errors_m: list[float] = []
    maximum_scale_deviations: list[float] = []
    for prediction in summary.predictions:
        correction = prediction.marker_depth_scale_correction
        scale = float(correction["scale"])
        scales.append(scale)
        maximum_scale_deviations.append(
            float(correction["maximum_relative_deviation"])
        )
        for observation in cast(list[dict[str, Any]], correction["observations"]):
            expected = float(observation["expected_camera_depth_m"])
            corrected = float(observation["predicted_depth_m"]) * scale
            marker_errors_m.append(abs(expected - corrected))

    automated_gate_checks = {
        "both_camera_clouds_non_empty": all(
            camera_points[camera_id].shape[0] > 0 for camera_id in CAMERA_IDS
        ),
        "fused_cloud_non_empty": fused_points.shape[0] > 0,
        "cross_camera_overlap_passes": (
            a_in_b >= minimum_overlap and b_in_a >= minimum_overlap
        ),
        "all_points_are_finite": bool(
            np.isfinite(fused_points).all()
            and all(
                np.isfinite(camera_points[camera_id]).all()
                for camera_id in CAMERA_IDS
            )
        ),
        "fused_extent_within_declared_bounds": within_bounds,
        "marker_scale_observations_pass": max(maximum_scale_deviations) <= 0.05,
        "rerun_contains_both_cameras": set(rerun_summary.logged_camera_ids)
        == set(CAMERA_IDS),
        "rerun_visual_evidence_present": screenshot_path.stat().st_size > 0,
    }
    if not all(automated_gate_checks.values()):
        failed = [
            name for name, passed in automated_gate_checks.items() if not passed
        ]
        raise RuntimeError(f"S02 automated gate checks failed: {failed}")

    geometry_path = project_root / summary.previews.geometry_png_ref
    report = StaticSceneVerificationReport(
        schema_version=1,
        stage="S02",
        status="passed",
        source_summary_ref=_relative(summary_path, project_root),
        source_summary_sha256=_sha256(summary_path),
        rerun_export_summary_ref=_relative(rerun_summary_path, project_root),
        rerun_export_summary_sha256=_sha256(rerun_summary_path),
        rerun_recording_ref=rerun_summary.recording_ref,
        rerun_recording_sha256=rerun_summary.recording_sha256,
        visual_evidence={
            "geometry": {
                "ref": _relative(geometry_path, project_root),
                "sha256": _sha256(geometry_path),
            },
            "rerun": {
                "ref": _relative(screenshot_path, project_root),
                "sha256": _sha256(screenshot_path),
            },
        },
        verified_artifact_count=len({path.resolve() for path in verified_paths}),
        point_counts={
            "camera_a": int(camera_points["camera_a"].shape[0]),
            "camera_b": int(camera_points["camera_b"].shape[0]),
            "fused": int(fused_points.shape[0]),
        },
        cross_camera_overlap={
            "radius_m": overlap_radius,
            "minimum_accepted_fraction": minimum_overlap,
            "camera_a_within_radius_of_camera_b_fraction": a_in_b,
            "camera_b_within_radius_of_camera_a_fraction": b_in_a,
        },
        marker_scale_diagnostics={
            "minimum_scale": min(scales),
            "maximum_scale": max(scales),
            "scale_range": max(scales) - min(scales),
            "maximum_observation_relative_deviation": max(
                maximum_scale_deviations
            ),
            "median_corrected_marker_depth_error_m": float(
                np.median(marker_errors_m)
            ),
            "maximum_corrected_marker_depth_error_m": max(marker_errors_m),
        },
        world_extent_m={
            "minimum": (
                float(actual_minimum[0]),
                float(actual_minimum[1]),
                float(actual_minimum[2]),
            ),
            "maximum": (
                float(actual_maximum[0]),
                float(actual_maximum[1]),
                float(actual_maximum[2]),
            ),
        },
        automated_gate_checks=automated_gate_checks,
        limitations=(
            "Living-room recognizability is a recorded visual QA judgement.",
            "The 0.10 m overlap check measures shared visible surfaces, not full coverage.",
            "Room bounds remain a conservative crop rather than surveyed walls.",
        ),
    )
    output_path.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    print(report.model_dump_json(indent=2))
    return 0


def _verify_summary_artifacts(
    *,
    project_root: Path,
    summary: StaticSceneRunSummary,
) -> list[Path]:
    pairs: list[tuple[str, str]] = [
        (
            summary.input_provenance.synchronization_manifest_ref,
            summary.input_provenance.synchronization_manifest_sha256,
        ),
        (
            summary.input_provenance.pose_calibration_ref,
            summary.input_provenance.pose_calibration_sha256,
        ),
        (
            summary.input_provenance.scene_metadata_ref,
            summary.input_provenance.scene_metadata_sha256,
        ),
        (summary.previews.geometry_png_ref, summary.previews.geometry_png_sha256),
        (summary.previews.glb_ref, summary.previews.glb_sha256),
    ]
    for prediction in summary.predictions:
        pairs.extend(
            (
                (prediction.raw_prediction_ref, prediction.raw_prediction_sha256),
                (
                    prediction.depth_confidence_preview_ref,
                    prediction.depth_confidence_preview_sha256,
                ),
            )
        )
    for record in summary.point_clouds.values():
        pairs.append((record.ply_ref, record.ply_sha256))

    verified: list[Path] = []
    for ref, expected in pairs:
        path = project_root / ref
        if _sha256(path) != expected:
            raise ValueError(f"S02 artifact hash mismatch: {ref}")
        verified.append(path)
    return verified


def _read_points(path: Path) -> FloatArray:
    vertices = PlyData.read(path)["vertex"]
    points = np.column_stack(
        (vertices["x"], vertices["y"], vertices["z"])
    ).astype(np.float64)
    if points.shape[0] == 0 or not np.isfinite(points).all():
        raise ValueError(f"point cloud must be non-empty and finite: {path}")
    return points


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
