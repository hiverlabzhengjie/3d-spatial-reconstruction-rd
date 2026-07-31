"""Export one accepted S02 point cloud and calibrated cameras to Rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from plyfile import PlyData  # type: ignore[import-untyped]

from spatial_reconstruction.geometry import (
    StaticSceneRerunExportSummary,
    StaticSceneRunSummary,
)

os.environ.setdefault("RERUN_TELEMETRY", "off")
import rerun as rr  # noqa: E402
import rerun.blueprint as rrb  # noqa: E402

FloatArray = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]
CAMERA_IDS = ("camera_a", "camera_b")
FUSED_VISUALIZATION_POINT_LIMIT = 30_000
CAMERA_VISUALIZATION_POINT_LIMIT = 2_000
CAMERA_IMAGE_MAXIMUM_DIMENSION = 960


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
        "--pose-calibration",
        type=Path,
        default=Path(
            "artifacts/s01/calibration/empty_room_pose/camera_calibration.json"
        ),
    )
    parser.add_argument(
        "--scene-metadata",
        type=Path,
        default=Path("artifacts/s01/scene_metadata.json"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--omit-camera-images",
        action="store_true",
        help="Retain camera transforms and pinholes but omit image textures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    summary_path = (project_root / args.run_summary).resolve()
    calibration_path = (project_root / args.pose_calibration).resolve()
    scene_metadata_path = (project_root / args.scene_metadata).resolve()
    summary = StaticSceneRunSummary.model_validate(
        _load_json(summary_path)
    ).model_dump(mode="json")
    calibration = _load_json(calibration_path)
    scene_metadata = _load_json(scene_metadata_path)
    _validate_inputs(
        project_root=project_root,
        summary=summary,
        calibration=calibration,
        scene_metadata=scene_metadata,
    )

    output_path = (
        (project_root / args.output).resolve()
        if args.output is not None
        else summary_path.parent / "static_scene.rrd"
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite Rerun recording: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    point_clouds = cast(dict[str, dict[str, Any]], summary["point_clouds"])
    source_fused_points, source_fused_colors = _read_colored_ply(
        project_root / str(point_clouds["fused"]["ply_ref"])
    )
    source_camera_clouds = {
        camera_id: _read_colored_ply(
            project_root / str(point_clouds[camera_id]["ply_ref"])
        )
        for camera_id in CAMERA_IDS
    }
    fused_points, fused_colors = _deterministic_visualization_sample(
        source_fused_points,
        source_fused_colors,
        maximum_points=FUSED_VISUALIZATION_POINT_LIMIT,
    )
    camera_clouds = {
        camera_id: _deterministic_visualization_sample(
            *source_camera_clouds[camera_id],
            maximum_points=CAMERA_VISUALIZATION_POINT_LIMIT,
        )
        for camera_id in CAMERA_IDS
    }
    rr.init(
        "spatial_reconstruction_s02",
        recording_id=(
            f"{summary['capture_session_id']}_{summary_path.parent.name}"
        ),
        spawn=False,
        strict=True,
    )
    rr.save(output_path)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    _log_point_cloud_chunks(
        entity_prefix="world/static_scene/fused",
        points=fused_points,
        colors=fused_colors,
        radius=0.006,
    )
    for camera_id, (points, colors) in camera_clouds.items():
        _log_point_cloud_chunks(
            entity_prefix=f"world/static_scene/by_camera/{camera_id}",
            points=points,
            colors=colors,
            radius=0.004,
        )

    room_bounds = cast(dict[str, Any], scene_metadata["room_bounds"])
    minimum = np.asarray(room_bounds["minimum_world_xyz_m"], dtype=np.float64)
    maximum = np.asarray(room_bounds["maximum_world_xyz_m"], dtype=np.float64)
    rr.log(
        "world/processing_bounds",
        rr.Boxes3D(
            centers=[(minimum + maximum) / 2.0],
            half_sizes=[(maximum - minimum) / 2.0],
            colors=[[120, 170, 255, 28]],
            radii=0.008,
        ),
        static=True,
    )

    marker_model = cast(dict[str, Any], calibration["marker_model"])
    accepted_marker_ids = tuple(
        int(value) for value in marker_model["pose_anchor_marker_ids"]
    )
    placements = cast(list[dict[str, Any]], marker_model["placements"])
    accepted_placements = tuple(
        record
        for record in placements
        if int(record["marker_id"]) in accepted_marker_ids
    )
    rr.log(
        "world/calibration/accepted_markers",
        rr.Points3D(
            [
                cast(list[float], record["center_world_m"])
                for record in accepted_placements
            ],
            colors=[[255, 210, 0]],
            radii=0.035,
            labels=[
                f"M{int(record['marker_id'])}" for record in accepted_placements
            ],
            show_labels=True,
        ),
        static=True,
    )

    camera_records = cast(dict[str, dict[str, Any]], calibration["cameras"])
    first_prediction = cast(dict[str, Any], summary["predictions"][0])
    first_prediction_cameras = cast(
        dict[str, dict[str, Any]],
        first_prediction["cameras"],
    )
    for camera_id in CAMERA_IDS:
        record = camera_records[camera_id]
        pose = cast(dict[str, Any], record["pose"])
        intrinsics = cast(dict[str, Any], record["intrinsics"])
        T_world_from_camera = np.asarray(
            pose["T_world_from_camera"],
            dtype=np.float64,
        )
        entity_path = f"world/cameras/{camera_id}"
        rr.log(
            entity_path,
            rr.Transform3D(
                translation=T_world_from_camera[:3, 3],
                mat3x3=T_world_from_camera[:3, :3],
                relation=rr.components.TransformRelation.ParentFromChild,
                axis_length=0.25,
            ),
            static=True,
        )
        rr.log(
            entity_path,
            rr.Pinhole(
                image_from_camera=[
                    [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
                    [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
                    [0.0, 0.0, 1.0],
                ],
                resolution=[
                    int(intrinsics["image_width"]),
                    int(intrinsics["image_height"]),
                ],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.35,
            ),
            static=True,
        )
        if not args.omit_camera_images:
            keyframe_path = _resolve_keyframe_path(
                project_root=project_root,
                run_dir=summary_path.parent,
                bundle_id=str(first_prediction["bundle_id"]),
                camera_id=camera_id,
                camera_record=first_prediction_cameras[camera_id],
            )
            with Image.open(keyframe_path) as image:
                image.thumbnail(
                    (
                        CAMERA_IMAGE_MAXIMUM_DIMENSION,
                        CAMERA_IMAGE_MAXIMUM_DIMENSION,
                    ),
                    Image.Resampling.LANCZOS,
                )
                rr.log(
                    entity_path,
                    rr.Image(np.asarray(image.convert("RGB"))),
                    static=True,
                )

    zones = cast(dict[str, Any], scene_metadata["zones"])
    for zone_name in ("pickup_blue_bed", "dropoff_white_floor"):
        zone = cast(dict[str, Any], zones[zone_name])
        centre = np.asarray(zone["center_world_m"], dtype=np.float64)
        radius = float(zone["radius_m"])
        angles = np.linspace(0.0, 2.0 * np.pi, num=129)
        ring = np.column_stack(
            (
                centre[0] + radius * np.cos(angles),
                centre[1] + radius * np.sin(angles),
                np.full_like(angles, centre[2]),
            )
        )
        color = [30, 110, 255] if zone_name.startswith("pickup") else [240, 240, 240]
        rr.log(
            f"world/zones/{zone_name}",
            rr.LineStrips3D([ring], colors=[color], radii=0.012),
            static=True,
        )

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Spatial3DView(
                origin="world",
                name="S02 Metric Static Scene",
                background=[18, 18, 18],
            ),
            collapse_panels=True,
        )
    )
    rr.disconnect()

    recording_sha256 = _sha256(output_path)
    result = {
        "schema_version": 1,
        "status": "passed",
        "stage": "S02",
        "rerun_sdk_version": rr.__version__,
        "source_summary_ref": _relative(summary_path, project_root),
        "source_summary_sha256": _sha256(summary_path),
        "recording_ref": _relative(output_path, project_root),
        "recording_sha256": recording_sha256,
        "recording_bytes": output_path.stat().st_size,
        "source_fused_point_count": int(source_fused_points.shape[0]),
        "source_camera_point_counts": {
            camera_id: int(source_camera_clouds[camera_id][0].shape[0])
            for camera_id in CAMERA_IDS
        },
        "logged_fused_point_count": int(fused_points.shape[0]),
        "logged_camera_point_counts": {
            camera_id: int(camera_clouds[camera_id][0].shape[0])
            for camera_id in CAMERA_IDS
        },
        "logged_camera_ids": list(CAMERA_IDS),
        "maximum_points_per_rerun_entity": 4_000,
        "camera_image_maximum_dimension": (
            None if args.omit_camera_images else CAMERA_IMAGE_MAXIMUM_DIMENSION
        ),
        "world_coordinates": "right-handed, metres, Z up",
        "camera_coordinates": "OpenCV X right, Y down, Z forward",
        "includes": [
            "deterministically sampled fused static point cloud",
            "deterministically sampled per-camera point clouds",
            "calibrated camera transforms and pinholes",
            *(
                []
                if args.omit_camera_images
                else ["representative downscaled undistorted camera images"]
            ),
            "accepted M40-M42 anchors",
            "conservative processing bounds",
            *(
                ["bounded D027 low-confidence static door supplement"]
                if bool(
                    cast(dict[str, Any], summary["processing"])
                    .get("static_inclusion", {})
                    .get("enabled", False)
                )
                else []
            ),
            "pickup and drop-off zone rings",
        ],
    }
    validated_result = StaticSceneRerunExportSummary.model_validate(
        result
    ).model_dump(mode="json")
    result_path = output_path.with_name(f"{output_path.stem}_export_summary.json")
    result_path.write_text(
        json.dumps(validated_result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validated_result, indent=2))
    return 0


def _log_point_cloud_chunks(
    *,
    entity_prefix: str,
    points: FloatArray,
    colors: UInt8Array,
    radius: float,
    maximum_points_per_entity: int = 4_000,
) -> None:
    if maximum_points_per_entity <= 0:
        raise ValueError("Rerun point-cloud chunk size must be positive")
    for chunk_index, start in enumerate(
        range(0, points.shape[0], maximum_points_per_entity)
    ):
        end = min(start + maximum_points_per_entity, points.shape[0])
        rr.log(
            f"{entity_prefix}/chunk_{chunk_index:03d}",
            rr.Points3D(
                points[start:end],
                colors=colors[start:end],
                radii=radius,
            ),
            static=True,
        )


def _deterministic_visualization_sample(
    points: FloatArray,
    colors: UInt8Array,
    *,
    maximum_points: int,
) -> tuple[FloatArray, UInt8Array]:
    if maximum_points <= 0:
        raise ValueError("visualization point limit must be positive")
    if points.shape[0] <= maximum_points:
        return points, colors
    indices = np.linspace(
        0,
        points.shape[0] - 1,
        num=maximum_points,
        dtype=np.int64,
    )
    return points[indices], colors[indices]


def _validate_inputs(
    *,
    project_root: Path,
    summary: dict[str, Any],
    calibration: dict[str, Any],
    scene_metadata: dict[str, Any],
) -> None:
    if summary.get("stage") != "S02":
        raise ValueError("Rerun export requires an S02 summary")
    if summary.get("status") != "completed_pending_visual_qa":
        raise ValueError("S02 source run is not complete enough for visual QA")
    if summary.get("pose_version_id") != calibration.get("pose_version_id"):
        raise ValueError("S02 summary and calibration pose versions differ")
    if (
        summary.get("pose_version_id")
        != cast(dict[str, str], scene_metadata["accepted_pose_versions"])["empty_room"]
    ):
        raise ValueError("S02 summary does not use the accepted empty-room pose")
    input_provenance = cast(dict[str, Any], summary["input_provenance"])
    expected_hashes = {
        "pose_calibration_sha256": _sha256(
            project_root / str(input_provenance["pose_calibration_ref"])
        ),
        "scene_metadata_sha256": _sha256(
            project_root / str(input_provenance["scene_metadata_ref"])
        ),
    }
    for field, actual in expected_hashes.items():
        if input_provenance.get(field) != actual:
            raise ValueError(f"S02 input provenance hash mismatch: {field}")
    point_clouds = cast(dict[str, dict[str, Any]], summary["point_clouds"])
    for key in (*CAMERA_IDS, "fused"):
        path = project_root / str(point_clouds[key]["ply_ref"])
        if _sha256(path) != point_clouds[key]["ply_sha256"]:
            raise ValueError(f"S02 point-cloud hash mismatch: {key}")


def _resolve_keyframe_path(
    *,
    project_root: Path,
    run_dir: Path,
    bundle_id: str,
    camera_id: str,
    camera_record: dict[str, Any],
) -> Path:
    explicit = camera_record.get("undistorted_keyframe_ref")
    if explicit is not None:
        path = project_root / str(explicit)
        expected_hash = camera_record.get("undistorted_keyframe_sha256")
        if expected_hash is not None and _sha256(path) != expected_hash:
            raise ValueError(f"keyframe hash mismatch for {camera_id}")
        return path
    matches = tuple(
        (run_dir / "keyframes").glob(f"{bundle_id[:12]}_{camera_id}_frame_*.png")
    )
    if len(matches) != 1:
        raise ValueError(f"could not resolve one retained keyframe for {camera_id}")
    return matches[0]


def _read_colored_ply(path: Path) -> tuple[FloatArray, UInt8Array]:
    ply = PlyData.read(path)
    vertices = ply["vertex"]
    points = np.column_stack(
        (vertices["x"], vertices["y"], vertices["z"])
    ).astype(np.float64)
    colors = np.column_stack(
        (vertices["red"], vertices["green"], vertices["blue"])
    ).astype(np.uint8)
    if points.shape[0] == 0 or not np.isfinite(points).all():
        raise ValueError(f"Rerun PLY must contain finite points: {path}")
    return points, colors


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], payload)


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
