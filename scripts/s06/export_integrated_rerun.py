"""Export synchronized S01-S05 evidence to one file-backed S06 Rerun recording."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from plyfile import PlyData  # type: ignore[import-untyped]

from spatial_reconstruction.orchestration import (
    ArtifactRole,
    Stage06OrchestrationManifest,
    build_event_markers,
    point_style,
)

os.environ.setdefault("RERUN_TELEMETRY", "off")
import rerun as rr  # noqa: E402
import rerun.blueprint as rrb  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMERA_IDS = ("camera_a", "camera_b")
TARGETS = ("person", "backpack")
POINT_LIMIT = 40_000
FloatArray = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orchestration-summary",
        type=Path,
        default=Path("artifacts/s06/orchestration_contract_v2_20260805/summary.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    orchestration_summary_path = (PROJECT_ROOT / args.orchestration_summary).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite Rerun recording: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    orchestration_summary = _load_json(orchestration_summary_path)
    manifest_path = PROJECT_ROOT / str(orchestration_summary["manifest_ref"])
    if _sha256(manifest_path) != orchestration_summary["manifest_sha256"]:
        raise ValueError("orchestration manifest hash differs from its summary")
    manifest = Stage06OrchestrationManifest.model_validate(_load_json(manifest_path))
    artifacts = {artifact.role: artifact for artifact in manifest.artifacts}
    for artifact in artifacts.values():
        if _sha256(PROJECT_ROOT / artifact.source_ref) != artifact.source_sha256:
            raise ValueError(f"orchestration artifact hash differs: {artifact.source_ref}")
    for video in manifest.source_videos:
        if _sha256(PROJECT_ROOT / video.source_ref) != video.source_sha256:
            raise ValueError(f"source video hash differs: {video.source_ref}")

    static_summary = _artifact_json(artifacts, ArtifactRole.STATIC_SCENE)
    calibration = _artifact_json(artifacts, ArtifactRole.ACTION_CALIBRATION)
    scene_metadata = _artifact_json(artifacts, ArtifactRole.SCENE_METADATA)
    perception_summary = _artifact_json(artifacts, ArtifactRole.PERCEPTION_TIMELINE)
    temporal_summary = _artifact_json(artifacts, ArtifactRole.TEMPORAL_PRESENTATION)
    interaction_summary = _artifact_json(artifacts, ArtifactRole.INTERACTION_TIMELINE)
    qwen_plan = _artifact_json(artifacts, ArtifactRole.QWEN_EVENT_PLAN)
    qwen_execution = _artifact_json(artifacts, ArtifactRole.QWEN_EVENT_RESULTS)
    qwen_results_document = _load_json(PROJECT_ROOT / str(qwen_execution["final_results_ref"]))
    qwen_results = cast(list[dict[str, Any]], qwen_results_document["results"])
    event_markers = build_event_markers(qwen_plan["jobs"], qwen_results)

    rr.init(
        "spatial_reconstruction_s06",
        recording_id=f"{manifest.capture_session_id}_{manifest.manifest_id[:12]}",
        spawn=False,
        strict=True,
    )
    rr.save(output_path)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "provenance/orchestration",
        rr.TextDocument(
            json.dumps(manifest.model_dump(mode="json"), indent=2),
            media_type="text/markdown",
        ),
        static=True,
    )

    logged_points = _log_static_world(static_summary, calibration, scene_metadata)
    video_frame_counts = _log_videos(manifest)
    box_count, mask_frame_count = _log_perception(perception_summary)
    state_counts = _log_temporal_presentation(temporal_summary)
    segment_counts = _log_trajectories(temporal_summary)
    _log_interaction(interaction_summary)
    _log_events(event_markers)
    _send_blueprint()
    rr.disconnect()

    summary_path = output_path.with_name(f"{output_path.stem}_export_summary.json")
    summary = {
        "schema_version": 1,
        "stage": "S06",
        "status": "passed",
        "purpose": "integrated_file_backed_rerun_export",
        "rerun_sdk_version": rr.__version__,
        "source_orchestration_summary_ref": _relative(orchestration_summary_path),
        "source_orchestration_summary_sha256": _sha256(orchestration_summary_path),
        "manifest_id": manifest.manifest_id,
        "authoritative_timeline": manifest.policy.authoritative_timeline,
        "recording_ref": _relative(output_path),
        "recording_sha256": _sha256(output_path),
        "recording_bytes": output_path.stat().st_size,
        "logged_static_point_count": logged_points,
        "video_frame_reference_counts": video_frame_counts,
        "perception_box_count": box_count,
        "segmentation_frame_count": mask_frame_count,
        "presentation_record_count": len(temporal_summary["presentation_records"]),
        "presentation_state_counts": state_counts,
        "measured_segment_counts": segment_counts,
        "interaction_record_count": len(interaction_summary["records"]),
        "event_markers": [marker.model_dump(mode="json") for marker in event_markers],
        "source_transition_times_preserved": True,
        "qwen_review_times_logged_separately": True,
        "worker_completion_order_used": False,
        "model_inference_performed": False,
        "entity_roots": [
            "cameras/camera_a/video",
            "cameras/camera_b/video",
            "world/static_scene",
            "world/dynamic",
            "world/trajectories",
            "timeline/interaction",
            "timeline/localization",
            "events/transitions",
            "events/qwen_reviews",
        ],
        "limitations": [
            "Video assets use the retained H.264 synchronized MP4 files.",
            "Stale points are display-only and measured trajectories remain disconnected.",
            "Missing and occluded states carry no point position.",
            "This export does not perform model inference or RTSP ingestion.",
        ],
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _log_static_world(
    static_summary: dict[str, Any],
    calibration: dict[str, Any],
    scene_metadata: dict[str, Any],
) -> int:
    fused_ref = str(static_summary["point_clouds"]["fused"]["ply_ref"])
    points, colors = _read_colored_ply(PROJECT_ROOT / fused_ref)
    points, colors = _sample(points, colors, maximum_points=POINT_LIMIT)
    for chunk_index, start in enumerate(range(0, len(points), 4_000)):
        end = min(start + 4_000, len(points))
        rr.log(
            f"world/static_scene/chunk_{chunk_index:03d}",
            rr.Points3D(points[start:end], colors=colors[start:end], radii=0.006),
            static=True,
        )
    for camera_id in CAMERA_IDS:
        record = calibration["cameras"][camera_id]
        transform = np.asarray(record["pose"]["T_world_from_camera"], dtype=float)
        intrinsics = record["intrinsics"]
        path = f"world/cameras/{camera_id}"
        rr.log(
            path,
            rr.Transform3D(
                translation=transform[:3, 3],
                mat3x3=transform[:3, :3],
                relation=rr.components.TransformRelation.ParentFromChild,
                axis_length=0.25,
            ),
            static=True,
        )
        rr.log(
            path,
            rr.Pinhole(
                image_from_camera=[
                    [intrinsics["fx"], 0.0, intrinsics["cx"]],
                    [0.0, intrinsics["fy"], intrinsics["cy"]],
                    [0.0, 0.0, 1.0],
                ],
                resolution=[intrinsics["image_width"], intrinsics["image_height"]],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.35,
            ),
            static=True,
        )
    for zone_name in ("pickup_blue_bed", "dropoff_white_floor"):
        zone = scene_metadata["zones"][zone_name]
        centre = np.asarray(zone["center_world_m"], dtype=float)
        angles = np.linspace(0.0, 2.0 * np.pi, 129)
        radius = float(zone["radius_m"])
        ring = np.column_stack(
            [
                centre[0] + radius * np.cos(angles),
                centre[1] + radius * np.sin(angles),
                np.full_like(angles, centre[2]),
            ]
        )
        color = [30, 110, 255] if zone_name.startswith("pickup") else [245, 245, 245]
        rr.log(
            f"world/zones/{zone_name}",
            rr.LineStrips3D([ring], colors=[color], radii=0.012),
            static=True,
        )
    return int(len(points))


def _log_videos(manifest: Stage06OrchestrationManifest) -> dict[str, int]:
    counts: dict[str, int] = {}
    for video in manifest.source_videos:
        path = f"cameras/{video.camera_id}/video"
        asset = rr.AssetVideo(path=PROJECT_ROOT / video.source_ref)
        rr.log(path, asset, static=True)
        timestamps_ns = asset.read_frame_timestamps_ns()
        if len(timestamps_ns) != video.decoded_frame_count:
            raise ValueError(f"Rerun video frame count differs: {video.camera_id}")
        rr.send_columns(
            path,
            indexes=[rr.TimeNanosColumn("capture_time", timestamps_ns)],
            columns=rr.VideoFrameReference.columns_nanoseconds(timestamps_ns),
        )
        rr.log(
            path,
            rr.AnnotationContext([(1, "person", (40, 235, 90)), (2, "backpack", (30, 145, 255))]),
            static=True,
        )
        counts[video.camera_id] = len(timestamps_ns)
    return counts


def _log_perception(perception_summary: dict[str, Any]) -> tuple[int, int]:
    box_count = 0
    segmentation_frame_count = 0
    for camera_id in CAMERA_IDS:
        timeline_ref = perception_summary["camera_summaries"][camera_id]["timeline_ref"]
        timeline = _load_json(PROJECT_ROOT / str(timeline_ref))
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in timeline["records"]:
            grouped[int(record["frame_identity"]["source_frame_index"])].append(record)
        for records in grouped.values():
            records.sort(key=lambda item: TARGETS.index(str(item["target"])))
            identity = records[0]["frame_identity"]
            rr.set_time_seconds("capture_time", float(identity["capture_timestamp_seconds"]))
            segmentation = np.zeros(
                (int(identity["image_height"]), int(identity["image_width"])),
                dtype=np.uint8,
            )
            has_mask = False
            for record in records:
                target = str(record["target"])
                candidates = record["candidate_metrics"]
                path = f"cameras/{camera_id}/video/detections/{target}"
                if not candidates:
                    rr.log(path, rr.Clear(recursive=False))
                    continue
                boxes: list[list[float]] = []
                labels: list[str] = []
                class_ids: list[int] = []
                for metric in candidates:
                    candidate = metric["candidate"]
                    detection = candidate["source_detection"]
                    box = detection["box"]
                    boxes.append([box["x_min"], box["y_min"], box["x_max"], box["y_max"]])
                    labels.append(
                        f"{target} {detection['class_name']} "
                        f"{float(detection['confidence']):.2f} "
                        f"{detection['camera_local_track_id'] or 'untracked'}"
                    )
                    class_id = 1 if target == "person" else 2
                    class_ids.append(class_id)
                    mask = _read_mask(str(detection["mask_ref"]))
                    segmentation[mask] = class_id
                    has_mask = True
                rr.log(
                    path,
                    rr.Boxes2D(
                        array=boxes,
                        array_format=rr.Box2DFormat.XYXY,
                        labels=labels,
                        class_ids=class_ids,
                        show_labels=True,
                    ),
                )
                box_count += len(boxes)
            segmentation_path = f"cameras/{camera_id}/video/segmentation"
            if has_mask:
                rr.log(segmentation_path, rr.SegmentationImage(segmentation, opacity=0.38))
                segmentation_frame_count += 1
            else:
                rr.log(segmentation_path, rr.Clear(recursive=False))
    return box_count, segmentation_frame_count


def _log_temporal_presentation(temporal: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    records = sorted(
        temporal["presentation_records"],
        key=lambda item: (item["capture_timestamp_seconds"], TARGETS.index(item["target"])),
    )
    for record in records:
        target = str(record["target"])
        state = str(record["state"])
        counts[f"{target}:{state}"] += 1
        rr.set_time_seconds("capture_time", float(record["capture_timestamp_seconds"]))
        style = point_style(
            target=target,
            state=state,
            anchor_kind=record["anchor_kind"],
        )
        point_path = f"world/dynamic/{target}/current"
        if style.show_position:
            xyz = record["presentation_world_xyz_m"]
            if xyz is None:
                raise ValueError("visible presentation state lacks presentation XYZ")
            age = record["measurement_age_seconds"]
            age_text = "" if age is None else f" age={float(age):.1f}s"
            rr.log(
                point_path,
                rr.Points3D(
                    [xyz],
                    colors=[style.color],
                    radii=style.radius_m,
                    labels=[f"{style.label_prefix}{age_text}"],
                    show_labels=True,
                ),
            )
        else:
            if (
                record["raw_world_xyz_m"] is not None
                or record["presentation_world_xyz_m"] is not None
            ):
                raise ValueError("missing/occluded presentation state contains XYZ")
            rr.log(point_path, rr.Clear(recursive=False))
        state_code = {"missing": 0.0, "occluded": 1.0, "stale": 2.0, "measured": 3.0}[state]
        scalar_path = f"timeline/localization/{target}/state_code"
        rr.log(scalar_path, rr.Scalar(state_code))
        rr.log(
            f"timeline/localization/{target}/log",
            rr.TextLog(f"{state}: {record['reason']} ({record['visual_style_id']})"),
        )
    return dict(counts)


def _log_trajectories(temporal: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    colors = {
        "person_footpoint": [40, 235, 90],
        "person_lower_body_surface": [255, 215, 40],
        "person_upper_body_surface": [200, 90, 255],
        "backpack_visible_cluster": [30, 145, 255],
    }
    for segment in temporal["measured_trajectory_segments"]:
        anchor = str(segment["anchor_kind"])
        target = str(segment["target"])
        rr.log(
            f"world/trajectories/{target}/{anchor}/{segment['segment_id'][:12]}",
            rr.LineStrips3D(
                [[segment["start_world_xyz_m"], segment["end_world_xyz_m"]]],
                colors=[colors[anchor]],
                radii=0.018,
            ),
            static=True,
        )
        counts[target] += 1
    return dict(counts)


def _log_interaction(interaction: dict[str, Any]) -> None:
    phase_codes = {"unknown": 0.0, "at_pickup": 1.0, "pickup": 2.0, "carry": 3.0, "place": 4.0}
    visibility_codes = {
        "unknown": 0.0,
        "visible": 1.0,
        "partially_occluded": 2.0,
        "fully_occluded": 3.0,
        "out_of_view": 4.0,
    }
    localization_codes = {"unavailable": 0.0, "stale": 1.0, "measured": 2.0}
    for record in interaction["records"]:
        rr.set_time_seconds("capture_time", float(record["capture_timestamp_seconds"]))
        rr.log("timeline/interaction/phase_code", rr.Scalar(phase_codes[record["phase"]]))
        rr.log(
            "timeline/interaction/visibility_code",
            rr.Scalar(visibility_codes[record["visibility_state"]]),
        )
        rr.log(
            "timeline/interaction/localization_code",
            rr.Scalar(localization_codes[record["localization_availability"]]),
        )
        rr.log(
            "timeline/interaction/log",
            rr.TextLog(
                f"phase={record['phase']} visibility={record['visibility_state']} "
                f"localization={record['localization_availability']} "
                f"authority={record['phase_authority']}"
            ),
        )


def _log_events(markers: tuple[Any, ...]) -> None:
    for marker in markers:
        rr.set_time_seconds("capture_time", marker.transition_timestamp_seconds)
        rr.log(
            f"events/transitions/{marker.event_kind}",
            rr.TextLog(
                f"{marker.event_kind} transition frame={marker.transition_frame_index}",
                level="INFO",
            ),
        )
        rr.set_time_seconds("capture_time", marker.review_timestamp_seconds)
        rr.log(
            f"events/qwen_reviews/{marker.event_kind}",
            rr.TextLog(
                f"Qwen {marker.qwen_event_label} match={marker.qwen_matches_candidate}: "
                f"{marker.qwen_summary}; review_frame={marker.review_frame_index}",
                level="INFO",
            ),
        )


def _send_blueprint() -> None:
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Spatial2DView(origin="cameras/camera_a/video", name="Camera A"),
            rrb.Spatial2DView(origin="cameras/camera_b/video", name="Camera B"),
            rrb.Spatial3DView(origin="world", name="Metric Digital Twin", background=[18, 18, 18]),
            rrb.TimeSeriesView(origin="timeline", name="State and interaction timeline"),
            rrb.TextLogView(origin="events", name="Pickup, carry, place events"),
            auto_layout=True,
            collapse_panels=True,
        )
    )


def _artifact_json(artifacts: dict[ArtifactRole, Any], role: ArtifactRole) -> dict[str, Any]:
    return _load_json(PROJECT_ROOT / artifacts[role].source_ref)


def _read_mask(mask_ref: str) -> NDArray[np.bool_]:
    file_ref, separator, fragment = mask_ref.partition("#mask_")
    if not separator or not fragment.isdigit():
        raise ValueError(f"invalid retained mask reference: {mask_ref}")
    with np.load(PROJECT_ROOT / file_ref, allow_pickle=False) as payload:
        masks = payload["source_sized_masks"]
        index = int(fragment)
        if index >= len(masks):
            raise ValueError(f"mask index outside retained array: {mask_ref}")
        return cast(NDArray[np.bool_], np.asarray(masks[index], dtype=bool))


def _read_colored_ply(path: Path) -> tuple[FloatArray, UInt8Array]:
    vertex = PlyData.read(path)["vertex"].data
    points = np.column_stack([vertex[axis] for axis in ("x", "y", "z")]).astype(np.float64)
    colors = np.column_stack([vertex[channel] for channel in ("red", "green", "blue")]).astype(
        np.uint8
    )
    return points, colors


def _sample(
    points: FloatArray,
    colors: UInt8Array,
    *,
    maximum_points: int,
) -> tuple[FloatArray, UInt8Array]:
    if len(points) <= maximum_points:
        return points, colors
    indices = np.linspace(0, len(points) - 1, maximum_points, dtype=np.int64)
    return points[indices], colors[indices]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
