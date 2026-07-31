"""Verify deterministic S01 file replay and synchronized frame-bundle identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from spatial_reconstruction.contracts import (
    FrameBundleStatus,
    SynchronizedFrameBundle,
)
from spatial_reconstruction.ingestion import (
    FileFrameSource,
    TimestampTransform,
    build_synchronized_bundles,
    restore_capture_order,
)


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


def _ordered_digest(bundle_ids: list[str]) -> str:
    payload = ("\n".join(bundle_ids) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_replay(
    *,
    project_root: Path,
    synchronization_manifest_path: Path,
    pose_calibration_path: Path,
    output_path: Path,
    pairing_tolerance_seconds: float,
) -> dict[str, Any]:
    """Run two real file replays and retain bounded deterministic evidence."""

    manifest = _load_json(synchronization_manifest_path)
    calibration = _load_json(pose_calibration_path)
    if not str(calibration["calibration_status"]).startswith("accepted_"):
        raise ValueError("pose calibration is not accepted")

    manifest_ref = str(synchronization_manifest_path.relative_to(project_root))
    manifest_sha256 = _sha256(synchronization_manifest_path)
    capture_session_id = str(manifest["capture_session_id"])
    reference_camera_id = str(manifest["reference_camera_id"])
    derived_outputs = cast(dict[str, dict[str, Any]], manifest["derived_outputs"])
    source_records = cast(dict[str, dict[str, Any]], manifest["sources"])
    camera_ids = tuple(
        camera_id
        for camera_id in ("camera_a", "camera_b")
        if camera_id in derived_outputs
    )
    if camera_ids != ("camera_a", "camera_b"):
        raise ValueError("verification requires camera_a and camera_b outputs")

    sources = {
        camera_id: FileFrameSource(
            path=project_root / str(derived_outputs[camera_id]["path"]),
            capture_session_id=capture_session_id,
            camera_id=camera_id,
            source_ref=str(derived_outputs[camera_id]["path"]),
            expected_sha256=str(derived_outputs[camera_id]["sha256"]),
            synchronization_manifest_ref=manifest_ref,
            synchronization_manifest_sha256=manifest_sha256,
            pose_version_id=str(calibration["pose_version_id"]),
            timestamp_transform=TimestampTransform(),
            expected_width=int(source_records[camera_id]["image_width"]),
            expected_height=int(source_records[camera_id]["image_height"]),
        )
        for camera_id in camera_ids
    }
    pixel_decode_smoke: dict[str, dict[str, Any]] = {}
    for camera_id in camera_ids:
        frame_iterator = sources[camera_id].iter_frames()
        first_decoded = next(frame_iterator)
        close_iterator = getattr(frame_iterator, "close", None)
        if callable(close_iterator):
            close_iterator()
        if first_decoded.image_bgr.flags.writeable:
            raise RuntimeError("decoded frame pixels are unexpectedly mutable")
        pixel_decode_smoke[camera_id] = {
            "first_frame_id": first_decoded.identity.frame_id,
            "shape": list(first_decoded.image_bgr.shape),
            "dtype": str(first_decoded.image_bgr.dtype),
            "writeable": bool(first_decoded.image_bgr.flags.writeable),
        }

    def replay() -> tuple[SynchronizedFrameBundle, ...]:
        return tuple(
            build_synchronized_bundles(
                {
                    camera_id: sources[camera_id].iter_identities()
                    for camera_id in camera_ids
                },
                expected_camera_ids=camera_ids,
                reference_camera_id=reference_camera_id,
                pairing_tolerance_seconds=pairing_tolerance_seconds,
            )
        )

    first = replay()
    second = replay()
    if not first:
        raise RuntimeError("real file replay produced no frame bundles")
    first_ids = [bundle.bundle_id for bundle in first]
    second_ids = [bundle.bundle_id for bundle in second]
    replay_match = first_ids == second_ids
    if not replay_match:
        raise RuntimeError("same-input replay produced different bundle identities")
    if len(set(first_ids)) != len(first_ids):
        raise RuntimeError("real file replay produced duplicate bundle identities")

    completed_in_reverse = list(reversed(first_ids))
    restored = restore_capture_order(
        completed_in_reverse,
        bundle_id_of=lambda bundle_id: bundle_id,
        bundles=first,
    )
    reverse_completion_restored = list(restored) == first_ids
    if not reverse_completion_restored:
        raise RuntimeError("worker completion order changed capture ordering")

    complete_count = sum(
        bundle.status is FrameBundleStatus.COMPLETE for bundle in first
    )
    incomplete_count = len(first) - complete_count
    missing_counts = {
        camera_id: sum(
            camera_id in bundle.missing_camera_ids for bundle in first
        )
        for camera_id in camera_ids
    }
    if incomplete_count:
        raise RuntimeError(
            f"accepted synchronized pair produced {incomplete_count} incomplete bundles"
        )

    for bundle in (first[0], first[-1]):
        restored_bundle = SynchronizedFrameBundle.model_validate_json(
            bundle.model_dump_json()
        )
        if restored_bundle != bundle:
            raise RuntimeError("bundle JSON schema round trip changed identity")

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "capture_session_id": capture_session_id,
        "pose_version_id": calibration["pose_version_id"],
        "input_provenance": {
            "synchronization_manifest": manifest_ref,
            "synchronization_manifest_sha256": manifest_sha256,
            "pose_calibration": str(pose_calibration_path.relative_to(project_root)),
            "pose_calibration_sha256": _sha256(pose_calibration_path),
            "sources": {
                camera_id: {
                    "path": derived_outputs[camera_id]["path"],
                    "sha256": derived_outputs[camera_id]["sha256"],
                }
                for camera_id in camera_ids
            },
        },
        "pairing_policy": {
            "expected_camera_ids": list(camera_ids),
            "reference_camera_id": reference_camera_id,
            "pairing_tolerance_seconds": pairing_tolerance_seconds,
            "algorithm": (
                "earliest-unconsumed capture-time bundling with deterministic "
                "camera order and no frame reuse"
            ),
        },
        "replay_results": {
            "bundle_count": len(first),
            "complete_bundle_count": complete_count,
            "incomplete_bundle_count": incomplete_count,
            "missing_bundle_count_by_camera": missing_counts,
            "maximum_frame_time_difference_seconds": max(
                bundle.max_frame_time_difference_seconds for bundle in first
            ),
            "first_capture_timestamp_seconds": first[0].capture_timestamp_seconds,
            "last_capture_timestamp_seconds": first[-1].capture_timestamp_seconds,
            "first_bundle_id": first[0].bundle_id,
            "last_bundle_id": first[-1].bundle_id,
            "ordered_bundle_id_sha256": _ordered_digest(first_ids),
            "same_input_replay_identity_and_order_match": replay_match,
            "reverse_worker_completion_restores_capture_order": (
                reverse_completion_restored
            ),
            "duplicate_bundle_id_count": len(first_ids) - len(set(first_ids)),
            "persistent_schema_round_trip_passed": True,
            "pixel_decode_smoke": pixel_decode_smoke,
        },
        "failure_behaviour_evidence": {
            "automated_tests": [
                "missing camera produces an explicit incomplete bundle",
                "duplicate frame IDs and non-increasing frame indices are rejected",
                "non-increasing capture timestamps are rejected",
                "mixed synchronization provenance is rejected",
                "duplicate and unknown worker completion results are rejected",
                "RTSP persistent references omit credentials and query strings",
            ]
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--synchronization-manifest",
        type=Path,
        default=Path(
            "artifacts/s01/action_take_01/synchronized/synchronization_manifest.json"
        ),
    )
    parser.add_argument(
        "--pose-calibration",
        type=Path,
        default=Path(
            "artifacts/s01/calibration/action_take_01_pose/camera_calibration.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/s01/ingestion/action_take_01_frame_bundle_replay.json"
        ),
    )
    parser.add_argument(
        "--pairing-tolerance-seconds",
        type=float,
        default=1.0 / 60.0,
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    result = verify_replay(
        project_root=project_root,
        synchronization_manifest_path=(
            project_root / args.synchronization_manifest
        ).resolve(),
        pose_calibration_path=(project_root / args.pose_calibration).resolve(),
        output_path=(project_root / args.output).resolve(),
        pairing_tolerance_seconds=float(args.pairing_tolerance_seconds),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
