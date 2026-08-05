"""Build the hash-bound S06 integrated offline orchestration entry manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spatial_reconstruction.orchestration import (
    ArtifactRole,
    OrchestrationArtifact,
    SourceVideo,
    Stage06OrchestrationManifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QWEN_PLAN_REF = "artifacts/s05/qwen_event_job_plan_v4_20260804/summary.json"
ARTIFACT_REFS = {
    ArtifactRole.ACTION_SYNCHRONIZATION: (
        "artifacts/s01/action_take_01/synchronized/synchronization_manifest.json"
    ),
    ArtifactRole.ACTION_CALIBRATION: (
        "artifacts/s01/calibration/action_take_01_pose/camera_calibration.json"
    ),
    ArtifactRole.SCENE_METADATA: "artifacts/s01/scene_metadata.json",
    ArtifactRole.STATIC_SCENE: ("artifacts/s02/door_inclusive_candidate_20260731/summary.json"),
    ArtifactRole.PERCEPTION_TIMELINE: ("artifacts/s03/target_timeline_5fps_20260801/summary.json"),
    ArtifactRole.TEMPORAL_PRESENTATION: (
        "artifacts/s04/temporal_presentation_occlusion_aware_20260803_v2/summary.json"
    ),
    ArtifactRole.INTERACTION_TIMELINE: (
        "artifacts/s05/semantic_interaction_v2_20260803/summary.json"
    ),
    ArtifactRole.QWEN_EVENT_PLAN: QWEN_PLAN_REF,
    ArtifactRole.QWEN_EVENT_RESULTS: (
        "artifacts/s05/qwen_event_execution_v5_20260804/summary.json"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    plan = _load_json(PROJECT_ROOT / QWEN_PLAN_REF)
    jobs = plan["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("accepted Qwen plan contains no jobs")
    capture_session_id = str(jobs[0]["capture_session_id"])
    synchronization_ref = str(plan["source_synchronization_manifest_ref"])
    synchronization_sha256 = str(plan["source_synchronization_manifest_sha256"])
    if synchronization_ref != ARTIFACT_REFS[ArtifactRole.ACTION_SYNCHRONIZATION]:
        raise ValueError("accepted Qwen plan uses an unexpected synchronization manifest")
    if _sha256(PROJECT_ROOT / synchronization_ref) != synchronization_sha256:
        raise ValueError("action synchronization manifest hash differs from Qwen plan")

    source_videos = tuple(SourceVideo.model_validate(video) for video in plan["video_sources"])
    if len(source_videos) != 2:
        raise ValueError("accepted Qwen plan must bind exactly two source videos")
    typed_videos = (source_videos[0], source_videos[1])
    for video in typed_videos:
        if _sha256(PROJECT_ROOT / video.source_ref) != video.source_sha256:
            raise ValueError(f"source video hash differs: {video.source_ref}")

    artifacts = tuple(
        OrchestrationArtifact(
            role=role,
            source_ref=source_ref,
            source_sha256=_sha256(PROJECT_ROOT / source_ref),
        )
        for role, source_ref in ARTIFACT_REFS.items()
    )
    manifest = Stage06OrchestrationManifest.create(
        capture_session_id=capture_session_id,
        synchronization_manifest_ref=synchronization_ref,
        synchronization_manifest_sha256=synchronization_sha256,
        source_videos=typed_videos,
        artifacts=artifacts,
    )
    manifest_path = output_dir / "orchestration_manifest.json"
    _write_json(manifest_path, manifest.model_dump(mode="json"))
    summary = {
        "schema_version": 1,
        "stage": "S06",
        "status": "completed",
        "purpose": "integrated_offline_orchestration_entry_contract",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "manifest_id": manifest.manifest_id,
        "manifest_ref": _relative(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "capture_session_id": manifest.capture_session_id,
        "source_video_count": len(manifest.source_videos),
        "artifact_count": len(manifest.artifacts),
        "policy": manifest.policy.model_dump(mode="json"),
        "model_inference_performed": False,
        "rerun_recording_written": False,
        "limitations": [
            "WP1 binds accepted file inputs but does not yet assemble the final Rerun recording.",
            "RTSP reconnect testing is assigned to a later S06 work package.",
            (
                "The process supervisor is verified with synthetic child processes, "
                "not a repeated Qwen model run."
            ),
        ],
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


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
