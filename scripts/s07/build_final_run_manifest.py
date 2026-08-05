"""Build the hash-bound S07 final demonstration entry manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spatial_reconstruction.finalization import (
    FinalArtifactRole,
    FinalRunArtifact,
    Stage07FinalRunManifest,
)
from spatial_reconstruction.orchestration import Stage06OrchestrationManifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
S06_MANIFEST_REF = (
    "artifacts/s06/orchestration_contract_v2_20260805/orchestration_manifest.json"
)
ARTIFACT_REFS = {
    FinalArtifactRole.ORCHESTRATION_SUMMARY: (
        "artifacts/s06/orchestration_contract_v2_20260805/summary.json"
    ),
    FinalArtifactRole.INTEGRATED_RERUN: (
        "artifacts/s06/integrated_rerun_20260805/digital_twin_stage06_v2.rrd"
    ),
    FinalArtifactRole.RERUN_EXPORT_SUMMARY: (
        "artifacts/s06/integrated_rerun_20260805/"
        "digital_twin_stage06_v2_export_summary.json"
    ),
    FinalArtifactRole.INTEGRATED_REPLAY_SUMMARY: (
        "artifacts/s06/integrated_replay_v2_20260805/summary.json"
    ),
    FinalArtifactRole.RTSP_SMOKE_SUMMARY: (
        "artifacts/s06/rtsp_smoke_v4_20260805/summary.json"
    ),
    FinalArtifactRole.TRACK_EVENT_EXPORT_SUMMARY: (
        "artifacts/s06/exports_20260805/summary.json"
    ),
    FinalArtifactRole.STAGE06_GATE_AUDIT: (
        "artifacts/s06/stage_close_audit_20260805/summary.json"
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

    s06_manifest_path = PROJECT_ROOT / S06_MANIFEST_REF
    s06_manifest = Stage06OrchestrationManifest.model_validate(_load_json(s06_manifest_path))
    _verify_s06_sources(s06_manifest)

    artifacts = tuple(
        FinalRunArtifact(
            role=role,
            source_ref=source_ref,
            source_sha256=_sha256(PROJECT_ROOT / source_ref),
        )
        for role, source_ref in ARTIFACT_REFS.items()
    )
    manifest = Stage07FinalRunManifest.create(
        source_stage06_manifest_id=s06_manifest.manifest_id,
        source_videos=s06_manifest.source_videos,
        artifacts=artifacts,
    )
    manifest_path = output_dir / "final_run_manifest.json"
    _write_json(manifest_path, manifest.model_dump(mode="json"))
    summary = {
        "schema_version": 1,
        "stage": "S07",
        "work_package": 1,
        "status": "completed",
        "purpose": "final_recording_selection_and_reproducible_run_entry_contract",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "manifest_id": manifest.manifest_id,
        "manifest_ref": _relative(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "source_stage06_manifest_id": manifest.source_stage06_manifest_id,
        "recording_name": manifest.recording.recording_name,
        "recording_selected_by_user": manifest.recording.selected_by_user,
        "source_video_count": len(manifest.source_videos),
        "source_video_frame_counts": {
            video.camera_id: video.decoded_frame_count for video in manifest.source_videos
        },
        "artifact_count": len(manifest.artifacts),
        "recapture_required": manifest.recording.recapture_required,
        "recalibration_required": manifest.recording.recalibration_required,
        "model_inference_performed": False,
        "final_rerun_written": False,
        "demo_video_written": False,
        "limitations": [
            "WP1 locks and verifies final inputs; it does not yet generate final outputs.",
            (
                "The retained S06 replay timings are virtual-time scheduling evidence, "
                "not measured M1 throughput."
            ),
            "The accepted backpack localization gap remains unavailable and must not be filled.",
        ],
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _verify_s06_sources(manifest: Stage06OrchestrationManifest) -> None:
    for video in manifest.source_videos:
        if _sha256(PROJECT_ROOT / video.source_ref) != video.source_sha256:
            raise ValueError(f"source video hash differs: {video.source_ref}")
    for artifact in manifest.artifacts:
        if _sha256(PROJECT_ROOT / artifact.source_ref) != artifact.source_sha256:
            raise ValueError(f"accepted S06 input hash differs: {artifact.source_ref}")


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
