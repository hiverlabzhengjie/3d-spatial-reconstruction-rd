"""Independently verify the S07 final demonstration entry manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from spatial_reconstruction.finalization import FinalArtifactRole, Stage07FinalRunManifest
from spatial_reconstruction.orchestration import Stage06OrchestrationManifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
S06_MANIFEST_REF = (
    "artifacts/s06/orchestration_contract_v2_20260805/orchestration_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite verification: {output}")
    summary_path = args.summary.resolve()
    summary = _load_json(summary_path)
    manifest_path = PROJECT_ROOT / str(summary["manifest_ref"])
    if _sha256(manifest_path) != summary["manifest_sha256"]:
        raise ValueError("final-run manifest hash differs from summary")
    manifest = Stage07FinalRunManifest.model_validate(_load_json(manifest_path))

    s06_manifest = Stage06OrchestrationManifest.model_validate(
        _load_json(PROJECT_ROOT / S06_MANIFEST_REF)
    )
    if manifest.source_stage06_manifest_id != s06_manifest.manifest_id:
        raise ValueError("final run is not bound to the accepted S06 manifest")
    if manifest.source_videos != s06_manifest.source_videos:
        raise ValueError("final source videos differ from the accepted S06 videos")
    for video in manifest.source_videos:
        if _sha256(PROJECT_ROOT / video.source_ref) != video.source_sha256:
            raise ValueError(f"final source video hash differs: {video.source_ref}")
    for artifact in manifest.artifacts:
        if _sha256(PROJECT_ROOT / artifact.source_ref) != artifact.source_sha256:
            raise ValueError(f"final-run artifact hash differs: {artifact.source_ref}")

    artifacts = {artifact.role: artifact for artifact in manifest.artifacts}
    rerun_summary = _load_json(
        PROJECT_ROOT / artifacts[FinalArtifactRole.RERUN_EXPORT_SUMMARY].source_ref
    )
    if rerun_summary["recording_sha256"] != artifacts[
        FinalArtifactRole.INTEGRATED_RERUN
    ].source_sha256:
        raise ValueError("Rerun recording hash differs from its export summary")
    gate = _load_json(PROJECT_ROOT / artifacts[FinalArtifactRole.STAGE06_GATE_AUDIT].source_ref)
    if gate.get("completion_gate_passed") is not True or gate.get(
        "completion_gate_weakened"
    ) is not False:
        raise ValueError("S06 completion gate is not accepted without weakening")
    if gate.get("source_manifest_id") != manifest.source_stage06_manifest_id:
        raise ValueError("S06 gate audit uses a different orchestration manifest")

    report = {
        "schema_version": 1,
        "stage": "S07",
        "work_package": 1,
        "status": "passed",
        "purpose": "final_recording_selection_and_entry_verification",
        "source_summary_ref": _relative(summary_path),
        "source_summary_sha256": _sha256(summary_path),
        "manifest_regenerated": True,
        "manifest_id": manifest.manifest_id,
        "source_stage06_manifest_id": manifest.source_stage06_manifest_id,
        "recording_name": manifest.recording.recording_name,
        "recording_selected_by_user": manifest.recording.selected_by_user,
        "source_video_count": len(manifest.source_videos),
        "source_video_frame_counts": {
            video.camera_id: video.decoded_frame_count for video in manifest.source_videos
        },
        "artifact_count": len(manifest.artifacts),
        "s06_completion_gate_passed": True,
        "s06_completion_gate_weakened": False,
        "recapture_required": manifest.recording.recapture_required,
        "recalibration_required": manifest.recording.recalibration_required,
        "unavailable_xyz_must_remain_null": manifest.policy.preserve_null_unavailable_xyz,
        "measured_trajectories_must_remain_disconnected": (
            manifest.policy.preserve_disconnected_measured_trajectories
        ),
        "qwen_has_spatial_authority": manifest.policy.qwen_has_spatial_authority,
        "demonstrated_live_capacity": manifest.policy.demonstrated_live_capacity,
        "model_inference_performed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


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
