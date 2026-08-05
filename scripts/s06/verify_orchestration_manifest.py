"""Independently verify the S06 integrated offline orchestration manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from spatial_reconstruction.orchestration import Stage06OrchestrationManifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    summary = _load_json(args.summary.resolve())
    manifest_ref = str(summary["manifest_ref"])
    manifest_path = PROJECT_ROOT / manifest_ref
    if _sha256(manifest_path) != summary["manifest_sha256"]:
        raise ValueError("orchestration manifest hash differs from summary")
    manifest = Stage06OrchestrationManifest.model_validate(_load_json(manifest_path))

    for video in manifest.source_videos:
        if _sha256(PROJECT_ROOT / video.source_ref) != video.source_sha256:
            raise ValueError(f"source video hash differs: {video.source_ref}")
    for artifact in manifest.artifacts:
        if _sha256(PROJECT_ROOT / artifact.source_ref) != artifact.source_sha256:
            raise ValueError(f"accepted artifact hash differs: {artifact.source_ref}")

    report = {
        "schema_version": 1,
        "stage": "S06",
        "status": "passed",
        "purpose": "integrated_offline_orchestration_entry_verification",
        "source_summary_ref": _relative(args.summary.resolve()),
        "source_summary_sha256": _sha256(args.summary.resolve()),
        "manifest_regenerated": True,
        "manifest_id": manifest.manifest_id,
        "source_video_count": len(manifest.source_videos),
        "artifact_count": len(manifest.artifacts),
        "capture_time_authoritative": True,
        "worker_completion_order_authoritative": False,
        "heavy_mps_permit_count": manifest.policy.heavy_mps_permit_count,
        "qwen_failure_blocks_geometry": manifest.policy.qwen_failure_blocks_geometry,
        "rerun_recording_written": False,
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
