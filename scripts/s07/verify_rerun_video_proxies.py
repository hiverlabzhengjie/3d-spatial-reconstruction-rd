"""Verify S07 seekable Rerun presentation-video proxies and source identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Literal, cast

from spatial_reconstruction.finalization import Stage07FinalRunManifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite verification: {output_path}")
    proxy_manifest = _load_json(manifest_path)
    if proxy_manifest["policy_id"] != "s07_rerun_seekable_h264_v1":
        raise ValueError("unexpected proxy policy")

    final_summary_path = PROJECT_ROOT / proxy_manifest["source_final_run_summary_ref"]
    if _sha256(final_summary_path) != proxy_manifest["source_final_run_summary_sha256"]:
        raise ValueError("final-run summary hash differs")
    final_summary = _load_json(final_summary_path)
    final_manifest = Stage07FinalRunManifest.model_validate(
        _load_json(PROJECT_ROOT / final_summary["manifest_ref"])
    )
    if final_manifest.manifest_id != proxy_manifest["source_final_run_manifest_id"]:
        raise ValueError("proxy manifest uses a different final-run identity")

    source_by_camera = {video.camera_id: video for video in final_manifest.source_videos}
    verified: dict[str, dict[str, Any]] = {}
    for record in proxy_manifest["videos"]:
        camera_id = cast(Literal["camera_a", "camera_b"], str(record["camera_id"]))
        source = source_by_camera[camera_id]
        if record["source_ref"] != source.source_ref:
            raise ValueError(f"proxy source reference differs: {camera_id}")
        if _sha256(PROJECT_ROOT / source.source_ref) != source.source_sha256:
            raise ValueError(f"original source hash differs: {camera_id}")
        proxy_path = PROJECT_ROOT / record["proxy_ref"]
        if _sha256(proxy_path) != record["proxy_sha256"]:
            raise ValueError(f"proxy hash differs: {camera_id}")
        probe = _probe(proxy_path)
        for field in (
            "codec",
            "pixel_format",
            "width",
            "height",
            "decoded_frame_count",
            "keyframe_count",
            "maximum_keyframe_gap_frames",
        ):
            if probe[field] != record[field]:
                raise ValueError(f"proxy probe differs for {camera_id}: {field}")
        if probe["decoded_frame_count"] != source.decoded_frame_count:
            raise ValueError(f"proxy frame count differs from source: {camera_id}")
        if probe["maximum_keyframe_gap_frames"] > 30:
            raise ValueError(f"proxy keyframe gap exceeds policy: {camera_id}")
        verified[camera_id] = probe

    if tuple(verified) != ("camera_a", "camera_b"):
        raise ValueError("proxy verification requires Camera A then Camera B")
    report = {
        "schema_version": 1,
        "stage": "S07",
        "work_package": 2,
        "status": "passed",
        "purpose": "seekable_rerun_presentation_video_proxy_verification",
        "source_manifest_ref": _relative(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_final_run_manifest_id": final_manifest.manifest_id,
        "video_count": len(verified),
        "videos": verified,
        "raw_sources_modified": False,
        "spatial_outputs_modified": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _probe(path: Path) -> dict[str, Any]:
    metadata_result = subprocess.run(  # noqa: S603 - fixed ffprobe argv
        (
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,nb_read_frames",
            "-of",
            "json",
            str(path),
        ),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60.0,
    )
    stream = json.loads(metadata_result.stdout)["streams"][0]
    keyframe_result = subprocess.run(  # noqa: S603 - fixed ffprobe argv
        (
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=key_frame",
            "-of",
            "csv=p=0",
            str(path),
        ),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60.0,
    )
    keyframes = [
        index
        for index, value in enumerate(keyframe_result.stdout.splitlines())
        if value.strip().split(",", maxsplit=1)[0] == "1"
    ]
    gaps = [
        following - current
        for current, following in zip(keyframes, keyframes[1:], strict=False)
    ]
    return {
        "codec": str(stream["codec_name"]),
        "pixel_format": str(stream["pix_fmt"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "decoded_frame_count": int(stream["nb_read_frames"]),
        "keyframe_count": len(keyframes),
        "maximum_keyframe_gap_frames": max(gaps, default=0),
    }


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
