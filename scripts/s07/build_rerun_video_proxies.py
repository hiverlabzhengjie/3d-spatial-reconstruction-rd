"""Build seekable H.264 presentation proxies for stable dual-camera Rerun playback."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from spatial_reconstruction.finalization import Stage07FinalRunManifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_ID = "s07_rerun_seekable_h264_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-run-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = (PROJECT_ROOT / args.final_run_summary).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    summary = _load_json(summary_path)
    manifest_path = PROJECT_ROOT / str(summary["manifest_ref"])
    if _sha256(manifest_path) != summary["manifest_sha256"]:
        raise ValueError("final-run manifest hash differs from summary")
    manifest = Stage07FinalRunManifest.model_validate(_load_json(manifest_path))

    videos: list[dict[str, Any]] = []
    for source in manifest.source_videos:
        source_path = PROJECT_ROOT / source.source_ref
        if _sha256(source_path) != source.source_sha256:
            raise ValueError(f"source video hash differs: {source.camera_id}")
        proxy_path = output_dir / f"{source.camera_id}_rerun_playback.mp4"
        command = (
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            "fps=30",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "30",
            "-keyint_min",
            "30",
            "-sc_threshold",
            "0",
            "-movflags",
            "+faststart",
            str(proxy_path),
        )
        subprocess.run(  # noqa: S603 - fixed FFmpeg argv and validated source path
            command,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=300.0,
        )
        probe = _probe(proxy_path)
        if probe["decoded_frame_count"] != source.decoded_frame_count:
            raise ValueError(f"proxy frame count differs: {source.camera_id}")
        videos.append(
            {
                "camera_id": source.camera_id,
                "source_ref": source.source_ref,
                "source_sha256": source.source_sha256,
                "proxy_ref": _relative(proxy_path),
                "proxy_sha256": _sha256(proxy_path),
                **probe,
            }
        )

    proxy_manifest = {
        "schema_version": 1,
        "stage": "S07",
        "work_package": 2,
        "status": "completed",
        "purpose": "seekable_rerun_presentation_video_proxies",
        "policy_id": POLICY_ID,
        "source_final_run_manifest_id": manifest.manifest_id,
        "source_final_run_summary_ref": _relative(summary_path),
        "source_final_run_summary_sha256": _sha256(summary_path),
        "codec": "h264",
        "pixel_format": "yuv420p",
        "frame_rate_fps": 30.0,
        "maximum_keyframe_interval_frames": 30,
        "audio_removed": True,
        "raw_sources_modified": False,
        "videos": videos,
        "limitations": [
            "Proxies are presentation-only and do not replace synchronized source identity.",
            "Spatial overlays, coordinates, calibration, and capture timestamps are unchanged.",
        ],
    }
    _write_json(output_dir / "proxy_manifest.json", proxy_manifest)
    print(json.dumps(proxy_manifest, indent=2, sort_keys=True))
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
            "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames:format=duration",
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
    metadata = json.loads(metadata_result.stdout)
    stream = metadata["streams"][0]
    numerator, denominator = str(stream["avg_frame_rate"]).split("/", maxsplit=1)
    frame_rate = float(numerator) / float(denominator)
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
        "duration_seconds": float(metadata["format"]["duration"]),
        "frame_rate_fps": frame_rate,
        "keyframe_count": len(keyframes),
        "maximum_keyframe_gap_frames": max(gaps, default=0),
    }


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
