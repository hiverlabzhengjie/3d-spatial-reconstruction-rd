"""Verify D025 S04 action-pair scaling artifacts and immutable raw inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from spatial_reconstruction.localization import (
    ActionDepthRunSummary,
    ActionMarkerScaleObservation,
    ActionPairScalePolicy,
    estimate_action_pair_scale,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-qa-passed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    summary_path = _resolve(root, args.summary)
    output_path = _resolve(root, args.output)
    if output_path.exists():
        raise FileExistsError(f"verification output already exists: {output_path}")
    summary = cast(
        dict[str, Any], json.loads(summary_path.read_text(encoding="utf-8"))
    )
    if summary.get("stage") != "S04" or summary.get("status") != "passed":
        raise ValueError("action-depth scale summary is not an accepted S04 run")
    if not args.visual_qa_passed:
        raise ValueError("explicit marker diagnostic visual QA is required")
    policy = ActionPairScalePolicy.model_validate(summary["policy"])
    raw_summary_path = _resolve(root, Path(summary["raw_action_depth_summary_ref"]))
    _require_hash(raw_summary_path, summary["raw_action_depth_summary_sha256"])
    raw_summary = ActionDepthRunSummary.model_validate_json(
        raw_summary_path.read_text(encoding="utf-8")
    )
    raw_by_job = {item.job.job_id: item for item in raw_summary.predictions}
    verified_jobs: list[dict[str, Any]] = []
    source_indices: list[int] = []
    for record in cast(list[dict[str, Any]], summary["jobs"]):
        raw_record = raw_by_job.get(str(record["job_id"]))
        if raw_record is None:
            raise ValueError("scaled job is absent from retained raw action depth")
        raw_path = _resolve(root, Path(raw_record.raw_prediction_ref))
        _require_hash(raw_path, raw_record.raw_prediction_sha256)
        corrected_path = _resolve(root, Path(record["corrected_prediction_ref"]))
        diagnostic_path = _resolve(root, Path(record["diagnostic_ref"]))
        _require_hash(corrected_path, record["corrected_prediction_sha256"])
        _require_hash(diagnostic_path, record["diagnostic_sha256"])
        estimate_record = cast(dict[str, Any], record["scale_estimate"])
        observations = tuple(
            ActionMarkerScaleObservation.model_validate(item)
            for item in cast(list[dict[str, Any]], estimate_record["observations"])
        )
        recomputed = estimate_action_pair_scale(observations, policy=policy)
        if recomputed.model_dump(mode="json") != estimate_record:
            raise ValueError("stored pair scale differs from independently applied gate")
        with np.load(raw_path, allow_pickle=False) as raw_arrays:
            raw_depth = np.asarray(raw_arrays["depth_m"], dtype=np.float32)
            confidence = np.asarray(raw_arrays["confidence"])
        with np.load(corrected_path, allow_pickle=False) as corrected_arrays:
            corrected = np.asarray(corrected_arrays["corrected_depth_m"])
            if str(corrected_arrays["job_id"].item()) != raw_record.job.job_id:
                raise ValueError("corrected depth has mismatched job identity")
            corrected_source_hash = str(
                corrected_arrays["raw_prediction_sha256"].item()
            )
            if corrected_source_hash != raw_record.raw_prediction_sha256:
                raise ValueError("corrected depth has mismatched raw source hash")
            if not bool(corrected_arrays["raw_da3_depth_preserved"].item()):
                raise ValueError("corrected artifact does not declare raw preservation")
            if not bool(corrected_arrays["confidence_unchanged"].item()):
                raise ValueError("corrected artifact does not preserve confidence")
            if str(corrected_arrays["raw_confidence_sha256"].item()) != hashlib.sha256(
                confidence.tobytes()
            ).hexdigest():
                raise ValueError("raw confidence evidence changed")
        expected = raw_depth * np.float32(recomputed.scale)
        if corrected.dtype != np.float32 or not np.array_equal(corrected, expected):
            raise ValueError("corrected depth is not exact float32 raw-times-shared-scale")
        source_index = raw_record.job.bundle.frames[0].source_frame_index
        if source_index != int(record["source_frame_index"]):
            raise ValueError("corrected depth source frame identity changed")
        source_indices.append(source_index)
        verified_jobs.append(
            {
                "job_id": raw_record.job.job_id,
                "source_frame_index": source_index,
                "scale": recomputed.scale,
                "marker_count_by_camera": recomputed.marker_count_by_camera,
                "maximum_relative_deviation": recomputed.maximum_relative_deviation,
                "raw_prediction_sha256": raw_record.raw_prediction_sha256,
                "corrected_prediction_sha256": record["corrected_prediction_sha256"],
            }
        )
    if len(verified_jobs) != len(raw_summary.predictions):
        raise ValueError("not every selected raw action pair has corrected evidence")
    if source_indices != sorted(source_indices):
        raise ValueError("corrected depth jobs are not in capture order")
    for ref_key, hash_key in (
        ("marker_observations_csv_ref", "marker_observations_csv_sha256"),
        ("contact_sheet_ref", "contact_sheet_sha256"),
        ("pose_calibration_ref", "pose_calibration_sha256"),
    ):
        _require_hash(_resolve(root, Path(summary[ref_key])), summary[hash_key])
    verification = {
        "schema_version": 1,
        "stage": "S04",
        "status": "passed",
        "purpose": "d025_action_pair_marker_scaling_verification",
        "source_summary_ref": _relative(summary_path, root),
        "source_summary_sha256": _sha256(summary_path),
        "visual_qa_passed": True,
        "all_raw_prediction_hashes_unchanged": True,
        "all_confidence_arrays_unchanged": True,
        "all_pairs_use_one_shared_scale": True,
        "camera_specific_fallback_absent": True,
        "capture_order_passed": True,
        "job_count": len(verified_jobs),
        "jobs": verified_jobs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))
    return 0


def _require_hash(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"artifact hash changed for {path}: {actual} != {expected}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
