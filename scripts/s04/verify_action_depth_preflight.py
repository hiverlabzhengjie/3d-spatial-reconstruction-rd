"""Verify retained S04 raw action-depth artifacts and policy boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from spatial_reconstruction.localization import ActionDepthRunSummary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/s04/action_depth_preflight_20260801/summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/s04/action_depth_preflight_20260801/verification.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    summary_path = _resolve(project_root, args.summary)
    output_path = _resolve(project_root, args.output)
    if output_path.exists():
        raise FileExistsError(f"verification output already exists: {output_path}")
    summary = ActionDepthRunSummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    _verify_input_provenance(project_root, summary.input_provenance)

    predictions: list[dict[str, Any]] = []
    for record in summary.predictions:
        raw_path = _resolve(project_root, Path(record.raw_prediction_ref))
        preview_path = _resolve(
            project_root, Path(record.depth_confidence_preview_ref)
        )
        _require_hash(raw_path, record.raw_prediction_sha256)
        _require_hash(preview_path, record.depth_confidence_preview_sha256)
        with np.load(raw_path, allow_pickle=False) as arrays:
            expected_shape = (2, 280, 504)
            depth = np.asarray(arrays["depth_m"])
            confidence = np.asarray(arrays["confidence"])
            if depth.shape != expected_shape or confidence.shape != expected_shape:
                raise ValueError("retained action depth/confidence shape changed")
            if str(arrays["job_id"].item()) != record.job.job_id:
                raise ValueError("raw action-depth job ID differs from summary")
            if str(arrays["bundle_id"].item()) != record.job.bundle.bundle_id:
                raise ValueError("raw action-depth bundle ID differs from summary")
            expected_frame_ids = [
                frame.frame_id for frame in record.job.bundle.frames
            ]
            if arrays["frame_ids"].tolist() != expected_frame_ids:
                raise ValueError("raw action-depth frame IDs differ from job")
            if bool(arrays["s02_corrections_applied"].item()):
                raise ValueError("raw S04 depth unexpectedly records an S02 correction")
            if float(arrays["depth_scale_applied"].item()) != 1.0:
                raise ValueError("raw S04 action depth has a non-unit scale")
            if not bool(arrays["is_metric"].item()):
                raise ValueError("retained S04 action depth is not metric")
            finite_positive_counts = [
                int(np.count_nonzero(np.isfinite(view) & (view > 0)))
                for view in depth
            ]
            finite_confidence_counts = [
                int(np.count_nonzero(np.isfinite(view))) for view in confidence
            ]
        for camera_index, camera_id in enumerate(("camera_a", "camera_b")):
            camera = record.cameras[camera_id]
            if camera["finite_positive_depth_count"] != finite_positive_counts[camera_index]:
                raise ValueError("summary finite-positive depth count changed")
            if camera["finite_confidence_count"] != finite_confidence_counts[camera_index]:
                raise ValueError("summary finite-confidence count changed")
            keyframe = _resolve(
                project_root, Path(str(camera["undistorted_keyframe_ref"]))
            )
            _require_hash(keyframe, str(camera["undistorted_keyframe_sha256"]))
        predictions.append(
            {
                "job_id": record.job.job_id,
                "bundle_id": record.job.bundle.bundle_id,
                "phase_id": record.job.phase_id,
                "source_frame_index": (
                    record.job.bundle.frames[0].source_frame_index
                ),
                "raw_prediction_sha256": record.raw_prediction_sha256,
                "finite_positive_depth_count_by_camera": dict(
                    zip(
                        ("camera_a", "camera_b"),
                        finite_positive_counts,
                        strict=True,
                    )
                ),
                "finite_confidence_count_by_camera": dict(
                    zip(
                        ("camera_a", "camera_b"),
                        finite_confidence_counts,
                        strict=True,
                    )
                ),
            }
        )

    verification = {
        "schema_version": 1,
        "stage": "S04",
        "status": "passed",
        "purpose": "raw_action_depth_artifact_and_policy_verification",
        "source_summary_ref": _relative(summary_path, project_root),
        "source_summary_sha256": _sha256(summary_path),
        "prediction_count": len(predictions),
        "schema_round_trip_passed": (
            ActionDepthRunSummary.model_validate_json(summary.model_dump_json())
            == summary
        ),
        "capture_order_passed": [
            item["source_frame_index"] for item in predictions
        ]
        == sorted(item["source_frame_index"] for item in predictions),
        "all_depth_finite_positive": all(
            count == 280 * 504
            for item in predictions
            for count in item["finite_positive_depth_count_by_camera"].values()
        ),
        "all_confidence_finite": all(
            count == 280 * 504
            for item in predictions
            for count in item["finite_confidence_count_by_camera"].values()
        ),
        "s02_corrections_absent": True,
        "mask_localization_not_yet_performed": True,
        "predictions": predictions,
    }
    if not all(
        (
            verification["schema_round_trip_passed"],
            verification["capture_order_passed"],
            verification["all_depth_finite_positive"],
            verification["all_confidence_finite"],
        )
    ):
        raise RuntimeError("S04 raw action-depth verification did not pass")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))
    return 0


def _verify_input_provenance(
    project_root: Path, provenance: dict[str, Any]
) -> None:
    for key, value in provenance.items():
        if not key.endswith("_ref") or key == "selection_ref":
            continue
        stem = key.removesuffix("_ref")
        expected = provenance.get(stem + "_sha256")
        if expected is None:
            raise ValueError(f"missing input hash for {key}")
        _require_hash(_resolve(project_root, Path(str(value))), str(expected))
    selection_ref = _resolve(project_root, Path(str(provenance["selection_ref"])))
    _require_hash(selection_ref, str(provenance["selection_sha256"]))


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


def _resolve(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _relative(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
