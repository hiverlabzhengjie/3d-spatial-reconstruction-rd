"""Independently verify retained S05 Qwen execution evidence and boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from spatial_reconstruction.interaction import (
    QwenEventJobPlanRunSummary,
    QwenEventResult,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    summary_path = _resolve(root, args.summary)
    output_path = _resolve(root, args.output)
    if output_path.exists():
        raise FileExistsError(f"verification output already exists: {output_path}")
    summary = _read_object(summary_path)
    if (
        summary.get("stage") != "S05"
        or summary.get("status") != "completed_pending_verification"
        or summary.get("device") != "mps"
        or summary.get("precision") != "float16"
        or summary.get("model_load_count") != 1
        or summary.get("spatial_state_mutated") is not False
    ):
        raise ValueError("Qwen execution summary violates the S05 execution boundary")

    plan_path = _resolve(root, Path(str(summary["source_plan_summary_ref"])))
    _require_hash(plan_path, str(summary["source_plan_summary_sha256"]))
    plan = QwenEventJobPlanRunSummary.model_validate_json(
        plan_path.read_text(encoding="utf-8")
    )
    if (
        summary.get("model_id") != plan.policy.model_id
        or summary.get("model_revision") != plan.policy.model_revision
    ):
        raise ValueError("executed Qwen model identity differs from the verified plan")

    named_artifacts = {
        "frames": ("frame_manifest_ref", "frame_manifest_sha256"),
        "attempts": ("attempt_results_ref", "attempt_results_sha256"),
        "final": ("final_results_ref", "final_results_sha256"),
        "contact_sheet": ("contact_sheet_ref", "contact_sheet_sha256"),
    }
    paths: dict[str, Path] = {}
    for name, (ref_key, hash_key) in named_artifacts.items():
        path = _resolve(root, Path(str(summary[ref_key])))
        _require_hash(path, str(summary[hash_key]))
        paths[name] = path

    frame_values = _read_object(paths["frames"]).get("frames")
    if not isinstance(frame_values, list) or len(frame_values) != 18:
        raise ValueError("Qwen execution must retain 18 frame artifacts")
    expected_frames = {
        (job.job_id, item.sequence_index): item for job in plan.jobs for item in job.frame_inputs
    }
    for frame in frame_values:
        if not isinstance(frame, dict):
            raise ValueError("Qwen frame manifest contains a non-object")
        key = (str(frame["job_id"]), int(frame["sequence_index"]))
        expected = expected_frames.pop(key, None)
        if expected is None:
            raise ValueError("Qwen frame manifest contains an unplanned frame")
        if (
            frame.get("camera_id") != expected.camera_id
            or frame.get("sample_role") != expected.sample_role.value
            or frame.get("source_frame_index") != expected.source_frame_index
            or abs(
                float(frame["decoded_timestamp_seconds"])
                - expected.capture_timestamp_seconds
            )
            > 0.05
        ):
            raise ValueError("retained Qwen frame differs from the plan")
        image_path = _resolve(root, Path(str(frame["artifact_ref"])))
        _require_hash(image_path, str(frame["artifact_sha256"]))
        with Image.open(image_path) as image:
            if image.size != (int(frame["width"]), int(frame["height"])):
                raise ValueError("retained Qwen frame dimensions differ")
    if expected_frames:
        raise ValueError("Qwen frame manifest omits planned frames")

    attempt_values = _read_object(paths["attempts"]).get("attempts")
    final_values = _read_object(paths["final"]).get("results")
    if not isinstance(attempt_values, list) or not isinstance(final_values, list):
        raise ValueError("Qwen result artifacts must contain lists")
    attempts = tuple(QwenEventResult.model_validate(value) for value in attempt_values)
    finals = tuple(QwenEventResult.model_validate(value) for value in final_values)
    if len(finals) != 3 or tuple(result.job.event_kind for result in finals) != tuple(
        job.event_kind for job in plan.jobs
    ):
        raise ValueError("Qwen final results must cover pickup, carry, and place")
    if len({result.job.deduplication_key for result in finals}) != 3:
        raise ValueError("Qwen final results must cover three logical events")
    latest: dict[str, QwenEventResult] = {}
    for result in attempts:
        latest[result.job.deduplication_key] = result
    if tuple(latest[job.deduplication_key] for job in plan.jobs) != finals:
        raise ValueError("Qwen final results are not the latest retained attempts")
    if any(result.job.attempt > plan.policy.maximum_attempts for result in attempts):
        raise ValueError("Qwen execution exceeded the repair-attempt bound")
    if int(summary["attempt_count"]) != len(attempts):
        raise ValueError("Qwen execution attempt count differs")
    if plan.policy.policy_id == "s05_qwen_event_review_v4":
        expected_labels = ("pickup", "carry", "place")
        if tuple(result.interpretation.event_label.value for result in finals) != (
            expected_labels
        ):
            raise ValueError("Qwen v4 final labels differ from pickup-carry-place")
        if (
            len(attempts) != 3
            or any(result.outcome.value != "completed" for result in attempts)
            or any(not result.interpretation.matches_candidate for result in finals)
            or any(result.response_normalization.value != "none" for result in attempts)
        ):
            raise ValueError("Qwen v4 requires three direct unnormalized candidate matches")

    verification = {
        "schema_version": 1,
        "stage": "S05",
        "status": "passed",
        "purpose": "qwen_event_execution_verification",
        "source_summary_ref": _relative(summary_path, root),
        "source_summary_sha256": _sha256(summary_path),
        "model_identity_verified": True,
        "mps_float16_verified": True,
        "model_load_count": 1,
        "frame_artifact_count": len(frame_values),
        "attempt_count": len(attempts),
        "final_result_count": len(finals),
        "final_event_labels": {
            result.job.event_kind.value: result.interpretation.event_label.value
            for result in finals
        },
        "all_candidates_matched": all(
            result.interpretation.matches_candidate for result in finals
        ),
        "response_normalizations": [
            result.response_normalization.value for result in attempts
        ],
        "raw_tokens_retained": all(
            result.output_token_count is not None
            for result in attempts
            if result.outcome.value in {"completed", "invalid_output"}
        ),
        "spatial_write_interface_absent": True,
        "contact_sheet_decodable": True,
    }
    with Image.open(paths["contact_sheet"]) as image:
        image.verify()
    output_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))
    return 0


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _require_hash(path: Path, expected: str) -> None:
    if _sha256(path) != expected:
        raise ValueError(f"artifact hash differs: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
