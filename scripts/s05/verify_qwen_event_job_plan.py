"""Independently verify the S05 Qwen event job plan before model execution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from spatial_reconstruction.interaction import (
    BoundedQwenEventQueue,
    QwenEventJob,
    QwenEventJobPlanRunSummary,
    QwenVideoSource,
    SemanticInteractionRunSummary,
    build_qwen_event_jobs,
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
    summary = QwenEventJobPlanRunSummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    source_paths = {
        "semantic": _resolve(root, Path(summary.source_semantic_summary_ref)),
        "semantic_verification": _resolve(
            root, Path(summary.source_semantic_verification_ref)
        ),
        "synchronization": _resolve(
            root, Path(summary.source_synchronization_manifest_ref)
        ),
        "qwen_gate": _resolve(root, Path(summary.source_qwen_gate_summary_ref)),
    }
    source_hashes = {
        "semantic": summary.source_semantic_summary_sha256,
        "semantic_verification": summary.source_semantic_verification_sha256,
        "synchronization": summary.source_synchronization_manifest_sha256,
        "qwen_gate": summary.source_qwen_gate_summary_sha256,
    }
    for key, path in source_paths.items():
        _require_hash(path, source_hashes[key])
    semantic_verification = _read_object(source_paths["semantic_verification"])
    if (
        semantic_verification.get("status") != "passed"
        or semantic_verification.get("source_summary_sha256")
        != source_hashes["semantic"]
    ):
        raise ValueError("Qwen plan semantic verification differs")
    qwen_gate = _read_object(source_paths["qwen_gate"])
    if (
        qwen_gate.get("outcome") != "passed"
        or qwen_gate.get("model_id") != summary.policy.model_id
        or qwen_gate.get("model_revision") != summary.policy.model_revision
        or qwen_gate.get("device") != "mps"
        or qwen_gate.get("fallback") is not None
    ):
        raise ValueError("Qwen plan model gate differs")
    semantic = SemanticInteractionRunSummary.model_validate_json(
        source_paths["semantic"].read_text(encoding="utf-8")
    )
    synchronization = _read_object(source_paths["synchronization"])
    video_sources = _regenerate_video_sources(root=root, summary=summary)
    if video_sources != summary.video_sources:
        raise ValueError("Qwen video sources do not regenerate")
    regenerated_jobs = build_qwen_event_jobs(
        candidates=semantic.event_candidates,
        video_sources=video_sources,
        capture_session_id=str(synchronization["capture_session_id"]),
        synchronization_manifest_ref=summary.source_synchronization_manifest_ref,
        synchronization_manifest_sha256=source_hashes["synchronization"],
        policy=summary.policy,
        created_processing_seconds=0.0,
    )
    if regenerated_jobs != summary.jobs:
        raise ValueError("Qwen event jobs do not regenerate")
    queue = BoundedQwenEventQueue(
        capacity=summary.policy.queue_capacity,
        overflow_policy=summary.policy.overflow_policy,
        maximum_attempts=summary.policy.maximum_attempts,
    )
    submissions = tuple(queue.submit(job) for job in regenerated_jobs) + (
        queue.submit(regenerated_jobs[0]),
    )
    if submissions != summary.queue_submissions or queue.diagnostics != (
        summary.queue_diagnostics
    ):
        raise ValueError("Qwen queue plan does not regenerate")

    artifact_paths = {
        "jobs": _resolve(root, Path(summary.jobs_ref)),
        "prompts": _resolve(root, Path(summary.prompt_manifest_ref)),
        "csv": _resolve(root, Path(summary.review_csv_ref)),
    }
    artifact_hashes = {
        "jobs": summary.jobs_sha256,
        "prompts": summary.prompt_manifest_sha256,
        "csv": summary.review_csv_sha256,
    }
    for key, path in artifact_paths.items():
        _require_hash(path, artifact_hashes[key])
    job_values = _read_object(artifact_paths["jobs"]).get("jobs")
    if not isinstance(job_values, list):
        raise ValueError("persistent Qwen jobs are not a list")
    if tuple(QwenEventJob.model_validate(value) for value in job_values) != (
        regenerated_jobs
    ):
        raise ValueError("persistent Qwen jobs differ")
    prompt_values = _read_object(artifact_paths["prompts"]).get("prompts")
    if not isinstance(prompt_values, list) or prompt_values != [
        job.prompt.model_dump(mode="json") for job in regenerated_jobs
    ]:
        raise ValueError("persistent Qwen prompts differ")
    with artifact_paths["csv"].open(encoding="utf-8", newline="") as handle:
        if len(list(csv.DictReader(handle))) != 3:
            raise ValueError("Qwen review CSV coverage differs")
    forbidden_job_fields = {
        "world_xyz_m",
        "track_identity",
        "zone_membership",
        "spatial_authority",
    }
    if any(forbidden_job_fields.intersection(job.model_dump()) for job in regenerated_jobs):
        raise ValueError("Qwen job exposes a forbidden spatial-write field")
    transition_frames = {
        job.event_kind.value: job.source_frame_index for job in regenerated_jobs
    }
    if transition_frames != {"pickup": 462, "carry": 468, "place": 666}:
        raise ValueError("Qwen event plan boundaries differ from S05 v2")
    review_frames = {
        job.event_kind.value: (
            job.source_frame_index
            if job.review_frame_index is None
            else job.review_frame_index
        )
        for job in regenerated_jobs
    }
    if summary.policy.policy_id == "s05_qwen_event_review_v4" and review_frames != {
        "pickup": 462,
        "carry": 567,
        "place": 666,
    }:
        raise ValueError("Qwen v4 review centres differ from the verified evidence")

    verification = {
        "schema_version": 1,
        "stage": "S05",
        "status": "passed",
        "purpose": "qwen_event_job_plan_verification",
        "source_summary_ref": _relative(summary_path, root),
        "source_summary_sha256": _sha256(summary_path),
        "jobs_regenerated": True,
        "queue_plan_regenerated": True,
        "job_count": len(regenerated_jobs),
        "event_kinds": [job.event_kind.value for job in regenerated_jobs],
        "transition_frames": transition_frames,
        "review_frames": review_frames,
        "frame_inputs_per_job": [len(job.frame_inputs) for job in regenerated_jobs],
        "unique_job_id_count": len({job.job_id for job in regenerated_jobs}),
        "unique_deduplication_key_count": len(
            {job.deduplication_key for job in regenerated_jobs}
        ),
        "queue_capacity": queue.diagnostics.capacity,
        "accepted_count": queue.diagnostics.accepted_count,
        "duplicate_coalesced_count": queue.diagnostics.duplicate_coalesced_count,
        "throttled_count": queue.diagnostics.throttled_count,
        "dropped_count": queue.diagnostics.dropped_oldest_count,
        "result_count": 0,
        "model_inference_performed": False,
        "forbidden_spatial_job_field_count": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))
    return 0


def _regenerate_video_sources(
    *,
    root: Path,
    summary: QwenEventJobPlanRunSummary,
) -> tuple[QwenVideoSource, QwenVideoSource]:
    sources: list[QwenVideoSource] = []
    for source in summary.video_sources:
        path = _resolve(root, Path(source.source_ref))
        _require_hash(path, source.source_sha256)
        sources.append(source)
    return cast(tuple[QwenVideoSource, QwenVideoSource], tuple(sources))


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


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
    return str(path.relative_to(root))


if __name__ == "__main__":
    raise SystemExit(main())
