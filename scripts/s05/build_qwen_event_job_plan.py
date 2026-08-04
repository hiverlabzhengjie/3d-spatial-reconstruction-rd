"""Build the verified S05 Qwen pickup/carry/place job plan without inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from spatial_reconstruction.interaction import (
    BoundedQwenEventQueue,
    QwenEventJob,
    QwenEventJobPlanRunSummary,
    QwenEventReviewPolicy,
    QwenQueueSubmission,
    QwenVideoSource,
    SemanticInteractionRunSummary,
    build_qwen_event_jobs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--semantic-summary",
        type=Path,
        default=Path("artifacts/s05/semantic_interaction_v2_20260803/summary.json"),
    )
    parser.add_argument(
        "--semantic-verification",
        type=Path,
        default=Path(
            "artifacts/s05/semantic_interaction_v2_20260803/verification.json"
        ),
    )
    parser.add_argument(
        "--synchronization-manifest",
        type=Path,
        default=Path(
            "artifacts/s01/action_take_01/synchronized/synchronization_manifest.json"
        ),
    )
    parser.add_argument(
        "--qwen-gate-summary",
        type=Path,
        default=Path("artifacts/s00/wp8/qwen_gate_20260728/summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    paths = {
        "semantic": _resolve(root, args.semantic_summary),
        "semantic_verification": _resolve(root, args.semantic_verification),
        "synchronization": _resolve(root, args.synchronization_manifest),
        "qwen_gate": _resolve(root, args.qwen_gate_summary),
    }
    output_dir = _resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    semantic = SemanticInteractionRunSummary.model_validate_json(
        paths["semantic"].read_text(encoding="utf-8")
    )
    semantic_verification = _read_object(paths["semantic_verification"])
    synchronization = _read_object(paths["synchronization"])
    qwen_gate = _read_object(paths["qwen_gate"])
    policy = QwenEventReviewPolicy()
    _verify_prerequisites(
        semantic_path=paths["semantic"],
        semantic_verification=semantic_verification,
        synchronization_path=paths["synchronization"],
        semantic=semantic,
        qwen_gate=qwen_gate,
        policy=policy,
    )
    video_sources = _load_video_sources(root=root, synchronization=synchronization)
    jobs = build_qwen_event_jobs(
        candidates=semantic.event_candidates,
        video_sources=video_sources,
        capture_session_id=str(synchronization["capture_session_id"]),
        synchronization_manifest_ref=_relative(paths["synchronization"], root),
        synchronization_manifest_sha256=_sha256(paths["synchronization"]),
        policy=policy,
        created_processing_seconds=0.0,
    )
    queue = BoundedQwenEventQueue(
        capacity=policy.queue_capacity,
        overflow_policy=policy.overflow_policy,
        maximum_attempts=policy.maximum_attempts,
    )
    submissions: list[QwenQueueSubmission] = [queue.submit(job) for job in jobs]
    submissions.append(queue.submit(jobs[0]))

    jobs_path = output_dir / "qwen_event_jobs.json"
    jobs_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "S05",
                "policy_id": policy.policy_id,
                "jobs": [job.model_dump(mode="json") for job in jobs],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    prompt_path = output_dir / "qwen_event_prompt_manifest.json"
    prompt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "S05",
                "policy_id": policy.policy_id,
                "prompts": [job.prompt.model_dump(mode="json") for job in jobs],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "qwen_event_job_review.csv"
    _write_review_csv(jobs, csv_path)

    summary = QwenEventJobPlanRunSummary(
        status="completed_pending_execution",
        created_at_utc=datetime.now(UTC),
        policy=policy,
        source_semantic_summary_ref=_relative(paths["semantic"], root),
        source_semantic_summary_sha256=_sha256(paths["semantic"]),
        source_semantic_verification_ref=_relative(
            paths["semantic_verification"], root
        ),
        source_semantic_verification_sha256=_sha256(
            paths["semantic_verification"]
        ),
        source_synchronization_manifest_ref=_relative(
            paths["synchronization"], root
        ),
        source_synchronization_manifest_sha256=_sha256(paths["synchronization"]),
        source_qwen_gate_summary_ref=_relative(paths["qwen_gate"], root),
        source_qwen_gate_summary_sha256=_sha256(paths["qwen_gate"]),
        video_sources=video_sources,
        jobs=jobs,
        queue_submissions=tuple(submissions),
        queue_diagnostics=queue.diagnostics,
        jobs_ref=_relative(jobs_path, root),
        jobs_sha256=_sha256(jobs_path),
        prompt_manifest_ref=_relative(prompt_path, root),
        prompt_manifest_sha256=_sha256(prompt_path),
        review_csv_ref=_relative(csv_path, root),
        review_csv_sha256=_sha256(csv_path),
        limitations=(
            "This artifact plans Qwen work but performs no model inference.",
            "Six sparse frames may not fully capture a brief or ambiguous action.",
            "Evidence strength is qualitative and is not a calibrated probability.",
            "Qwen results cannot change coordinates, identity, timestamps, zones, "
            "or spatial authority.",
        ),
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "job_count": len(jobs),
                "event_kinds": [job.event_kind.value for job in jobs],
                "frame_inputs_per_job": [len(job.frame_inputs) for job in jobs],
                "queue_capacity": queue.diagnostics.capacity,
                "accepted_count": queue.diagnostics.accepted_count,
                "duplicate_coalesced_count": (
                    queue.diagnostics.duplicate_coalesced_count
                ),
                "model_inference_performed": False,
            },
            indent=2,
        )
    )
    return 0


def _verify_prerequisites(
    *,
    semantic_path: Path,
    semantic_verification: dict[str, Any],
    synchronization_path: Path,
    semantic: SemanticInteractionRunSummary,
    qwen_gate: dict[str, Any],
    policy: QwenEventReviewPolicy,
) -> None:
    if (
        semantic_verification.get("status") != "passed"
        or semantic_verification.get("source_summary_sha256")
        != _sha256(semantic_path)
        or semantic_verification.get("candidate_kinds")
        != ["pickup", "carry", "place"]
    ):
        raise ValueError("Qwen plan requires matching passed S05 v2 verification")
    if semantic.source_synchronization_manifest_sha256 != _sha256(
        synchronization_path
    ):
        raise ValueError("Qwen plan synchronization manifest differs from S05 v2")
    if (
        qwen_gate.get("outcome") != "passed"
        or qwen_gate.get("model_id") != policy.model_id
        or qwen_gate.get("model_revision") != policy.model_revision
        or qwen_gate.get("device") != "mps"
        or qwen_gate.get("fallback") is not None
    ):
        raise ValueError("Qwen plan requires the exact passed S00 MPS model gate")


def _load_video_sources(
    *,
    root: Path,
    synchronization: dict[str, Any],
) -> tuple[QwenVideoSource, QwenVideoSource]:
    outputs = synchronization.get("derived_outputs")
    if not isinstance(outputs, dict):
        raise ValueError("synchronization manifest lacks derived outputs")
    sources: list[QwenVideoSource] = []
    for camera_id in ("camera_a", "camera_b"):
        value = outputs.get(camera_id)
        if not isinstance(value, dict):
            raise ValueError(f"synchronization manifest lacks {camera_id}")
        source_path = _resolve(root, Path(str(value["path"])))
        source_sha256 = str(value["sha256"])
        if _sha256(source_path) != source_sha256:
            raise ValueError(f"synchronized {camera_id} video hash differs")
        sources.append(
            QwenVideoSource(
                camera_id=camera_id,
                source_ref=_relative(source_path, root),
                source_sha256=source_sha256,
                decoded_frame_count=int(value["decoded_frame_count"]),
                duration_seconds=float(value["duration_seconds"]),
                nominal_frame_rate_fps=30.0,
            )
        )
    return cast(tuple[QwenVideoSource, QwenVideoSource], tuple(sources))


def _write_review_csv(jobs: tuple[QwenEventJob, ...], path: Path) -> None:
    fields = (
        "job_id",
        "deduplication_key",
        "event_kind",
        "source_frame_index",
        "capture_timestamp_seconds",
        "clip_start_timestamp_seconds",
        "clip_end_timestamp_seconds",
        "review_frame_index",
        "review_timestamp_seconds",
        "review_clip_start_timestamp_seconds",
        "review_clip_end_timestamp_seconds",
        "frame_count",
        "frame_sequence",
        "prompt_sha256",
        "model_id",
        "model_revision",
        "max_new_tokens",
        "timeout_seconds",
        "attempt",
        "priority",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "job_id": job.job_id,
                    "deduplication_key": job.deduplication_key,
                    "event_kind": job.event_kind.value,
                    "source_frame_index": job.source_frame_index,
                    "capture_timestamp_seconds": f"{job.capture_timestamp_seconds:.9f}",
                    "clip_start_timestamp_seconds": (
                        f"{job.clip_start_timestamp_seconds:.9f}"
                    ),
                    "clip_end_timestamp_seconds": (
                        f"{job.clip_end_timestamp_seconds:.9f}"
                    ),
                    "review_frame_index": job.review_frame_index,
                    "review_timestamp_seconds": job.review_timestamp_seconds,
                    "review_clip_start_timestamp_seconds": (
                        job.review_clip_start_timestamp_seconds
                    ),
                    "review_clip_end_timestamp_seconds": (
                        job.review_clip_end_timestamp_seconds
                    ),
                    "frame_count": len(job.frame_inputs),
                    "frame_sequence": ";".join(
                        f"{item.sequence_index}:{item.sample_role.value}:"
                        f"{item.camera_id}:{item.source_frame_index}"
                        for item in job.frame_inputs
                    ),
                    "prompt_sha256": job.prompt.prompt_sha256,
                    "model_id": job.model_id,
                    "model_revision": job.model_revision,
                    "max_new_tokens": job.max_new_tokens,
                    "timeout_seconds": job.timeout_seconds,
                    "attempt": job.attempt,
                    "priority": job.priority,
                }
            )


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


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
