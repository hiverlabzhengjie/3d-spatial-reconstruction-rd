"""Real D028 YOLO/ByteTrack processor for the bounded perception worker."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spatial_reconstruction.models import (
    YOLOSegAdapter,
    normalize_yolo_result,
    select_perception_candidates,
)
from spatial_reconstruction.perception.worker import (
    PerceptionProcessingOutput,
    PerceptionWorkItem,
)
from spatial_reconstruction.runtime import DeviceName


class YOLOByteTrackProcessor:
    """Persist raw tracking outputs and return D028 perception candidates."""

    def __init__(
        self,
        *,
        adapter: YOLOSegAdapter,
        project_root: Path,
        output_dir: Path,
        device: DeviceName,
        image_size: int,
        confidence_threshold: float,
        tracked_class_ids: tuple[int, ...],
        bag_class_aliases: tuple[str, ...],
        excluded_bag_classes: tuple[str, ...],
        policy_id: str,
    ) -> None:
        self._adapter = adapter
        self._project_root = project_root.resolve()
        self._output_dir = output_dir.resolve()
        self._output_dir.mkdir(parents=True, exist_ok=False)
        self._device = device
        self._image_size = image_size
        self._confidence_threshold = confidence_threshold
        self._tracked_class_ids = tracked_class_ids
        self._bag_class_aliases = bag_class_aliases
        self._excluded_bag_classes = excluded_bag_classes
        self._policy_id = policy_id

    def process(self, item: PerceptionWorkItem) -> PerceptionProcessingOutput:
        """Track one frame and atomically identify its persistent artifact names."""

        job = item.job
        if job.model_revision != self._adapter.weight_sha256:
            raise ValueError("perception job model revision differs from loaded checkpoint")
        if job.policy_id != self._policy_id:
            raise ValueError("perception job policy differs from processor policy")
        stem = (
            f"frame_{job.frame_identity.source_frame_index:06d}_"
            f"{job.job_id[:12]}"
        )
        mask_path = self._output_dir / f"{stem}_raw.npz"
        detection_path = self._output_dir / f"{stem}_detections.json"
        if mask_path.exists() or detection_path.exists():
            raise FileExistsError(f"perception frame artifacts already exist: {stem}")
        mask_ref = str(mask_path.relative_to(self._project_root))
        vendor_result = self._adapter.track(
            image_rgb=item.image_rgb,
            frame=job.frame_identity.as_frame_ref(),
            device=self._device,
            image_size=self._image_size,
            confidence_threshold=self._confidence_threshold,
            class_ids=self._tracked_class_ids,
        )
        normalized = normalize_yolo_result(
            vendor_result,
            frame=job.frame_identity.as_frame_ref(),
            mask_artifact_ref=mask_ref,
            require_track_ids=False,
        )
        candidates = select_perception_candidates(
            normalized,
            bag_class_aliases=self._bag_class_aliases,
            excluded_bag_classes=self._excluded_bag_classes,
            policy_id=self._policy_id,
        )
        np.savez_compressed(
            mask_path,
            source_sized_masks=normalized.masks,
            raw_masks=normalized.raw_masks,
            raw_boxes_xyxy=normalized.raw_boxes_xyxy,
            raw_class_ids=normalized.raw_class_ids,
            raw_confidence=normalized.raw_confidence,
            raw_track_ids=normalized.raw_track_ids,
        )
        detection_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "job": job.model_dump(mode="json"),
                    "detections": [
                        detection.model_dump(mode="json")
                        for detection in normalized.detections
                    ],
                    "candidates": [
                        candidate.model_dump(mode="json") for candidate in candidates
                    ],
                    "native_speed_ms": normalized.speed_ms,
                    "raw_mask_artifact": mask_ref,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return PerceptionProcessingOutput(
            candidates=candidates,
            raw_artifact_refs=(
                mask_ref,
                str(detection_path.relative_to(self._project_root)),
            ),
        )
