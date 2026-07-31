"""Deterministic capture-time frame bundling and replay ordering."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import TypeVar

from spatial_reconstruction.contracts import FrameIdentity, SynchronizedFrameBundle

T = TypeVar("T")


def build_synchronized_bundles(
    frame_streams: Mapping[str, Iterable[FrameIdentity]],
    *,
    expected_camera_ids: tuple[str, ...],
    reference_camera_id: str,
    pairing_tolerance_seconds: float,
) -> Iterator[SynchronizedFrameBundle]:
    """Greedily bundle the earliest unconsumed frames in deterministic order."""

    if not expected_camera_ids:
        raise ValueError("expected_camera_ids must not be empty")
    if len(set(expected_camera_ids)) != len(expected_camera_ids):
        raise ValueError("expected_camera_ids must be unique")
    if reference_camera_id not in expected_camera_ids:
        raise ValueError("reference_camera_id must be expected")
    if pairing_tolerance_seconds <= 0:
        raise ValueError("pairing_tolerance_seconds must be positive")
    unexpected = set(frame_streams) - set(expected_camera_ids)
    if unexpected:
        raise ValueError(f"unexpected frame streams: {sorted(unexpected)}")
    if not frame_streams:
        raise ValueError("at least one camera stream is required")

    iterators = {
        camera_id: iter(frame_streams.get(camera_id, ()))
        for camera_id in expected_camera_ids
    }
    last_frames: dict[str, FrameIdentity] = {}
    seen_frame_ids: set[str] = set()

    def next_valid(camera_id: str) -> FrameIdentity | None:
        try:
            frame = next(iterators[camera_id])
        except StopIteration:
            return None
        if frame.camera_id != camera_id:
            raise ValueError(
                f"frame stream {camera_id} yielded camera {frame.camera_id}"
            )
        if frame.frame_id in seen_frame_ids:
            raise ValueError(f"duplicate frame_id detected: {frame.frame_id}")
        previous = last_frames.get(camera_id)
        if previous is not None:
            if frame.source_frame_index <= previous.source_frame_index:
                raise ValueError(
                    f"duplicate or non-increasing frame index in {camera_id}"
                )
            if frame.capture_timestamp_seconds <= previous.capture_timestamp_seconds:
                raise ValueError(
                    f"duplicate or non-increasing capture timestamp in {camera_id}"
                )
        seen_frame_ids.add(frame.frame_id)
        last_frames[camera_id] = frame
        return frame

    heads = {
        camera_id: next_valid(camera_id)
        for camera_id in expected_camera_ids
    }
    bundle_index = 0
    common_session_id: str | None = None
    common_manifest_ref: str | None = None
    common_manifest_sha256: str | None = None

    while any(frame is not None for frame in heads.values()):
        available_heads = [frame for frame in heads.values() if frame is not None]
        anchor_timestamp = min(
            frame.capture_timestamp_seconds for frame in available_heads
        )
        selected: list[FrameIdentity] = []
        for camera_id in expected_camera_ids:
            frame = heads[camera_id]
            if (
                frame is not None
                and frame.capture_timestamp_seconds - anchor_timestamp
                <= pairing_tolerance_seconds
            ):
                selected.append(frame)
                heads[camera_id] = next_valid(camera_id)

        if not selected:
            raise RuntimeError("synchronizer made no progress")
        first = selected[0]
        if common_session_id is None:
            common_session_id = first.capture_session_id
            common_manifest_ref = first.synchronization_manifest_ref
            common_manifest_sha256 = first.synchronization_manifest_sha256
        for frame in selected:
            if frame.capture_session_id != common_session_id:
                raise ValueError("all streams must share capture_session_id")
            if (
                frame.synchronization_manifest_ref != common_manifest_ref
                or frame.synchronization_manifest_sha256
                != common_manifest_sha256
            ):
                raise ValueError("all streams must share synchronization provenance")

        reference_frame = next(
            (
                frame
                for frame in selected
                if frame.camera_id == reference_camera_id
            ),
            None,
        )
        capture_timestamp = (
            reference_frame.capture_timestamp_seconds
            if reference_frame is not None
            else anchor_timestamp
        )
        yield SynchronizedFrameBundle.create(
            bundle_index=bundle_index,
            capture_session_id=first.capture_session_id,
            capture_timestamp_seconds=capture_timestamp,
            reference_camera_id=reference_camera_id,
            expected_camera_ids=expected_camera_ids,
            frames=tuple(selected),
            pairing_tolerance_seconds=pairing_tolerance_seconds,
            synchronization_manifest_ref=first.synchronization_manifest_ref,
            synchronization_manifest_sha256=first.synchronization_manifest_sha256,
        )
        bundle_index += 1


def restore_capture_order(
    completed_items: Iterable[T],
    *,
    bundle_id_of: Callable[[T], str],
    bundles: Sequence[SynchronizedFrameBundle],
) -> tuple[T, ...]:
    """Restore worker results to authoritative bundle capture order."""

    order = {bundle.bundle_id: bundle.bundle_index for bundle in bundles}
    if len(order) != len(bundles):
        raise ValueError("bundle sequence contains duplicate bundle IDs")
    by_bundle_id: dict[str, T] = {}
    for item in completed_items:
        bundle_id = bundle_id_of(item)
        if bundle_id not in order:
            raise ValueError(f"worker result references unknown bundle_id: {bundle_id}")
        if bundle_id in by_bundle_id:
            raise ValueError(f"duplicate worker result for bundle_id: {bundle_id}")
        by_bundle_id[bundle_id] = item
    return tuple(
        sorted(
            by_bundle_id.values(),
            key=lambda item: order[bundle_id_of(item)],
        )
    )
