from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from spatial_reconstruction.contracts import (
    FrameBundleStatus,
    FrameIdentity,
    FrameSourceKind,
    SourceFingerprintKind,
    SynchronizedFrameBundle,
)
from spatial_reconstruction.ingestion import (
    DecodedFrame,
    FileFrameSource,
    RTSPAttemptOutcome,
    RTSPFrameSource,
    RTSPReconnectPolicy,
    TimestampTransform,
    build_synchronized_bundles,
    read_rtsp_with_reconnect,
    restore_capture_order,
    sanitize_rtsp_ref,
)

SOURCE_HASH = "a" * 64
MANIFEST_HASH = "b" * 64
MANIFEST_REF = "artifacts/s01/synchronized/synchronization_manifest.json"
EXPECTED_CAMERAS = ("camera_a", "camera_b")


def make_identity(
    camera_id: str,
    frame_index: int,
    capture_timestamp_seconds: float,
    *,
    manifest_hash: str = MANIFEST_HASH,
) -> FrameIdentity:
    return FrameIdentity.create(
        capture_session_id="session_01",
        camera_id=camera_id,
        source_kind=FrameSourceKind.FILE,
        source_frame_index=frame_index,
        source_timestamp_seconds=capture_timestamp_seconds,
        capture_timestamp_seconds=capture_timestamp_seconds,
        source_ref=f"derived/{camera_id}.mp4",
        source_fingerprint=SOURCE_HASH,
        source_fingerprint_kind=SourceFingerprintKind.CONTENT_SHA256,
        synchronization_manifest_ref=MANIFEST_REF,
        synchronization_manifest_sha256=manifest_hash,
        pose_version_id=f"session_01:{camera_id}:pose:v1",
        image_width=4,
        image_height=3,
    )


def build_complete_fixture() -> tuple[SynchronizedFrameBundle, ...]:
    streams = {
        "camera_a": [
            make_identity("camera_a", index, timestamp)
            for index, timestamp in enumerate((0.0, 0.033, 0.066))
        ],
        "camera_b": [
            make_identity("camera_b", index, timestamp)
            for index, timestamp in enumerate((0.002, 0.035, 0.068))
        ],
    }
    return tuple(
        build_synchronized_bundles(
            streams,
            expected_camera_ids=EXPECTED_CAMERAS,
            reference_camera_id="camera_a",
            pairing_tolerance_seconds=0.01,
        )
    )


def test_frame_identity_is_stable_serializable_and_tamper_evident() -> None:
    first = make_identity("camera_a", 3, 0.1)
    replay = make_identity("camera_a", 3, 0.1)

    assert first.frame_id == replay.frame_id
    assert FrameIdentity.model_validate_json(first.model_dump_json()) == first
    assert first.as_frame_ref().timestamp_seconds == 0.1

    tampered = first.model_dump()
    tampered["capture_timestamp_seconds"] = 0.2
    with pytest.raises(ValidationError, match="frame_id does not match"):
        FrameIdentity.model_validate(tampered)

    with pytest.raises(ValidationError):
        first.source_frame_index = 4


def test_decoded_frame_copies_pixels_and_makes_them_read_only() -> None:
    source = np.zeros((3, 4, 3), dtype=np.uint8)
    decoded = DecodedFrame(
        identity=make_identity("camera_a", 0, 0.0),
        image_bgr=source,
    )

    source[0, 0] = 255
    assert np.all(decoded.image_bgr[0, 0] == 0)
    with pytest.raises(ValueError):
        decoded.image_bgr[0, 0] = 1


def test_timestamp_transform_maps_source_pts_and_rejects_invalid_results() -> None:
    transform = TimestampTransform(scale=0.999, offset_seconds=1.0)
    assert transform.apply(2.0) == pytest.approx(2.998)

    with pytest.raises(ValueError, match="positive"):
        TimestampTransform(scale=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        TimestampTransform(offset_seconds=-1.0).apply(0.5)


def test_deterministic_replay_produces_same_bundle_ids_and_order() -> None:
    first = build_complete_fixture()
    second = build_complete_fixture()

    assert [bundle.bundle_id for bundle in first] == [bundle.bundle_id for bundle in second]
    assert [bundle.capture_timestamp_seconds for bundle in first] == [
        0.0,
        0.033,
        0.066,
    ]
    assert all(bundle.status is FrameBundleStatus.COMPLETE for bundle in first)
    assert all(bundle.missing_camera_ids == () for bundle in first)
    assert SynchronizedFrameBundle.model_validate_json(first[0].model_dump_json()) == first[0]


def test_missing_camera_is_explicit_and_does_not_fabricate_a_frame() -> None:
    streams = {
        "camera_a": [
            make_identity("camera_a", index, timestamp)
            for index, timestamp in enumerate((0.0, 0.033, 0.066))
        ],
        "camera_b": [
            make_identity("camera_b", index, timestamp)
            for index, timestamp in enumerate((0.002, 0.068))
        ],
    }

    bundles = tuple(
        build_synchronized_bundles(
            streams,
            expected_camera_ids=EXPECTED_CAMERAS,
            reference_camera_id="camera_a",
            pairing_tolerance_seconds=0.01,
        )
    )

    assert len(bundles) == 3
    assert bundles[1].status is FrameBundleStatus.INCOMPLETE
    assert bundles[1].missing_camera_ids == ("camera_b",)
    assert tuple(frame.camera_id for frame in bundles[1].frames) == ("camera_a",)


def test_duplicate_or_non_monotonic_frames_are_rejected() -> None:
    duplicate = make_identity("camera_a", 0, 0.0)
    with pytest.raises(ValueError, match="duplicate frame_id"):
        tuple(
            build_synchronized_bundles(
                {"camera_a": [duplicate, duplicate]},
                expected_camera_ids=EXPECTED_CAMERAS,
                reference_camera_id="camera_a",
                pairing_tolerance_seconds=0.01,
            )
        )

    non_monotonic = [
        make_identity("camera_a", 0, 0.1),
        make_identity("camera_a", 1, 0.05),
    ]
    with pytest.raises(ValueError, match="non-increasing capture timestamp"):
        tuple(
            build_synchronized_bundles(
                {"camera_a": non_monotonic},
                expected_camera_ids=EXPECTED_CAMERAS,
                reference_camera_id="camera_a",
                pairing_tolerance_seconds=0.01,
            )
        )


def test_mixed_synchronization_provenance_is_rejected() -> None:
    streams = {
        "camera_a": [make_identity("camera_a", 0, 0.0)],
        "camera_b": [make_identity("camera_b", 0, 0.0, manifest_hash="c" * 64)],
    }
    with pytest.raises(ValueError, match="synchronization provenance"):
        tuple(
            build_synchronized_bundles(
                streams,
                expected_camera_ids=EXPECTED_CAMERAS,
                reference_camera_id="camera_a",
                pairing_tolerance_seconds=0.01,
            )
        )


def test_reversed_worker_completion_restores_capture_order() -> None:
    bundles = build_complete_fixture()
    completed = [
        SimpleNamespace(bundle_id=bundle.bundle_id, value=bundle.bundle_index)
        for bundle in reversed(bundles)
    ]

    ordered = restore_capture_order(
        completed,
        bundle_id_of=lambda item: str(item.bundle_id),
        bundles=bundles,
    )

    assert [item.value for item in ordered] == [0, 1, 2]
    with pytest.raises(ValueError, match="duplicate worker result"):
        restore_capture_order(
            [completed[0], completed[0]],
            bundle_id_of=lambda item: str(item.bundle_id),
            bundles=bundles,
        )
    with pytest.raises(ValueError, match="unknown bundle_id"):
        restore_capture_order(
            [SimpleNamespace(bundle_id="f" * 64)],
            bundle_id_of=lambda item: str(item.bundle_id),
            bundles=bundles,
        )


def test_bundle_rejects_tampered_identity() -> None:
    bundle = build_complete_fixture()[0]
    payload = bundle.model_dump()
    payload["bundle_index"] = 99

    with pytest.raises(ValidationError, match="bundle_id does not match"):
        SynchronizedFrameBundle.model_validate(payload)


def test_file_source_rejects_content_hash_mismatch(tmp_path: Path) -> None:
    video_path = tmp_path / "not-a-video.mp4"
    video_path.write_bytes(b"unchanged fixture bytes")

    with pytest.raises(ValueError, match="content hash mismatch"):
        FileFrameSource(
            path=video_path,
            capture_session_id="session_01",
            camera_id="camera_a",
            source_ref="capture/camera_a.mp4",
            expected_sha256=SOURCE_HASH,
            synchronization_manifest_ref=MANIFEST_REF,
            synchronization_manifest_sha256=MANIFEST_HASH,
            pose_version_id="session_01:pose:v1",
        )


def test_rtsp_source_persists_no_credentials_or_query() -> None:
    url = "rtsp://alice:secret@example.test:8554/live?token=private"
    source = RTSPFrameSource(
        url=url,
        capture_session_id="live_session",
        camera_id="camera_a",
        synchronization_manifest_ref=MANIFEST_REF,
        synchronization_manifest_sha256=MANIFEST_HASH,
        pose_version_id="live_session:pose:v1",
    )

    assert source.source_ref == "rtsp://example.test:8554/live"
    assert "alice" not in source.source_ref
    assert "secret" not in source.source_ref
    assert "private" not in source.source_ref
    assert sanitize_rtsp_ref(url) == source.source_ref
    with pytest.raises(ValueError, match="rtsp"):
        sanitize_rtsp_ref("https://example.test/live")


class FakeReconnectSource:
    def __init__(
        self,
        timestamps: tuple[float, ...],
        *,
        failure: Exception | None = None,
        source_kind: FrameSourceKind = FrameSourceKind.RTSP,
    ) -> None:
        self.camera_id = "camera_a"
        self.source_kind = source_kind
        self.source_ref = "rtsp://127.0.0.1:8554/test"
        self._timestamps = timestamps
        self._failure = failure

    def iter_identities(self) -> Iterator[FrameIdentity]:
        for frame in self.iter_frames():
            yield frame.identity

    def iter_frames(self) -> Iterator[DecodedFrame]:
        for index, timestamp in enumerate(self._timestamps):
            identity = FrameIdentity.create(
                capture_session_id="rtsp_test",
                camera_id=self.camera_id,
                source_kind=self.source_kind,
                source_frame_index=index,
                source_timestamp_seconds=timestamp,
                capture_timestamp_seconds=timestamp,
                source_ref=self.source_ref,
                source_fingerprint="d" * 64,
                source_fingerprint_kind=(SourceFingerprintKind.STREAM_CONFIGURATION_SHA256),
                synchronization_manifest_ref=MANIFEST_REF,
                synchronization_manifest_sha256=MANIFEST_HASH,
                pose_version_id="rtsp_test:pose:v1",
                image_width=4,
                image_height=3,
            )
            yield DecodedFrame(
                identity=identity,
                image_bgr=np.zeros((3, 4, 3), dtype=np.uint8),
            )
        if self._failure is not None:
            raise self._failure


def test_bounded_rtsp_reconnect_rebases_reset_timestamps_and_identity() -> None:
    sources = (
        FakeReconnectSource((0.0, 0.1), failure=RuntimeError("publisher stopped")),
        FakeReconnectSource((0.0, 0.1)),
    )
    clock_value = 0.0
    sleeps: list[float] = []

    def clock() -> float:
        nonlocal clock_value
        clock_value += 0.01
        return clock_value

    frames, result = read_rtsp_with_reconnect(
        lambda attempt: sources[attempt - 1],
        target_frame_count=4,
        policy=RTSPReconnectPolicy(
            maximum_connection_attempts=2,
            reconnect_delay_seconds=0.25,
            minimum_timestamp_step_seconds=0.1,
        ),
        clock=clock,
        sleeper=sleeps.append,
    )

    assert len(frames) == 4
    assert tuple(frame.identity.source_frame_index for frame in frames) == (0, 1, 2, 3)
    assert tuple(frame.identity.capture_timestamp_seconds for frame in frames) == pytest.approx(
        (0.0, 0.1, 0.2, 0.3)
    )
    assert len({frame.identity.frame_id for frame in frames}) == 4
    assert result.diagnostics.target_reached is True
    assert result.diagnostics.reconnect_count == 1
    assert tuple(attempt.outcome for attempt in result.diagnostics.attempts) == (
        RTSPAttemptOutcome.FAILED,
        RTSPAttemptOutcome.TARGET_REACHED,
    )
    assert result.diagnostics.attempts[1].observed_reconnect_gap_seconds == pytest.approx(0.1)
    assert result.diagnostics.attempts[1].capture_timestamp_offset_seconds == (pytest.approx(0.2))
    assert sleeps == [0.25]


def test_rtsp_reconnect_exhaustion_is_explicit_and_bounded() -> None:
    policy = RTSPReconnectPolicy(
        maximum_connection_attempts=3,
        reconnect_delay_seconds=0.0,
    )

    def unavailable(_attempt: int) -> FakeReconnectSource:
        raise ConnectionError("local stream unavailable")

    frames, result = read_rtsp_with_reconnect(
        unavailable,
        target_frame_count=2,
        policy=policy,
        clock=lambda: 0.0,
        sleeper=lambda _seconds: None,
    )

    assert frames == ()
    assert result.diagnostics.exhausted is True
    assert result.diagnostics.reconnect_count == 2
    assert len(result.diagnostics.attempts) == 3
    assert all(
        attempt.outcome is RTSPAttemptOutcome.FAILED for attempt in result.diagnostics.attempts
    )


def test_rtsp_reconnect_rejects_a_non_rtsp_factory() -> None:
    _frames, result = read_rtsp_with_reconnect(
        lambda _attempt: FakeReconnectSource((), source_kind=FrameSourceKind.FILE),
        target_frame_count=1,
        policy=RTSPReconnectPolicy(maximum_connection_attempts=1),
        clock=lambda: 0.0,
        sleeper=lambda _seconds: None,
    )

    assert result.diagnostics.exhausted is True
    attempt = result.diagnostics.attempts[0]
    assert attempt.outcome is RTSPAttemptOutcome.FAILED
    assert attempt.error_type == "ValueError"
    assert "RTSP source" in str(attempt.error_message)
