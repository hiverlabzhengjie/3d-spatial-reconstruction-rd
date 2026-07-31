"""PyAV-backed file and RTSP frame-source boundaries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlsplit, urlunsplit

import av
import numpy as np
from numpy.typing import NDArray

from spatial_reconstruction.contracts import (
    FrameIdentity,
    FrameSourceKind,
    SourceFingerprintKind,
)

UInt8Array = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class TimestampTransform:
    """Affine mapping from source PTS seconds to synchronized capture time."""

    scale: float = 1.0
    offset_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("timestamp scale must be finite and positive")
        if not np.isfinite(self.offset_seconds):
            raise ValueError("timestamp offset must be finite")

    def apply(self, source_timestamp_seconds: float) -> float:
        """Map one source timestamp into the capture timeline."""

        capture_timestamp = self.scale * source_timestamp_seconds + self.offset_seconds
        if not np.isfinite(capture_timestamp) or capture_timestamp < 0:
            raise ValueError("mapped capture timestamp must be finite and non-negative")
        return float(capture_timestamp)


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    """One immutable decoded BGR frame and its persistent source identity."""

    identity: FrameIdentity
    image_bgr: UInt8Array

    def __post_init__(self) -> None:
        image = np.asarray(self.image_bgr)
        expected_shape = (
            self.identity.image_height,
            self.identity.image_width,
            3,
        )
        if image.dtype != np.uint8 or image.shape != expected_shape:
            raise ValueError(
                "decoded frame must be uint8 BGR with dimensions matching its identity"
            )
        immutable = np.ascontiguousarray(image).copy()
        immutable.setflags(write=False)
        object.__setattr__(self, "image_bgr", immutable)


class FrameSource(Protocol):
    """Transport-neutral frame source consumed by deterministic ingestion."""

    @property
    def camera_id(self) -> str:
        """Return the immutable camera identity."""

    @property
    def source_kind(self) -> FrameSourceKind:
        """Return the source transport kind."""

    @property
    def source_ref(self) -> str:
        """Return the credential-free persistent source reference."""

    def iter_identities(self) -> Iterator[FrameIdentity]:
        """Decode and yield frame identities without pixel conversion."""

    def iter_frames(self) -> Iterator[DecodedFrame]:
        """Decode and yield immutable pixel frames."""


@dataclass(frozen=True, slots=True)
class _SourceContext:
    capture_session_id: str
    camera_id: str
    source_ref: str
    source_fingerprint: str
    source_fingerprint_kind: SourceFingerprintKind
    synchronization_manifest_ref: str
    synchronization_manifest_sha256: str
    pose_version_id: str
    timestamp_transform: TimestampTransform
    expected_width: int | None
    expected_height: int | None


class _PyAVFrameSource:
    """Shared decoder implementation for file and RTSP transports."""

    def __init__(
        self,
        *,
        location: str,
        source_kind: FrameSourceKind,
        context: _SourceContext,
        open_options: dict[str, str] | None = None,
    ) -> None:
        self._location = location
        self._source_kind = source_kind
        self._context = context
        self._open_options = dict(open_options or {})

    @property
    def camera_id(self) -> str:
        return self._context.camera_id

    @property
    def source_kind(self) -> FrameSourceKind:
        return self._source_kind

    @property
    def source_ref(self) -> str:
        return self._context.source_ref

    def iter_identities(self) -> Iterator[FrameIdentity]:
        for identity, _ in self._decode(include_pixels=False):
            yield identity

    def iter_frames(self) -> Iterator[DecodedFrame]:
        for identity, image in self._decode(include_pixels=True):
            if image is None:
                raise RuntimeError("pixel decoding unexpectedly returned no image")
            yield DecodedFrame(identity=identity, image_bgr=image)

    def _decode(
        self,
        *,
        include_pixels: bool,
    ) -> Iterator[tuple[FrameIdentity, UInt8Array | None]]:
        with av.open(self._location, options=self._open_options) as container:
            video_streams = container.streams.video
            if not video_streams:
                raise RuntimeError(f"frame source has no video stream: {self.source_ref}")
            stream = video_streams[0]
            for source_frame_index, frame in enumerate(container.decode(stream)):
                if frame.pts is None or frame.time_base is None:
                    raise RuntimeError(
                        f"decoded frame lacks a source timestamp: {self.source_ref}"
                    )
                source_timestamp = float(frame.pts * frame.time_base)
                if source_timestamp < 0 or not np.isfinite(source_timestamp):
                    raise RuntimeError(
                        f"decoded frame has invalid source timestamp: {self.source_ref}"
                    )
                width = int(frame.width)
                height = int(frame.height)
                if (
                    self._context.expected_width is not None
                    and width != self._context.expected_width
                ):
                    raise RuntimeError("decoded frame width differs from source manifest")
                if (
                    self._context.expected_height is not None
                    and height != self._context.expected_height
                ):
                    raise RuntimeError("decoded frame height differs from source manifest")

                identity = FrameIdentity.create(
                    capture_session_id=self._context.capture_session_id,
                    camera_id=self._context.camera_id,
                    source_kind=self._source_kind,
                    source_frame_index=source_frame_index,
                    source_timestamp_seconds=source_timestamp,
                    capture_timestamp_seconds=self._context.timestamp_transform.apply(
                        source_timestamp
                    ),
                    source_ref=self._context.source_ref,
                    source_fingerprint=self._context.source_fingerprint,
                    source_fingerprint_kind=self._context.source_fingerprint_kind,
                    synchronization_manifest_ref=(
                        self._context.synchronization_manifest_ref
                    ),
                    synchronization_manifest_sha256=(
                        self._context.synchronization_manifest_sha256
                    ),
                    pose_version_id=self._context.pose_version_id,
                    image_width=width,
                    image_height=height,
                )
                image = (
                    cast(UInt8Array, frame.to_ndarray(format="bgr24"))
                    if include_pixels
                    else None
                )
                yield identity, image


class FileFrameSource(_PyAVFrameSource):
    """Deterministic local-file decoder with content-hash validation."""

    def __init__(
        self,
        *,
        path: Path,
        capture_session_id: str,
        camera_id: str,
        source_ref: str,
        expected_sha256: str,
        synchronization_manifest_ref: str,
        synchronization_manifest_sha256: str,
        pose_version_id: str,
        timestamp_transform: TimestampTransform | None = None,
        expected_width: int | None = None,
        expected_height: int | None = None,
    ) -> None:
        resolved_path = path.resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(resolved_path)
        actual_sha256 = _sha256(resolved_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"source content hash mismatch for {source_ref}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        super().__init__(
            location=str(resolved_path),
            source_kind=FrameSourceKind.FILE,
            context=_SourceContext(
                capture_session_id=capture_session_id,
                camera_id=camera_id,
                source_ref=source_ref,
                source_fingerprint=actual_sha256,
                source_fingerprint_kind=SourceFingerprintKind.CONTENT_SHA256,
                synchronization_manifest_ref=synchronization_manifest_ref,
                synchronization_manifest_sha256=synchronization_manifest_sha256,
                pose_version_id=pose_version_id,
                timestamp_transform=timestamp_transform or TimestampTransform(),
                expected_width=expected_width,
                expected_height=expected_height,
            ),
        )


class RTSPFrameSource(_PyAVFrameSource):
    """RTSP decoder boundary sharing the same frame identity contract."""

    def __init__(
        self,
        *,
        url: str,
        capture_session_id: str,
        camera_id: str,
        synchronization_manifest_ref: str,
        synchronization_manifest_sha256: str,
        pose_version_id: str,
        timestamp_transform: TimestampTransform | None = None,
        expected_width: int | None = None,
        expected_height: int | None = None,
        open_options: dict[str, str] | None = None,
    ) -> None:
        safe_ref = sanitize_rtsp_ref(url)
        fingerprint_payload = json.dumps(
            {
                "capture_session_id": capture_session_id,
                "camera_id": camera_id,
                "source_ref": safe_ref,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        options = {"rtsp_transport": "tcp", **(open_options or {})}
        super().__init__(
            location=url,
            source_kind=FrameSourceKind.RTSP,
            context=_SourceContext(
                capture_session_id=capture_session_id,
                camera_id=camera_id,
                source_ref=safe_ref,
                source_fingerprint=hashlib.sha256(fingerprint_payload).hexdigest(),
                source_fingerprint_kind=(
                    SourceFingerprintKind.STREAM_CONFIGURATION_SHA256
                ),
                synchronization_manifest_ref=synchronization_manifest_ref,
                synchronization_manifest_sha256=synchronization_manifest_sha256,
                pose_version_id=pose_version_id,
                timestamp_transform=timestamp_transform or TimestampTransform(),
                expected_width=expected_width,
                expected_height=expected_height,
            ),
            open_options=options,
        )


def sanitize_rtsp_ref(url: str) -> str:
    """Return a credential- and query-free RTSP reference for persistence."""

    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"rtsp", "rtsps"}:
        raise ValueError("RTSP source URL must use rtsp:// or rtsps://")
    if parsed.hostname is None:
        raise ValueError("RTSP source URL must include a host")
    hostname = parsed.hostname
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path, "", ""))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
