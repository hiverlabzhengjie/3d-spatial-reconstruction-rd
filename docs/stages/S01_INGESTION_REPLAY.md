# S01 Deterministic Ingestion and Frame-Bundle Replay

**Date:** 2026-07-31

**Stage:** S01 - Capture, Synchronization, and Calibration

**Status:** Completion-gate implementation passed

## Purpose

Provide the transport boundary and immutable capture-time identities required
for S01 and later independent DA3, YOLO/ByteTrack, and Qwen workers. Worker
completion time and completion order are excluded from frame and bundle
identity.

## Implemented contracts

`FrameIdentity` persistently records:

- capture session and camera;
- source transport kind and decoded source-frame index;
- original source PTS and synchronized capture timestamp;
- credential-free source reference and source fingerprint;
- synchronization-manifest reference and SHA-256;
- capture-specific pose-version ID;
- image dimensions; and
- a deterministic SHA-256 frame ID derived from those immutable fields.

`SynchronizedFrameBundle` records:

- deterministic bundle index, bundle ID, and capture timestamp;
- expected, present, and missing cameras;
- complete or incomplete status;
- pairing tolerance and observed camera-time difference; and
- synchronization provenance shared by all member frames.

Frame and bundle JSON round trips revalidate their IDs. Changing a timestamp,
source, pose version, or other identity field without regenerating the ID is
rejected.

Decoded BGR arrays are copied and made read-only before worker handoff. The
existing `FrameRef` worker contract can be derived from a `FrameIdentity`
without changing its authoritative capture timestamp.

## Source boundary

- `FileFrameSource` verifies the complete input SHA-256 before decoding with
  PyAV.
- `RTSPFrameSource` implements the same transport-neutral `FrameSource`
  protocol and timestamp transform.
- Persistent RTSP references strip username, password, and query parameters.
  Because a live stream has no finite content hash, its fingerprint is
  explicitly a stream-configuration SHA-256 rather than a media-content hash.
- Actual local RTSP serving, reconnect, jitter, and packet-loss testing remain
  assigned to S06 by `docs/MODEL_SCHEDULING.md`.

No new dependency or non-baseline reconstruction method was introduced.

## Pairing and ordering policy

The synchronizer consumes each camera in strictly increasing capture-time
order. It selects the earliest unconsumed timestamp, adds at most one frame per
camera within the declared tolerance, and never reuses a frame.

For accepted 30 FPS derived clips, the pairing tolerance is half a nominal
frame: `1/60 s` (`16.667 ms`). An unavailable camera produces an explicit
incomplete bundle; no frame or XYZ value is fabricated.

Duplicate frame IDs, duplicate or decreasing frame indices, duplicate or
decreasing timestamps, mixed capture sessions, and mixed synchronization
provenance are rejected. A helper restores worker results to authoritative
bundle order and rejects duplicate or unknown result IDs.

## Real replay results

| Capture | Bundles | Complete | Missing | Maximum A/B time difference | Ordered-ID digest |
| --- | ---: | ---: | ---: | ---: | --- |
| Empty room | `1,220` | `1,220` | `0` | `3.333 ms` | `ceb374cb37a3a951dece8872d721c2decd8a15fabdbb9d2d7787b75b9d40261b` |
| Preferred action | `1,047` | `1,047` | `0` | `6.667 ms` | `e22685d6b70b36adf5551cf28b6bbfd5879be9f188768ccc7b54c511c5c477c5` |

Each accepted pair was decoded twice. Both replays produced exactly the same
bundle IDs and order. Simulated reverse worker completion restored to the same
capture order. The first real 1920x1080 BGR frame from every source also passed
the immutable-pixel smoke check.

Retained summaries:

- `artifacts/s01/ingestion/empty_room_frame_bundle_replay.json`
- `artifacts/s01/ingestion/action_take_01_frame_bundle_replay.json`

## Failure behaviour verified

Automated tests prove that:

- one absent camera produces an incomplete bundle naming that camera;
- no synthetic partner frame is inserted;
- duplicate frame IDs and non-increasing frame indices are rejected;
- non-increasing capture timestamps are rejected;
- mixed synchronization provenance is rejected;
- duplicate and unknown worker completion results are rejected;
- tampered persistent frame and bundle IDs fail schema validation; and
- RTSP credentials and query values are absent from persisted references.

## Capture notes and room metadata

The local capture session now includes:

- `data/raw/s01_capture_20260729/capture_notes.json`
- `artifacts/s01/scene_metadata.json`

The scene metadata defines a conservative processing AABB of
`(-0.5, -0.5, 0.0)` to `(3.0, 4.5, 3.0) m`. It is intentionally labelled
approximate and not surveyed. Its purpose is S02 gross-outlier rejection, not
precise wall or ceiling reconstruction.

## Reproduction

```bash
.venv/bin/python scripts/ingestion/verify_frame_bundles.py

.venv/bin/python scripts/ingestion/verify_frame_bundles.py \
  --synchronization-manifest \
    artifacts/s01/empty_room/synchronized/synchronization_manifest.json \
  --pose-calibration \
    artifacts/s01/calibration/empty_room_pose/camera_calibration.json \
  --output artifacts/s01/ingestion/empty_room_frame_bundle_replay.json

.venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/mypy src scripts/calibration/estimate_fixed_poses.py \
  scripts/calibration/estimate_zones.py \
  scripts/ingestion/verify_frame_bundles.py
git diff --check
```

## S01 completion-gate assessment

- Synchronized pair previews are visually correct: passed.
- Same-input frame-bundle IDs and order reproduce exactly: passed for
  empty-room and preferred-action captures.
- Ordering is independent of downstream completion order: passed with reverse
  completion.
- Marker reprojections align plausibly in both cameras: passed for fixed,
  empty-room, and preferred-action pose versions.
- Camera frustums point into the declared room: passed.
- No material camera movement invalidates calibration: passed through D023
  capture-specific marker checks within the fixed-reference thresholds.

No S01 completion-gate condition was weakened.
