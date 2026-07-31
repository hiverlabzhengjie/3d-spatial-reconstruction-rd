# S01 Handoff - Capture, Synchronization, and Calibration

**Stage:** S01 - Capture, Synchronization, and Calibration

**Status:** Complete

**Started:** 2026-07-29

**Closed:** 2026-07-31

## Stage Goal

Produce synchronized two-camera recordings, stable intrinsic and fixed-pose
calibration in one local metric world frame, accepted room/zone metadata, and
deterministic immutable frame bundles suitable for independent downstream
workers.

## Entry Inputs

- Completed S00 Python 3.11 environment, model adapters, geometry utilities,
  runtime diagnostics, and worker-scheduling contract.
- Two iPhone 16 Pro Max devices on stable temporary mounts using their
  13 mm-equivalent ultrawide lenses at 1920x1080 and nominal 30 FPS.
- One printed, measured, rigid matte A4 ChArUco board.
- Four printed 180 mm ArUco floor markers and approximate tape-measured
  centres.
- Intrinsic, fixed-world-pose, empty-room, and two dynamic action capture
  takes.
- Visible flash and audible clap at the beginning and end of paired captures.

## Work Completed

- Created the canonical ChArUco board and 180 mm floor-marker assets and
  recorded their physical dimensions.
- Accepted shared Camera A/B intrinsics from the matched-phone ChArUco capture
  under D021, with `0.280 px` calibration RMS and successful Camera B
  world-marker validation.
- Synchronized the fixed-world-pose pair with offset/drift correction.
- Solved fixed Camera A/B poses from common non-collinear M40-M42 anchors.
- Retained M43 as an excluded failed coordinate diagnostic under D022 without
  rewriting its physical measurement from imagery.
- Adopted D023 versioned capture-specific pose correction and accepted:
  - empty-room pose `s01_capture_20260729:empty_room:v1`;
  - action pose `s01_capture_20260729:action_take_01:v1`.
- Synchronized the empty-room pair and restricted S02 use to the stable
  `22.0-38.0 s` window.
- Estimated the blue pickup and white drop-off rope-circle zones from both
  empty-room views under D024; automated checks passed and the user validated
  their physical plausibility.
- Reviewed both dynamic takes, retained take 02 unchanged as backup, and
  selected/synchronized take 01 as the baseline pickup-carry-place recording.
- Added local capture notes and conservative approximate room bounds.
- Added deterministic immutable frame and synchronized-bundle contracts with
  source, timestamp, synchronization, and pose-version provenance.
- Added content-hash-validated file ingestion and a credential-safe
  RTSP-compatible source boundary.
- Added explicit missing-camera bundles plus duplicate, non-monotonic,
  mixed-provenance, and tamper rejection.
- Proved same-input identity/order replay for both accepted empty-room and
  action pairs, independent of reversed downstream completion order.

## Changed Files

Important project-owned additions and updates include:

- `.gitignore`;
- `src/spatial_reconstruction/contracts.py`;
- `src/spatial_reconstruction/calibration/`;
- `src/spatial_reconstruction/ingestion/`;
- `scripts/calibration/`;
- `scripts/ingestion/verify_frame_bundles.py`;
- `tests/test_fixed_pose.py`;
- `tests/test_zones.py`;
- `tests/test_ingestion.py`;
- `docs/DECISIONS.md`;
- `docs/STATUS.md`;
- `docs/stages/S01_WORLD_POSE_CALIBRATION.md`;
- `docs/stages/S01_EMPTY_ROOM_SYNC_QA.md`;
- `docs/stages/S01_ZONE_ESTIMATION.md`;
- `docs/stages/S01_ACTION_SYNC_QA.md`;
- `docs/stages/S01_INGESTION_REPLAY.md`;
- this handoff.

Raw captures, local capture notes, generated targets, derived synchronized
videos, calibration outputs, and diagnostics remain excluded from Git.

## Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Capture notes | `data/raw/s01_capture_20260729/capture_notes.json` | Hardware, settings, physical references, file hashes, and movement record |
| Fixed-pose synchronization | `artifacts/s01/world_pose/synchronized/` | Offset/drift-corrected calibration pair and manifest |
| Fixed camera calibration | `artifacts/s01/calibration/fixed_pose/` | Physical reference poses and reprojection diagnostics |
| Empty-room synchronization | `artifacts/s01/empty_room/synchronized/` | Accepted static-input pair, manifest, and stable-window declaration |
| Empty-room pose | `artifacts/s01/calibration/empty_room_pose/` | D023 pose version for S02 |
| Zone metadata | `artifacts/s01/zones/estimated_zones.json` | Accepted video-estimated pickup/drop-off circles |
| Scene metadata | `artifacts/s01/scene_metadata.json` | World convention, approximate bounds, zones, and accepted pose versions |
| Action synchronization | `artifacts/s01/action_take_01/synchronized/` | Preferred synchronized dynamic pair and visual preview |
| Action pose | `artifacts/s01/calibration/action_take_01_pose/` | D023 pose version for later dynamic stages |
| Empty replay evidence | `artifacts/s01/ingestion/empty_room_frame_bundle_replay.json` | Deterministic 1,220-bundle replay |
| Action replay evidence | `artifacts/s01/ingestion/action_take_01_frame_bundle_replay.json` | Deterministic 1,047-bundle replay |

## Verification

### Commands

```text
.venv/bin/python scripts/calibration/estimate_fixed_poses.py \
  --input-config artifacts/s01/calibration/action_take_01_pose_inputs.json \
  --output-dir artifacts/s01/calibration/action_take_01_pose

.venv/bin/python scripts/calibration/estimate_zones.py

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
uv --cache-dir /private/tmp/spatial-reconstruction-uv-cache lock --check
uv --cache-dir /private/tmp/spatial-reconstruction-uv-cache sync --check
git diff --check
```

### Results

- `128` automated tests passed.
- Ruff passed.
- Strict mypy passed across `22` source/script files.
- Lockfile and installed environment consistency checks passed.
- Raw capture SHA-256 values remained unchanged.
- Fixed, empty-room, and action marker reprojection/stability/frustum checks
  passed.
- Empty-room replay: `1,220` complete bundles, zero missing, maximum
  inter-camera difference `3.333 ms`.
- Action replay: `1,047` complete bundles, zero missing, maximum inter-camera
  difference `6.667 ms`.
- Both real replays reproduced identical ordered IDs and restored capture order
  after reverse simulated worker completion.
- Missing-camera, duplicate, non-monotonic, mixed-provenance, tamper, unknown
  completion, and credential-safe RTSP behaviours are covered by tests.
- User validated the video-estimated zone positions and approximately
  `0.60 m` bed-zone height.
- No completion-gate check was skipped or weakened.

## Physical Setup and Observations

- Both devices are iPhone 16 Pro Max units using the same selected
  13 mm-equivalent ultrawide lens and recording configuration.
- Focus, exposure, and white balance were locked; automatic lens switching and
  enhanced stabilization were disabled.
- The temporary rigid mounts were accepted as stable for this prototype.
- M40 is the world origin; X follows the selected primary wall, Y extends into
  the room, and Z points upward.
- Floor-marker centre uncertainty is approximately `+/-0.05 m`.
- M40-M42 remain the accepted pose anchors.
- Effective image pose changed slightly between recordings despite stable
  mounts. D023 preserves the physical reference and records bounded
  capture-specific pose versions rather than silently reusing one transform.
- Any future material camera/lens movement invalidates the relevant
  calibration and requires marker-based recalibration.

## Problems and Limitations

- Shared intrinsics are a bounded matched-phone approximation, not proof that
  the two physical ultrawide lenses are identical.
- M43's recorded centre is inconsistent with imagery and remains excluded; it
  must be physically remeasured if later geometry exposes a material
  world-alignment issue.
- Synchronization is millisecond-accurate for the prototype, not hardware
  genlock.
- Pickup/drop-off centres are video-estimated and user-validated, not surveyed.
- Room bounds `(-0.5, -0.5, 0.0)` to `(3.0, 4.5, 3.0) m` are a conservative
  filtering envelope, not exact wall/ceiling coordinates.
- The stable S02 empty-room interval is only `16 s`; setup frames outside
  `22.0-38.0 s` contain the operator and must not enter static reconstruction.
- RTSP has an implemented protocol-compatible source boundary, but actual
  local RTSP reconnect/jitter/loss testing remains S06 work.
- Production synchronization, monitoring, and deployment remain out of scope.

## Decisions Made

- D021 - shared intrinsic estimate for the matched phone pair.
- D022 - fixed camera poses use common markers M40-M42.
- D023 - versioned capture-specific pose correction.
- D024 - video-estimated pickup and drop-off zones.

No optional static-reconstruction method was activated in S01.

## Prerequisites for the Next Stage

Software:

- Preserve the exact DA3 vendor snapshot and S00 MPS adapter.
- Use `DA3NESTED-GIANT-LARGE-1.1` in pose-conditioned multi-view metric mode.
- Use the immutable S01 bundle identities and capture timestamps.
- Use empty-room pose version
  `s01_capture_20260729:empty_room:v1`, not the fixed reference or action pose.
- Use the accepted synchronized empty-room inputs and manifest hashes.
- Use the conservative room bounds only for gross outlier filtering.
- Retain DA3 raw depth/confidence outputs and frame provenance.

Input selection:

- Use only synchronized empty-room times `22.0-38.0 s`.
- Exclude setup/operator frames.
- Begin with S00's provisional DA3 process resolution `504`; revisit the
  static keyframe selection from calibrated scene evidence.

Physical/user actions:

- No new recording is required for the baseline S02 run.
- Do not move or repurpose the calibrated recordings.
- If the room reconstruction materially contradicts marker alignment, stop and
  diagnose calibration before proceeding.

## Version Control

- GitHub repository:
  `https://github.com/hiverlabzhengjie/3d-spatial-reconstruction-rd`
- Stage-close commit:
  `9d2bb08778c3a6fe014c8e300ab511d9dafa6b4a`
- Stage-close GitHub URL:
  `https://github.com/hiverlabzhengjie/3d-spatial-reconstruction-rd/commit/9d2bb08778c3a6fe014c8e300ab511d9dafa6b4a`
- Annotated tag: `None`
- Remote push verified: `Yes`; `refs/heads/main` resolved to
  `9d2bb08778c3a6fe014c8e300ab511d9dafa6b4a` before this provenance-only
  handoff update.
- Vendor/model revisions:
  - DA3 vendor fingerprint:
    `683cad1fec1186cd2a22f2b6d083b73d4c83c7ab1140f45ba24876612bc51d43`;
  - DA3 model revision:
    `b2359bdf726fb44ef62acca04d629dcf158053e7`;
  - YOLO checkpoint SHA-256:
    `a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c`;
  - Qwen model revision:
    `89644892e4d85e24eaac8bacfd4f463576704203`.

This handoff's final commit reference is recorded in a separate documentation
commit because a Git commit cannot contain its own final hash.

## Exact Next Action

After explicit user approval to begin S02, select deterministic synchronized
empty-room keyframe bundles only from `22.0-38.0 s` and run the first
pose-conditioned DA3 metric-depth reconstruction at process resolution `504`.
