# S01 Empty-Room Synchronization and QA

**Stage:** S01 - Capture, Synchronization, and Calibration

**Status:** Synchronization and capture-specific poses accepted

**Date:** 2026-07-30

## Scope

This record covers the empty-room recording pair's synchronization and
read-only content/calibration QA. It does not begin S02 DA3 reconstruction.

## Inputs

- `data/raw/s01_capture_20260729/empty_room/camera_a_empty_room.mp4`
  - SHA-256:
    `c8f3e689e9208bceef48c0f939c9b4cd1e2deb97b1bf5b45440b86c7019ce751`
  - 1920x1080, approximately 30 FPS, H.264, AAC 48 kHz
- `data/raw/s01_capture_20260729/empty_room/camera_b_empty_room.mp4`
  - SHA-256:
    `a3bd8d99c99a0ade60edcee426af42e3485649f2427002fc5384c131231b7632`
  - 1920x1080, approximately 30 FPS, HEVC, AAC 48 kHz

Both raw files remained unchanged.

## Synchronization

Sample-level clap onsets were used as the numerical anchors:

| Anchor | Camera A source time | Camera B source time |
|---|---:|---:|
| Start | `30.256938 s` | `4.246792 s` |
| End | `68.932438 s` | `42.922771 s` |

Camera A leads Camera B by approximately `26.010 s`. The measured relative
clock drift is `12.389 ppm`. One second of pre-roll and post-roll was retained,
and Camera B timestamps/audio were corrected to Camera A.

The two derived clips each decode to 1,220 frames and have a `40.675 s`
duration. The residual clap disagreement is `0.667 ms` at the start and
`0.583 ms` at the end, comfortably below one 30 FPS frame.

## Content QA

The entire synchronized interval is not empty. The operator remains visible
during setup in sampled frames at 10-18 seconds. The operator has left by
20 seconds, and the door/scene is settled by 22 seconds.

The candidate static interval for S02 is therefore restricted to synchronized
time `22.0-38.0 s`, providing 16 seconds of stable, person-free imagery. S02
must not use the setup portion as empty-room geometry input.

The blue pickup circle on the bed and white drop-off circle on the floor are
visible in both cameras. M40-M42 are also repeatedly detectable.

## Fixed-Pose Carryover Check

Directly reusing the accepted world-pose calibration produces sampled marker
corner errors above the existing `5 px` threshold:

| Check | Camera A | Camera B |
|---|---:|---:|
| Existing-pose marker error p95 | `10.032 px` | `6.853 px` |
| Empty-room refit centre difference | `0.006 m` | `0.010 m` |
| Empty-room refit rotation difference | `0.598 deg` | `0.337 deg` |
| Empty-room refit marker RMS | `1.294 px` | `1.137 px` |

The marker geometry is still coherent and the physical centre differences are
small, but the image orientation shifted slightly between recordings. This is
consistent with a small mount change or per-recording phone stabilization
state. The prior `5 px` threshold is not weakened, and the original pose is not
silently reused.

D023 adopts versioned capture-specific pose correction from stationary
M40-M42. The original world-pose calibration remains the physical reference;
it is not overwritten. The empty-room correction is stored as
`s01_capture_20260729:empty_room:v1`.

The correction passes all marker, stability, height, optical-axis,
floor-intersection, and reference-difference checks:

| Check | Camera A | Camera B |
|---|---:|---:|
| Corrected camera centre XYZ (m) | `(0.129, 4.002, 2.160)` | `(2.183, 3.672, 2.206)` |
| Aggregate marker RMS | `1.403 px` | `1.146 px` |
| Sampled-corner error p95 | `2.183 px` | `2.160 px` |
| Centre displacement from reference | `0.009 m` | `0.010 m` |
| Rotation difference from reference | `0.574 deg` | `0.342 deg` |
| Per-frame corrected centre stability p95 | `0.012 m` | `0.003 m` |
| Per-frame corrected rotation stability p95 | `0.166 deg` | `0.046 deg` |

Both centre displacements are below D023's `0.05 m` limit and both rotation
differences are below its `1.0 degree` limit.

## Artifacts

- `artifacts/s01/empty_room/synchronized/camera_a_empty_room_synced.mp4`
- `artifacts/s01/empty_room/synchronized/camera_b_empty_room_synced.mp4`
- `artifacts/s01/empty_room/synchronized/empty_room_pair_preview.jpg`
- `artifacts/s01/empty_room/synchronized/synchronization_manifest.json`
- `artifacts/s01/calibration/empty_room_pose_inputs.json`
- `artifacts/s01/calibration/empty_room_pose/camera_calibration.json`
- `artifacts/s01/calibration/empty_room_pose/camera_pair_reprojection_preview.jpg`

Generated artifacts remain local and ignored by Git.

## Reproduction and Verification

```text
.venv/bin/python scripts/calibration/estimate_fixed_poses.py \
  --input-config artifacts/s01/calibration/empty_room_pose_inputs.json \
  --output-dir artifacts/s01/calibration/empty_room_pose

.venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/mypy src scripts/calibration/estimate_fixed_poses.py
git diff --check
```

- The empty-room pose workflow produced byte-identical JSON and preview hashes
  on two consecutive runs.
- `112` project tests passed.
- Ruff and strict mypy passed.
- Both synchronization/calibration JSON files parsed successfully.
- `git diff --check` passed.
- The final empty-room pose JSON SHA-256 is
  `f20a3dbed7409d23ad706bdbdaf13f234baac877a876ddae7acc7004cd11b737`.
- The pair reprojection preview SHA-256 is
  `202f7ba9a3cefefa2674f31148fdf0eecab37296ab1dd75f207ba72c168df2c5`.
- Raw empty-room input hashes remained unchanged.

## Next Stage-01 Action

Obtain approximate world coordinates for the centres of the two
`0.30 m`-radius zones. The blue bed zone requires `(X, Y, bed-surface Z)` and
the white floor zone requires `(X, Y, 0)`. Only after zone validation and the
remaining S01 software gates pass may S02 begin.
