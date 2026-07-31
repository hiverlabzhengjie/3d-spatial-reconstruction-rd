# S01 Fixed World-Pose Calibration

**Stage:** S01 - Capture, Synchronization, and Calibration

**Status:** Fixed poses accepted with documented M43 exclusion; S01 remains in
progress

**Date:** 2026-07-30

## Scope

This record covers only the fixed Camera A/B pose calculation and its
reprojection/frustum checks. It does not close S01. Empty-room and dynamic
captures, room bounds/zones, deterministic synchronized frame bundles, and the
remaining S01 completion-gate checks are still pending.

## Prerequisites Verified

- S01 is the current stage in `docs/STATUS.md`.
- Both raw world-pose recordings remain under
  `data/raw/s01_capture_20260729/world_pose/`.
- Their synchronized derived clips and offset/drift manifest are present.
- Both clips decode at 1920x1080 and contain 3,188 frames.
- The accepted shared intrinsic estimate uses 25 ChArUco poses and has
  `0.280 px` calibration RMS.
- The floor-marker model is DICT_5X5_100 with 180 mm markers, page top aligned
  to world `+Y`, and page right aligned to world `+X`.
- The manually measured marker centres and stated `+/-0.05 m` uncertainty are:
  M40 `(0.00, 0.00, 0.00)`, M41 `(1.23, 0.45, 0.00)`,
  M42 `(0.00, 2.20, 0.00)`, and M43 `(1.10, 3.70, 0.00)` metres.

## Method

OpenCV ArUco observations were sampled deterministically every 15th decoded
frame from 5-101 seconds in each synchronized recording. The median pixel
location of each marker corner was used to reduce subpixel jitter. Markers
eligible to anchor the pose had to be detected at least ten times in both
cameras.

M40-M42 were the resulting common anchors. Their twelve planar corners were
solved with OpenCV SQPnP and refined with Levenberg-Marquardt. Each fixed
solution was then checked against all sampled observations, independent
per-frame pose solves, camera height, optical-axis direction, and the optical
axis's floor intersection. The implementation retains explicit
`T_world_from_camera` and `T_camera_from_world` transforms and validates that
they are rigid mutual inverses.

This is the approved OpenCV baseline, not an added reconstruction method.

## Results

| Check | Camera A | Camera B |
|---|---:|---:|
| Camera centre XYZ (m) | `(0.131, 3.999, 2.151)` | `(2.176, 3.670, 2.201)` |
| Aggregate M40-M42 reprojection RMS | `1.527 px` | `1.481 px` |
| Sampled-corner error, 95th percentile | `2.402 px` | `2.280 px` |
| Per-frame centre difference, 95th percentile | `0.011 m` | `0.011 m` |
| Per-frame rotation difference, 95th percentile | `0.202 deg` | `0.148 deg` |
| Successful independent frame poses | 177 | 173 |
| Downward optical axis | Pass | Pass |
| Optical axis intersects marked floor envelope | Pass | Pass |

The two estimated camera centres are `2.071 m` apart. Both frustums point
downward into the shared marked area. Camera B's shared intrinsic estimate
therefore passes the fixed-world-pose validation required by D021.

## M43 Limitation

M43 is not used to fit either camera:

- Camera B contains only a clipped marker portion, so it does not yield a
  complete ArUco detection.
- Camera A detects M43, but its observed corners disagree with reprojection
  from the recorded centre `(1.10, 3.70, 0.00) m` by `100.044 px` RMS.

The implementation does not infer and substitute a new M43 coordinate from the
image. D022 records the bounded decision to retain M43 as a failed diagnostic
and accept M40-M42 for this exploratory prototype.

## Artifacts

- `artifacts/s01/calibration/world_pose_inputs.json`
- `artifacts/s01/calibration/fixed_pose/camera_calibration.json`
- `artifacts/s01/calibration/fixed_pose/camera_a_reprojection_preview.jpg`
- `artifacts/s01/calibration/fixed_pose/camera_b_reprojection_preview.jpg`
- `artifacts/s01/calibration/fixed_pose/camera_pair_reprojection_preview.jpg`

These generated outputs remain local and are ignored by Git.

## Reproduction

```text
.venv/bin/python scripts/calibration/estimate_fixed_poses.py \
  --input-config artifacts/s01/calibration/world_pose_inputs.json \
  --output-dir artifacts/s01/calibration/fixed_pose

.venv/bin/pytest -q tests/test_fixed_pose.py tests/test_transforms.py
.venv/bin/ruff check src/spatial_reconstruction/calibration \
  scripts/calibration/estimate_fixed_poses.py tests/test_fixed_pose.py
.venv/bin/mypy src/spatial_reconstruction/calibration \
  scripts/calibration/estimate_fixed_poses.py
```

## Verification Results

- The pose workflow was run twice from the same inputs.
- Both runs produced byte-identical calibration JSON and JPEG hashes.
- `112` project tests passed.
- Ruff passed across `src`, `tests`, and `scripts`.
- Strict mypy passed across the `16` checked source files.
- Both input/output JSON files parsed successfully.
- `git diff --check` passed.
- The two raw world-pose input hashes remained:
  - Camera A:
    `687b8b5def98d43f42a0b5d92d3d777a37ada9ddd792dbeaed9415a1183bedf4`;
  - Camera B:
    `13e11e0cb45f38af496f604f9b5f869a3c1ffabb6f9bec98eb10832ce0d43662`.
- Reproduced artifact hashes:
  - `camera_calibration.json`:
    `f711850714353fa31311d877d1a7e62a2f3bfaf5b932e4c6fd83058115da3b1c`;
  - pair reprojection preview:
    `6517717632f16ccb26ce756fb9b7fa9f10b92f9ebcdd559dd7a7f141d66b775e`.

## Interpretation and Next Action

These are prototype-grade metric poses because marker centres were measured to
approximately `+/-0.05 m`, not surveyed. Their low pixel residuals show that
M40-M42 and the camera models are internally coherent; they do not turn the
physical measurements into survey-grade ground truth.

Do not move either camera or the accepted M40-M42 anchors. The next physical
step is the synchronized empty-room capture, followed by pickup/drop-off zone
definition and the synchronized dynamic action capture. S01 software work must
also implement deterministic synchronized frame bundles and complete the
remaining failure/replay tests before the stage can close.
