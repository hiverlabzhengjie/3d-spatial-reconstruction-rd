# S02 Handoff - DA3 Static Room Geometry

**Stage:** S02 - DA3 Static Room Geometry

**Status:** Complete

**Started:** 2026-07-31

**Closed:** 2026-07-31

## Stage Goal

Generate recognizable static living-room geometry from accepted synchronized
empty-room frames using pose-conditioned DA3 metric depth, the calibrated
shared world frame, confidence and conservative room-bound filtering, and
deterministic point-cloud fusion.

## Entry Inputs

- Completed S01 synchronization, calibration, scene metadata, and immutable
  frame-bundle contracts.
- Stable empty-room interval `22.0-38.0 s`.
- Empty-room pose version `s01_capture_20260729:empty_room:v1`.
- Conservative processing bounds `(-0.5, -0.5, 0.0)` to
  `(3.0, 4.5, 3.0) m`.
- Exact DA3 Nested 1.1 model revision
  `b2359bdf726fb44ef62acca04d629dcf158053e7`.
- Exact unmodified DA3 vendor fingerprint
  `683cad1fec1186cd2a22f2b6d083b73d4c83c7ab1140f45ba24876612bc51d43`.

## Work Completed

- Added deterministic explicit-time and interval keyframe selection that
  accepts only complete bundles inside the S01 stable window.
- Added finite-depth, adaptive-confidence, and calibrated world-bound
  filtering with inspectable counts and honest empty/missing behaviour.
- Added deterministic metric voxel fusion and colored PLY output.
- Added strict persistent schemas for the S02 run and Rerun export summaries.
- Ran the exact pose-conditioned multi-view DA3 model on native Apple MPS with
  supplied calibrated OpenCV poses and no CPU fallback.
- Preserved raw depth/confidence, processed RGB, selected source identity,
  timestamps, intrinsics, poses, and diagnostic images.
- Diagnosed a consistent `14-16%` raw depth underestimate against accepted
  M40-M42 camera-Z depths without modifying raw DA3 output.
- Adopted and implemented D025: one bounded shared marker-anchored scalar per
  synchronized pair for derived S02 static geometry only.
- Reconstructed three deterministic pairs nearest `22`, `30`, and `38 s`.
- Produced per-camera and fused point clouds plus PNG, GLB, and Rerun previews.
- Added exact bidirectional radius-overlap verification without a new
  reconstruction dependency.
- Added a localhost-only Rerun web-viewer QA path after isolating a
  Rerun 0.22.1 native screenshot helper issue on Retina/Metal.
- Reopened S02 after user inspection identified missing in-bounds room
  surfaces, diagnosed confidence filtering rather than X/Y bounds as the
  cause, and adopted D026 after the user approved the `20th`-percentile,
  `Z >= 0` comparison.
- Reopened S02 again after the door behind M40 remained absent, confirmed that
  its valid depth was below p20 confidence in both cameras, and adopted D027:
  a p5 supplement restricted to the video-estimated static door volume.

## Accepted Result

- Confidence percentile: `20th` under D026; the original X/Y/Z bounds remain
  unchanged.
- D027 door supplement: p5 only inside `(-0.35, -0.40, 0.00)` to
  `(0.90, -0.12, 2.10) m`.
- Supplemental retained samples: Camera A `18,930`; Camera B `9,426`.
- Camera A point cloud: `52,006` points.
- Camera B point cloud: `43,561` points.
- Fused point cloud: `81,709` points.
- Door-volume fused points: `10,126`; Rerun visualization sample: `3,719`.
- Camera A points within `0.10 m` of Camera B: `73.363%`.
- Camera B points within `0.10 m` of Camera A: `86.297%`.
- Marker-derived scales: `1.164136`, `1.157311`, and `1.157667`.
- Maximum scale-observation deviation: `1.606%` against the `5%` limit.
- Corrected marker camera-depth error: `0.020 m` median and `0.054 m` maximum.
- Fused world extent: approximately `(-0.385, -0.330, 0.000)` to
  `(2.841, 4.152, 2.004) m`.
- The living room, bed, floor/wall structure, furniture, and door behind M40
  are recognizable.
- Both calibrated cameras and the point cloud are displayed together in one
  right-handed, metre, Z-up Rerun scene.

## Changed Files

Important project-owned additions and updates include:

- `src/spatial_reconstruction/geometry/static_scene.py`;
- `src/spatial_reconstruction/geometry/__init__.py`;
- `scripts/s02/reconstruct_static_scene.py`;
- `scripts/s02/export_rerun_static_scene.py`;
- `scripts/s02/verify_static_scene.py`;
- `tests/test_static_scene.py`;
- `pyproject.toml`;
- `uv.lock`;
- `docs/DECISIONS.md`;
- `docs/STATUS.md`;
- `docs/stages/S02_DA3_STATIC_GEOMETRY.md`;
- this handoff.

Raw captures, model weights, vendor source, generated geometry, predictions,
previews, Rerun recordings, environments, and caches remain excluded from Git.

## Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Accepted run summary | `artifacts/s02/door_inclusive_candidate_20260731/summary.json` | Schema-validated model/input/filter/runtime provenance |
| Raw DA3 predictions | `artifacts/s02/door_inclusive_candidate_20260731/predictions/` | Depth, confidence, processed RGB, intrinsics, and poses |
| Retained keyframes | `artifacts/s02/door_inclusive_candidate_20260731/keyframes/` | Undistorted synchronized frame evidence |
| Camera A PLY | `artifacts/s02/door_inclusive_candidate_20260731/camera_a_static_scene.ply` | Camera A world-space geometry |
| Camera B PLY | `artifacts/s02/door_inclusive_candidate_20260731/camera_b_static_scene.ply` | Camera B world-space geometry |
| Fused PLY | `artifacts/s02/door_inclusive_candidate_20260731/static_scene.ply` | Accepted static scene |
| Geometry PNG | `artifacts/s02/door_inclusive_candidate_20260731/previews/static_scene_geometry.png` | Recognizability and camera-frame QA |
| Geometry GLB | `artifacts/s02/door_inclusive_candidate_20260731/previews/static_scene_with_cameras.glb` | Interactive general 3D preview |
| Rerun recording | `artifacts/s02/door_inclusive_candidate_20260731/static_scene_door_inclusive_v2.rrd` | Door-inclusive Digital Twin-style geometry and camera view |
| Rerun screenshot | `artifacts/s02/door_inclusive_candidate_20260731/previews/rerun_static_scene_door_inclusive.png` | Accepted viewer evidence |
| Verification report | `artifacts/s02/door_inclusive_candidate_20260731/verification_v3.json` | Hashes, overlap, extents, and bounded-inclusion checks |

The prior p20 completeness revision remains under
`artifacts/s02/completeness_accepted_v2_20260731/`; the prior
`40th`-percentile accepted baseline remains under
`artifacts/s02/candidate_three_keyframes_20260731/`. The first raw metric
diagnostic and corrected single-pair diagnostic remain separately retained
under `artifacts/s02/first_calibrated_20260731/` and
`artifacts/s02/corrected_calibrated_20260731/`.

## Verification

### Commands

```text
.venv/bin/python scripts/s02/reconstruct_static_scene.py \
  --target-time-seconds 22 30 38 \
  --confidence-percentile 20 \
  --static-inclusion-confidence-percentile 5 \
  --static-inclusion-world-bounds -0.35 -0.40 0.00 0.90 -0.12 2.10 \
  --output-dir artifacts/s02/door_inclusive_candidate_20260731

.venv/bin/python scripts/s02/export_rerun_static_scene.py \
  --run-summary \
    artifacts/s02/door_inclusive_candidate_20260731/summary.json \
  --output \
    artifacts/s02/door_inclusive_candidate_20260731/static_scene_door_inclusive_v2.rrd

.venv/bin/python scripts/s02/verify_static_scene.py \
  --run-summary \
    artifacts/s02/door_inclusive_candidate_20260731/summary.json \
  --rerun-export-summary \
    artifacts/s02/door_inclusive_candidate_20260731/static_scene_door_inclusive_v2_export_summary.json \
  --rerun-screenshot \
    artifacts/s02/door_inclusive_candidate_20260731/previews/rerun_static_scene_door_inclusive.png \
  --output \
    artifacts/s02/door_inclusive_candidate_20260731/verification_v3.json

.venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/mypy --strict src scripts/s02
uv --cache-dir /private/tmp/spatial-reconstruction-uv-cache lock --check
uv --cache-dir /private/tmp/spatial-reconstruction-uv-cache sync --check
.venv/bin/rerun rrd print \
  artifacts/s02/door_inclusive_candidate_20260731/static_scene_door_inclusive_v2.rrd
git diff --check
```

Artifact-producing commands refuse to overwrite existing accepted outputs. Use
a new output directory and Rerun filename for a reproduction run.

### Results

- `143` automated tests passed.
- Ruff passed.
- Strict mypy passed across `23` S02-relevant source/script files.
- Lockfile and installed environment checks passed.
- The accepted Rerun recording parsed successfully with `58` chunks.
- The verification report checked `16` hashed artifacts.
- All automated geometry, overlap, finite-data, room-bound, marker-scale, and
  Rerun-presence checks passed. D027-specific checks also confirmed a valid
  bounded policy, supplemental points from both cameras, `10,126` fused
  door-volume points, and `3,719` logged Rerun door-volume points.
- Visual inspection confirmed recognizable geometry, the door plane behind
  M40, and correctly oriented camera frustums.
- No completion-gate check was skipped or weakened.

## Completion Gate

- Recognizable living-room point cloud: **passed**.
- Plausibly aligned Camera A/B points in one world frame: **passed**.
- Invalid and out-of-room points filtered: **passed**.
- Camera poses and point cloud displayed together correctly: **passed**.

## Problems and Limitations

- The result is exploratory, not survey-grade.
- Thin, reflective, texture-poor, occluded, and camera-exclusive surfaces
  remain incomplete or noisy.
- Cross-camera overlap is a shared-surface measure, not complete-room coverage.
- D025 is allowed only for derived S02 static geometry. Raw DA3 outputs remain
  unchanged, and S04 dynamic localization must not inherit this scalar.
- D026 lowers only the derived S02 confidence percentile. It preserves raw
  predictions and the original room bounds, and it may retain more noisy
  low-confidence samples.
- D027 retains p5 samples only inside a video-estimated door volume. It is a
  one-time static-scene completeness measure and is excluded from S03/S04.
- The two cameras retain the bounded shared-intrinsic assumption from D021.
- Room bounds remain a conservative processing crop rather than surveyed
  surfaces.
- Rerun SDK is pinned to `0.22.1` for NumPy 1 compatibility. Its native
  `--screenshot-to` helper exceeds the Metal texture limit on this Retina
  configuration; normal native viewing, the localhost-only web viewer, and
  the `.rrd` recording are unaffected.

## Reusable Static-Reconstruction Technique

When an important static feature is visible and has valid depth but is removed
by the global confidence filter, preserve the global threshold and introduce a
smaller world-space inclusion volume with a lower regional percentile. This is
safer than lowering confidence across the whole scene and more consistent
across calibrated cameras than maintaining separate pixel masks.

Required guardrails are: finite positive depth, inclusion bounds wholly inside
the global room bounds, persisted thresholds/bounds/counts, supplemental
points from relevant views, unchanged raw outputs, overlap and integrity
verification, visual confirmation, and proof that the feature remains in the
Rerun sample. Treat each region as an explicit static-only exception; do not
carry it into S03 perception or S04 dynamic localization by default.

## Decisions Made

- D025 - marker-anchored scalar correction for S02 static depth.
- D026 - lower static-scene confidence percentile for room completeness.
- D027 - bounded low-confidence supplement for the static room door.

No optional COLMAP, SfM, MVS, stereo, triangulation, Gaussian Splatting, or
other independent reconstruction method was activated.

## Prerequisites for S03

Software:

- Preserve the exact accepted `action_take_01` synchronization and action pose
  version `s01_capture_20260729:action_take_01:v1`.
- Use `yolov8n-seg.pt` through the project-owned adapter and add ByteTrack as
  the camera-local tracker.
- Preserve immutable frame/bundle identities, capture timestamps, and raw
  detector/segmentation outputs.
- Keep perception independent from DA3 geometry and Qwen semantics.
- Use bounded queues and deterministic capture-time ordering.

Inputs and physical state:

- Use the synchronized preferred action pair, not the backup take.
- Confirm the person and backpack remain visible enough in both cameras for
  the selected processing interval.
- Keep the accepted fixed camera/lens setup and calibration provenance. Any
  material camera or lens movement invalidates the relevant pose.
- The backpack is the primary target. Use the approved bottle fallback only if
  the standard YOLO model cannot detect the backpack reliably.

## Version Control

- GitHub repository:
  `https://github.com/hiverlabzhengjie/3d-spatial-reconstruction-rd`
- Stage-close commit:
  `4084c34e9c1d26d6dae0294fa0321ec238824704`
- Stage-close GitHub URL:
  `https://github.com/hiverlabzhengjie/3d-spatial-reconstruction-rd/commit/4084c34e9c1d26d6dae0294fa0321ec238824704`
- D026 completeness-revision commit:
  `e163b4e72c90ac798e84df264162b93541922a3c`
- D026 completeness-revision GitHub URL:
  `https://github.com/hiverlabzhengjie/3d-spatial-reconstruction-rd/commit/e163b4e72c90ac798e84df264162b93541922a3c`
- D027 door-inclusive revision commit:
  `9226a85911cba0e032bdf76b5d32bb9828ff1997`
- D027 door-inclusive revision GitHub URL:
  `https://github.com/hiverlabzhengjie/3d-spatial-reconstruction-rd/commit/9226a85911cba0e032bdf76b5d32bb9828ff1997`
- Annotated tag: `None`
- Remote push verified: `Yes`; `refs/heads/main` resolved to
  `9226a85911cba0e032bdf76b5d32bb9828ff1997` before this final
  provenance-only handoff update.
- DA3 vendor fingerprint:
  `683cad1fec1186cd2a22f2b6d083b73d4c83c7ab1140f45ba24876612bc51d43`
- DA3 model revision:
  `b2359bdf726fb44ef62acca04d629dcf158053e7`

This handoff's final commit reference is recorded in this separate
provenance-only documentation update because a Git commit cannot contain its
own final hash.

## Exact Next Action

Stop. Begin S03 only after explicit user instruction.
