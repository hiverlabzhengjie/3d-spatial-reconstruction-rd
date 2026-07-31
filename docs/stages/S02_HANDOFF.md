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

## Accepted Result

- Camera A point cloud: `30,239` points.
- Camera B point cloud: `22,332` points.
- Fused point cloud: `45,919` points.
- Camera A points within `0.10 m` of Camera B: `69.295%`.
- Camera B points within `0.10 m` of Camera A: `86.638%`.
- Marker-derived scales: `1.164240`, `1.157371`, and `1.157654`.
- Maximum scale-observation deviation: `1.606%` against the `5%` limit.
- Corrected marker camera-depth error: `0.020 m` median and `0.054 m` maximum.
- Fused world extent: approximately `(-0.370, 0.272, 0.000)` to
  `(2.841, 4.044, 2.001) m`.
- The living room, bed, floor/wall structure, and furniture are recognizable.
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
| Accepted run summary | `artifacts/s02/candidate_three_keyframes_20260731/summary.json` | Schema-validated model/input/filter/runtime provenance |
| Raw DA3 predictions | `artifacts/s02/candidate_three_keyframes_20260731/predictions/` | Depth, confidence, processed RGB, intrinsics, and poses |
| Retained keyframes | `artifacts/s02/candidate_three_keyframes_20260731/keyframes/` | Undistorted synchronized frame evidence |
| Camera A PLY | `artifacts/s02/candidate_three_keyframes_20260731/camera_a_static_scene.ply` | Camera A world-space geometry |
| Camera B PLY | `artifacts/s02/candidate_three_keyframes_20260731/camera_b_static_scene.ply` | Camera B world-space geometry |
| Fused PLY | `artifacts/s02/candidate_three_keyframes_20260731/static_scene.ply` | Accepted static scene |
| Geometry PNG | `artifacts/s02/candidate_three_keyframes_20260731/previews/static_scene_geometry.png` | Recognizability and camera-frame QA |
| Geometry GLB | `artifacts/s02/candidate_three_keyframes_20260731/previews/static_scene_with_cameras.glb` | Interactive general 3D preview |
| Rerun recording | `artifacts/s02/candidate_three_keyframes_20260731/static_scene_accepted_v2.rrd` | Digital Twin-style geometry and camera view |
| Rerun screenshot | `artifacts/s02/candidate_three_keyframes_20260731/previews/rerun_static_scene_accepted.png` | Accepted viewer evidence |
| Verification report | `artifacts/s02/candidate_three_keyframes_20260731/verification.json` | Hashes, overlap metrics, extents, and automated gate checks |

The first raw metric diagnostic and the corrected single-pair diagnostic remain
separately retained under `artifacts/s02/first_calibrated_20260731/` and
`artifacts/s02/corrected_calibrated_20260731/`.

## Verification

### Commands

```text
.venv/bin/python scripts/s02/reconstruct_static_scene.py \
  --target-time-seconds 22 30 38 \
  --output-dir artifacts/s02/candidate_three_keyframes_20260731

.venv/bin/python scripts/s02/export_rerun_static_scene.py \
  --run-summary \
    artifacts/s02/candidate_three_keyframes_20260731/summary.json \
  --output \
    artifacts/s02/candidate_three_keyframes_20260731/static_scene_accepted_v2.rrd

.venv/bin/python scripts/s02/verify_static_scene.py

.venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/mypy --strict src scripts/s02
uv --cache-dir /private/tmp/spatial-reconstruction-uv-cache lock --check
uv --cache-dir /private/tmp/spatial-reconstruction-uv-cache sync --check
.venv/bin/rerun rrd print \
  artifacts/s02/candidate_three_keyframes_20260731/static_scene_accepted_v2.rrd
git diff --check
```

Artifact-producing commands refuse to overwrite existing accepted outputs. Use
a new output directory and Rerun filename for a reproduction run.

### Results

- `141` automated tests passed.
- Ruff passed.
- Strict mypy passed across `23` S02-relevant source/script files.
- Lockfile and installed environment checks passed.
- The accepted Rerun recording parsed successfully with `58` chunks.
- The verification report checked `16` hashed artifacts.
- All automated geometry, overlap, finite-data, room-bound, marker-scale, and
  Rerun-presence checks passed.
- Visual inspection confirmed recognizable geometry and correctly oriented
  camera frustums.
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
- The two cameras retain the bounded shared-intrinsic assumption from D021.
- Room bounds remain a conservative processing crop rather than surveyed
  surfaces.
- Rerun SDK is pinned to `0.22.1` for NumPy 1 compatibility. Its native
  `--screenshot-to` helper exceeds the Metal texture limit on this Retina
  configuration; normal native viewing, the localhost-only web viewer, and
  the `.rrd` recording are unaffected.

## Decisions Made

- D025 - marker-anchored scalar correction for S02 static depth.

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
- Stage-close commit: `Pending`
- Stage-close GitHub URL: `Pending`
- Annotated tag: `None`
- Remote push verified: `Pending`
- DA3 vendor fingerprint:
  `683cad1fec1186cd2a22f2b6d083b73d4c83c7ab1140f45ba24876612bc51d43`
- DA3 model revision:
  `b2359bdf726fb44ef62acca04d629dcf158053e7`

The final stage-close commit reference will be recorded in a separate
provenance-only documentation commit because a Git commit cannot contain its
own final hash.

## Exact Next Action

Stop after the S02 stage-close push. Begin S03 person/backpack perception only
after explicit user instruction.
