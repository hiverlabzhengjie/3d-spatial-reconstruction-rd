# S02 DA3 Static Room Geometry Record

**Stage:** S02 - DA3 Static Room Geometry

**Status:** Complete

**Started:** 2026-07-31

## Stage Goal

Generate recognizable static living-room geometry from accepted synchronized
empty-room frames using pose-conditioned DA3 metric depth, the calibrated
shared world frame, confidence filtering, conservative room-bound filtering,
and deterministic point-cloud fusion.

## Accepted Inputs

- Synchronization manifest:
  `artifacts/s01/empty_room/synchronized/synchronization_manifest.json`
- Stable synchronized interval: `22.0-38.0 s`
- Pose version: `s01_capture_20260729:empty_room:v1`
- Scene bounds: `(-0.5, -0.5, 0.0)` to `(3.0, 4.5, 3.0) m`
- Model: `depth-anything/DA3NESTED-GIANT-LARGE-1.1`
- Model revision: `b2359bdf726fb44ef62acca04d629dcf158053e7`
- DA3 vendor fingerprint:
  `683cad1fec1186cd2a22f2b6d083b73d4c83c7ab1140f45ba24876612bc51d43`
- Initial process resolution: `504`

## First Calibrated Run

The first run selected the complete synchronized bundle nearest `30.0 s`:

- bundle index: `900`
- bundle timestamp: `30.013333 s`
- inter-camera difference: `3.333 ms`
- device: native Apple MPS
- precision: float16 autocast
- model load: `15.206 s`
- pair inference: `2.018 s`

Raw depth and confidence were coherent and finite. The confidence threshold
used DA3's documented `40th` percentile convention. After confidence,
room-bound, and `0.02 m` voxel filtering, the uncorrected run contained:

- Camera A: `22,875` points
- Camera B: `14,960` points
- fused: `35,114` points

The room, bed, floor, walls, and furniture were recognizable, and the
calibrated cameras pointed into the reconstructed room.

## Metric-Diagnostic Finding

The raw run is retained at:

`artifacts/s02/first_calibrated_20260731/`

It is diagnostic rather than accepted final geometry. Both cameras placed the
known floor about `0.23 m` above world `Z=0`. Sampling DA3 depth at the
projected accepted M40-M42 centres produced expected/predicted depth ratios:

- Camera A: `1.154`, `1.161`, `1.142`
- Camera B: `1.164`, `1.164`, `1.139`

This consistent two-view bias motivated D025, a bounded shared
marker-anchored scalar correction for derived S02 geometry. Raw DA3 outputs
remain unchanged.

## Accepted Three-Keyframe Reconstruction

The accepted run selected the complete synchronized bundles nearest `22.0`,
`30.0`, and `38.0 s`:

| Target | Selected time | Bundle index | Maximum pair difference |
|---|---:|---:|---:|
| `22.0 s` | `22.010000 s` | `660` | `3.333 ms` |
| `30.0 s` | `30.013333 s` | `900` | `3.333 ms` |
| `38.0 s` | `37.983333 s` | `1139` | `3.333 ms` |

The model loaded once on native Apple MPS in `7.424 s`. Pair inference took
`1.638 s`, `1.374 s`, and `1.357 s`; there was no CPU fallback. Raw depth,
confidence, processed RGB, immutable source identities, timestamps, and
retained keyframes remain inspectable for every prediction.

D025 produced one shared two-camera scale per bundle:

- `22.010000 s`: `1.164240`;
- `30.013333 s`: `1.157371`;
- `37.983333 s`: `1.157654`.

The scale range across the stable interval is `0.006870`. The maximum
single-observation relative deviation is `1.606%`, below the `5%` rejection
limit. Across all eighteen M40-M42 observations, corrected marker camera-depth
error is `0.020 m` median and `0.054 m` maximum.

After confidence, finite-depth, conservative room-bound, and `0.02 m` voxel
filtering, the accepted point clouds contain:

- Camera A: `30,239` points;
- Camera B: `22,332` points;
- fused: `45,919` points.

The fused extent is approximately `(-0.370, 0.272, 0.000)` to
`(2.841, 4.044, 2.001) m`, wholly inside the declared processing bounds.

## Cross-Camera and Visual QA

An exact spatial-hash radius check on the retained per-camera PLY files found:

- `69.295%` of Camera A points within `0.10 m` of Camera B;
- `86.638%` of Camera B points within `0.10 m` of Camera A.

Both exceed the recorded `65%` minimum for shared visible surfaces. The
asymmetry is expected because Camera A sees surfaces outside Camera B's useful
coverage; this metric is overlap, not full-scene completeness.

The geometry PNG and GLB show a recognizable living room with the bed,
floor/wall structure, and furniture in one Z-up metric frame. The Rerun
recording displays the fused and sampled per-camera clouds with both calibrated
camera transforms/pinholes, representative images, M40-M42, processing bounds,
and pickup/drop-off rings. The accepted local web-viewer capture rendered
without errors. Rerun 0.22.1's native `--screenshot-to` path triggered a
Retina/Metal off-screen texture-limit error; this did not affect the normal
native viewer, local web viewer, `.rrd` structure, or geometry. The diagnostic
capture therefore uses Rerun's localhost-only web viewer.

## Generated Artifacts

The accepted artifact directory is:

`artifacts/s02/candidate_three_keyframes_20260731/`

Important outputs:

- `summary.json` - schema-validated run, source, model, filtering, timing, and
  artifact provenance;
- `predictions/*.npz` - raw DA3 depth/confidence, processed RGB, intrinsics,
  and poses;
- `keyframes/*.png` - retained undistorted synchronized source frames;
- `camera_a_static_scene.ply` and `camera_b_static_scene.ply`;
- `static_scene.ply` - accepted fused geometry;
- `previews/static_scene_geometry.png`;
- `previews/static_scene_with_cameras.glb`;
- `static_scene_accepted_v2.rrd`;
- `static_scene_accepted_v2_export_summary.json`;
- `previews/rerun_static_scene_accepted.png`;
- `verification.json` - schema-validated hashes and automated gate evidence.

The raw uncorrected diagnostic remains under
`artifacts/s02/first_calibrated_20260731/`; the corrected single-pair
diagnostic remains under `artifacts/s02/corrected_calibrated_20260731/`.

## Reproduction Commands

```text
.venv/bin/python scripts/s02/reconstruct_static_scene.py \
  --target-time-seconds 22 30 38 \
  --output-dir artifacts/s02/candidate_three_keyframes_20260731

.venv/bin/python scripts/s02/export_rerun_static_scene.py \
  --run-summary \
    artifacts/s02/candidate_three_keyframes_20260731/summary.json \
  --output \
    artifacts/s02/candidate_three_keyframes_20260731/static_scene_accepted_v2.rrd

.venv/bin/rerun \
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

Artifact-producing commands refuse to overwrite an existing output directory
or Rerun recording. Use a new output path when reproducing the run.

## Final Verification

- `141` automated tests passed.
- Ruff passed across `src`, `tests`, and `scripts`.
- Strict mypy passed across `23` S02-relevant source/script files.
- Lockfile and installed-environment consistency checks passed.
- The accepted Rerun recording parsed successfully with `58` chunks.
- The verification script checked `16` hashed inputs/outputs and all automated
  gate checks passed.
- Transform, projection, invalid-depth, confidence, bounds, missing-data,
  deterministic selection, voxel fusion, PLY, persistent-summary schema, and
  overlap failure behaviours are covered.
- The accepted heavy run used the exact model/vendor identities and native MPS
  without CPU fallback.
- No optional COLMAP, SfM, MVS, stereo, triangulation, or Gaussian method ran.

## Completion Gate

- Point cloud recognizable as the living room: **passed** by inspection of the
  geometry and Rerun previews.
- Both cameras occupy one plausibly aligned world frame: **passed** by visual
  inspection and bidirectional `0.10 m` overlap.
- Invalid and out-of-room points filtered: **passed** by tests, recorded
  filtering counts, finite checks, and accepted fused extent.
- Camera poses and point cloud displayed together correctly: **passed** in the
  accepted Rerun recording and retained diagnostic capture.

## Limitations

- This remains exploratory geometry, not survey-grade reconstruction.
- Thin, reflective, texture-poor, occluded, and view-exclusive surfaces remain
  incomplete or noisy.
- The D025 scalar is authorized only for derived S02 static geometry; it does
  not alter raw DA3 output and must not be reused for S04 dynamic localization.
- Camera intrinsics remain the bounded shared estimate accepted under D021.
- Room bounds are a conservative crop rather than measured wall/ceiling
  surfaces.
- Rerun is pinned to `0.22.1` because later available SDK lines require NumPy
  2 while this project's Open3D-compatible environment remains on NumPy 1.

## Exact Next Action

Create the S02 stage handoff, close and push the stage, then stop. S03 must not
begin until the user explicitly requests it.
