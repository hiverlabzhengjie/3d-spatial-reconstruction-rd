# S03 Handoff - Person and Backpack Perception

**Stage:** S03 - Person and Backpack Perception

**Status:** Complete with known backpack-detection limitation

**Started:** 2026-07-31

**Closed:** 2026-08-01

## Stage Goal

Detect, segment, and track the person and backpack independently in each
accepted action camera, preserving exact source-frame identity, confidence,
mask area, visibility, camera-local track IDs, explicit missing/failure state,
and deterministic bounded-worker behaviour.

## Entry Inputs

- Completed S01 synchronization/calibration and completed S02 static geometry.
- Accepted preferred action pair `action_take_01`, with `1,047` complete
  synchronized frame bundles and no missing camera frames.
- Action pose version `s01_capture_20260729:action_take_01:v1`.
- Exact `yolov8n-seg.pt` checkpoint SHA-256
  `a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c`.
- Baseline confidence threshold `0.25` and source videos at 1920x1080,
  approximately 30 FPS.
- Existing immutable frame identities, file ingestion, YOLO normalization, and
  project configuration contracts.

## Work Completed

- Ran sparse and dense synchronized-frame perception diagnostics with retained
  source identities, raw/normalized masks, boxes, classes, confidences,
  timings, and annotated previews.
- Confirmed that literal `backpack` labels alone are unusable across the
  intended action, while `handbag` often masks the same physical bag and
  `suitcase` produces a Camera A bed-region false positive.
- Adopted D028: select only vendor `backpack` and `handbag` as candidates for
  the one canonical physical backpack while preserving every original label
  and excluding `suitcase`.
- Added persistent camera-local ByteTrack calls, one tracker instance per
  camera, normalized track IDs, and explicit null IDs for tentative detections.
- Added and locked `lap==0.5.13`, required by Ultralytics ByteTrack assignment.
- Ran the approved nominal 5 FPS camera-local tracking smoke over `160`
  synchronized bundles per camera.
- Adopted D029: do not run a continuous 10 FPS comparison; keep the backpack as
  a representative proof-of-concept object and prioritize the end-to-end
  spatial/event pipeline.
- Added immutable perception jobs, bounded FIFO queues, offline throttling,
  future-live drop-oldest accounting, failure conversion, cancellation, stable
  capture-derived job IDs, and strict source-order/camera isolation.
- Ran the real D028 tracker through independent capacity-eight queues for both
  cameras and retained every model result/artifact.
- Added typed per-target observed, untracked, ambiguous, missing, and failed
  states with source-mask area and frame-edge visibility.
- Derived camera timelines and contiguous state intervals without rerunning
  inference or guessing occlusion.

## Accepted Result

- Camera B person track `camera_b:1`: `152/160` observations from approximately
  `1.0-32.8 s`.
- Camera A backpack track `camera_a:13`: `32` stationary/pickup observations
  from approximately `6.8-13.6 s`.
- Camera B backpack track `camera_b:17`: eight placement-phase observations
  from approximately `26.0-28.6 s`, retaining both `handbag` and `backpack`
  vendor labels under one ByteTrack identity.
- Each bounded queue: `160` accepted, popped, and completed jobs; `152`
  explicit throttle-and-drain events; zero drops, failures, or cancellations;
  zero final pending/in-flight work.
- Camera A target states:
  - person: `117` observed, `5` untracked, `38` missing;
  - backpack: `35` observed, `2` untracked, `123` missing.
- Camera B target states:
  - person: `152` observed, `3` untracked, `5` missing;
  - backpack: `13` observed, `1` untracked, `146` missing.
- Both cameras: zero ambiguous and zero failed target states.
- Cross-camera availability:
  - person observed in at least one camera in `154/160` frames;
  - backpack observed in at least one camera in `48/160` frames;
  - both cameras missing the backpack in `109/160` frames.
- The main two-camera backpack absence from approximately `17.2-22.0 s` is
  persisted as missing. Occlusion is not inferred from absence and no track or
  position is fabricated.
- Bottle fallback not activated because D028 provides usable stationary,
  pickup, movement, and placement segments for the proof-of-concept gate.

## Changed Files

Important project-owned additions and updates include:

- `configs/default.yaml`;
- `src/spatial_reconstruction/config.py`;
- `src/spatial_reconstruction/contracts.py`;
- `src/spatial_reconstruction/models/yolo_adapter.py`;
- `src/spatial_reconstruction/models/__init__.py`;
- `src/spatial_reconstruction/perception/`;
- `scripts/s03/`;
- `tests/test_config.py`;
- `tests/test_yolo_adapter.py`;
- `tests/test_perception_worker.py`;
- `tests/test_perception_timeline.py`;
- `pyproject.toml` and `uv.lock`;
- `docs/DECISIONS.md`;
- `docs/STATUS.md`;
- `docs/stages/S03_PERSON_BACKPACK_PERCEPTION.md`; and
- this handoff.

Raw captures, model weights, generated perception artifacts, environments, and
caches remain excluded from Git.

## Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Sparse entry preflight | `artifacts/s03/entry_preflight_native_20260731/` | Representative source-preserving YOLO diagnostic |
| Dense viability diagnostic | `artifacts/s03/backpack_viability_dense_20260731/` | Threshold and vendor-class evidence for D028 |
| Tracking smoke | `artifacts/s03/tracking_smoke_5fps_v2_20260731/` | Camera-local ByteTrack tracks, raw outputs, and annotated preview |
| Tracking contact sheet | `artifacts/s03/tracking_smoke_5fps_v2_20260731/tracking_preview_contact_sheet.jpg` | Visual person/bag mask and track-ID QA |
| Bounded replay | `artifacts/s03/bounded_replay_5fps_20260731/` | Complete job, queue, mask, detection, and timing provenance |
| Explicit target timelines | `artifacts/s03/target_timeline_5fps_20260801/` | Typed per-camera states, mask metrics, visibility, and missing intervals |

## Verification

### Commands

```text
.venv/bin/python scripts/s03/run_perception_preflight.py \
  --output-dir artifacts/s03/entry_preflight_native_<new-run-id>

.venv/bin/python scripts/s03/run_perception_preflight.py \
  --purpose dense_backpack_viability_diagnostic \
  --confidence-threshold 0.10 \
  --target-time-seconds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    20 21 22 23 24 25 26 27 28 29 30 31 32 33 \
  --output-dir artifacts/s03/backpack_viability_dense_<new-run-id>

.venv/bin/python scripts/s03/analyze_backpack_viability.py \
  --run-summary \
    artifacts/s03/backpack_viability_dense_<new-run-id>/summary.json \
  --output \
    artifacts/s03/backpack_viability_dense_<new-run-id>/viability_analysis.json

.venv/bin/python scripts/s03/run_tracking_smoke.py \
  --output-dir artifacts/s03/tracking_smoke_5fps_<new-run-id>

.venv/bin/python scripts/s03/run_bounded_perception_replay.py \
  --output-dir artifacts/s03/bounded_replay_5fps_<new-run-id>

.venv/bin/python scripts/s03/derive_perception_timeline.py \
  --bounded-replay-summary \
    artifacts/s03/bounded_replay_5fps_<new-run-id>/summary.json \
  --output-dir artifacts/s03/target_timeline_5fps_<new-run-id>

.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/mypy --strict src/spatial_reconstruction \
  scripts/s03/analyze_backpack_viability.py \
  scripts/s03/derive_perception_timeline.py \
  scripts/s03/run_bounded_perception_replay.py \
  scripts/s03/run_perception_preflight.py \
  scripts/s03/run_tracking_smoke.py
uv --cache-dir .cache/uv lock --check
uv --cache-dir .cache/uv sync --check
git diff --check
```

Artifact-producing commands refuse to overwrite existing outputs. Native model
commands require Apple MPS; the timeline derivation itself performs no model
inference.

### Results

- `162` automated tests passed.
- Ruff passed across source, tests, and scripts.
- Strict mypy passed across `29` source/S03 script files.
- The lockfile resolved `128` packages; the installed `106` packages are
  current and synchronization would make no changes.
- All `320` bounded worker results retained unique, stable job IDs in exact
  camera capture order.
- All `640` bounded raw/detection artifacts exist and their array/detection/job
  content was validated during the accepted replay.
- Timeline derivation validated and read all `320` retained source-mask NPZs
  before writing `640` target states.
- Raw source SHA-256 values remain unchanged:
  - Camera A: `1e7064fa2d4911dcf2ac82803dd95fa5b9ece332906589c0f8627232bb526136`;
  - Camera B: `da5bd4eeaeac0da78cc71f14a43326d5d60c5c216f2609c85553cba720e40d5a`.
- Visual inspection confirmed sensible person and bag masks, distinct
  camera-local IDs, the stationary/pickup bag, carry/placement views, and
  preserved label transitions.
- No completion-gate check was skipped or weakened.

## Completion Gate

- Person tracked through a representative sequence: **passed**.
- Bounded/deterministic worker preserves exact source identity and reports
  overload/failure behaviour explicitly: **passed**.
- Backpack detected while stationary and during at least part of movement:
  **passed**.
- Occluded or missing backpack observations explicitly represented:
  **passed as missing/untracked intervals without unproven occlusion labels**.
- Bottle fallback if backpack evidence is unusable: **not applicable; D028
  evidence is usable for the representative gate**.

## Physical Setup and Observations

- No physical camera, lens, marker, mount, or room change was made in S03.
- Only the accepted synchronized `action_take_01` derived videos were read;
  raw and derived recordings were not modified.
- The fixed camera/lens provenance and action pose version remain required.
- Any physical camera or selected-lens movement invalidates the applicable
  pose calibration for future captures.

## Problems and Limitations

- Backpack detections are sparse and fragmented. S03 proves representative
  segments, not uninterrupted single-object identity across the full action.
- Image-plane absence does not distinguish occlusion, detector failure,
  viewpoint loss, or physical absence; the timeline therefore says missing.
- The Camera A and Camera B bag track IDs are local and must never be compared
  as global identities without later geometric fusion.
- Camera A person tracking fragments into several IDs, while Camera B provides
  the accepted representative person track.
- D028 is specific to the known one-bag demonstration. It is not a general
  multi-bag re-identification policy.
- The intentional capacity-eight offline burst produces roughly `1.5 s`
  median queue waits; this is overload/backpressure evidence, not live latency.
- Two-camera 5 FPS serialized perception already consumes most of one M1 Max
  real-time compute budget before downstream DA3/Qwen work. Production live
  scheduling remains future scope.
- No custom detector training, bottle fallback, multi-person re-ID, continuous
  10 FPS run, or production RTSP deployment was introduced.

## Decisions Made

- D028 - guarded backpack and handbag perception policy.
- D029 - five-FPS perception cadence and representative object boundary.

No optional model, custom detector, or non-baseline perception method was
activated.

## Prerequisites for S04

Software:

- Preserve immutable S01 frame identities, action pose version, D028 target
  selection, raw masks, confidences, and explicit target states.
- Use pose-conditioned multi-view DA3 metric depth on synchronized action
  frames containing dynamic entities; empty-room depth is not a substitute.
- Back-project only finite positive mask depth passing confidence filters.
- Keep raw per-camera observations separate from smoothed/fused presentation
  state and never fabricate XYZ for missing/untracked frames.
- Do not apply S02's D025 static marker scale, D026 global percentile, or D027
  door supplement to S04 dynamic depth without a new explicit decision and
  evidence.
- Retain explicit `T_world_from_camera` / `T_camera_from_world` naming and rerun
  transform round-trip, reprojection, back-projection, invalid-depth, and
  missing-data tests.

Inputs and physical state:

- Use the accepted synchronized preferred action pair and pose version
  `s01_capture_20260729:action_take_01:v1`.
- Use selected frames where retained person/backpack masks exist; do not infer
  object positions for the approximately `17.2-22.0 s` two-camera backpack
  gap.
- The fixed camera/lens setup and calibration provenance must remain valid.

## Version Control

- GitHub repository:
  `https://github.com/hiverlabzhengjie/3d-spatial-reconstruction-rd`
- Stage-close commit: pending creation after this handoff and final gate check.
- Stage-close GitHub URL: pending.
- Annotated tag: `None`.
- Remote push verified: pending.
- YOLO model SHA-256:
  `a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c`.
- DA3 vendor fingerprint required by S04:
  `683cad1fec1186cd2a22f2b6d083b73d4c83c7ab1140f45ba24876612bc51d43`.
- DA3 model revision required by S04:
  `b2359bdf726fb44ef62acca04d629dcf158053e7`.

## Exact Next Action

Stop. Begin S04 only after explicit user instruction. First select synchronized
action frames with retained masks, run pose-conditioned DA3 metric depth, and
back-project only valid mask depth into raw per-camera world observations.
