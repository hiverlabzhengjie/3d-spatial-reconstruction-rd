# S03 Person and Backpack Perception Record

**Stage:** S03 - Person and Backpack Perception

**Status:** Complete

**Started:** 2026-07-31

## Stage Goal

Detect, segment, and track the person and backpack independently in each
accepted action camera while preserving immutable capture identity, explicit
missing/occluded states, deterministic offline ordering, and bounded worker
behaviour.

## Entry Prerequisites

- S02 is complete and its D027 revision is remotely verified.
- Accepted dynamic input is `action_take_01`; the backup take is excluded from
  baseline processing.
- Both synchronized action videos retain `1,047` decoded frames and match the
  SHA-256 values in their synchronization manifest.
- All `1,047` S01 frame bundles are complete, with no missing camera and a
  maximum inter-camera timestamp difference of `6.667 ms`.
- Capture-specific pose version
  `s01_capture_20260729:action_take_01:v1` remains accepted for both cameras.
- The retained pair preview shows the person and backpack during pickup,
  movement, and placement in both cameras.
- The cached `yolov8n-seg.pt` checkpoint matches the S00 fingerprint
  `a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c`.
- Existing YOLO adapter, ingestion, and persistent-contract tests pass (`38`
  focused tests).

No physical input or calibration blocker is known at S03 entry.

## First Work Step - Representative Perception Preflight

Before adding ByteTrack or queue orchestration:

1. select a deterministic, action-spanning set of synchronized complete frame
   bundles from the accepted action pair;
2. run the exact baseline YOLO segmentation checkpoint on both camera frames;
3. retain source identities, raw boxes/confidences/masks, normalized masks,
   timings, and annotated previews;
4. summarize person and backpack detections by camera and action phase;
5. record explicit zero-detection results; and
6. use the evidence to set the initial S03 tracking interval and determine
   whether the backpack baseline is viable before considering the approved
   bottle fallback.

This preflight does not assign track IDs and does not use static or dynamic DA3
depth, Qwen semantics, or S02 confidence exceptions.

### Result

Completed on native Apple MPS using synchronized target times `1`, `7`, `13`,
`19`, `25`, and `31 s` in both cameras (`12` frames total).

- Exact checkpoint SHA-256:
  `a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c`.
- Configuration: `640 px`, confidence threshold `0.25`, float32, tracking off.
- Person detections: Camera A `3`; Camera B `5`.
- Backpack detections: Camera A `0`; Camera B `0`.
- The physical backpack was instead detected as `handbag` in two Camera A
  samples and as `suitcase` in one Camera A sample.
- Visual review confirmed that these alternate labels overlap the physical
  backpack. Other scene false positives include `keyboard`, `knife`, and
  `hot dog`.
- Median inference wall time across the twelve calls was `0.097 s`; the first
  cold call was the `2.413 s` maximum. These are entry observations, not the
  final S03 throughput measurement.
- Both source videos remained byte-identical to their accepted manifest hashes.
- The restricted-process attempt stopped before model loading because MPS was
  unavailable; no CPU fallback or inference occurred. The accepted run used a
  fresh artifact directory and native MPS.

Artifacts are retained under
`artifacts/s03/entry_preflight_native_20260731/`, including per-frame raw and
source-sized masks, boxes/classes/confidences, immutable frame identity,
annotated previews, a pair contact sheet, and `summary.json`.

The result is sufficient to continue evaluating the backpack baseline, but not
to declare it unusable from six sparse times. Do not silently relabel
`handbag` or `suitcase` as `backpack`; a denser diagnostic and an explicit
class/fallback decision are required first.

### Reproduction Command

```text
.venv/bin/python scripts/s03/run_perception_preflight.py \
  --output-dir artifacts/s03/entry_preflight_native_20260731
```

## Dense Backpack Viability Diagnostic

The unchanged checkpoint was run at every integer target time from `1-33 s`
in both accepted camera streams. The inference floor was lowered to `0.10` for
diagnosis only; the project default remains `0.25`. All 66 camera-frame results
retain vendor labels and confidences, so thresholds `0.10`, `0.15`, `0.20`,
and `0.25` can be assessed without rerunning the model.

At the baseline `0.25` threshold:

- Camera A detected a person in `26/33` sampled frames and Camera B in `31/33`;
  at least one camera detected the person in `31/33` synchronized bundles.
- True `backpack` labels occurred in `0` Camera A frames and only `2` Camera B
  frames, at approximately `26.012 s` and `28.012 s` near placement.
- Camera A produced `handbag` detections in `13` frames, predominantly while
  the same physical backpack was stationary on the bed and during pickup.
- Camera B produced `handbag` detections in `3` frames near the carry/place
  transition.
- Camera A `suitcase` detections were visually rejected as a large bed-region
  false positive. The vendor label is preserved in raw outputs.

At the diagnostic `0.10` floor, true `backpack` labels appeared in only five
frames: Camera A at approximately `14.007` and `17.007 s`, and Camera B at
`26.012`, `27.012`, and `28.012 s`. Only two of these five exceed `0.25`.
Lowering the threshold alone therefore does not make the literal `backpack`
class reliable across the intended sequence.

Visual review found that the Camera A `handbag` masks generally cover the
physical backpack cleanly during its stationary/pickup phase. Camera B changes
between `handbag` and `backpack` around placement, with useful object masks.
This supports a possible guarded bag-class alias policy, but does not authorize
one. `suitcase` cannot be globally aliased because it also masks most of the
bed in Camera A.

Generated evidence:

- `artifacts/s03/backpack_viability_dense_20260731/summary.json`;
- `artifacts/s03/backpack_viability_dense_20260731/viability_analysis.json`;
- per-frame detection JSON, masks, and annotated previews in the same folder;
- `annotated_pair_contact_sheet.jpg` covering all 33 synchronized times.

### Reproduction Commands

```text
.venv/bin/python scripts/s03/run_perception_preflight.py \
  --purpose dense_backpack_viability_diagnostic \
  --confidence-threshold 0.10 \
  --target-time-seconds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    20 21 22 23 24 25 26 27 28 29 30 31 32 33 \
  --output-dir artifacts/s03/backpack_viability_dense_20260731

.venv/bin/python scripts/s03/analyze_backpack_viability.py \
  --run-summary artifacts/s03/backpack_viability_dense_20260731/summary.json \
  --output artifacts/s03/backpack_viability_dense_20260731/viability_analysis.json
```

## Decisions

- D028 - guarded `backpack` plus `handbag` physical-bag candidate policy. The
  user approved this policy after dense diagnostic review. `suitcase` remains
  excluded, raw vendor labels remain unchanged, and ByteTrack continuity is
  required before accepting one camera-local backpack track.

## D028 Policy and ByteTrack Boundary Implementation

- Added typed perception configuration for policy
  `d028_guarded_backpack_handbag_v1`, allowed aliases `backpack` and
  `handbag`, and excluded class `suitcase`.
- Added immutable `PerceptionCandidate` records that retain the complete vendor
  `SegmentationDetection`, including its original class, confidence, mask
  reference, and optional camera-local track ID.
- Candidate selection maps `person` to the person target and only
  `backpack`/`handbag` to the canonical backpack target. Other detections are
  ignored without changing the retained raw YOLO result.
- Extended the project-owned YOLO adapter with a persistent
  `bytetrack.yaml` tracking call. One adapter instance binds permanently to one
  camera ID and rejects attempts to mix camera streams.
- Tracking requests are explicitly class-filtered, use persistent ByteTrack
  state, and require normalized non-negative integer track IDs when detections
  are present.
- Normalized raw arrays now include track IDs (`-1` only for ordinary
  non-tracking predictions); camera-local persistent IDs use the explicit form
  `<camera_id>:<vendor_track_id>`.
- Added tests for D028 configuration invariants, source-label preservation,
  `suitcase` exclusion, ByteTrack arguments, track-ID normalization, and
  cross-camera tracker rejection.

This boundary does not yet prove track continuity on the accepted action
recording. A native-MPS representative sequence smoke check is the next gate.

## Camera-Local ByteTrack Smoke - 5 FPS

Ultralytics ByteTrack required the assignment package `lap`, which was absent
from the environment. Added and locked baseline runtime dependency
`lap==0.5.13`; no tracking method or model changed.

The accepted smoke processed 160 synchronized bundles from approximately
`1.0-32.8 s`, every sixth frame (nominal 5 FPS), independently for each camera.
It used native MPS, the exact S00 YOLO checkpoint, threshold `0.25`, class IDs
for `person`, `backpack`, and `handbag`, and `bytetrack.yaml` with persistent
per-camera state.

Camera A:

- processed frames: `160`;
- person-candidate frames: `122`, including `5` candidates not yet assigned a
  track ID;
- backpack-candidate frames: `37`, including `2` untracked candidates;
- primary stationary/pickup backpack track `camera_a:13`: `32` observations
  from `6.803-13.607 s`, vendor class `handbag`, with a longest consecutive run
  of `24` processed frames;
- later backpack track `camera_a:22`: `3` observations from
  `15.407-15.807 s`;
- no frame contained multiple simultaneous backpack candidates.

Camera B:

- processed frames: `160`;
- person track `camera_b:1`: `152` observations from `1.000-32.813 s`, with a
  longest consecutive run of `57` processed frames;
- backpack-candidate frames: `14`, including `1` untracked candidate;
- placement backpack track `camera_b:17`: `8` observations from
  `26.012-28.612 s`; its preserved vendor classes include both `handbag` and
  `backpack`, demonstrating that D028 can keep one ByteTrack identity across a
  class-label change;
- two shorter `handbag` tracks cover parts of `22.210-24.410 s`;
- no frame contained multiple simultaneous backpack candidates.

The first native attempt stopped when a valid detection had no ByteTrack ID.
This is a legitimate tentative-tracker boundary, not malformed detector data.
The accepted implementation now preserves such detections explicitly with a
null camera-local track ID and counts them as untracked candidates; they cannot
contribute to confirmed track continuity until ByteTrack assigns an ID.

Visual review of the retained contact sheet confirmed sensible person and bag
masks, separated camera-local IDs, the stationary bed-side backpack, pickup and
carry views, and the placed backpack. Raw source hashes remained unchanged.

Accepted artifacts are under
`artifacts/s03/tracking_smoke_5fps_v2_20260731/`, including:

- `summary.json`;
- per-camera `frame_results.json`;
- per-frame source-sized and raw masks, boxes, class IDs, confidences, and track
  IDs;
- representative annotated frames; and
- `tracking_preview_contact_sheet.jpg`.

### Reproduction Command

```text
.venv/bin/python scripts/s03/run_tracking_smoke.py \
  --output-dir artifacts/s03/tracking_smoke_5fps_v2_20260731
```

### Verification

- All 320 selected camera frames preserve unique immutable frame identities in
  strict capture-time order.
- Every raw/mask artifact exists and its array counts match the persistent
  detection records.
- D028 candidates contain only `person`, `backpack`, or `handbag`; `suitcase`
  is absent.
- Explicit no-candidate and untracked-candidate frames are retained.
- `146` project tests passed.
- Ruff and strict mypy passed.
- Lockfile and installed environment checks passed with 128 resolved and 106
  installed packages.
- Artifact validation and `git diff --check` passed.

The smoke proves that the baseline person and backpack can be tracked in
representative segments. It does not yet establish one uninterrupted backpack
identity across expected occlusion, nor the bounded worker/overload behaviour
required by the S03 completion gate.

## Five-FPS Cadence and Representative Object Boundary

D029 records the user-approved decision not to run the proposed continuous
10 FPS comparison. The two camera streams required approximately `30.01 s` of
summed inference time across about `31.81 s` of accepted capture interval at
5 FPS per camera, already consuming roughly 94% of a serialized real-time
inference budget before pipeline overhead. Continuous 10 FPS per camera is not
a credible single-M1 live default.

The backpack remains the S03 demonstration object and its missing/fragmented
states must remain honest. However, high-quality backpack classification is not
the core long-term value of this proof of concept, and an eventual application
may use different object categories. Work therefore proceeds to bounded worker,
3D observation, event-state, and Digital Twin integration rather than further
heavy cadence experiments or object-specific optimization.

## Bounded Perception Worker Contracts

Added a project-owned bounded worker runtime under
`src/spatial_reconstruction/perception/` with:

- deterministic job IDs derived from immutable frame identity, model identity,
  model revision, D028 policy, attempt, and priority;
- separate recorded processing-creation time so timing diagnostics do not make
  replay job identity nondeterministic;
- immutable RGB work items whose dimensions must match their frame identity;
- a one-camera-per-queue rule and strictly increasing source-frame/capture-time
  submissions;
- FIFO processing independent of model completion metadata;
- offline `throttle` overflow behavior that preserves the full queue and drops
  nothing;
- future-live `drop_oldest` behavior that explicitly reports the exact dropped
  job ID;
- accepted, popped, completed, failed, throttled, dropped, cancelled, pending,
  and in-flight diagnostics;
- explicit completed results with zero candidates for valid missing detections;
- explicit failed results with the original job identity and error details but
  no fabricated candidates or model artifacts;
- rejection of processor candidates that reference a different source frame;
  and
- explicit cancellation/drain accounting for pending shutdown work.

Eight focused worker tests cover stable/tamper-evident identity, immutable
pixels, offline saturation, future-live dropping, FIFO order, duplicate/mixed/
non-monotonic rejection, empty successful results, model failure, wrong-frame
output, and cancellation.

Full verification after this addition:

- `154` project tests passed;
- Ruff passed across `src`, `tests`, and `scripts`;
- strict mypy passed across `25` source and S03 script files;
- the 128-package lockfile and 106-package installed environment are
  consistent; and
- `git diff --check` passed.

The generic bounded runtime is verified, but it has not yet executed the real
D028 YOLO/ByteTrack processor through the queue. That integrated deterministic
replay is the next S03 step.

## Integrated Bounded Five-FPS Replay

Added `YOLOByteTrackProcessor`, which runs the real camera-local tracker,
normalizes D028 candidates, preserves original vendor labels and track IDs, and
writes source-sized masks plus raw masks, boxes, class IDs, confidences, track
IDs, detection JSON, job identity, and native model timing for every processed
frame.

The accepted replay used two independent capacity-eight offline queues over the
same 160 synchronized 5 FPS bundles as the tracking smoke. The producer
deliberately filled each queue; every later full-queue submission returned
`throttle_required`, drained the earliest job, and retried the same unchanged
job. No work was silently discarded.

Both cameras independently recorded:

- accepted jobs: `160`;
- popped jobs: `160`;
- completed results: `160`;
- failed results: `0`;
- explicit throttle events: `152`;
- dropped-oldest jobs: `0`;
- cancelled jobs: `0`;
- final pending depth: `0`;
- final in-flight count: `0`;
- stable job identities when recreated with different processing-creation
  times: passed; and
- result order exactly matching authoritative capture order: passed.

Camera A produced person candidates in `122` frames, backpack candidates in
`37`, and completely empty candidate results in `17`. Camera B produced person
candidates in `155` frames, backpack candidates in `14`, and empty results in
`5`. Unassigned ByteTrack candidates remain explicit: Camera A person `5` and
backpack `2`; Camera B person `3` and backpack `1`.

Median per-job processing duration, including raw artifact persistence, was
`0.119 s` for Camera A and `0.122 s` for Camera B. Total processing duration was
`20.487 s` and `19.651 s`. These measurements reinforce D029: this deterministic
offline implementation is not evidence that two continuous 5 FPS streams plus
all downstream work fit a live single-M1 latency budget. The observed median
queue waits of about `1.48-1.51 s` result from the intentional capacity-eight
burst/throttle stress pattern and are not represented as production latency.

Artifacts are retained under
`artifacts/s03/bounded_replay_5fps_20260731/`:

- `summary.json`;
- per-camera `worker_results.json` and `queue_submissions.json`; and
- 320 pairs of raw NPZ and detection JSON artifacts under the per-camera `raw/`
  directories.

### Reproduction Command

```text
.venv/bin/python scripts/s03/run_bounded_perception_replay.py \
  --output-dir artifacts/s03/bounded_replay_5fps_20260731
```

### Verification

- All 320 accepted jobs have unique stable IDs and immutable frame provenance.
- All results are completed, source-ordered, and tied to their exact job.
- All 640 referenced raw/detection artifacts exist; array/detection counts
  agree and embedded job/candidate records match worker results.
- Both unchanged source hashes match the accepted synchronization manifest.
- Offline overload produced only explicit throttling and no drops.
- `155` project tests passed.
- Ruff, strict mypy across `27` source/S03 script files, lockfile/environment,
  artifact validation, and `git diff --check` passed.

The bounded-worker portion of the S03 gate passes.

## Explicit Per-Target Timeline

Added immutable per-target frame records with five honest image-plane states:
`observed`, `untracked`, `ambiguous`, `missing`, and `failed`. An observed state
requires exactly one candidate with a camera-local ByteTrack ID; an untracked
state requires exactly one candidate without an assigned ID; multiple
candidates become ambiguous; and a completed result without a candidate is
missing. Worker errors become failed states for both targets with their error
details. No state fabricates a track continuation.

Every selected candidate retains its original confidence, vendor label, box,
mask reference, and camera-local track ID. The derived record adds source-sized
mask area in pixels and as an image fraction, plus `fully_in_frame` or
`frame_edge_truncated` visibility based on mask/box contact with the source
image boundary. The derivation validates the retained mask dimensions and
detection indices before writing a state.

The accepted derivation reused the bounded replay artifacts and performed no
model inference:

- Camera A person: `117` observed, `5` untracked, `38` missing, `0` ambiguous,
  and `0` failed frames. Of its `122` candidates, `39` touch a frame edge.
- Camera B person: `152` observed, `3` untracked, `5` missing, `0` ambiguous,
  and `0` failed frames. Of its `155` candidates, `5` touch a frame edge.
- Camera A backpack: `35` observed, `2` untracked, `123` missing, `0`
  ambiguous, and `0` failed frames. All `37` candidates retain the vendor
  label `handbag` and are fully in frame.
- Camera B backpack: `13` observed, `1` untracked, `146` missing, `0`
  ambiguous, and `0` failed frames. Its `14` fully-in-frame candidates retain
  seven `handbag` and seven `backpack` labels.
- At least one camera has an observed person in `154/160` synchronized
  processed frames and a person candidate in `156/160`.
- At least one camera has an observed backpack in `48/160` frames and a bag
  candidate in `51/160`; both cameras are missing the bag in `109/160`.

The longest important two-camera backpack absence spans Camera A's missing
interval beginning at approximately `17.207 s` through Camera B's first
untracked candidate at approximately `22.005 s`. It is recorded as missing,
not automatically labelled occluded: image-plane absence alone does not prove
the physical cause. S04 may use synchronized depth and multi-view evidence but
must not convert these missing frames into fabricated coordinates.

### Post-close D037 clarification (2026-08-03)

The retained S03 result remains correct as detector evidence: `missing` means
no selected bag candidate and does not establish why. The derivation contract
now states explicitly that missing detections require separate synchronized-
video visibility evidence and that neither visibility nor occlusion evidence
may supply XYZ. S05 owns a versioned overlay rather than rewriting these
closed S03 artifacts. For frames `468-660`, that overlay records the user's
affirmative synchronized-video review as `partially_occluded` while preserving
both cameras' original missing/observed/untracked detector states.

Artifacts are retained under
`artifacts/s03/target_timeline_5fps_20260801/`:

- `summary.json`, including state/visibility/track/label/mask-area counts and
  contiguous state intervals;
- `camera_a_target_timeline.json`; and
- `camera_b_target_timeline.json`.

### Reproduction Command

```text
.venv/bin/python scripts/s03/derive_perception_timeline.py \
  --bounded-replay-summary \
    artifacts/s03/bounded_replay_5fps_20260731/summary.json \
  --output-dir artifacts/s03/target_timeline_5fps_<new-run-id>
```

The command refuses to overwrite an existing output directory.

### Final Verification

- `162` project tests passed, including seven focused target-timeline tests
  covering observed mask metrics, frame-edge visibility, untracked,
  ambiguous, missing, failed, invalid-mask-shape, invalid-mask-type, and
  invalid-index behaviour.
- Ruff passed across `src`, `tests`, and `scripts`.
- Strict mypy passed across `29` source and S03 script files.
- The lockfile resolves `128` packages, the environment has the expected `106`
  installed packages, and synchronization would make no changes.
- Timeline derivation schema-validated all `320` worker results and read all
  `320` retained raw mask artifacts before producing `640` target records.
- Source videos remain byte-identical to their accepted hashes.
- `git diff --check` passed.

## Completion Gate

- Person tracked through a representative sequence: **passed**. Camera B's
  primary track contains `152/160` observations from approximately
  `1.0-32.8 s`; missing/untracked frames remain explicit.
- Bounded/deterministic worker with source identity and explicit overload or
  failure behaviour: **passed**. Both queues completed all `160` accepted jobs
  in capture order with explicit throttling and zero silent drops.
- Backpack detected while stationary and during at least part of movement:
  **passed**. Camera A track `camera_a:13` covers `32` stationary/pickup
  observations; Camera B track `camera_b:17` covers eight placement-phase
  observations across preserved `handbag`/`backpack` labels.
- Occluded or missing backpack observations represented explicitly:
  **passed** as missing/untracked intervals. No unproven occlusion cause or XYZ
  is invented.
- Bottle fallback: **not activated** because the guarded D028 bag evidence is
  usable for the representative stationary, pickup, movement, and placement
  segments required by this proof of concept.

The S03 completion gate is met without weakening it and without the rejected
continuous 10 FPS experiment.

## Exact Next Action

Stop. Begin S04 only after explicit user instruction. Its first step is to run
pose-conditioned DA3 metric depth on selected synchronized action frames that
contain the retained person/backpack masks, then back-project only valid mask
depth into raw per-camera world observations without inheriting S02's static
depth scale correction.
