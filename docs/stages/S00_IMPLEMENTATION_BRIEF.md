# S00 Implementation Brief - Project Setup and MPS Model Gate

**Stage:** S00 - Project Setup and MPS Model Gate
**Stage state at authoring:** Ready to begin
**Document type:** Approved implementation plan; no implementation has been
performed by this document

## 1. Purpose

S00 establishes a small, reproducible Python 3.11 project around the existing
DA3 vendor checkout and proves that each approved heavy model can run
independently on the Apple M1 Max:

- `depth-anything/DA3NESTED-GIANT-LARGE-1.1` in two-view,
  pose-conditioned metric-depth mode
- Ultralytics `yolov8n-seg.pt`
- `Qwen/Qwen3-VL-2B-Instruct` on a small multi-frame input

The stage also establishes the minimum typed interfaces, coordinate utilities,
tests, configuration, and diagnostic conventions needed by later stages. S00
is a compatibility and reproducibility gate. It is not an attempt to reconstruct
the room, calibrate cameras, track an action, or interpret pickup and placement
events.

## 2. Boundaries

Work in S00 is limited to:

- creating a project-owned Python 3.11 environment and deterministic lockfile;
- adding a minimal project-owned package, configuration, adapters, smoke-test
  entry points, and tests;
- adapting device selection, precision, and unsupported-operation handling
  outside the DA3 vendor source;
- running the three models separately and retaining compact diagnostic outputs;
- observing cold/warm runtime and memory use;
- recommending a provisional DA3 processing resolution and offline keyframe
  interval for later validation;
- defining and testing coordinate and missing-data fundamentals without using
  them to localize real objects.

The existing vendor checkout must remain unmodified. It is located at
`Depth-Anything-3-main/`. The project configuration must make this path
explicit rather than copying or editing the checkout.

Project decision D015 permits supporting methodologies and tools when they
have a concrete, proportionate benefit and are highlighted to the user before
use. S00 still validates the exact baseline models; an auxiliary method cannot
silently replace one of its completion gates.

## 3. Entry Assumptions and Available Inputs

### Software and machine

- macOS on a MacBook Pro with M1 Max, 32-core integrated GPU, and 64 GB unified
  memory.
- Apple MPS is the intended accelerator.
- Python 3.11 is the required runtime, but no project-owned environment,
  package metadata, or lockfile currently exists.
- The required scientific and model packages are not assumed to be present in
  the system Python.
- The DA3 vendor source is available at `Depth-Anything-3-main/`.
- Vendor fixtures currently available:
  - `assets/examples/SOH/000.png`
  - `assets/examples/SOH/010.png`
  - `assets/examples/robot_unitree.mp4`
- Model weights must be downloaded during the later implementation turn unless
  already present in a local model cache. Weight availability is not assumed.

### Physical and scene inputs

- No representative living-room image is currently present.
- No synchronized living-room action sample is currently present.
- Phones, stable mounts, ChArUco board, floor markers, room axes, and zones are
  planned but not yet confirmed or defined.
- S00 does not depend on camera calibration or final capture. However, the
  YOLO completion-gate run requires one user-supplied representative
  living-room image, and the most representative Qwen check benefits from a
  short user-supplied action sample when available.

### Interpretation of the model gate

- Vendor media may establish import, device, input, output, and repeatability
  compatibility before living-room media exists.
- The two SOH images have no canonical project calibration. Provisional,
  explicitly labelled test intrinsics and two valid OpenCV
  `T_camera_from_world` matrices may therefore be used only to exercise DA3's
  pose-conditioned interface. Their results must not be described as calibrated
  geometry or accuracy evidence.
- A frame set extracted from the vendor video may exercise Qwen's multi-frame
  interface, but it is not evidence of pickup-carry-place understanding.
- The YOLO roadmap gate is not satisfied until `yolov8n-seg.pt` has run on a
  representative living-room image.

## 4. Exact S00 Deliverables

The later implementation turn must produce all of the following:

1. A project-owned Python 3.11 virtual environment, `pyproject.toml`, and
   deterministic lockfile.
2. A minimal importable package with Pydantic configuration and typed core
   contracts.
3. Project-owned adapters for DA3, YOLO, and Qwen3-VL 2B Instruct.
4. A DA3 MPS compatibility layer that does not modify files below
   `Depth-Anything-3-main/`.
5. Independent, directly runnable smoke checks for:
   - DA3 pose-conditioned two-view inference;
   - YOLOv8n segmentation on one representative image;
   - Qwen3-VL 2B Instruct on a small multi-frame input.
6. Machine-readable smoke summaries containing model identity, device,
   precision, input description, output shapes/status, cold/warm timings,
   memory observations, warnings, and fallback use.
7. Compact representative raw outputs sufficient to diagnose each smoke run,
   without storing unnecessary large tensors or altering source media.
8. A measured recommendation for DA3 processing resolution and a provisional
   offline keyframe interval, including the evidence and headroom rule used.
9. Initial automated tests for schemas, transform inversion and round trips,
   projection/back-projection, invalid depth/confidence handling, and missing
   camera/object observations.
10. One documented project test command that passes.
11. A dependency and model-licence record suitable for the project's
    non-commercial research constraint.
12. At stage close only: updated `docs/STATUS.md`, any necessary appended
    decision entries in `docs/DECISIONS.md`, and
    `docs/stages/S00_HANDOFF.md`.

## 5. Proposed Project-Owned Scaffold

The following is the planned scaffold. It is not created by this brief.

```text
.
├── pyproject.toml
├── uv.lock
├── README.md
├── configs/
│   └── default.yaml
├── src/
│   └── spatial_reconstruction/
│       ├── __init__.py
│       ├── config.py
│       ├── contracts.py
│       ├── geometry/
│       │   ├── __init__.py
│       │   ├── transforms.py
│       │   └── projection.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── da3_adapter.py
│       │   ├── da3_mps.py
│       │   ├── yolo_adapter.py
│       │   └── qwen3_vl_adapter.py
│       └── diagnostics/
│           ├── __init__.py
│           ├── memory.py
│           └── timing.py
├── scripts/
│   └── smoke/
│       ├── da3_two_view.py
│       ├── yolo_seg.py
│       └── qwen3_vl_multiframe.py
├── tests/
│   ├── test_contracts.py
│   ├── test_transforms.py
│   ├── test_projection.py
│   └── test_missing_data.py
├── artifacts/
│   └── s00/
│       ├── environment/
│       ├── da3/
│       ├── yolo/
│       └── qwen3_vl/
└── docs/
    ├── licences/
    │   └── MODEL_AND_LIBRARY_LICENCES.md
    └── stages/
        ├── S00_IMPLEMENTATION_BRIEF.md
        └── S00_HANDOFF.md        # created only after the gate passes
```

`configs/default.yaml` should hold paths, exact model identifiers, device and
precision policy, DA3 resolution candidates, confidence rules, and artifact
locations. It must not contain ambiguous `extrinsics` fields; persistent and
internal interfaces use `T_world_from_camera` and
`T_camera_from_world`. DA3/OpenCV naming and layout conversion remains inside
the DA3 adapter.

## 6. Ordered Implementation Work Packages

### WP1 - Environment and reproducibility

1. Confirm an arm64 Python 3.11 interpreter and the selected lock tool are
   available.
2. Create `.venv` at the project root and initialize project metadata.
3. Declare only the dependencies needed for S00 and approved later-stage
   foundations. Lock justified optional tools separately and do not install
   them in the main MPS environment unless the current work package needs
   them. Do not install optional DA3 Gaussian-splatting or web-app extras.
4. Resolve and commit the lockfile.
5. Record macOS, machine, architecture, Python, PyTorch, MPS availability, and
   key package versions.
6. Record applicable licences, with special attention to model
   non-commercial restrictions.

Acceptance: a fresh shell can invoke the project Python and import the local
package using documented commands.

### WP2 - Configuration and core contracts

1. Add validated configuration for model IDs, vendor path, device preference,
   precision policy, input paths, output paths, and smoke parameters.
2. Add the minimal contracts listed in Section 9.
3. Reject invalid matrix shapes, non-finite transforms, invalid image sizes,
   negative timestamps, and fabricated XYZ states.
4. Add schema serialization round-trip tests.

Acceptance: valid fixtures round-trip; invalid and missing-data fixtures fail
or serialize with the declared missing state as intended.

### WP3 - Coordinate foundation

1. Implement explicit transform inversion and point transformation.
2. Implement OpenCV pinhole projection and pixel/depth back-projection.
3. Keep DA3 pose layout/convention conversion isolated in its adapter.
4. Add deterministic synthetic tests before any output is treated as trusted.

Acceptance: all Section 9 coordinate tests pass within stated floating-point
tolerances.

### WP4 - Shared device, timing, and memory diagnostics

1. Select `mps` only when PyTorch reports it built and available; otherwise
   report an actionable failure rather than silently claiming an MPS run.
2. Provide device-specific synchronization around timed inference.
3. Measure cold and warm phases consistently.
4. Sample process and MPS memory as described in Section 8.
5. Write one summary schema used by all three model smoke checks.

Acceptance: a no-model diagnostic test can emit a valid summary and clearly
distinguish MPS, CPU fallback, and failure.

### WP5 - DA3 adapter and independent smoke check

1. Import the unmodified vendor package from its configured checkout.
2. Load exactly `depth-anything/DA3NESTED-GIANT-LARGE-1.1`.
3. Convert explicit project transforms into the vendor's expected OpenCV
   world-to-camera array only inside the adapter.
4. Apply the project-owned MPS compatibility policy from Section 8.
5. Run the independent test from Section 7 and retain its compact outputs.
6. Run the resolution ladder and record a provisional keyframe recommendation.

Acceptance: the DA3 part of the roadmap gate passes on MPS, or a precise,
reproducible MPS workaround is documented with its limitations.

### WP6 - YOLO adapter and independent smoke check

1. Load exactly `yolov8n-seg.pt`.
2. Run inference with segmentation enabled and tracking disabled for this
   independent model gate.
3. Normalize results into project-owned typed detections without introducing
   cross-camera or ByteTrack logic.
4. Run the Section 7 check first on an available fixture and then on the
   required representative living-room image.

Acceptance: the representative-image run completes, produces a valid
segmentation result structure even if no target class is detected, and writes
an annotated preview plus summary.

### WP7 - Qwen3-VL adapter and independent smoke check

1. Load exactly `Qwen/Qwen3-VL-2B-Instruct`; no prior-generation model is in
   scope.
2. Build a bounded multi-frame request and deterministic generation settings.
3. Keep the adapter callable independently and suitable for asynchronous use
   later; do not integrate it into geometry processing in S00.
4. Run the Section 7 check and retain prompt metadata, decoded response, and
   summary.

Acceptance: the model accepts multiple ordered frames and returns non-empty
text without blocking or modifying any spatial state.

### WP8 - Integrated verification and stage-close evidence

1. Run unit tests and each model smoke check as separate processes so model
   memory does not overlap.
2. Review summaries, warnings, raw outputs, and representative previews.
3. Confirm all four roadmap completion-gate statements in Section 10.
4. Record exact commands, package lock identity, model revisions or cache
   identifiers, results, failures, and fallbacks.
5. Close S00 following Section 14, then stop.

## 7. Independent Smoke-Test Design

All model checks must run in separate processes. Each must return a non-zero
exit status on an unmet assertion, write a machine-readable summary, and avoid
depending on another model's output.

### 7.1 DA3 pose-conditioned two-view inference

**Inputs**

- The two vendor SOH PNGs, copied only by reference and never modified.
- A synthetic but valid pair of identical pinhole intrinsics scaled to the
  processed image dimensions.
- Two explicit OpenCV `T_camera_from_world` matrices: the first identity and
  the second with a small declared horizontal baseline and no rotation.
- Model: `depth-anything/DA3NESTED-GIANT-LARGE-1.1`.
- `infer_gs=False`; no GLB, Gaussian, pose-estimation-only, or alternate-depth
  path.

**Procedure**

1. Start a fresh process and capture environment and baseline memory.
2. Load the model to MPS through the project adapter.
3. Run one cold two-view inference at the lowest resolution candidate.
4. Synchronize MPS, validate output, then run at least two warm repetitions
   with identical inputs.
5. Repeat one run at each higher resolution candidate only after the lower
   level succeeds and memory headroom is acceptable.
6. Save a compact NPZ containing depth, confidence, returned camera parameters,
   and input metadata for the selected recommendation; save a depth/confidence
   preview and JSON summary.

**Assertions**

- Exactly two processed views are returned.
- Depth and confidence are finite arrays with matching `[2, H, W]` shapes.
- A non-zero fraction of depth samples is finite and strictly positive.
- Returned intrinsics and poses have the expected count, shapes, and finite
  values.
- The run metadata proves that intrinsics and poses were supplied.
- The adapter's pose conversion round-trips independently.
- Cold and warm durations and memory observations are present.

This is an execution gate, not a metric-accuracy or camera-calibration claim.

### 7.2 YOLOv8n-seg

**Inputs**

- Required final gate input: one unmodified representative living-room image
  supplied by the user.
- Optional early API fixture: one available vendor image.
- Model: `yolov8n-seg.pt`.

**Procedure**

1. Start a fresh process and load the model on MPS.
2. Run one cold inference and at least two warm repetitions with fixed image
   size and thresholds.
3. Validate every returned box, class, confidence, and mask shape.
4. Save the native result summary, normalized detection JSON, and an annotated
   preview.

**Assertions**

- Inference completes on the representative image.
- The result contains a valid image-size record and a well-formed list of zero
  or more detections.
- Any detection has finite confidence in `[0, 1]`, an in-bounds box, and a mask
  that can be associated with the source image.
- The preview is written without changing the source image.
- Cold/warm durations and memory observations are present.

Detection of a backpack is observed but is not an S00 pass requirement; its
reliability is evaluated in S03. The bottle fallback is not activated in S00.

### 7.3 Qwen3-VL 2B Instruct multi-frame input

**Inputs**

- Four to eight ordered, uniformly sampled frames from a short local video.
  Until a user action sample exists, use the vendor `robot_unitree.mp4`.
- A concise prompt asking for an ordered factual description of visible action,
  explicitly allowing `unknown`.
- Model: `Qwen/Qwen3-VL-2B-Instruct`.

**Procedure**

1. Extract frames to an S00 artifact directory without modifying the video.
2. Start a fresh process, load the processor and model on MPS, and build one
   multi-image message in temporal order.
3. Run one cold generation and one bounded warm generation using the same
   request, a fixed maximum number of new tokens, and deterministic decoding.
4. Save the prompt manifest, ordered frame metadata, raw text response, and
   JSON summary.

**Assertions**

- The exact approved model ID and more than one input frame are recorded.
- The request is accepted and returns non-empty decoded text within the token
  bound.
- No coordinate, track identity, timestamp, or zone field is accepted from or
  written by this adapter.
- Cold/warm durations and memory observations are present.

The S00 test validates multi-frame model operation only. Structured event
schema, repair attempt, `unknown` fallback, triggered clips, and asynchronous
pipeline integration belong to S05.

## 8. Apple MPS Compatibility and Memory Observation

### Device and precision policy

- Prefer MPS and record the actual device used for every phase.
- Do not silently fall back to CPU. Any CPU execution must be marked as a
  workaround in the summary and handoff.
- Use inference/evaluation mode and synchronize MPS before stopping a timer.
- Select autocast by device capability rather than calling CUDA-only capability
  functions on MPS.
- Begin DA3 with conservative processing resolutions that preserve its expected
  patch multiple. Use a planned ladder of 336, 420, and 504 pixels, stopping
  before a higher level if the previous level lacks safe headroom.
- Load DA3, YOLO, and Qwen sequentially in separate processes. They do not need
  to coexist in unified memory.

The inspected DA3 vendor API currently selects autocast using a CUDA capability
query even when the image device may be MPS. The project-owned compatibility
layer must handle this device branch outside the vendor checkout, for example
through a narrow adapter override of the forward/autocast boundary. The exact
workaround and why it is safe must be tested and documented.

### Memory and timing observations

For each independent smoke check, record:

- process RSS before import/load, after model load, immediately before
  inference, and after inference;
- peak process RSS using a short polling interval during model load and
  inference;
- `torch.mps.current_allocated_memory()` and
  `torch.mps.driver_allocated_memory()` where supported;
- the MPS recommended maximum memory value where supported;
- input dimensions, frame/view count, tensor precision, model load duration,
  cold inference duration, each warm duration, and output dimensions;
- macOS memory-pressure or out-of-memory symptoms and any fallback used.

The recommendation should select the highest DA3 resolution that completes
repeatedly while retaining conservative unified-memory headroom and avoiding
material swap pressure. The keyframe interval is provisional: report the
measured seconds per two-view pair and recommend a simple offline interval for
30 FPS source video (for example, one pair every 1, 2, or 5 seconds). Choose
among these only from observed throughput and useful temporal coverage; S02
must revisit the choice using calibrated room imagery.

### Bounded compatibility escalation

Apply fallbacks in this order and record each attempt:

1. project-owned device-specific autocast correction;
2. full float32 for the failing model or operation;
3. lower DA3 processing resolution or fewer Qwen frames/tokens;
4. `PYTORCH_ENABLE_MPS_FALLBACK=1` for unsupported operations, with warnings
   and actual device behaviour recorded;
5. CPU execution only as a documented model-gate workaround when MPS cannot
   complete.

No fallback or supporting method may be used to claim that an exact S00 model
gate passed when that model did not run. A CPU-only result does not prove MPS
performance; the handoff must state that limitation plainly. Any additional
method introduced under D015 must have separate evidence and provenance.

## 9. Minimal Core Data Contracts and Coordinate Tests

### Planned contracts

- `FrameRef`: camera ID, frame index, timestamp in seconds, immutable source
  path/reference, image width, and image height.
- `CameraIntrinsics`: camera ID, `fx`, `fy`, `cx`, `cy`, image size, distortion
  metadata if known, and units/convention notes.
- `CameraPose`: camera ID, finite 4-by-4 `T_world_from_camera` and
  `T_camera_from_world`, right-handed world convention, and validation that the
  transforms are mutual inverses.
- `DepthPrediction`: frame reference, metric depth array reference, confidence
  array reference, invalid-value policy, model identity, processing size, and
  raw-output reference.
- `SegmentationDetection`: frame reference, camera-local class ID/name,
  confidence, pixel box, mask reference, and optional camera-local track ID.
- `SpatialObservation`: entity type/ID, timestamp, optional raw world XYZ,
  observation state (`observed`, `missing`, `occluded`, or `stale`), source
  camera IDs, confidence, and provenance. `missing` and `occluded` must not
  carry a fabricated XYZ; a stale state may reference a separately identified
  last observation.
- `ModelRunObservation`: exact model/revision, device, precision, input
  description, timing, memory, outcome, warnings, fallback, and artifact paths.

These are the minimal stable concepts, not a commitment to implement later
tracking, fusion, state-machine, event, or Rerun schemas during S00.

### Required synthetic tests

1. Invert a known rigid `T_world_from_camera`; verify multiplication by
   `T_camera_from_world` yields identity within tolerance.
2. Transform synthetic camera points to world and back; recover the original
   points within tolerance.
3. Project known positive-Z OpenCV camera points to pixels and back-project
   those pixels with the original depths; recover the camera points.
4. Verify the principal point back-projects to the optical axis.
5. Reject zero, negative, NaN, and infinite depth.
6. Reject non-finite confidence and filter confidence below a declared
   threshold without manufacturing a replacement position.
7. Reject wrong transform shapes, non-rigid final rows, and singular rotation
   blocks.
8. Verify camera-pose serialization preserves the explicit transform names and
   numerical values.
9. Verify a missing camera fails with a typed error or explicit unavailable
   result.
10. Verify a missing/occluded object contains no XYZ, while stale presentation
    state remains distinguishable from a raw observation.
11. Verify persistent contract validation rejects ambiguous fields named only
    `extrinsics`.

## 10. Verification Checklist Mapped to the S00 Completion Gate

### Gate A - DA3 two-view inference runs or has a documented MPS workaround

- [ ] Exact `DA3NESTED-GIANT-LARGE-1.1` identity/revision recorded.
- [ ] Two views, intrinsics, and explicit OpenCV
      `T_camera_from_world` inputs recorded.
- [ ] Metric-depth and confidence output assertions pass.
- [ ] MPS device, precision, cold/warm timing, and memory are recorded.
- [ ] Any compatibility workaround is project-owned, reproducible, and leaves
      vendor files unchanged.
- [ ] Resolution and provisional keyframe recommendations cite measured runs.

### Gate B - YOLOv8n-seg runs on a representative image

- [ ] User supplies or identifies one representative living-room image.
- [ ] Exact `yolov8n-seg.pt` identity/revision recorded.
- [ ] MPS inference and output-structure assertions pass on that image.
- [ ] Annotated preview, normalized output, timing, and memory summary exist.
- [ ] Zero detections, missed backpack, and invalid result handling are
      represented honestly.

### Gate C - Qwen3-VL-2B accepts a small multi-frame input and returns text

- [ ] Exact `Qwen/Qwen3-VL-2B-Instruct` identity/revision recorded; no
      prior-generation dependency or identifier is present.
- [ ] More than one temporally ordered frame is accepted.
- [ ] Non-empty bounded text is returned and saved.
- [ ] Timing, memory, input count, and fallback state are recorded.
- [ ] The adapter cannot write geometry, identity, timestamps, or zones.

### Gate D - The project test command runs successfully

- [ ] One exact documented command runs all non-heavy automated tests.
- [ ] Schema, transform, projection/back-projection, invalid-data, missing
      camera/object, and serialization tests pass.
- [ ] Independent heavy smoke commands are documented separately and their
      results are recorded.
- [ ] A fresh environment can reproduce the commands from the lockfile and
      locally available model weights/inputs.

The stage is not complete if Gate B lacks a representative living-room image,
or if any other gate is merely assumed from an attractive preview.

## 11. Likely Failure Modes and Bounded Fallbacks

| Failure mode | Required observation | Bounded fallback |
|---|---|---|
| Python/package lacks arm64 or Python 3.11 support | Package, version, resolver error, and architecture | Pin a compatible released version and regenerate the lock; do not change the approved model |
| Model weights cannot be downloaded | Exact model ID, cache state, and network/auth error | Accept a user-provided local cache/path for the same model and record its revision |
| DA3 invokes CUDA-only capability or memory code on MPS | Full traceback and failing boundary | Handle the branch in `da3_mps.py` or a narrow adapter override; do not edit vendor source |
| MPS operation is unsupported | Operation, dtype, input shape, and fallback warning | Try float32, then explicit PyTorch MPS fallback; document CPU portions |
| Unified-memory pressure or process termination | Resolution/frame count, RSS/MPS samples, swap/pressure symptoms | Lower DA3 resolution, reduce Qwen frames/tokens, and keep model runs in separate processes |
| DA3 output has invalid or empty depth | Output statistics, pose/intrinsic inputs, precision, and resolution | Retry float32 and a lower resolution; if still invalid, fail the gate and preserve diagnostics |
| Synthetic DA3 poses are geometrically inconsistent with vendor images | Label fixtures and result limitations | Treat the run only as an API/MPS gate; wait for S01 calibration before geometry claims |
| YOLO returns no detections | Image, thresholds, and raw empty output | Pass the structural smoke if valid, but record the miss; do not lower standards or activate the bottle fallback until S03 |
| Representative living-room image is absent | Missing input recorded | Stop Gate B closure and request one unmodified image; vendor media cannot satisfy this gate |
| Qwen processor/model API mismatch | Exact versions, prompt manifest, and traceback | Pin mutually compatible Transformers/model revisions for the same Qwen3-VL model |
| Qwen returns empty, repetitive, or excessive text | Raw response and generation settings | Reduce frames, bound tokens, use deterministic decoding, and retry once; otherwise fail the smoke |
| Transform convention confusion | Failing round-trip/reprojection fixture | Fix only adapter conversion and explicit naming; do not introduce an ambiguous `extrinsics` contract |
| Raw input or output accidentally overwritten | Path and write operation | Fail the run; all diagnostics must go below `artifacts/s00/` and raw media remains immutable |

The required S00 smoke gates remain DA3, YOLOv8n-seg, and Qwen3-VL 2B
Instruct. Supporting methods such as COLMAP, SfM, MVS, stereo, triangulation,
or floor-plane localization may be evaluated only under D015: highlight the
need to the user first, isolate the dependency, record provenance, and do not
use the result as a substitute for a failed baseline gate. No such need has
been identified for S00 model compatibility.

## 12. User Inputs Needed During S00

Required before S00 can close:

- One representative, unmodified living-room image and permission to use it
  locally for the YOLO gate.
- Approval for the later implementation turn to create the Python environment,
  resolve/install dependencies, and download the three exact model weights if
  they are not already cached.

Helpful but not required for initial model compatibility:

- One short local multi-frame living-room action sample for a more
  representative Qwen smoke check.
- Confirmation of the intended Python dependency/lock tool if the user does not
  accept the proposed `uv`/`uv.lock` scaffold.

The phones, calibration board, markers, room measurements, axes, and zones are
S01 inputs and must not block the non-calibration portions of S00.

## 13. Explicit Non-Goals

S00 does not:

- modify, patch, reformat, relocate, or commit generated content into the DA3
  vendor checkout;
- run a full room reconstruction or generate the final point cloud;
- calibrate camera intrinsics/poses or define the surveyed world origin, axes,
  room bounds, pickup zone, or drop-off zone;
- capture or alter raw recordings;
- synchronize videos, implement MP4/RTSP ingestion, or test RTSP;
- run continuous dense depth at video frame rate;
- implement ByteTrack, cross-camera association, re-identification, 3D
  localization, track fusion, trajectory smoothing, or zone membership;
- implement pickup-carry-place state logic, structured Qwen event repair, or
  let Qwen alter spatial facts;
- build Rerun presentation, Open3D scene fusion, a custom web frontend, or 3D
  Gaussian Splatting;
- run an unmotivated method-comparison programme or claim survey-grade
  accuracy; a targeted supporting method still requires the D015 process;
- train or fine-tune a detector, or activate the bottle fallback;
- begin S01 work.

## 14. Stage-Close Records

Only after every completion-gate item has been reviewed:

1. Run and record the exact test and smoke commands.
2. List environment/lock identity, representative diagnostic artifacts, model
   revisions, timings, memory observations, the selected DA3 resolution, and
   provisional keyframe interval.
3. Record successful and failed attempts, missing-data behaviour, fallbacks,
   and any weakened gate.
4. Confirm the vendor checkout has no project-authored modifications.
5. Update `docs/STATUS.md` with the actual result and one next action for S01.
6. Append to `docs/DECISIONS.md` only if S00 introduces a scope,
   architecture, physical-setup, or interpretation decision. Do not rewrite
   existing entries.
7. Create `docs/stages/S00_HANDOFF.md` from
   `docs/stages/HANDOFF_TEMPLATE.md`, including generated artifacts,
   reproduction commands, known limitations, and S01 software/physical
   prerequisites.
8. Create the descriptive S00 stage-close commit and record its hash in the
   handoff.
9. Optionally create and record an annotated S00 tag.
10. Push the commit to `origin/main`, push the optional tag, and verify both on
    the public GitHub remote.
11. Stop. Do not start S01 without an explicit request.

This implementation brief is not a completion handoff, and its creation does
not change `docs/STATUS.md`.

## 15. Exact Next Action for the Later Implementation Turn

WP1 and WP2 are complete. Begin WP3 by implementing explicit transform
inversion, point transformation, OpenCV projection, and depth
back-projection with the deterministic synthetic tests defined in Section 9.
