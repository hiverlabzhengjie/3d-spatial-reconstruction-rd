# S00 WP2 Configuration and Contracts Record

**Work package:** WP2 - Configuration and core contracts
**Date:** 2026-07-27
**Result:** Complete

## Outcome

- Added `configs/default.yaml` as the single baseline configuration.
- Added `spatial_reconstruction.config` for strict YAML loading, baseline model
  identity checks, path resolution, and runtime/DA3/perception settings.
- Added `spatial_reconstruction.contracts` with immutable Pydantic contracts.
- Added automated positive, failure, missing-data, and serialization tests.
- Added one documented project test command to `README.md`.

## Configuration

The default configuration records:

- the unmodified DA3 vendor location and generated-artifact root;
- exact DA3, YOLO, and Qwen3-VL baseline identifiers;
- preferred MPS device and explicit CPU-fallback policy;
- DA3 resolution candidates aligned to its 14-pixel patch multiple;
- provisional keyframe interval candidates;
- YOLO confidence and target-class defaults; and
- raw-model-output preservation.

Unknown keys are rejected. Relative paths resolve from the project root.
Changing a baseline model identifier requires an explicit code/config change
and project decision; supporting tools under D015 remain separate from the
baseline model fields.

## Core Contracts

| Contract | Purpose and key invariant |
|---|---|
| `FrameRef` | Camera/frame/timestamp and file-or-stream source reference; dimensions and time must be valid |
| `CameraIntrinsics` | OpenCV pinhole parameters in pixels with finite values |
| `CameraPose` | Explicit `T_world_from_camera` and `T_camera_from_world`; finite rigid 4-by-4 matrices that are mutual inverses |
| `DepthPrediction` | References to metric depth, confidence, and preserved raw output |
| `PixelBox` | Positive-area pixel box |
| `SegmentationDetection` | Camera-local class, confidence, in-frame box, mask reference, and optional local track ID |
| `SpatialObservation` | Raw XYZ only when observed; missing, occluded, and stale states cannot carry fabricated raw XYZ |
| `TimingObservation` | Named non-negative duration |
| `MemoryObservation` | Named non-negative process/MPS memory sample |
| `ModelRunObservation` | Exact model/device/precision, inputs, timings, memory, outcome, fallbacks, warnings, and artifact references |

All contracts reject unknown fields and are frozen after validation.
Persistent camera poses cannot use a field named only `extrinsics`.

## Verification

Commands:

```text
uv run pytest
uv run ruff check src tests
uv run mypy src tests
uv lock --check
uv sync --check
```

Results:

- pytest: 25 passed.
- Ruff: pass.
- mypy strict mode: pass across five source/test files.
- Lockfile: current.
- Main environment: synchronized; 102 installed packages.

Test coverage includes:

- default configuration and exact model IDs;
- relative path resolution and unknown-key rejection;
- DA3 resolution validation;
- JSON serialization round-trips;
- frozen contracts and invalid frame values;
- missing frame validation;
- non-finite/invalid intrinsics;
- rigid transform shape, final row, rotation, inverse, and naming checks;
- depth-output references;
- bounding-box extents and image bounds;
- observed/missing/occluded/stale spatial state rules; and
- duplicate timing/memory phase prevention.

## Deferred to WP3 and Later Work

- WP3 implements transform inversion, point transformation, OpenCV projection,
  depth back-projection, and numerical round-trip tests.
- Confidence/depth array filtering operates on model arrays and is implemented
  with geometry/model adapters later; WP2 only declares the persistent policy.
- Cross-camera fusion, smoothing, event schemas, and Rerun contracts remain in
  their roadmap stages.
- Model weights, model inference, and model-generated artifacts were not
  created in WP2.

## Decisions

No new project decision was required. WP2 implements the existing coordinate,
missing-data, baseline-model, and D015 policies.

## Exact Next Action

Begin WP3 by implementing project-owned transform and OpenCV
projection/back-projection utilities with deterministic synthetic tests.
