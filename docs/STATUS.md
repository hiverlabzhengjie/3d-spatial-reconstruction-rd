# Project Status

**Last updated:** 2026-07-27
**Overall phase:** Implementation
**Current stage:** S00 - Project Setup and MPS Model Gate
**Stage state:** In progress - WP1 and WP2 complete

## Completed

- Reviewed the original CCTV spatial-reconstruction proposal.
- Clarified that its main Digital Twin value is live spatial event context over
  a calibrated 3D scene.
- Refined the work into a focused, exploratory two-camera prototype.
- Selected DA3 Nested 1.1 as the geometry and depth backbone.
- Selected YOLOv8 segmentation/tracking, Qwen3-VL-2B semantic interpretation,
  OpenCV calibration, and Rerun presentation.
- Chosen a one-person, one-backpack pickup-carry-place demonstration.
- Reduced the roadmap to eight sequential stages.
- Created the initial project continuity pack.
- Prepared `docs/stages/S00_IMPLEMENTATION_BRIEF.md` without starting
  technical implementation.
- Completed S00 WP1:
  - installed `uv` and created the native arm64 Python 3.11 `.venv`;
  - added `pyproject.toml`, `uv.lock`, minimal package metadata, VS Code
    settings, and the initial licence record;
  - verified that the locked environment is reproducible and Apple MPS is
    available to a native process;
  - recorded exact results in `docs/stages/S00_WP1_ENVIRONMENT.md`.
- Relocated the unmodified DA3 checkout to `Depth-Anything-3-main/` and removed
  the obsolete `Glossary/` folder at the user's request.
- Adopted D015: useful supporting methods such as COLMAP may be introduced when
  their benefit and complexity are highlighted to the user and recorded.
- Completed S00 WP2:
  - added validated `configs/default.yaml` and project configuration loading;
  - added immutable typed contracts for frames, cameras, depth, detections,
    spatial observations, and model-run diagnostics;
  - enforced explicit camera transform names and honest missing, occluded, and
    stale states;
  - passed 25 automated tests plus Ruff and strict mypy checks;
  - recorded exact results in `docs/stages/S00_WP2_CONTRACTS.md`.
- Initialized project-level Git history under D016 with an initial
  research-plan checkpoint and vendor-source fingerprint.

## Current Blockers and Unknowns

- Coordinate transform, projection, and back-projection utilities and their
  synthetic numerical tests have not yet been implemented.
- DA3 two-view pose-conditioned MPS inference has not been verified inside the
  project workflow.
- The project-owned DA3 adapter must isolate the vendor's COLMAP exporter from
  the main MPS process to avoid loading a second OpenMP runtime. PyCOLMAP is
  available as a locked optional extra for isolated future use.
- YOLOv8n-seg and Qwen3-VL-2B model smoke tests have not been run here.
- No living-room calibration images or synchronized action videos are present.
- ChArUco board dimensions, floor markers, room axes, and zones are not yet
  defined.

## Available Software Inputs

- `Depth-Anything-3-main/`
- DA3 example images and video inside its vendor checkout
- Native arm64 Python 3.11 `.venv` resolved by `uv.lock`
- Installed S00 model libraries; model weights are not downloaded
- Optional PyCOLMAP dependency locked but not installed in the main `.venv`
- MacBook Pro with M1 Max, 32-core GPU, and 64 GB unified memory

## Available or Planned Physical Inputs

- Two fixed phone cameras: planned
- Stable mounts/tripods: not yet confirmed
- ChArUco board: not yet prepared
- Printed floor markers: not yet prepared
- Tape measurements: not yet recorded
- Target backpack: planned
- Empty-room recording: not yet captured
- Pickup-carry-place recording: not yet captured

## Exact Next Action

Begin S00 WP3 only: implement explicit transform, projection, and
back-projection utilities with deterministic synthetic tests.
