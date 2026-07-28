# Project Status

**Last updated:** 2026-07-28
**Overall phase:** Implementation
**Current stage:** S00 - Project Setup and MPS Model Gate
**Stage state:** Complete - all four completion gates passed

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
- Created the public GitHub repository
  `hiverlabzhengjie/3d-spatial-reconstruction-rd`, configured it as `origin`,
  and adopted D017 for verified stage-close pushes.
- Adopted D018 to require action-frame depth for dynamic localization:
  empty-room depth cannot place a foreground person or backpack, two-camera
  overlap is redundancy rather than a stale-depth correction, and elevated
  downward-looking cameras reduce but do not eliminate occlusion and
  ray-depth ambiguity.
- Completed S00 WP3:
  - added explicit rigid-transform inversion and camera/world point utilities;
  - added OpenCV projection, metric depth back-projection, and
    depth/confidence filtering without placeholder XYZ values;
  - passed 49 project tests plus Ruff and strict mypy checks;
  - recorded exact results in `docs/stages/S00_WP3_GEOMETRY.md`.
- Completed S00 WP4:
  - added explicit device selection with honest MPS, CPU fallback, and failure
    provenance;
  - added synchronized cold/warm timing, current/peak process RSS, and
    supported MPS memory observations;
  - added a reusable no-model JSON diagnostic and verified it in restricted
    failure, explicit CPU-fallback, and native MPS modes;
  - passed 66 project tests plus Ruff and strict mypy checks;
  - recorded exact results in
    `docs/stages/S00_WP4_RUNTIME_DIAGNOSTICS.md`.
- Completed S00 WP5:
  - added a project-owned DA3 adapter that isolates MPS autocast and optional
    COLMAP import compatibility without modifying vendor source;
  - adopted D019 for the vendor's degenerate exact-two-view Umeyama
    post-alignment boundary;
  - ran the exact DA3 Nested 1.1 checkpoint on native MPS with supplied
    synthetic camera conditions at 336, 420, and 504;
  - selected 504 and a provisional two-second keyframe interval with
    substantial observed MPS memory headroom;
  - passed 83 project tests plus Ruff and strict mypy checks;
  - recorded exact evidence in `docs/stages/S00_WP5_DA3_MPS.md`.
- Completed S00 WP6:
  - added a project-owned, replaceable `yolov8n-seg.pt` adapter with
    source-sized mask normalization and strict output validation;
  - ran one cold and two warm predictions on the user-supplied representative
    image using actual Apple MPS with no CPU fallback;
  - preserved the source image byte-for-byte and retained normalized
    detections, raw arrays, an annotated preview, timings, and memory evidence;
  - observed two `bed` detections and honestly recorded zero `person` and
    `backpack` detections in the unstaged scene;
  - passed 92 project tests plus Ruff and strict mypy checks;
  - recorded exact evidence in `docs/stages/S00_WP6_YOLO_MPS.md`.
- Completed S00 WP7:
  - added an asynchronous, project-owned adapter for exactly
    `Qwen/Qwen3-VL-2B-Instruct` with no spatial-state write interface;
  - extracted four uniformly ordered frames from the unchanged vendor
    `robot_unitree.mp4` fixture and retained prompt/input provenance;
  - ran deterministic cold and warm generation on actual Apple MPS with no CPU
    fallback, producing identical non-empty 41-token descriptions;
  - recorded exact revision, prompt, raw token IDs, timings, memory, and the
    MP4 header-versus-decodable-frame discrepancy;
  - passed 106 project tests plus Ruff and strict mypy checks;
  - recorded exact evidence in `docs/stages/S00_WP7_QWEN_MPS.md`.
- Completed S00 WP8:
  - re-ran DA3, YOLO, and Qwen as three separate native Apple MPS processes;
  - confirmed all three exact model identities, outputs, timings, memory
    records, raw-input preservation, and absence of CPU fallback;
  - re-ran 106 automated tests plus Ruff, strict mypy, lockfile, environment,
    and whitespace checks;
  - restored and re-verified the exact 161-file DA3 vendor fingerprint after
    removing three transient Finder metadata files;
  - reviewed representative DA3, YOLO, and Qwen diagnostics;
  - confirmed the public stage-close scope excludes raw inputs, vendor source,
    model weights, caches, environments, and generated artifacts;
  - recorded exact evidence in `docs/stages/S00_WP8_VERIFICATION.md`.

## Current Blockers and Unknowns

- No S00 completion-gate blocker remains.
- No living-room calibration images or synchronized action videos are present.
- ChArUco board dimensions, floor markers, room axes, and zones are not yet
  defined.

## Available Software Inputs

- `Depth-Anything-3-main/`
- DA3 example images and video inside its vendor checkout
- Native arm64 Python 3.11 `.venv` resolved by `uv.lock`
- Installed S00 model libraries
- Exact DA3 Nested 1.1, YOLOv8n-seg, and Qwen3-VL 2B revisions cached locally
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
- Representative room-view image: supplied and used for WP6

## Exact Next Action

After explicit user approval to begin S01, inventory the two phone/lens
configurations and confirm rigid mounts plus the printed ChArUco board's exact
dimensions before recording any calibration session.
