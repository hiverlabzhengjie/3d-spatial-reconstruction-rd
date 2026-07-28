# S00 Handoff - Project Setup and MPS Model Gate

**Stage:** S00 - Project Setup and MPS Model Gate

**Status:** Complete

**Started:** 2026-07-27

**Closed:** 2026-07-28

## Stage Goal

Create the Python 3.11 project foundation and prove that the exact DA3, YOLO,
and Qwen3-VL models can run independently on the M1 Max, while establishing
minimal typed contracts, coordinate utilities, tests, and runtime diagnostics.

## Entry Inputs

- Unmodified local `Depth-Anything-3-main/` vendor snapshot.
- DA3 vendor sample images and `robot_unitree.mp4`.
- User-supplied representative room image
  `stage_00_WP6_sample_image.jpeg`, kept outside the repository.
- MacBook Pro with M1 Max, 32-core integrated GPU, and 64 GB unified memory.
- VS Code, Docker Desktop, Ollama, and GitHub CLI available to the operator;
  only the native Python/MPS and GitHub CLI paths were needed in S00.
- No calibration recordings, action recordings, room measurements, or
  prepared physical calibration targets.

## Work Completed

- Created the native arm64 Python 3.11 `.venv`, `uv.lock`, package metadata,
  VS Code interpreter/test settings, and licence record.
- Added validated configuration with exact baseline model identifiers.
- Added immutable contracts for frames, intrinsics, explicit camera transforms,
  metric depth, segmentation detections, honest spatial observation states,
  timing, memory, and model-run provenance.
- Added and tested rigid transforms, projection/back-projection, and invalid
  depth/confidence filtering.
- Added device selection, synchronized timings, process/MPS memory
  observations, and explicit failure/fallback provenance.
- Added project-owned replaceable adapters for:
  - DA3 pose-conditioned two-view metric depth;
  - YOLOv8n segmentation;
  - asynchronous Qwen3-VL ordered multi-frame text.
- Kept the DA3 vendor source unmodified. D019 contains the bounded exact
  two-view post-alignment compatibility boundary.
- Ran all three exact models independently on Apple MPS, then re-ran them in
  separate WP8 close processes.
- Selected DA3 resolution 504 and a provisional two-second/60-frame keyframe
  interval for 30 FPS offline processing.
- Initialized Git history, created the public GitHub remote, and established
  the verified stage-close publishing workflow.
- Passed the integrated S00 gate without a CPU or alternate-model fallback.

## Changed Files

Important project-owned additions and updates include:

- `pyproject.toml`, `uv.lock`, `.python-version`, `.vscode/settings.json`;
- `configs/default.yaml`;
- `src/spatial_reconstruction/config.py`;
- `src/spatial_reconstruction/contracts.py`;
- `src/spatial_reconstruction/geometry/`;
- `src/spatial_reconstruction/runtime/`;
- `src/spatial_reconstruction/models/`;
- `scripts/diagnose_runtime.py`;
- `scripts/smoke/`;
- `tests/`;
- `docs/PROJECT_BRIEF.md`;
- `docs/ROADMAP.md`;
- `docs/STATUS.md`;
- `docs/DECISIONS.md`;
- `docs/VENDOR_DEPENDENCIES.md`;
- `docs/licences/MODEL_AND_LIBRARY_LICENCES.md`;
- `docs/stages/S00_IMPLEMENTATION_BRIEF.md`;
- `docs/stages/S00_WP1_ENVIRONMENT.md` through
  `docs/stages/S00_WP8_VERIFICATION.md`;
- this handoff.

Generated caches, model weights, vendor source, raw media, and experiment
artifacts are intentionally excluded from Git.

## Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Runtime diagnostics | `artifacts/s00/runtime/` | Device/fallback and memory evidence |
| DA3 WP5 evidence | `artifacts/s00/da3/` | Compatibility iterations and selected 504 result |
| YOLO WP6 evidence | `artifacts/s00/yolo/` | Representative-image masks and preview |
| Qwen WP7 evidence | `artifacts/s00/qwen/` | Ordered frames, prompts, tokens, and text |
| DA3 close gate | `artifacts/s00/wp8/da3_gate_20260728/` | Final Gate A arrays, previews, manifest, summary |
| YOLO close gate | `artifacts/s00/wp8/yolo_gate_20260728/` | Final Gate B masks, detections, preview, summary |
| Qwen close gate | `artifacts/s00/wp8/qwen_gate_20260728/` | Final Gate C frames, responses, manifest, summary |

Exact artifact hashes and interpretation are recorded in the corresponding
WP4-WP8 stage records. All locations are local and ignored by Git.

## Verification

### Commands

```text
.venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/mypy src scripts/diagnose_runtime.py \
  scripts/smoke/da3_two_view.py scripts/smoke/yolo_seg.py \
  scripts/smoke/qwen_multiframe.py
uv --cache-dir /private/tmp/spatial-reconstruction-uv-cache lock --check
uv --cache-dir /private/tmp/spatial-reconstruction-uv-cache sync --check
git diff --check

.venv/bin/python scripts/smoke/da3_two_view.py \
  --revision b2359bdf726fb44ef62acca04d629dcf158053e7 \
  --output-dir artifacts/s00/wp8/da3_gate_20260728

.venv/bin/python scripts/smoke/yolo_seg.py \
  --image <path-to-representative-room-image> \
  --output-dir artifacts/s00/wp8/yolo_gate_20260728

.venv/bin/python scripts/smoke/qwen_multiframe.py \
  --output-dir artifacts/s00/wp8/qwen_gate_20260728
```

### Results

- 106 automated tests passed.
- Ruff passed.
- Strict mypy passed across 17 source/script files.
- Lockfile and installed environment passed consistency checks.
- Gate A passed: DA3 revision
  `b2359bdf726fb44ef62acca04d629dcf158053e7`, MPS float16,
  two pose-conditioned views, finite metric output, selected 504.
- Gate B passed: YOLO checkpoint SHA-256
  `a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c`,
  MPS float32, valid source-sized segmentation results.
- Gate C passed: Qwen revision
  `89644892e4d85e24eaac8bacfd4f463576704203`, MPS float16,
  four ordered frames, bounded non-empty text.
- Representative DA3, YOLO, and Qwen visual outputs were inspected.
- Raw user image and vendor video hashes remained unchanged.
- DA3 vendor fingerprint matched
  `683cad1fec1186cd2a22f2b6d083b73d4c83c7ab1140f45ba24876612bc51d43`.
- No completion-gate check was skipped or weakened.

## Physical Setup and Observations

- No cameras were mounted, moved, calibrated, or recorded during S00.
- The user-supplied room image is an approximate intended viewpoint, not a
  surveyed final camera pose.
- No phone model/lens settings, resolution/FPS locks, stabilization choices,
  or synchronization events have yet been recorded.
- No ChArUco board dimensions, floor markers, world origin/axes, room bounds,
  pickup zone, or drop-off zone have yet been established.
- S01 must treat all future calibration as invalid if a camera or selected lens
  moves afterward.

## Problems and Limitations

- DA3's vendor autocast logic required a project-owned MPS boundary.
- Exact two-view supplied-camera inference required D019 because two camera
  centres cannot determine the vendor's full post-inference Sim(3) alignment.
  This does not weaken the MPS/API gate and does not claim calibrated accuracy.
- YOLO saw two bed regions but no person/backpack in the unstaged room image;
  target reliability remains S03 work.
- The vendor MP4 header reports 175 frames while 174 decode; the Qwen sampler
  now samples against the decodable count.
- Qwen used a vendor robot action, not the future living-room
  pickup-carry-place recording.
- DA3 504 and the two-second keyframe interval are provisional until calibrated
  room imagery is available in S02.
- Physical S01 inputs are not yet prepared.

## Decisions Made

- D015 - controlled use of supporting methodologies and tools.
- D016 - project-level version and experiment history.
- D017 - public GitHub remote and stage publishing.
- D018 - dynamic localization requires action-frame depth.
- D019 - exact two-view DA3 post-alignment stays in the project adapter.

No optional supporting methodology was activated in S00.

## Prerequisites for the Next Stage

Software:

- Preserve the project `.venv`, `uv.lock`, model caches, and exact vendor
  snapshot.
- Continue using explicit `T_world_from_camera` and `T_camera_from_world`.
- Keep captures and generated calibration artifacts out of public Git.

Physical/user actions:

- Identify the two phone models and exact selected lenses.
- Prepare two rigid mounts/tripods and continuous power.
- Print a ChArUco board on rigid matte backing and record its exact square and
  marker dimensions.
- Print at least four floor markers.
- Prepare tape measure, painter's tape, and the target backpack.
- Choose stable landscape camera placements with useful overlap and downward
  floor visibility.
- Be ready to lock resolution, 30 FPS, lens, focus, exposure, white balance,
  and stabilization where possible.
- Do not move either camera/lens after fixed-pose calibration.

## Version Control

- GitHub repository:
  `https://github.com/hiverlabzhengjie/3d-spatial-reconstruction-rd`
- Stage-close commit: `<pending stage-close commit>`
- Annotated tag: `None`
- Remote push verified: `Pending stage-close commit`
- Vendor/model revisions:
  - DA3 vendor:
    `683cad1fec1186cd2a22f2b6d083b73d4c83c7ab1140f45ba24876612bc51d43`;
  - DA3 model: `b2359bdf726fb44ef62acca04d629dcf158053e7`;
  - YOLO checkpoint:
    `a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c`;
  - Qwen model: `89644892e4d85e24eaac8bacfd4f463576704203`.

## Exact Next Action

After explicit user approval to begin S01, inventory the two phone/lens
configurations and confirm rigid mounts plus the printed ChArUco board's exact
dimensions before recording any calibration session.
