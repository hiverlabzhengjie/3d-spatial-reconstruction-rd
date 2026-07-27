# Project Instructions for Codex

## Mission

Build an exploratory, DA3-centred prototype that converts two synchronized,
overlapping living-room camera recordings into:

1. A visually coherent metric 3D scene.
2. 3D tracks for one person and one movable backpack.
3. A pickup-carry-place event sequence.
4. A synchronized Digital Twin-style Rerun recording.

This is a focused proof of concept, not a production surveillance system or a
survey-grade reconstruction project.

## Required Reading

Before planning or changing this project, read:

1. `docs/PROJECT_BRIEF.md`
2. `docs/ROADMAP.md`
3. `docs/STATUS.md`
4. `docs/DECISIONS.md`
5. The latest applicable file in `docs/stages/`

Treat these files and the implementation in the workspace as the canonical
project record. Do not rely on memory from another Codex task.

## Baseline Methodology and Tools

- Depth and geometry: `DA3NESTED-GIANT-LARGE-1.1`
- DA3 mode: multi-view, pose-conditioned metric depth
- Primary 3D localization: DA3 depth back-projection
- Detection and segmentation: Ultralytics `yolov8n-seg.pt`
- Camera-local tracking: ByteTrack
- Semantic interpretation: `Qwen/Qwen3-VL-2B-Instruct`
- Calibration and geometry utilities: OpenCV and ChArUco/ArUco markers
- Video and RTSP ingestion: PyAV and FFmpeg
- Point-cloud processing: Open3D
- Visualization and timeline: Rerun
- Configuration and schemas: Pydantic
- Primary language/runtime: Python 3.11 on Apple Silicon

Keep the existing `Depth-Anything-3-main` source as an unmodified vendor
dependency. Put Apple MPS compatibility work in a project-owned adapter.

Additional methodologies and tools, including COLMAP, SfM, MVS, stereo,
triangulation, and floor-plane methods, may be used when they provide a clear
project benefit without unnecessary complexity. Before using one, highlight it
to the user, state its purpose and added operational cost, and record the choice
in `docs/DECISIONS.md` and the applicable stage record. Keep optional methods
isolated behind adapters, processes, or dependency groups where practical.

## Coordinate Conventions

- The world frame is right-handed and measured in metres.
- Z points upward.
- The origin is a surveyed floor marker near a room corner.
- X follows the selected primary wall.
- Y extends across the room.
- OpenCV image/camera convention is X right, Y down, Z forward.
- Internally use explicit transform names:
  - `T_world_from_camera`
  - `T_camera_from_world`
- Do not use an ambiguous field named only `extrinsics`.
- Isolate conversion to DA3/OpenCV pose formats inside their adapters.
- Add synthetic round-trip and reprojection tests before trusting new transform
  code.

## Scope Boundaries

The following remain outside the baseline and require a concrete benefit,
advance notice to the user, and an explicit update to `docs/DECISIONS.md`:

- Method-comparison or survey-grade evaluation programmes
- Multi-person re-identification
- Multiple rooms, floors, BIM, or GIS
- Coverage maps or blind-spot analysis
- Continuous dense reconstruction at full video frame rate
- A custom web frontend
- Production RTSP deployment
- Required 3D Gaussian Splatting output
- Custom detector training

The primary target object is a backpack. A bottle is the approved fallback only
if the standard YOLO model cannot detect the backpack reliably.

## Data and Physical-World Rules

- Never overwrite or modify raw recordings.
- Store each capture session separately with camera files and capture notes.
- Record camera model, lens, resolution, frame rate, lock settings,
  synchronization events, and any physical camera movement.
- Treat calibration as invalid if a camera or selected lens moves after pose
  calibration.
- Never fabricate missing XYZ observations.
- Mark temporarily unavailable object positions as missing, occluded, or stale.
- Separate raw positions from any smoothed or inferred presentation state.
- Keep all image and video processing local.

## Implementation Expectations

- Work on only the current stage recorded in `docs/STATUS.md`.
- Inspect existing code and inputs before editing.
- Prefer small, testable modules and typed data contracts.
- Keep model adapters replaceable. Add a non-baseline model or tool only under
  the controlled expansion rule above.
- Heavy models may run sequentially; they do not need to remain resident
  together.
- DA3 may operate on offline keyframes rather than every video frame.
- Qwen processing must be asynchronous and must not block geometry processing.
- Qwen may describe an event but may not change spatial coordinates, track
  identity, timestamps, or zone membership.
- Preserve raw model outputs needed for troubleshooting.

## Verification Requirements

For every stage:

- Run the stage-specific automated tests and smoke tests.
- Record exact reproduction commands.
- Save representative diagnostic outputs.
- Verify failure and missing-data behaviour, not only the successful path.
- Do not claim stage completion based only on a visually attractive result.
- If physical inputs are inadequate, stop dependent implementation and provide
  precise recapture or adjustment instructions.

Minimum recurring tests include:

- Transform inversion and coordinate round trips
- Pixel/depth back-projection
- Invalid-depth and confidence filtering
- Missing camera and missing object handling
- Schema validation for persistent outputs

## Version Control

- Use the project-root Git repository as the durable software and research
  history.
- Keep raw captures, model weights, generated artifacts, virtual environments,
  caches, and the unmodified DA3 vendor checkout out of Git.
- Record vendor source identity and a reproducible fingerprint in
  `docs/VENDOR_DEPENDENCIES.md`.
- Use descriptive commit messages that state what changed and why.
- Create at least one dedicated stage-close commit after each S00-S07
  completion gate and handoff are complete. Use the form
  `stage(SNN): concise outcome`.
- Do not create a stage-close commit or tag before its completion gate passes.
- Add an annotated stage tag when it improves experiment provenance, using a
  lowercase name such as `stage-03-da3-geometry`.
- Record the stage-close commit hash and optional tag in the stage handoff.
- Keep unrelated changes out of a stage-close commit.

## Stage Close Requirements

Before marking a stage complete:

1. Verify its completion gate in `docs/ROADMAP.md`.
2. Run relevant tests and record the results.
3. Update `docs/STATUS.md`.
4. Add any new decision to `docs/DECISIONS.md`.
5. Create the matching `docs/stages/SNN_HANDOFF.md` from the handoff template.
6. List generated artifacts and reproduction commands.
7. State physical and software prerequisites for the next stage.
8. Create the descriptive stage-close commit.
9. Optionally create an annotated stage tag and record it in the handoff.
10. Stop; do not begin the next stage unless explicitly requested.
