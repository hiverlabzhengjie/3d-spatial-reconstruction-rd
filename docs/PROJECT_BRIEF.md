# Project Brief

## Stable Objective

Develop an exploratory end-to-end prototype that converts two synchronized,
overlapping fixed-camera recordings of one living-room scene into:

- A visually coherent metric 3D representation of the visible room
- 3D observations and trajectories for one person and one backpack
- A semantic pickup-carry-place event sequence
- A synchronized Digital Twin-style Rerun recording

The project uses a streamlined DA3-centred baseline. Supporting methods and
tools may be added when they solve a concrete problem or materially improve
reliability, provided their benefit and complexity are highlighted to the user
and recorded. It does not attempt to prove that one method is best or make
survey-grade accuracy claims.

## Intended Demonstration

The final recording should show:

1. A backpack initially stationary inside a pickup zone.
2. A person approaching the backpack.
3. The person picking it up.
4. The person carrying it through the overlapping camera scene.
5. Honest handling of temporary backpack occlusion.
6. The person placing the backpack inside a drop-off zone.
7. The person moving away.
8. The backpack remaining at its new position.

The presentation should combine:

- Two synchronized camera views
- YOLO masks, labels, and track IDs
- A DA3-derived static room point cloud
- Calibrated camera poses and frustums
- Pickup and drop-off zones
- Person and backpack 3D observations and trajectories
- Visibility and stale-position state
- Pickup, carry, and placement events
- Basic model timing and confidence diagnostics

The architecture diagram below is the current baseline, not a prohibition on
targeted supporting methods. Any addition should remain modular and should not
replace a baseline stage gate silently.

## Chosen Architecture

```text
Two fixed phone MP4 recordings
               |
Timestamp alignment and synchronized frame pairs
               |
Camera undistortion and fixed pose calibration
               |
DA3 pose-conditioned multi-view metric depth
        |                              |
Static room point cloud       Per-camera depth/confidence
                                       |
                          YOLO person/backpack masks
                                       |
                         DA3-depth back-projection
                                       |
                         Shared world XYZ observations
                                       |
                         Cross-camera confidence fusion
                                       |
                       Person/object tracks and state
                                       |
                  Triggered Qwen3-VL event interpretation
                                       |
                 Rerun video/3D/event timeline recording
```

## Meaning of Success

Success is a coherent and reproducible exploratory demonstration, not a formal
accuracy benchmark.

The project is successful when:

- The room point cloud is recognizable and plausibly aligned with the cameras.
- Person and backpack XYZ observations are spatially sensible.
- Their trajectories broadly agree with the synchronized videos.
- The backpack is shown moving from pickup zone to drop-off zone.
- Missing or occluded observations are represented honestly.
- Qwen3-VL identifies the principal pickup and placement actions.
- The final Rerun recording communicates useful Digital Twin context.
- Major limitations, failure cases, performance, and future alternatives are
  documented clearly.

Lightweight diagnostics should record runtime, missed detections, invalid depth,
occlusion periods, cross-camera disagreement, and a few measured-distance spot
checks. These are observations, not pass/fail accuracy targets.

## Hardware and Physical Setup

Primary development machine:

- MacBook Pro
- Apple M1 Max
- 32-core integrated GPU
- 64 GB unified memory
- Apple MPS inference

Physical equipment:

- Two smartphones
- Two stable tripods or rigid phone clamps
- Power/charging for sustained recording
- Printed ChArUco calibration board on rigid matte backing
- At least four printed floor markers
- Tape measure and painter's tape
- One medium-sized, visually distinctive backpack
- A visible flash and audible clap for synchronization
- Stable lighting with reflections and strong backlighting minimized

Recommended capture defaults:

- Both phones in fixed landscape poses
- Same nominal resolution and 30 FPS
- No digital zoom or lens switching
- Focus, exposure, white balance, and stabilization locked where possible
- Visible/audible synchronization event at the start and end
- Approximately 45-90 seconds per action recording

## Persistent Project Outputs

- Camera calibration JSON
- Synchronization manifest
- DA3 metric depth and confidence NPZ
- Static room PLY point cloud
- Per-frame track states
- Structured JSONL event log
- Rerun `.rrd` recording
- Annotated demonstration video
- Reproduction instructions
- Final findings and limitations report

## Version and Experiment Provenance

The project-root Git repository is the durable history for code, configuration,
decisions, tests, and stage handoffs. Each completed stage receives a
descriptive stage-close commit and may receive an annotated tag. Raw media,
model weights, generated artifacts, environments, caches, and the unmodified
DA3 vendor checkout remain local and are referenced through manifests,
fingerprints, model revisions, and reproduction records instead of being
committed.

## Non-Commercial Research Constraint

This prototype is for non-commercial research. Model and library licences must
be documented before any external or commercial reuse.
