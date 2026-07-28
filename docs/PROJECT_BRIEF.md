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

## Static Scene Versus Dynamic-Object Depth

A 2D detection does not by itself identify a 3D point. A pixel identifies a
camera ray; the depth selected along that ray determines the resulting XYZ.
Consequently, the empty-room/static reconstruction must not be used as the
depth lookup for a person or backpack detected later. If a person stands
between a camera and a table edge, reusing the static depth at that pixel would
incorrectly place the observation on the table.

Dynamic localization therefore uses:

1. a segmentation mask from the current action frame;
2. DA3 depth and confidence associated with that same camera and action-frame
   time, or with an explicitly bounded and recorded synchronization tolerance;
3. robust aggregation of valid depth samples inside the dynamic mask;
4. the calibrated camera transform to place the observation in the shared
   world frame; and
5. cross-camera validation and confidence-weighted fusion using only current,
   valid observations.

The static point cloud remains valuable as room context, a source of bounds and
zones, and a plausibility/occlusion reference. It is not evidence for the
current range of a foreground entity. Two-camera overlap improves visibility,
redundancy, and disagreement detection, but it does not make stale static depth
valid. If neither camera has current valid dynamic depth, the observation must
be missing, occluded, or stale rather than placed on the background surface.

Elevated, downward-looking CCTV viewpoints are helpful: they often preserve a
view of the floor around a person, reduce some furniture occlusions, and make a
person's lower-body or ground-contact region easier to interpret. This reduces
the frequency and severity of the problem but does not remove the underlying
ray-depth ambiguity. Feet may still be hidden, the person may occlude the
backpack, and a dynamic foreground pixel still cannot inherit its empty-room
depth.

DA3 may run on offline action-frame keyframes rather than at the detector's
full frame rate. A real-time 2D detection is therefore not automatically a
real-time measured XYZ observation. Depth-frame identity, timestamp, and
freshness must be retained. Any later temporal propagation is presentation or
inference state and must remain separate from raw measured XYZ.

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

The public remote is
`https://github.com/hiverlabzhengjie/3d-spatial-reconstruction-rd`. Each
stage-close commit and optional tag is pushed and verified there. Public
visibility does not relax the project's local-processing, data-protection, or
licence constraints.

## Non-Commercial Research Constraint

This prototype is for non-commercial research. Model and library licences must
be documented before any external or commercial reuse.
