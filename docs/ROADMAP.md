# Approved Eight-Stage Roadmap

Stages are sequential. Do not begin a stage until the previous stage's
completion gate is satisfied or an explicit decision records why the project is
proceeding with a known limitation.

The listed methods and outputs are the current streamlined baseline. A
supporting method or tool may be introduced when it has a concrete benefit,
the user is told before it is used, and the decision and affected stage record
are updated. Such an addition does not silently replace an existing completion
gate.

## S00 - Project Setup and MPS Model Gate

**Purpose**

Create the project scaffold and prove that each chosen model can run
independently on the M1 Max.

**Inputs**

- Existing DA3 source checkout
- DA3 sample images
- One representative living-room image
- One short multi-frame action sample, when available

**Outputs**

- Python 3.11 project environment and lockfile
- Project configuration and typed core interfaces
- DA3, YOLO, and Qwen adapters
- Cold/warm runtime and memory observations
- Recommended DA3 resolution and keyframe interval
- Initial automated test suite

**Completion gate**

- DA3 two-view inference runs or has a documented MPS workaround.
- YOLOv8n-seg runs on a representative image.
- Qwen3-VL-2B accepts a small multi-frame input and returns text.
- The project test command runs successfully.

## S01 - Capture, Synchronization, and Calibration

**Purpose**

Produce synchronized recordings and stable camera calibration in one local
world frame.

**Inputs**

- Two phones and rigid mounts
- ChArUco board and floor markers
- Tape measurements
- Short calibration and synchronization recordings

**Outputs**

- Raw capture-session structure and capture notes
- File and RTSP frame-source interfaces
- Immutable synchronized frame bundles with capture-session, camera, frame,
  capture-time, and synchronization provenance suitable for worker jobs
- Synchronization manifest with offset/drift correction
- Intrinsic calibration JSON for each phone
- Fixed camera poses
- Room bounds, pickup zone, and drop-off zone
- Reprojection and camera-pose diagnostic previews

**Completion gate**

- A synchronized frame-pair preview is visually correct.
- Replaying the same inputs produces the same frame-bundle identities and
  ordering, independent of downstream worker completion order.
- Marker reprojections align plausibly in both cameras.
- Camera frustums point into the room in the declared world frame.
- No camera has moved since the fixed-pose calibration.

## S02 - DA3 Static Room Geometry

**Purpose**

Generate the visible static room geometry with DA3 pose-conditioned multi-view
metric depth.

**Inputs**

- Synchronized empty-room frame pairs
- Camera intrinsics and fixed poses
- Room bounds

**Outputs**

- DA3 depth, confidence, and optional sky outputs
- Per-camera and fused world-space point clouds
- Confidence and room-bound filtering
- Downsampled `static_scene.ply`
- Rerun/Open3D geometry preview
- Notes on visible artifacts and incomplete surfaces

**Completion gate**

- The point cloud is recognizable as the living room.
- Both cameras' points occupy one plausibly aligned world frame.
- Invalid and out-of-room points are filtered.
- The camera poses and point cloud are displayed together correctly.

## S03 - Person and Backpack Perception

**Purpose**

Detect, segment, and track the person and backpack in each camera.

**Inputs**

- Synchronized action recordings
- `yolov8n-seg.pt`
- Person and backpack class configuration

**Outputs**

- Per-camera bounding boxes and segmentation masks
- Camera-local ByteTrack IDs
- Timestamped YOLO/ByteTrack worker results that retain the exact source-frame
  identity
- Confidence, mask area, and visibility fields
- Annotated 2D video/frame previews
- Missed-detection and occlusion observations

**Completion gate**

- The person is tracked through a representative sequence.
- The perception worker can consume a bounded queue or deterministic offline
  stream without losing source identity; overload/failure behavior is explicit.
- The backpack is detected while stationary and during at least part of the
  movement.
- Occluded or missing backpack observations are explicitly represented.
- If backpack detection is unusable, the documented bottle fallback is tested.

## S04 - DA3-Depth 3D Localization and Fusion

**Purpose**

Convert person and backpack masks into shared world XYZ observations using DA3
depth only.

**Inputs**

- YOLO masks and tracklets
- DA3 metric depth and confidence from synchronized action frames containing
  the dynamic entities; empty-room/static-reconstruction depth is not a valid
  substitute
- Camera intrinsics and poses
- Room bounds and zones

**Outputs**

- Raw per-camera person/backpack XYZ observations
- Robust mask-to-XYZ aggregation
- Explicit depth-frame identity, timestamp, and freshness provenance
- Strict join results between YOLO masks and temporally compatible DA3 depth,
  independent of which worker finishes first
- Cross-camera confidence-weighted fused observations
- Missing, occluded, and stale-position states
- Raw and presentation trajectories
- Diagnostics for invalid/stale depth, foreground-versus-static-surface
  conflicts, and cross-camera disagreement

**Completion gate**

- Person and backpack locations are qualitatively plausible in the 3D scene.
- The backpack moves from the pickup side toward the drop-off side.
- A foreground person/object cannot be localized by silently reusing the
  empty-room depth at the same pixel.
- Two-camera fusion uses only temporally compatible, valid dynamic
  observations; overlap is treated as redundancy, not as a correction for
  stale static depth.
- Out-of-order worker completion cannot associate a mask with the wrong depth
  frame or reorder the capture timeline.
- Missing observations do not generate fabricated XYZ positions.
- Coordinate and back-projection tests pass.

**Required localization considerations**

- A pixel defines a ray, not a unique XYZ. Back-projection must use depth
  belonging to the detected action-frame content.
- Include a controlled test in which a synthetic foreground entity occludes a
  farther static surface; the returned XYZ must follow the foreground depth,
  and deliberately stale/static depth must be rejected or flagged.
- Define the spatial anchor represented by each track. An in-mask depth
  aggregate estimates visible surface location; it is not automatically the
  person's anatomical centre. Evaluate a robust lower-body/ground-contact
  anchor for the person and a robust in-mask cluster/aggregate for the
  backpack.
- Elevated, downward-looking cameras should be exploited for floor visibility
  and ground-contact reasoning, while retaining failure handling for hidden
  feet, furniture occlusion, person-backpack occlusion, and steep viewing
  angles.
- If DA3 is evaluated only on action keyframes, detections between those
  keyframes must not inherit an old depth as a new raw measurement. Mark the
  XYZ unavailable/stale or keep any explicitly modelled temporal estimate
  separate from raw observations.
- A second camera can supply an independent current observation or expose
  disagreement. If both views lack valid current depth, do not intersect the
  detection rays with the static point cloud and report the result as the
  entity's measured position.

## S05 - Interaction State and Qwen Events

**Purpose**

Turn tracks into a pickup-carry-place state sequence and use Qwen for concise
semantic interpretation.

**Inputs**

- Person and backpack tracks
- Pickup and drop-off zones
- Candidate-event video frames
- `Qwen/Qwen3-VL-2B-Instruct`

**Outputs**

- Backpack interaction state machine
- Pickup, carry, place, occluded, and unknown states
- Triggered event clips
- Bounded, deduplicated Qwen job queue and asynchronous timestamped results
- Schema-validated Qwen event JSON
- Retry and `unknown` fallback handling
- Human-readable event summaries

**Completion gate**

- A representative recording produces a sensible pickup-carry-place sequence.
- Qwen delay, timeout, or failure does not block perception, depth, geometry,
  or spatial-state processing.
- Qwen output conforms to the event schema or safely becomes `unknown`.
- Qwen never changes coordinates, identities, timestamps, or zone membership.
- Occlusion is represented without invented object locations.

## S06 - Rerun Presentation and RTSP Compatibility

**Purpose**

Assemble the Digital Twin-style presentation and prove protocol-level RTSP
compatibility.

**Inputs**

- Synchronized videos
- Static point cloud and camera calibration
- 2D detections, 3D tracks, zones, and events

**Outputs**

- Synchronized Rerun 2D camera views
- 3D scene, camera frustums, zones, tracks, and trajectories
- Event and diagnostic timeline
- Integrated local worker orchestration with bounded queues, explicit
  backpressure/drop policy, and serialized MPS access by default
- Queue depth, queue wait, processing latency, dropped/coalesced job, worker
  failure, and accelerator-utilization diagnostics
- Shareable `.rrd` recording
- Track and event exports
- File/RTSP source-interface smoke test

**Completion gate**

- The complete recording can be replayed and scrubbed coherently.
- Video, geometry, tracks, and events share one timeline.
- Offline replay is deterministic even when worker results complete out of
  order.
- Qwen cannot block geometry, queues cannot grow without bound, and model
  failures produce explicit degraded states.
- The default single-M1 policy prevents unmeasured simultaneous heavy MPS
  inference; any concurrency change cites measured memory and throughput
  evidence.
- Missing/stale observations are visibly distinguishable.
- The RTSP adapter opens or reconnects to a local test stream.

## S07 - Final Capture, Refinement, and Reporting

**Purpose**

Produce the final demonstration and consolidate findings.

**Inputs**

- Completed pipeline
- Lessons from development captures
- Final calibrated action recording

**Outputs**

- Reproducible final run
- Final Rerun recording
- Short demonstration video
- Capture and calibration guide
- Reproduction commands
- Concise technical report covering value, limitations, failure cases, MPS
  performance, model/queue scheduling observations, and potential future
  alternatives

**Completion gate**

- Another task can reproduce the final run from documented commands and inputs.
- The demo shows the intended object movement and semantic event sequence.
- Limitations and failures are presented honestly.
- The report separates demonstrated offline throughput from projected live
  capacity and states the measured changes needed for a production deployment.
- `docs/STATUS.md`, `docs/DECISIONS.md`, and all stage handoffs are current.
