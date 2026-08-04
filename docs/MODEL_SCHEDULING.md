# Model Scheduling and Queue Architecture

**Adopted:** 2026-07-28

**Decision:** D020

**Current delivery boundary:** Local recorded-MP4 prototype with RTSP
compatibility testing; not a production RTSP service

## Purpose

This document defines how DA3, YOLO/ByteTrack, and Qwen cooperate as the
pipeline is implemented. The design provides production-relevant separation,
timestamp correctness, bounded resource use, and failure isolation without
building distributed production infrastructure inside the prototype.

The central rule is:

> Models are independent timestamped workers at different logical rates, while
> heavy MPS inference may remain serialized on the single M1 Max.

Asynchronous work means that one model does not need another model's result to
finish unrelated work. It does not imply that all models run on the GPU at the
same instant.

## Logical Dataflow

```text
File/RTSP source
       |
Capture-session manifest and timestamp synchronizer
       |
Immutable synchronized frame bundles
       |
       +----------------------+----------------------+
       |                      |                      |
YOLO/ByteTrack queue      DA3 queue              Event trigger
higher-rate frames        static/action pairs        |
       |                  at selected times           v
       |                      |                   Qwen clip queue
       |                      |                      |
       +---------------- timestamped results --------+
                              |
           Provenance-aware joins, localization, and state
                              |
                  Rerun and persistent exports
```

The capture timeline is authoritative. Processing start time, completion time,
and completion order are diagnostics only.

## Worker Responsibilities and Rates

### YOLO/ByteTrack worker

- Consumes camera frames with immutable source identity.
- Produces segmentation, detections, visibility, and camera-local tracklets.
- Runs at the highest practical configured rate.
- Must not wait for Qwen.
- May run ahead of DA3, but its masks become measured XYZ only when joined to
  valid, temporally compatible action-frame depth.

### DA3 worker

- Produces the static-room geometry from synchronized empty-room inputs.
- Produces current dynamic depth from selected synchronized action-frame
  keyframes.
- S04 replaced the provisional two-second interval with a mask-aware dense
  action profile: retain the original action boundaries and add roughly
  one-second observations where current person and backpack masks exist. The
  retained proof-of-concept cadence ranges from approximately `0.6-1.8 s`
  locally and does not schedule fabricated measurements inside the accepted
  two-camera backpack absence interval.
- Retains frame, camera, timestamp, confidence, model, and raw-output
  provenance.
- Never converts an old keyframe depth into a new raw observation.

### Qwen worker

- Receives only bounded clips created by candidate pickup/carry/place triggers.
- Produces schema-bounded semantic interpretation.
- Is lower-rate and latency-tolerant.
- Cannot write coordinates, identities, capture timestamps, or zones.
- On timeout or failure, returns or causes an explicit `unknown` result without
  blocking perception, depth, localization, or state processing.
- S05 D038 fixes the recorded-MP4 plan at three unique capture-ordered jobs,
  capacity three with throttle-and-drain, six ordered frames per job, a
  `96`-token bound, a `45 s` timeout, and at most one sequential repair
  attempt. Duplicate logical events are coalesced by candidate, prompt, model,
  and policy identity.

## Identity and Message Contract

Exact schemas are introduced in their assigned stages, but every job and result
must preserve enough immutable information to identify:

- capture-session ID;
- worker/job ID and attempt;
- camera ID or synchronized camera set;
- source frame index or indices;
- source capture timestamp or timestamps;
- synchronization-manifest identity and offset/drift provenance;
- model identity/revision and processing configuration;
- job priority and creation time;
- processing start/finish times and outcome; and
- artifact/raw-output references where applicable.

Capture timestamps must never be replaced by processing timestamps. Results are
joined by source identity. Any allowed time tolerance must be named,
configured, tested, and recorded.

## Queue and Backpressure Rules

All queues are bounded. Each queue records accepted, completed, failed,
cancelled, coalesced, and deliberately dropped jobs, plus queue wait and
processing duration.

Recorded-MP4 mode is the required prototype mode:

- Prefer throttling, deterministic batching, or pausing ingestion over dropping
  source work.
- Keep execution reproducible from the same inputs and configuration.
- Group work to amortize model loading when measurements show that this is
  beneficial.
- Allow results to complete out of order, but persist and present them in
  capture-time order.

Future live mode is a different execution policy over the same contracts:

- Protect a configured end-to-end freshness target with bounded queues.
- Coalesce or drop superseded ordinary work according to an explicit policy
  rather than accumulating unbounded delay.
- Give synchronization and current perception priority.
- Preserve event-relevant frames/clips according to a configured retention
  policy.
- Emit visible degraded states and drop/coalescing diagnostics.

The default priority order is:

1. ingestion, timestamping, and synchronization;
2. current YOLO/ByteTrack perception;
3. scheduled or event-prioritized DA3 action depth;
4. triggered Qwen interpretation; and
5. nonessential preview generation.

This priority is provisional and must be tuned from measured end-to-end
behavior, not assumed to be production-optimal.

## Accelerator and Model Residency

The current machine has one Apple MPS accelerator and unified memory. One
project-owned accelerator permit serializes heavy model inference by default.
This prevents accidental simultaneous peaks while allowing CPU-side work to
continue.

Three logical workers do not require three simultaneously resident models.
Model residency and batching are execution-policy choices:

- deterministic offline runs may perform model-sized batches in separate
  processes to avoid repeated load/unload costs;
- an interactive/local run may keep a model resident when measured memory
  permits;
- a future production deployment may assign workers to separate accelerators
  or hosts.

The S04 dense action profile is an evidence-quality choice, not a throughput
limit derived from the M1 Max. A future live deployment with stronger or
separate accelerators may run DA3 more frequently, but must preserve the same
exact-frame identity, D025 scale gate, bounded queues, and explicit missing
state when no valid observation exists.

Do not reload a heavy model for every frame. Do not enable simultaneous heavy
MPS calls merely because worker APIs are asynchronous. Any change to the
single-permit default requires representative measurements of peak memory,
swap pressure, throughput, tail latency, failures, and output integrity.

## Joining and State Rules

- Worker completion order never determines capture order.
- YOLO masks and DA3 action depth must match the exact source content or an
  explicit tested tolerance.
- A late but correctly identified result may update the historical capture
  time in offline output.
- A result that is too old for current live state may still be retained as
  historical evidence but cannot be labelled current.
- Missing, failed, or late depth produces missing/occluded/stale state, not a
  background-static or fabricated XYZ.
- Qwen results annotate already established deterministic phase/visibility
  evidence and spatial facts. They never revise coordinates, track identity,
  capture timestamps, zones, or spatial authority.
- D039 bounds each retained Qwen inference image to `768` pixels maximum
  dimension. Invalid or truncated responses retain raw token/tensor evidence
  but resolve to `unknown`, even when their prose appears plausible.
- D040 uses a hash-bound assistant JSON prefill and a `160`-token ceiling.
  Model output supplies five semantic fields; candidate matching and the
  no-spatial-claims boundary remain application-owned.
- Qwen review time is distinct from event-transition time. Carry retains its
  frame-468 onset but reviews the sustained interval around frame 567; this
  semantic sampling change cannot move the transition or supply XYZ.
- Raw observations, derived anchors, fused tracks, inferred state, and
  presentation smoothing remain distinguishable.

## Failure and Shutdown Behavior

Each worker reports failure independently. A worker crash, model error, timeout,
or invalid output must not silently stop other workers.

The orchestrator must support:

- explicit degraded operation;
- bounded retry rules;
- cancellation and clean shutdown;
- draining or recording the disposition of queued jobs;
- preservation of raw input identity;
- restart without treating duplicated results as new capture events; and
- an `unknown`, missing, occluded, or stale output where the relevant contract
  requires it.

## Verification Assigned Across Stages

### S01

- Deterministic frame-bundle identity and replay.
- Timestamp, offset, drift, missing-camera, and duplicate-frame behavior.

### S03

- Bounded perception queue and explicit overload/failure behavior.
- Preservation of source identity through detections and ByteTrack results.

### S04

- Correct exact-frame mask/depth joins despite out-of-order completion.
- Rejection of mismatched or stale action depth.
- Missing-depth and camera-loss behavior without fabricated XYZ.

### S05

- Trigger deduplication, bounded Qwen queue, timeout, retry, and `unknown`.
- Proof that delayed/failed Qwen work cannot block geometry.
- Treat the Qwen runner as a supervised process for hard timeout and restart;
  an asynchronous thread timeout alone cannot preempt an active MPS call.
- Verify source-transition and semantic-review frame identities separately so
  a valid semantic window cannot silently rewrite event capture time.

### S06

- Integrated accelerator arbitration and bounded queues.
- Deterministic offline replay with deliberately reordered worker completions.
- Queue saturation, cancellation, worker failure, restart/idempotency, and
  graceful shutdown.
- Diagnostics for queue wait, processing latency, end-to-end result latency,
  backlog, drop/coalescing count, and degraded state.
- File and local RTSP source behavior through the same worker contracts.

### S07

- Report measured throughput and latency by worker and end to end.
- Distinguish demonstrated offline capacity from projected live capacity.
- Record the exact production adjustments identified from evidence.

## Future Live-Production Adaptation

The prototype architecture deliberately creates useful production boundaries,
but future engineers must validate and supply the production system around
them. At minimum they must:

1. define service-level objectives for capture-to-detection, capture-to-XYZ,
   event latency, availability, and acceptable frame loss;
2. load-test representative camera counts, resolutions, scene activity, and
   event bursts;
3. choose model residency, batching, accelerator count, and worker placement
   from measured throughput and tail latency;
4. replace local in-memory queues with durable or distributed queues only when
   restart, scaling, or delivery requirements justify the operational cost;
5. implement process supervision, health checks, automatic restart,
   idempotency, and rolling deployment;
6. secure camera credentials, network transport, stored media, logs, and
   access to personally sensitive outputs;
7. define retention, privacy, audit, and incident-handling policies;
8. monitor clock synchronization, camera movement, calibration invalidation,
   queue age, dropped work, accelerator pressure, model drift, and output
   quality;
9. validate RTSP reconnect, jitter, packet loss, frame reordering, and partial
   camera outage under realistic conditions;
10. establish model/version rollout, rollback, and calibration/version
    compatibility rules; and
11. decide whether separate accelerators/hosts, optimized models, or other
    methods are required. Any change to this project's baseline methodology
    must follow the applicable decision process.

Production deployment remains outside the present roadmap. S06 proves only
protocol-level RTSP compatibility, and S07 must not represent offline results
as a demonstrated live-production service.
