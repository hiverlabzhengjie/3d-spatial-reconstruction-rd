# Final Technical Report

## Outcome

The project produced a reproducible exploratory Digital Twin demonstration for
one living room, two synchronized fixed cameras, one person, and one backpack.
The accepted interactive Rerun combines both camera views, a DA3-derived metric
static scene, calibrated cameras and zones, 2D detections, honest 3D
observations, progressive measured trajectories, localization/visibility
state, and pickup-carry-place semantics on one capture-time timeline.

The final presentation is
`artifacts/s07/final_run_v2_20260805/digital_twin_stage07_final.rrd`, SHA-256
`bcf84af987069151339427d57d7642cffd0e92b6c0ff05bbdbddb7c6143b64ca`.
Under D044, the user accepted this interactive recording as the final
demonstration instead of a separate rendered MP4.

## Demonstrated value

- A viewer can move between synchronized Camera A/B imagery and a calibrated
  world-space representation rather than inspecting isolated detections.
- The backpack is measured near the pickup zone before motion and near the
  drop-off zone after motion; the measured endpoints move 2.545 m.
- Pickup, carry, and place remain separate, timestamped event facts. Qwen
  reviews the visible semantics but cannot alter coordinates, identities,
  timestamps, zones, or spatial authority.
- Temporary object unavailability is visible. The 6.803-second carry gap is
  not filled with a plausible-looking invented path.
- Every retained observation can be traced to capture session, camera, source
  frame, timestamp, calibration, model revision, and processing policy.

This is useful Digital Twin context for reviewing where an interaction occurs,
how an object changes zones, what each camera saw, and when the spatial system
did not have a defensible coordinate.

## Final evidence

The 34.923-second accepted action pair contains 1,047 frames per camera. The
final Rerun retains:

- 1,047 synchronized video references per camera;
- 328 labelled boxes and 298 segmentation-overlay frames;
- 81,709 fused static-scene points before visualization sampling;
- 320 person/backpack presentation records;
- 16 person and 17 backpack measured 3D observations;
- eight person and 15 backpack same-anchor measured trajectory segments;
- 160 orthogonal interaction records; and
- pickup, carry, and place transitions, with carry onset at frame 468 and its
  separate sustained-carry Qwen review centred at frame 567.

Presentation-only H.264 proxies reduce the embedded-camera maximum keyframe gap
from 250 to 30 frames. Browser review found both views visible at early,
middle, and late seek points. Measured dots and valid segments appear at their
capture times, so paths grow during playback instead of being fully visible at
the beginning.

## Accuracy and interpretation boundary

The project has no surveyed dynamic ground-truth trajectory, so it does not
report an absolute localization error. Synthetic geometry tests verify the
mathematics, and calibrated marker checks constrain the static/world geometry,
but neither substitutes for a motion-capture reference.

Person coordinates mix explicit `person_footpoint`,
`person_lower_body_surface`, and `person_upper_body_surface` measurements when
feet are not consistently visible. Backpack points are visible-cluster centres,
not hidden physical centroids. Same-kind points may form measured segments;
different anchor kinds and unsupported gaps remain disconnected. No visual
smoothing or interpolation is applied because it would imply unsupported
continuity or accuracy.

The dynamic prototype uses a 0.35 m same-anchor cross-camera agreement gate and
an inspectable DA3-support/confidence/dispersion reliability score. These are
bounded policies for this capture, not calibrated probabilities or general
production tolerances.

## Measured Apple M1 Max observations

The following measurements come from separate retained experiments. They are
not one end-to-end run and must not be summed as though models were measured in
one simultaneous workload.

| Component | Demonstrated local measurement | Interpretation |
|---|---|---|
| DA3, two views at 504 | cached repeated pair mean 1.050 s; calibrated action warm pairs 1.330-1.401 s | Suitable for sparse offline keyframes, not full-frame-rate dense depth |
| DA3 residency | about 6.76 GB MPS allocation and 8.82-8.83 GB driver allocation | Isolated-model evidence only |
| YOLO single-image warm | 0.066-0.069 s after a 1.428 s cold inference | Representative S00 image, not dual-stream pipeline latency |
| YOLO accepted 5 FPS workload | about 30.01 s summed inference for 31.81 s of two-camera capture | Roughly 94% of one serialized real-time inference budget before other work |
| Qwen accepted event review | 4.76-5.13 s for each six-image request, 48-52 output tokens | Asynchronous semantic latency is acceptable offline but not immediate |
| Final retained-output assembly | 3.759 s for 34.923 s of retained content, or 9.291 captured seconds assembled per wall second | Hash validation and Rerun serialization only; no model inference |

The single-M1 execution policy therefore serializes heavy MPS work. The
logical workers remain independent and timestamped, while CPU decoding,
validation, queue handling, and output work may continue. Virtual-time S06
replay demonstrated bounded queues, deterministic capture ordering under two
different completion schedules, explicit degraded results, hard Qwen process
supervision, clean shutdown, and maximum accelerator occupancy one. Those
virtual timings validate scheduling logic, not M1 throughput or tail latency.

## Principal failures and retained limitations

- The baseline YOLO checkpoint labels the physical bag inconsistently as
  `backpack` and `handbag`. A guarded two-label policy works for this one-bag
  recording, but detection remains fragmented during carrying.
- The backpack has no defensible measured XYZ through the 6.803-second carry
  interval. Video review supports the semantic carry phase, but Qwen and stale
  display state do not fill the spatial gap.
- Three person views are bottom-truncated. They remain upper-body evidence,
  not inferred feet.
- One dense person pair disagrees by 0.377 m and is rejected beyond the 0.35 m
  gate rather than arbitrarily selecting a camera.
- Initial Qwen response formats repeatedly produced truncated or invalid JSON.
  Bounded assistant-prefilled JSON plus a sustained-carry review window yielded
  schema-valid results; all earlier failures remain diagnostic history.
- The original embedded H.264 camera streams had sparse keyframes and could
  black out during scrubbing. Presentation proxies fix seeking without
  changing raw source identity or spatial evidence.
- Zone centres are video-estimated and floor-marker centres have stated
  approximately +/-0.05 m measurement uncertainty.
- The static scene is recognizable and metrically constrained but incomplete,
  noisy, and not survey-grade. One bounded low-confidence door supplement is
  specific to this static reconstruction.
- RTSP evidence covers one unauthenticated localhost stream and one deliberate
  outage. It does not validate real CCTV networks, authentication, TLS, jitter,
  packet loss, multiple cameras, or long-duration service.
- Rerun playback depends on local FFmpeg support. Generated artifacts and raw
  media are deliberately excluded from the public Git repository.

## Demonstrated offline capacity versus projected live capacity

Demonstrated offline capacity consists of deterministic recorded-MP4
processing, isolated native-MPS model runs, virtual-time orchestration tests,
one localhost RTSP compatibility exercise, and measured retained-output Rerun
assembly. The project has not demonstrated a sustainable live two-camera
service, capture-to-XYZ service-level objective, production availability, or
end-to-end tail latency.

A live deployment is therefore a projection, not a result. On the current
single M1 Max, continuous two-camera 5 FPS YOLO already consumes most of one
serialized inference budget, DA3 adds approximately 1.3-1.4 seconds per
selected pair, and Qwen adds approximately five seconds per review. A useful
live design would likely need lower DA3 cadence, event-prioritized scheduling,
optimized models, or separate accelerators/hosts.

## Required production measurements and changes

Before production deployment, engineers must:

1. Define capture-to-detection, capture-to-XYZ, and event-latency SLOs plus
   availability, acceptable frame loss, and stale-state limits.
2. Measure one complete running system with representative camera count,
   resolution, activity, event bursts, model residency, queue age, peak memory,
   swap pressure, throughput, p95/p99 latency, and restart behavior.
3. Select model cadence, batching, quantization/optimization, accelerator count,
   and worker placement from those measurements. Preserve exact-frame joins and
   never turn an old depth result into a new measurement.
4. Improve bag perception with deployment-specific data and evaluation. Custom
   training or another detector is a future decision, not part of this
   prototype.
5. Add durable supervision, health checks, idempotent restart, deployment
   rollback, calibration/version compatibility, and observable degraded modes.
6. Validate real RTSP authentication, TLS, jitter, loss, reordering, clock
   drift, multi-camera outage, reconnection, and long-duration stability.
7. Monitor marker visibility, camera movement, calibration invalidation, model
   drift, disagreement rate, unavailable XYZ, and queue freshness.
8. Establish access control, encryption, retention, privacy, audit, incident
   response, and human-review policy for personally sensitive video and tracks.
9. Collect surveyed or motion-capture dynamic ground truth if numerical
   localization-accuracy claims or calibrated thresholds are required.

Potential future alternatives include separate compute for perception/depth/
semantics, an optimized detector for the actual object taxonomy, higher-rate
DA3 on capable hardware, or an independently justified localization method.
Any change must preserve provenance, explicit missing data, and the controlled
method-expansion decision process.

## Conclusion

The proof of concept meets its exploratory objective: it communicates a
coherent metric scene and pickup-carry-place sequence while making uncertainty,
occlusion, disagreement, and missing localization visible. Its strongest
result is not uninterrupted tracking; it is a reproducible spatial-event
record that distinguishes what was measured from what was only seen,
remembered, or semantically interpreted.
