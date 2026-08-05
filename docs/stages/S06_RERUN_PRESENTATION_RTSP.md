# S06 - Rerun Presentation and RTSP Compatibility

## Status

Complete with known prototype limitations. Work Packages 1-5 and all seven
completion-gate criteria passed on 2026-08-05. Stage-close publication is
pending.

## Entry Prerequisites

Verified before S06 changes:

- S05 is closed at commit `6cdcd12de055f0ffe357d1fd2e8fdcd6c077faab`
  with tag `stage-05-interaction-events`.
- Local `main` was clean and matched `origin/main` at
  `00b250b4356c48b1477e6347ab9bbcd49d750ab8`.
- The accepted S02 static scene, S04 occlusion-aware presentation, S05
  semantic interaction timeline, Qwen v4 plan, and Qwen v4 results passed
  fresh independent verification.
- The two synchronized action videos, action synchronization manifest, and
  accepted S01-S05 artifacts are present and hash-consistent.
- All 259 existing tests, Ruff, strict mypy across 86 source files, lockfile,
  environment, and whitespace checks passed.
- No new physical capture, calibration, model, or user action was required.

## Work Package 1 - Integrated Offline Contract and Worker Supervision

Completed without model inference or Rerun recording generation.

### Contract

Policy `s06_integrated_offline_orchestration_v1` establishes:

- recorded file input as the current execution mode;
- capture time as the sole authoritative ordering and Rerun timeline;
- Camera A then Camera B source ordering;
- bounded perception queues of eight per camera, DA3 capacity two, and Qwen
  capacity three;
- deterministic offline throttle-and-drain behavior with no silent source
  drops;
- exactly one heavy-MPS permit on the M1 Max;
- a 45-second Qwen process hard timeout and at most two process attempts; and
- explicit prohibition on Qwen failure blocking geometry.

The accepted v2 stable manifest binds these inputs by exact content hash:

1. action synchronization manifest;
2. action-specific camera calibration;
3. scene and zone metadata;
4. accepted S02 static scene summary;
5. accepted S03 five-FPS target timeline;
6. accepted S04 occlusion-aware temporal presentation;
7. accepted S05 orthogonal interaction timeline;
8. accepted Qwen v4 event plan; and
9. accepted Qwen v4 execution results.

It also binds both synchronized action MP4 files, their decoded frame counts,
durations, nominal frame rates, and content hashes. Worker completion order is
diagnostic only and cannot change the manifest or capture timeline.

### Supervisor Boundary

The new subprocess supervisor provides:

- argv-only process launch without a shell;
- one hard wall-clock timeout per attempt;
- terminate followed by kill after a bounded grace period;
- captured exit status, stdout, stderr, timeout, terminate, and kill evidence;
- contiguous attempt numbering and bounded automatic restart; and
- an explicit degraded final state when all attempts fail or time out.

This addresses D039's known limitation: an asynchronous thread timeout cannot
preempt an active MPS call. The Qwen model runner can now be placed behind a
process boundary in later integrated execution. WP1 does not repeat the model
run; it tests lifecycle behavior with short synthetic child processes.

### Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Run summary | `artifacts/s06/orchestration_contract_20260805/summary.json` | Policy, stable manifest identity, counts, and limitations |
| Orchestration manifest | `artifacts/s06/orchestration_contract_20260805/orchestration_manifest.json` | Hash-bound file/video and accepted-artifact inputs |
| Verification | `artifacts/s06/orchestration_contract_20260805/verification.json` | Independent source, hash, policy, and manifest regeneration |

Accepted v2 manifest ID:
`87a1c225049f167d6b5f87632d953d2d242ac7479eb20a33bfc24393f359a8f7`.
The earlier seven-artifact manifest remains retained as diagnostic history and
is not the WP2 input.

### Reproduction Commands

```text
.venv/bin/python scripts/s06/build_orchestration_manifest.py \
  --output-dir artifacts/s06/orchestration_contract_<new-run-id>

.venv/bin/python scripts/s06/verify_orchestration_manifest.py \
  --summary artifacts/s06/orchestration_contract_<new-run-id>/summary.json \
  --output artifacts/s06/orchestration_contract_<new-run-id>/verification.json

.venv/bin/python -m pytest -q tests/test_orchestration.py
```

Artifact-producing commands refuse to overwrite existing output paths.

### Verification

- The manifest independently reloaded and regenerated with two source videos,
  seven unique required artifact roles, and all current file hashes intact.
- Capture time is authoritative; worker completion order is not.
- The default heavy-MPS permit count is exactly one.
- Qwen failure is explicitly non-blocking for geometry.
- Focused tests cover stable/tamper-evident manifest identity, missing-role
  rejection, successful process completion, bounded restart after failure,
  hard termination/kill of a stuck process, and continued geometry execution.
- All `264` project tests passed. Ruff passed. Strict mypy passed across `91`
  source/script files. The lockfile resolves `128` packages and the installed
  `106`-package environment requires no changes. `git diff --check` passed.
- No raw capture, synchronized video, accepted model artifact, or vendor file
  was modified.

### Scope and Limitations

- WP1 defines and verifies the integrated entry boundary; it does not yet
  create the final `.rrd` recording.
- The supervisor is verified with synthetic child processes. A later S06
  integrated execution will place the real Qwen runner behind this boundary.
- Queue-wait, end-to-end latency, backlog, and accelerator-utilization
  diagnostics remain later S06 work.
- RTSP open/reconnect behavior remains later S06 work and does not represent a
  production service.
- No new decision was required: the work implements D020 and D039 without
  changing their scope or thresholds.

## Work Package 2 - Integrated File-backed Rerun Presentation

Completed without model inference, coordinate derivation, or RTSP ingestion.

### Accepted Recording

The refined `digital_twin_stage06_v2.rrd` contains:

- both complete synchronized H.264 action videos with `1,047` frame
  references per camera on the `capture_time` timeline;
- `328` labelled YOLO/ByteTrack boxes and `298` five-FPS segmentation-overlay
  frames across the two views;
- a deterministic `40,000`-point sample of the accepted S02 static scene;
- action-specific calibrated camera transforms and pinholes;
- accepted pickup and drop-off zone rings;
- all `320` D034 presentation records with distinct measured, stale,
  occluded, and missing behavior;
- person footpoint, lower-body, and upper-body measurements in distinct
  colors, plus a separate backpack visible-cluster style;
- all `23` exact measured trajectory segments: eight person and 15 backpack,
  with no interpolation, stale endpoints, or connection across the known gap;
- all `160` D037 interaction records as phase, visibility, and localization
  time series; and
- pickup, carry, and place transition logs plus separate Qwen review logs.

Carry preserves frame `468` / `15.606667 s` as its deterministic transition
and frame `567` / `18.900000 s` as its semantic review. Worker completion time
does not appear as timeline authority.

### Visual Semantics

- Measured person footpoints are green.
- Measured lower-body surfaces are yellow.
- Measured upper-body surfaces are purple.
- Measured backpack visible clusters are blue.
- Stale positions are smaller, translucent orange, retain their source age,
  and remain display-only.
- Missing and occluded states clear the current point and contain no XYZ.
- Exact measured trajectory segments remain disconnected and carry no current-
  position labels.

### Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Refined orchestration summary | `artifacts/s06/orchestration_contract_v2_20260805/summary.json` | Nine first-class accepted inputs and both source videos |
| Refined orchestration manifest | `artifacts/s06/orchestration_contract_v2_20260805/orchestration_manifest.json` | Hash-bound calibration, scene, video, and S02-S05 evidence |
| Accepted Rerun recording | `artifacts/s06/integrated_rerun_20260805/digital_twin_stage06_v2.rrd` | Shareable synchronized Digital Twin-style presentation |
| Export summary | `artifacts/s06/integrated_rerun_20260805/digital_twin_stage06_v2_export_summary.json` | Counts, sources, timeline, event identities, hash, and limitations |
| Strong verification | `artifacts/s06/integrated_rerun_20260805/verification_v3.json` | Independent regeneration, RRD parsing, semantics, hashes, and visual evidence |
| Web-viewer QA | `artifacts/s06/integrated_rerun_20260805/web_*.png` | Camera A/B, metric twin, timeline, and event-log visual evidence |

Accepted recording SHA-256:
`0ec24e52ee4ab592bb02d9c2c30bbca5f455129466421f8b2ee2bb612f8d1fe9`.
Its size is `51,928,057` bytes.

### Reproduction Commands

```text
.venv/bin/python scripts/s06/export_integrated_rerun.py \
  --orchestration-summary \
    artifacts/s06/orchestration_contract_v2_20260805/summary.json \
  --output artifacts/s06/integrated_rerun_<new-run-id>/digital_twin.rrd

.venv/bin/python scripts/s06/verify_integrated_rerun.py \
  --export-summary \
    artifacts/s06/integrated_rerun_<new-run-id>/digital_twin_export_summary.json \
  --visual-qa-dir artifacts/s06/integrated_rerun_<new-run-id> \
  --visual-qa-passed \
  --output artifacts/s06/integrated_rerun_<new-run-id>/verification.json

.venv/bin/rerun --serve-web \
  artifacts/s06/integrated_rerun_<new-run-id>/digital_twin.rrd
```

The Rerun viewer requires an FFmpeg executable for these H.264 assets. During
WP2, Homebrew FFmpeg `8.1.2_1` was installed after the first viewer attempt
reported that prerequisite missing.

### Verification

- The `.rrd` parses successfully with Rerun `0.22.1` and contains all 16
  required structural paths plus the `capture_time` timeline.
- Both video counts, box/mask counts, all presentation state counts, all
  measured segments, interaction records, and event identities independently
  regenerate from the hash-bound sources.
- Missing/occluded XYZ, stale raw XYZ, interpolated segments, stale segment
  endpoints, and worker-completion timeline use are all zero.
- The local Rerun web viewer visually passed Camera A, Camera B, the 3D metric
  twin, state/interaction plots, and event logs. Five `1280x720` screenshots
  and hashes are retained in `verification_v3.json`.
- The first native viewer screenshot exposed an unrelated macOS window-surface
  size of `80,000 px`, above the GPU's `16,384 px` limit. Those failed native
  previews remain diagnostic; visual acceptance uses the equivalent localhost
  web viewer over the same recording and blueprint.
- The first 3D visual review found cluttering static segment labels. The v2
  recording removes only those labels and retains all exact segment geometry.
- Project-wide verification passes: `266` tests, Ruff, strict mypy across `94`
  source files, `uv lock --check`, `uv sync --check`, and `git diff --check`.
- No raw capture, synchronized video, accepted model output, calibration,
  coordinate, timestamp, track identity, zone, or vendor file was modified.

### Scope and Limitations

- The recording presents retained offline evidence; it is not proof of live
  capacity or a production viewer deployment.
- The backpack trajectory remains disconnected across the accepted `6.803 s`
  localization gap. Qwen carry text supplies no point.
- H.264 decoding depends on a locally available FFmpeg executable.
- Queue wait, processing/end-to-end latency, backlog, restart/idempotency,
  shutdown, and accelerator-utilization evidence are exercised in Work Package
  3 below using explicitly labelled deterministic virtual time.
- RTSP open/reconnect compatibility remains a later S06 work package.

## Work Package 3 - Deterministic Integrated Replay and Diagnostics

Completed without model inference, coordinate derivation, video decoding, or
RTSP ingestion.

### Replay Contract

The accepted WP3 replay is bound to orchestration manifest
`87a1c225049f167d6b5f87632d953d2d242ac7479eb20a33bfc24393f359a8f7`.
It uses deterministic virtual time to test orchestration invariants; its
latencies are diagnostic simulation values and are not measured model
throughput or hardware-capacity evidence.

The exercise contains `34` immutable logical jobs and `35` attempts across:

- Camera A and Camera B perception queues, capacity eight each;
- the DA3 queue, capacity two;
- the Qwen queue, capacity three; and
- a bounded geometry queue, capacity four.

Every queue transition retains the logical job and source identity. The
accepted `112`-event stream records acceptance, saturation throttling,
draining, retry acceptance, popping, completed/failed terminal state, and
duplicate coalescing. All queues stay within capacity and drain to zero.

### Scheduling and Failure Evidence

- Two deliberately different completion schedules produce the same persisted
  capture-ordered output digest:
  `746c0f1175982dbd61a13514c7c4398f3de8ff65960ab39335692ea03a5ead9b`.
- Worker completion order differs from capture output order and remains
  non-authoritative.
- Nine submissions encounter full queues and follow deterministic
  throttle/drain/retry behavior; no accepted source work is dropped.
- One duplicate Qwen submission is coalesced.
- One Qwen attempt fails and restarts successfully; its logical result is
  persisted once.
- One perception job and one DA3 job end in explicit degraded failure states.
- Geometry completes independently before the Qwen retry finishes.
- All nine heavy-MPS intervals are serialized through one permit, with maximum
  observed virtual occupancy one and no interval overlap.
- The shutdown exercise accepts five jobs, lets one in-flight job finish,
  explicitly cancels four pending jobs, drains backlog and in-flight counts to
  zero, and releases the accelerator permit.

Queue wait, processing latency, end-to-end result latency, peak/final backlog,
throttle/coalescing/drop counts, attempts, outcomes, degraded state, and
shutdown disposition are retained per queue or job. These values validate the
scheduling contract, not real-time capacity.

### Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Replay summary | `artifacts/s06/integrated_replay_v2_20260805/summary.json` | Source identity, aggregate outcomes, digest, and limitations |
| Accepted replay report | `artifacts/s06/integrated_replay_v2_20260805/integrated_replay_report.json` | Jobs, attempts, queue events, latency diagnostics, serialized accelerator intervals, results, and shutdown evidence |
| Independent verification | `artifacts/s06/integrated_replay_v2_20260805/verification.json` | Exact regeneration and invariant verification |

The earlier `integrated_replay_20260805` artifact predates explicit per-event
queue lifecycle evidence and remains diagnostic rather than accepted.
The accepted report SHA-256 is
`6c1fc968c626aacda035979f6c5f27fa1b68780ef2dd1ecf3fad4b2938a371c8`.

### Reproduction Commands

```text
.venv/bin/python scripts/s06/run_integrated_replay.py \
  --orchestration-summary \
    artifacts/s06/orchestration_contract_v2_20260805/summary.json \
  --output-dir artifacts/s06/integrated_replay_<new-run-id>

.venv/bin/python scripts/s06/verify_integrated_replay.py \
  --summary artifacts/s06/integrated_replay_<new-run-id>/summary.json \
  --output artifacts/s06/integrated_replay_<new-run-id>/verification.json

.venv/bin/python -m pytest -q tests/test_orchestration.py
```

Artifact-producing commands refuse to overwrite existing output paths.

### Verification

- The independent verifier reloads every accepted video/artifact hash and
  regenerates the complete typed replay report exactly.
- Contract validation independently reconstructs queue depth from the event
  stream and rejects overflow, inconsistent depth, non-contiguous events,
  incomplete drain, and aggregate/event-count disagreement.
- Tests reject capture-output reordering and overlapping accelerator permit
  intervals in addition to checking success, failure, restart, idempotency,
  coalescing, cancellation, and shutdown behavior.
- Project-wide verification passes: `269` tests, Ruff, strict mypy across `97`
  source files, `uv lock --check`, `uv sync --check`, and `git diff --check`.
- No raw capture, synchronized video, accepted model output, calibration,
  coordinate, timestamp, track identity, zone, Rerun recording, or vendor file
  was modified.

### Scope and Limitations

- WP3 proves deterministic orchestration behavior with virtual time. It does
  not claim measured live throughput, tail latency, memory pressure, or model
  residency performance.
- The accepted S02-S05 outputs are referenced rather than recomputed, avoiding
  unnecessary heavy inference and any change to accepted spatial evidence.
- No new method, model, dependency, or decision was introduced.
- Local RTSP open/reconnect compatibility is exercised in Work Package 4
  below.

## Work Package 4 - Local RTSP Open and Bounded Reconnect

Completed without model inference, spatial recomputation, raw-capture changes,
or external-network streaming.

### Controlled Test Fixture

D041 adds MediaMTX `1.19.3` solely as a Homebrew-installed localhost RTSP test
server. FFmpeg `8.1.2` publishes a transient 640x360 H.264 stream derived from
the accepted synchronized Camera A MP4; PyAV `16.1.0` reads it through the
existing `RTSPFrameSource` contract.

The accepted tracked configuration binds RTSP/TCP only to
`127.0.0.1:18554`. RTMP, HLS, WebRTC, SRT, MoQ, playback, API, metrics, and
profiling are disabled. The server and both publishers are terminated after
the test. An earlier diagnostic run exposed unrelated default RTMP/MoQ
listeners and a disposable generated keypair; the keypair was removed and that
run is not accepted.

### Reconnect and Identity Contract

Policy `s06_rtsp_bounded_reconnect_v1` provides:

- at most eight RTSP connection attempts;
- a `0.25 s` bounded delay between attempts;
- explicit `target_reached`, `stream_ended`, or `failed` attempt outcomes;
- explicit exhaustion instead of an unlimited retry loop;
- one credential/query-free persistent RTSP reference and stable
  stream-configuration fingerprint;
- contiguous global frame indices and unique immutable frame IDs across
  connections; and
- per-attempt source-PTS offset, capture-time offset, and observed reconnect
  gap evidence.

If a replacement publisher restarts its PTS, the raw source offset remains
visible. Capture time remains strictly increasing and includes at least the
observed local wall-clock outage gap rather than silently compressing the
disconnect.

### Accepted Exercise

The accepted `rtsp_smoke_v4_20260805` run decoded `45` frames across four
connection attempts:

1. `27` frames, then `stream_ended` when the first publisher was terminated;
2. a failed open while the stream was absent;
3. a second failed open while the stream was absent; and
4. `18` frames from the replacement publisher, ending `target_reached`.

The observed reconnect gap retained in capture time is `1.747282 s`. All frame
IDs are unique, indices are `0-44`, capture timestamps are strictly increasing,
and the final frame validates as an ordinary `PerceptionJob` without running
YOLO. MediaMTX logs independently show the first publish/read, stream outage,
failed reads, second publish/read, and clean process teardown.

### Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Local fixture | `configs/mediamtx_s06_local.yml` | RTSP/TCP-only loopback server configuration |
| Smoke summary | `artifacts/s06/rtsp_smoke_v4_20260805/summary.json` | Tool versions, source binding, counts, hashes, identity and timeline results |
| Reconnect read | `artifacts/s06/rtsp_smoke_v4_20260805/reconnect_read.json` | All frame identities, attempt outcomes, offsets, observed gaps, and bounded policy |
| Worker-contract sample | `artifacts/s06/rtsp_smoke_v4_20260805/sample_perception_job.json` | Final RTSP frame carried unchanged into a standard worker job |
| Process logs | `artifacts/s06/rtsp_smoke_v4_20260805/process_logs.json` | MediaMTX and FFmpeg lifecycle evidence |
| Independent verification | `artifacts/s06/rtsp_smoke_v4_20260805/verification.json` | Hash, source, lifecycle, identity, timeline, and worker-contract checks |

Accepted hashes:

- summary:
  `eb5d092abb9cd985ff1208a79101d87baa63bdfca6a3bdc5d78afff317972b01`;
- reconnect read:
  `cb93aae87628bd7f67c36b1eee1d592d0905d85b04c6a9abbed160d4a63d21ba`;
  and
- verification:
  `dde09b1ad483976b1aceb1707682ec14bd08eb1497d93bac595061c3d95161c7`.

### Reproduction Commands

```text
brew install mediamtx

.venv/bin/python scripts/s06/run_rtsp_smoke.py \
  --orchestration-summary \
    artifacts/s06/orchestration_contract_v2_20260805/summary.json \
  --output-dir artifacts/s06/rtsp_smoke_<new-run-id>

.venv/bin/python scripts/s06/verify_rtsp_smoke.py \
  --summary artifacts/s06/rtsp_smoke_<new-run-id>/summary.json \
  --output artifacts/s06/rtsp_smoke_<new-run-id>/verification.json

.venv/bin/python -m pytest -q tests/test_ingestion.py
```

The smoke runner requires permission to bind the configured localhost port and
refuses non-localhost URLs or existing output directories.

### Verification

- The verifier reloads the accepted orchestration manifest and confirms the
  exact source MP4 and fixture hashes.
- Typed validation rejects unbounded or non-contiguous attempts, inconsistent
  frame ranges, duplicate identities, non-RTSP frames, non-monotonic capture
  time, missing timestamp-rebase evidence, and worker-contract mismatch.
- Failure tests cover bounded exhaustion and a non-RTSP factory.
- Project-wide verification passes: `272` tests, Ruff, strict mypy across `100`
  source files, `uv lock --check`, `uv sync --check`, and `git diff --check`.
- No RTSP process remains listening after the test, and no generated key or
  certificate remains in the workspace.

### Scope and Limitations

- This is a single-camera, protocol-level localhost compatibility proof, not a
  production service or live-capacity benchmark.
- Jitter, packet loss, authentication, TLS, multiple concurrent streams,
  camera clock drift, and long-duration operation are not tested.
- The 640x360 transient stream is chosen for a fast compatibility fixture; the
  accepted 1080p synchronized source file remains unchanged.
- MediaMTX is not part of the application runtime and adds no model or spatial
  authority.

## Work Package 5 - Dedicated Exports and Completion Audit

Completed without model inference, spatial recomputation, source-video
changes, or a new physical capture.

### Dedicated Exports

The accepted `exports_20260805` bundle provides machine-readable products
outside the Rerun recording:

- `320` typed track-state records copied exactly from the accepted D034
  occlusion-aware S04 presentation;
- `23` typed exact measured trajectory segments, eight person and 15
  backpack, with no interpolation or stale endpoint;
- three typed pickup, carry, and place events derived from the accepted S05
  candidates and Qwen v4 evidence; and
- a hash-bound summary linking the exports to the accepted orchestration
  manifest and all S04/S05/Qwen inputs.

The carry record preserves authoritative transition frame `468` separately
from semantic review frame `567`. Each event retains deterministic phase and
spatial authority, the source-state identity, Qwen job identity/outcome, and an
explicit false boundary for Qwen spatial mutation.

### Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Track states | `artifacts/s06/exports_20260805/track_states.jsonl` | Complete measured/stale/occluded/missing presentation history |
| Trajectory segments | `artifacts/s06/exports_20260805/trajectory_segments.jsonl` | Exact measured person/backpack segments only |
| Events | `artifacts/s06/exports_20260805/events.jsonl` | Pickup/carry/place transitions, review times, and authority provenance |
| Export summary | `artifacts/s06/exports_20260805/summary.json` | Source binding, counts, file hashes, and limitations |
| Stage-close audit | `artifacts/s06/stage_close_audit_20260805/` | Fresh five-layer verification and unified seven-criterion gate decision |

Accepted export hashes:

- track states:
  `bba5065c047eea740c67a19ff5b9719d64f6e977eace3912336f5d6804702552`;
- trajectory segments:
  `d6e28f93e6118f886a7fa538f6b7630903c04b93e37715c3ec3b8e91be65108f`;
- events:
  `86886fac05117d1ace88add2d209e5ca1eb0f2e01644b825568df05f334fad76`;
  and
- summary:
  `8a9e1072a31a8f383e8bdc70a8e90d29d5204964ef17515c2cade4bb2c6e27bc`.

### Reproduction Commands

```text
.venv/bin/python scripts/s06/export_tracks_events.py \
  --orchestration-summary \
    artifacts/s06/orchestration_contract_v2_20260805/summary.json \
  --output-dir artifacts/s06/exports_<new-run-id>

.venv/bin/python scripts/s06/verify_tracks_events.py \
  --summary artifacts/s06/exports_<new-run-id>/summary.json \
  --output artifacts/s06/exports_<new-run-id>/verification.json

.venv/bin/python scripts/s06/verify_stage06_gate.py \
  --orchestration <fresh-orchestration-verification.json> \
  --rerun <fresh-rerun-verification.json> \
  --replay <fresh-replay-verification.json> \
  --rtsp <fresh-rtsp-verification.json> \
  --exports <fresh-export-verification.json> \
  --output <new-stage-gate-summary.json>
```

All artifact-producing commands refuse to overwrite existing output paths.

### Verification

- Tracks, measured segments, and events regenerate exactly from their retained
  hash-bound sources.
- Non-measured raw XYZ, missing/occluded presentation XYZ, interpolated
  segments, stale segments, and Qwen spatial writes are all zero.
- The fresh close audit reran the orchestration, Rerun, replay, RTSP, and
  export verifiers before evaluating the completion gate.
- Project-wide verification passes: `273` tests, Ruff, strict mypy across
  `104` source/script files, `uv lock --check`, `uv sync --check`, and
  `git diff --check`.
- No raw capture, synchronized source, accepted model artifact, calibration,
  coordinate, track identity, timestamp, zone, model weight, vendor file, or
  accepted Rerun recording was modified.

## Completion Gate

1. Complete recording replay/scrubbing: **passed**. The 51.9 MB RRD parses,
   contains all 16 required paths and 1,047 references per camera, and passed
   five-view visual QA.
2. Common video/geometry/track/event timeline: **passed**. `capture_time` is
   authoritative, source transitions are preserved, and worker completion
   order is absent from presentation authority.
3. Deterministic out-of-order offline replay: **passed**. Two different
   completion schedules produce the same capture-ordered digest.
4. Non-blocking Qwen, bounded queues, and explicit degraded states:
   **passed**. Queue saturation, Qwen retry, independent geometry completion,
   two degraded results, and complete shutdown drain are verified.
5. Single-M1 serialized heavy inference policy: **passed**. One permit,
   maximum occupancy one, non-overlapping intervals, and release are verified.
6. Visible missing/stale distinction: **passed**. The Rerun styles and state
   time series are distinct; unavailable states have null XYZ and stale
   presentation never becomes raw spatial evidence.
7. Local RTSP open/reconnect: **passed**. The localhost fixture decodes before
   and after a deliberate outage through bounded attempts while preserving
   capture-time identity and worker compatibility.

No completion criterion was skipped or weakened. The gate auditor records
`completion_gate_passed=true` and `completion_gate_weakened=false`.

### Remaining Limitations

- The `.rrd` and dedicated exports present retained offline results; they are
  not a live-throughput or production deployment claim.
- Replay timing is deterministic virtual time, not measured M1 model latency,
  memory pressure, or sustainable rate evidence.
- RTSP evidence covers one unauthenticated localhost stream, not packet loss,
  jitter, TLS, multiple cameras, clock drift, or long-duration operation.
- The backpack remains unlocalized through the accepted `6.803 s` carry gap;
  semantic confirmation does not fill the trajectory.
- Detector reliability, qualitative Qwen evidence, video-estimated zones, and
  lack of surveyed dynamic ground truth remain documented prototype limits.

## Exact Next Action

Create and publish the S06 stage-close commit and annotated tag, record their
verified remote provenance in `docs/stages/S06_HANDOFF.md`, then stop. Do not
begin S07 without an explicit request.
