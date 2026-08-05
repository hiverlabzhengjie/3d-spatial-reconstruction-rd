# S07 - Final Capture, Refinement, and Reporting

## Status

Complete with known limitations. Work Packages 1-3 and the five-criterion S07
completion gate passed on 2026-08-05 without weakening a criterion. The
stage-close handoff is complete; version-control provenance is recorded there.

## Entry Prerequisites

Verified before S07 implementation changes:

- the user explicitly selected the already accepted `action_take_01` as the
  final demonstration recording;
- local `main` was clean and matched `origin/main` at
  `6e4bef08425f3cdd37775781c91a9b4337086bd4`;
- the accepted S06 stage-close implementation is commit
  `1ae09564934d322cf67e7b42e499431960c9b277`, with the later provenance-only
  documentation commit at local and remote `main`;
- all five retained S06 verification layers passed fresh independent checks;
- the unified S06 audit passed all seven completion criteria without weakening
  a criterion;
- the accepted S06 Rerun SHA-256 remained
  `0ec24e52ee4ab592bb02d9c2c30bbca5f455129466421f8b2ee2bb612f8d1fe9`;
  and
- no camera, lens, calibration, raw recording, or accepted model artifact
  changed.

## Work Package 1 - Final Recording Lock and Reproducible-run Entry Contract

Completed without model inference, Rerun regeneration, demo-video generation,
or modification of an accepted upstream artifact.

### Recording Selection

D042 records the user's decision to retain `action_take_01`. The final source
pair remains:

- Camera A: 1,047 decoded frames, SHA-256
  `1e7064fa2d4911dcf2ac82803dd95fa5b9ece332906589c0f8627232bb526136`;
- Camera B: 1,047 decoded frames, SHA-256
  `da5bd4eeaeac0da78cc71f14a43326d5d60c5c216f2609c85553cba720e40d5a`.

The accepted capture session remains `s01_capture_20260729`. No recapture or
recalibration is required for this retained pair. A future new capture would
still require synchronization and capture-specific pose validation.

### Contract

Policy `s07_reproducible_final_run_v1` preserves:

- recorded MP4 input and capture time as the authoritative timeline;
- reuse of verified model outputs for final assembly, without unnecessary
  model inference;
- null XYZ for unavailable states;
- disconnected measured trajectory segments across unsupported intervals;
- Qwen's lack of spatial authority; and
- the explicit boundary that no live-production capacity has been
  demonstrated.

The manifest binds the accepted S06 orchestration summary, 51.9 MB integrated
Rerun recording, Rerun export summary, deterministic replay summary, RTSP smoke
summary, dedicated track/event export summary, and S06 gate audit by exact
content hash. The S06 orchestration manifest transitively binds the accepted
calibration, scene, static geometry, perception, localization, semantic, Qwen,
and synchronized-video evidence.

Accepted S07 manifest ID:
`9bb49beb13262f8108f4b37dd0974de6cc23f7fa456214e0c14e6c540a31ba08`.

### Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Final-run manifest | `artifacts/s07/final_run_contract_20260805/final_run_manifest.json` | Hash-bound recording choice and accepted S06 evidence |
| Run summary | `artifacts/s07/final_run_contract_20260805/summary.json` | Selection, counts, boundaries, and limitations |
| Independent verification | `artifacts/s07/final_run_contract_20260805/verification.json` | Hash, S06 gate, recording, and policy audit |

### Reproduction Commands

```text
.venv/bin/python scripts/s07/build_final_run_manifest.py \
  --output-dir artifacts/s07/final_run_contract_<new-run-id>

.venv/bin/python scripts/s07/verify_final_run_manifest.py \
  --summary artifacts/s07/final_run_contract_<new-run-id>/summary.json \
  --output artifacts/s07/final_run_contract_<new-run-id>/verification.json

.venv/bin/pytest -q
.venv/bin/ruff check .
MYPYPATH=src .venv/bin/mypy --strict src scripts
uv --cache-dir .cache/uv lock --check
uv --cache-dir .cache/uv sync --check
git diff --check
```

Artifact-producing commands refuse to overwrite existing output paths.

### Verification

- The final-run manifest independently reloaded and regenerated its stable ID.
- All two source videos and seven final input artifacts matched their retained
  hashes.
- The final Rerun hash matched its export summary.
- The S06 audit remained passed and unweakened and used the same accepted S06
  orchestration manifest.
- Recapture and recalibration remain false for the selected recording.
- Unavailable XYZ must remain null, measured trajectories must remain
  disconnected, Qwen spatial authority remains false, and demonstrated live
  capacity remains false.
- Three focused finalization tests passed, including manifest tamper, missing
  role, duplicate role, wrong recording, and wrong frame-count rejection.
- All 276 project tests passed. Ruff passed. Strict mypy passed across 108
  source/script files. The 128-package lock and 106-package environment were
  unchanged. `git diff --check` passed.

### Scope and Limitations

- WP1 locks inputs but does not yet produce the final S07 Rerun or demo video.
- No measured worker or end-to-end M1 throughput was collected in WP1. S06
  virtual-time replay remains scheduling evidence only.
- The physical detector and localization limitations remain unchanged,
  including the accepted 6.803-second backpack localization gap.
- No completion-gate criterion has been claimed or weakened.

## Work Package 2 - Measured Final Assembly and Rerun Regeneration

Completed without YOLO, DA3, or Qwen inference and without changing any
accepted upstream coordinate, identity, timestamp, zone, or semantic fact.

### Measured Execution Contract

The typed execution requires exactly two successful, ordered steps:

1. independent verification of the WP1 final-run entry manifest; and
2. regeneration of the file-backed Rerun presentation from the accepted S06
   orchestration inputs.

Each step retains its exact argv, stdout/stderr logs and hashes, return code,
and measured wall time. The contract independently recomputes the step sum,
assembly-to-capture duration factor, and captured-seconds-per-assembly-second
rate. It fixes the evidence kind as `measured_retained_output_assembly`, model
inference as false, and demonstrated live capacity as false.

### Accepted Refined Final Recording

Interactive review of the initial 51.9 MB assembly found three presentation
problems: both embedded camera views could black out while seeking; current
timestamped XYZ was not prominent enough; and complete trajectories were
visible from capture start. The initial file remains immutable historical
evidence at `artifacts/s07/final_run_20260805/`.

The preferred refined S07 recording is:

- path:
  `artifacts/s07/final_run_v2_20260805/digital_twin_stage07_final.rrd`;
- size: `44,022,273` bytes; and
- SHA-256:
  `bcf84af987069151339427d57d7642cffd0e92b6c0ff05bbdbddb7c6143b64ca`.

D043 introduces presentation-only, source-hash-bound H.264 proxies. Both
original camera streams had five keyframes and maximum 250-frame gaps. Each
proxy retains 1,047 frames at 1920x1080/30 FPS and has 35 keyframes with a
maximum gap of 30 frames. The original synchronized videos, timestamps, and
all spatial evidence remain unchanged.

- Camera A proxy SHA-256:
  `b1ac9622a7ef513d2de05a7c10ea3de5603c644ec334e3629b468e08315b5b46`;
- Camera B proxy SHA-256:
  `af919ffb8c0468cca154fff98157ecf866c91929440e97f30b016959ce64b42d`;
- proxy manifest SHA-256:
  `58452a2bf280b4450a57748313f8bb977fddfe03e241c0049a0abcebc77d5809`.

The refined recording adds a dedicated coordinate/provenance view covering
all 320 presentation records. Measured rows show timestamp, frame, XYZ, anchor
kind, and contributing camera IDs. Stale values are explicitly display-only;
missing and occluded states expose no XYZ. Current 3D labels also show capture
time and XYZ.

All 16 person and 17 backpack measured observations and the accepted 8 person
and 15 backpack same-anchor segments are logged on `capture_time`. Measured
dots and segments therefore accumulate during playback instead of appearing
in full at the start. No smoothing, interpolation, cross-anchor joining, or gap
filling was added: doing so would overstate accuracy without surveyed dynamic
ground truth.

Rerun serialization is not byte-deterministic across exports, so acceptance
does not require the S07 file hash to equal the S06 file hash. Independent
verification instead requires the complete stable semantic export fields to
match, parses the new RRD directly, and uses new visual evidence from the new
file.

### Measured Results

| Step | Measured wall time |
|---|---:|
| S07 entry verification | `1.297895917 s` |
| Rerun export | `2.460805166 s` |
| Total retained-output assembly | `3.758701083 s` |

The source recording duration is `34.922667 s`. Therefore this assembly-only
run measured:

- `0.107629x` wall time per second of captured content; and
- `9.291153` seconds of captured content assembled per wall-clock second.

These values measure hash validation plus presentation assembly from retained
outputs. They do not include fresh video decoding through every model, model
loading, YOLO, DA3, Qwen, queue residency under a live workload, or RTSP
transport. They are not an end-to-end live-capacity result.

### Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Proxy manifest | `artifacts/s07/rerun_video_proxies_20260805/proxy_manifest.json` | Hash-bound seekable camera presentation assets |
| Proxy verification | `artifacts/s07/rerun_video_proxies_20260805/verification.json` | Independent source, codec, frame-count, and keyframe audit |
| Refined final Rerun | `artifacts/s07/final_run_v2_20260805/digital_twin_stage07_final.rrd` | Preferred synchronized interactive Digital Twin recording |
| Export summary | `artifacts/s07/final_run_v2_20260805/digital_twin_stage07_final_export_summary.json` | Stable semantics plus coordinate and progressive-trail policy |
| Measured run summary | `artifacts/s07/final_run_v2_20260805/summary.json` | Commands, log hashes, timings, and assembly-only throughput |
| Verification | `artifacts/s07/final_run_v2_20260805/verification.json` | Independent structure, semantics, timing, and visual-QA audit |
| Visual QA | `artifacts/s07/final_run_v2_20260805/visual_qa/` | Six browser-captured views from the refined RRD |

### Reproduction Commands

```text
.venv/bin/python scripts/s07/build_rerun_video_proxies.py \
  --final-run-summary artifacts/s07/final_run_contract_20260805/summary.json \
  --output-dir artifacts/s07/rerun_video_proxies_<new-run-id>

.venv/bin/python scripts/s07/verify_rerun_video_proxies.py \
  --manifest \
    artifacts/s07/rerun_video_proxies_<new-run-id>/proxy_manifest.json \
  --output \
    artifacts/s07/rerun_video_proxies_<new-run-id>/verification.json

.venv/bin/python scripts/s07/run_measured_final_assembly.py \
  --final-run-summary artifacts/s07/final_run_contract_20260805/summary.json \
  --presentation-video-manifest \
    artifacts/s07/rerun_video_proxies_<new-run-id>/proxy_manifest.json \
  --output-dir artifacts/s07/final_run_<new-run-id>

.venv/bin/rerun --serve-web \
  artifacts/s07/final_run_<new-run-id>/digital_twin_stage07_final.rrd

.venv/bin/python scripts/s07/verify_measured_final_assembly.py \
  --summary artifacts/s07/final_run_<new-run-id>/summary.json \
  --visual-qa-dir artifacts/s07/final_run_<new-run-id>/visual_qa \
  --visual-qa-passed \
  --output artifacts/s07/final_run_<new-run-id>/verification.json
```

Artifact-producing commands refuse to overwrite existing paths. Visual-QA
acceptance is explicit and cannot be inferred from file existence.

### Verification

- The refined RRD parsed and exposed all 20 required entity paths plus the
  authoritative `capture_time` timeline.
- All stable export fields matched accepted S06 semantics: 1,047 video-frame
  references per camera, 328 boxes, 298 segmentation frames, 320 presentation
  records, 23 measured trajectory segments, 160 interaction records, and the
  pickup/carry/place event identities.
- Independent proxy verification matched both original source hashes, proxy
  hashes, codec, pixel format, resolution, all 1,047 frames, and the 30-frame
  maximum keyframe gap. It records raw-source and spatial-output modification
  as false.
- Browser playback review scrubbed Camera A and Camera B at early, middle, and
  late capture times without an observed black frame. The metric-twin view
  showed no track at the start, partial measured trails mid-run, and the
  accumulated disconnected trails at the end.
- The coordinate view visibly exposed timestamped measured XYZ and provenance,
  display-only stale values, and unavailable XYZ for missing/occluded states.
- Carry retains transition frame 468 and separate Qwen review frame 567.
- Camera A, Camera B, the metric scene/cameras/zones/trajectories, coordinate
  log, state timeline, and event log passed new browser visual review.
- Missing/occluded XYZ, interpolated/stale trajectory segments,
  worker-completion timeline use, model inference, and live-capacity claims
  remain absent. Stale coordinates remain explicitly display-only.
- Focused presentation/finalization tests and all 279 project tests passed.
  Ruff and strict mypy across 113 source/script files passed. The lockfile and
  environment were unchanged, and `git diff --check` passed.

### Scope and Limitations

- The S07 Rerun is final presentation evidence, but the short demo video has
  not yet been produced.
- The timing is a single measured retained-output assembly run, not a
  distribution with warm/cold repetitions or tail latency.
- Accepted historical worker timings remain separate runs. A later S07 report
  must synthesize them without pretending they form one freshly measured
  end-to-end execution.
- The known backpack detector and 6.803-second localization gap remain visible
  limitations and are not filled.
- No S07 completion-gate criterion has been claimed or weakened.

## Work Package 3 - Reporting and Stage Close

Completed on 2026-08-05 after the user explicitly accepted the refined
interactive Rerun as the final demonstration.

### Final Demonstration Decision

D044 records that the verified interactive `.rrd` replaces the separate short
rendered-video deliverable. This is an explicit user-selected output
substitution, not a claim that a standalone demonstration MP4 was produced.
The completion gate itself remains unchanged: the accepted presentation must
show the intended synchronized movement and pickup-carry-place sequence.

### Durable Documentation

- `docs/CAPTURE_CALIBRATION_GUIDE.md` records equipment, fixed camera settings,
  ChArUco/marker procedure, world-frame pose gates, synchronization, recording
  structure, calibration invalidation, and acceptance checks.
- `docs/REPRODUCING_FINAL_DEMONSTRATION.md` records required local inputs and
  exact commands for entry verification, proxy generation/verification,
  retained-output assembly, playback, six-view visual QA, structural
  verification, and close audit.
- `docs/FINAL_TECHNICAL_REPORT.md` consolidates demonstrated value, final
  evidence, anchor/accuracy boundaries, failures, isolated M1 measurements,
  queue/model scheduling, offline-versus-live scope, potential alternatives,
  and exact production measurements/changes still required.
- `docs/stages/S07_HANDOFF.md` records the complete stage outcome, artifacts,
  commands, results, physical assumptions, limitations, decisions, and final
  version-control provenance.

### Completion Gate

Completion gate passed without weakening all five ROADMAP criteria:

1. The final run is reproducible from hash-bound inputs and documented
   commands. Fresh entry, proxy, and final-Rerun verifiers share one final-run
   manifest identity.
2. The six-view visually verified interactive demo shows both synchronized
   cameras, progressive measured movement from pickup toward drop-off, and the
   pickup-carry-place sequence.
3. Detector fragmentation, the null-XYZ carry gap, mixed person anchors,
   rejected disagreement, Qwen failures, proxy scope, RTSP limits, and absent
   dynamic ground truth are documented rather than concealed.
4. The report separates isolated offline model measurements, virtual-time
   scheduling, localhost RTSP evidence, and retained-output assembly from
   projected live capacity, and lists the SLO, load, tail-latency, memory,
   supervision, network, calibration, security, and ground-truth work required
   for production.
5. `docs/STATUS.md`, `docs/DECISIONS.md`, and all S00-S07 handoffs are current.

The close evidence is retained under
`artifacts/s07/stage_close_audit_20260805/`. All 279 tests, Ruff, strict mypy
across 114 source/script files, the 128-package lock, unchanged 106-package
environment, and whitespace checks passed.

## Exact Next Action

Stop. The approved S00-S07 roadmap is complete. Begin follow-on work only from
a new user-approved objective and scope.
