# S06 Handoff - Rerun Presentation and RTSP Compatibility

**Stage:** S06 - Rerun Presentation and RTSP Compatibility

**Status:** Complete with known offline, detector, localization, and RTSP-test limitations

**Started:** 2026-08-05

**Closed:** 2026-08-05

## Stage Goal

Assemble the Digital Twin-style presentation and prove protocol-level RTSP
compatibility while preserving capture-time authority, bounded worker
scheduling, serialized heavy-MPS access, and honest missing spatial evidence.

## Entry Inputs

- Closed S05 interaction chain at commit
  `6cdcd12de055f0ffe357d1fd2e8fdcd6c077faab` and tag
  `stage-05-interaction-events`.
- Local `main` matching `origin/main` at
  `00b250b4356c48b1477e6347ab9bbcd49d750ab8` with a clean worktree.
- Accepted synchronized 1,047-frame action videos, capture-specific camera
  poses, S02 static scene, S03 perception timeline, S04 occlusion-aware
  presentation, S05 orthogonal interaction timeline, and Qwen v4 results.
- Exact cached DA3, YOLO, and Qwen model identities and the unchanged DA3
  vendor checkout.
- No new physical capture, calibration, or user action was required.

## Work Completed

- Added a stable hash-bound orchestration manifest covering nine accepted
  artifact roles and both synchronized videos.
- Implemented a hard subprocess worker boundary with bounded restart,
  terminate/kill escalation, retained diagnostics, and explicit degraded
  outcomes; Qwen failure cannot block geometry.
- Produced and visually verified the 51.9 MB
  `digital_twin_stage06_v2.rrd` with synchronized Camera A/B video, calibrated
  scene/frustums/zones, detections, state-aware tracks, exact measured
  trajectories, interaction state, and pickup/carry/place logs.
- Implemented deterministic virtual-time replay across five bounded queues,
  including backpressure, retry, coalescing, idempotency, degraded results,
  graceful shutdown, and one serialized heavy-MPS permit.
- Proved that deliberately different worker completion schedules yield the
  same capture-ordered output digest.
- Added D041's isolated MediaMTX localhost fixture and a finite RTSP reconnect
  contract preserving identity, source/capture offsets, and the observed
  outage gap.
- Decoded 45 frames across a deliberate publisher outage and recovery, then
  validated the final frame through the ordinary perception-worker contract.
- Exported all 320 typed presentation records, 23 exact measured trajectory
  segments, and three typed events with transition/review and spatial-authority
  boundaries explicit.
- Reran all five independent S06 verifiers and passed one reproducible audit
  of all seven roadmap completion criteria without weakening any criterion.

## Changed Files

- `configs/mediamtx_s06_local.yml`;
- `src/spatial_reconstruction/orchestration/`;
- `src/spatial_reconstruction/ingestion/reconnect.py` and ingestion exports;
- S06 builders, runners, exporters, and independent verifiers in
  `scripts/s06/`;
- orchestration and ingestion tests in `tests/`;
- `docs/DECISIONS.md`, `docs/STATUS.md`, the S06 stage record, and this
  handoff.

Raw captures, synchronized videos, accepted model artifacts, cached weights,
the virtual environment, generated artifacts, and vendor source remain
excluded from Git.

## Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Orchestration contract | `artifacts/s06/orchestration_contract_v2_20260805/` | Hash-bound accepted inputs and single-MPS scheduling policy |
| Integrated Rerun | `artifacts/s06/integrated_rerun_20260805/` | Accepted `.rrd`, export summary, verification, and five visual-QA images |
| Integrated replay | `artifacts/s06/integrated_replay_v2_20260805/` | Deterministic queues, failures, ordering, diagnostics, and shutdown evidence |
| RTSP smoke | `artifacts/s06/rtsp_smoke_v4_20260805/` | Local open/outage/reconnect identity, process logs, and verification |
| Dedicated exports | `artifacts/s06/exports_20260805/` | Typed track states, measured trajectory segments, and events |
| Close audit | `artifacts/s06/stage_close_audit_20260805/` | Fresh five-layer verification and unified seven-criterion gate audit |

Key accepted identities:

- orchestration manifest:
  `87a1c225049f167d6b5f87632d953d2d242ac7479eb20a33bfc24393f359a8f7`;
- Rerun recording SHA-256:
  `0ec24e52ee4ab592bb02d9c2c30bbca5f455129466421f8b2ee2bb612f8d1fe9`;
- replay capture-output digest:
  `746c0f1175982dbd61a13514c7c4398f3de8ff65960ab39335692ea03a5ead9b`;
  and
- dedicated export summary SHA-256:
  `8a9e1072a31a8f383e8bdc70a8e90d29d5204964ef17515c2cade4bb2c6e27bc`.

## Verification

### Commands

```text
.venv/bin/python scripts/s06/verify_orchestration_manifest.py \
  --summary artifacts/s06/orchestration_contract_v2_20260805/summary.json \
  --output <new-orchestration-verification.json>

.venv/bin/python scripts/s06/verify_integrated_rerun.py \
  --export-summary \
    artifacts/s06/integrated_rerun_20260805/digital_twin_stage06_v2_export_summary.json \
  --visual-qa-dir artifacts/s06/integrated_rerun_20260805 \
  --visual-qa-passed --output <new-rerun-verification.json>

.venv/bin/python scripts/s06/verify_integrated_replay.py \
  --summary artifacts/s06/integrated_replay_v2_20260805/summary.json \
  --output <new-replay-verification.json>

.venv/bin/python scripts/s06/verify_rtsp_smoke.py \
  --summary artifacts/s06/rtsp_smoke_v4_20260805/summary.json \
  --output <new-rtsp-verification.json>

.venv/bin/python scripts/s06/verify_tracks_events.py \
  --summary artifacts/s06/exports_20260805/summary.json \
  --output <new-export-verification.json>

.venv/bin/python scripts/s06/verify_stage06_gate.py \
  --orchestration <new-orchestration-verification.json> \
  --rerun <new-rerun-verification.json> \
  --replay <new-replay-verification.json> \
  --rtsp <new-rtsp-verification.json> \
  --exports <new-export-verification.json> \
  --output <new-stage-gate-summary.json>

.venv/bin/pytest -q
.venv/bin/ruff check .
MYPYPATH=src .venv/bin/mypy --strict src scripts
uv --cache-dir .cache/uv lock --check
uv --cache-dir .cache/uv sync --check
git diff --check
```

### Results

- All five accepted S06 layers passed fresh independent verification and bind
  to orchestration manifest
  `87a1c225049f167d6b5f87632d953d2d242ac7479eb20a33bfc24393f359a8f7`.
- The RRD parses, carries 16 required paths and 1,047 frame references per
  camera, and passed visual QA for both videos, metric twin, state timeline,
  and event logs.
- Different worker completion schedules preserve one capture-ordered result;
  five queues remain bounded, explicit degraded states persist, Qwen is
  non-blocking, and shutdown drains cleanly.
- One heavy-MPS permit reaches maximum occupancy one with no overlapping
  intervals and is released.
- The RTSP fixture reconnects after a deliberate outage through four bounded
  attempts, retains a 1.747282-second observed gap, and produces a compatible
  worker job.
- Dedicated exports regenerate exactly: 320 track states, 23 measured
  segments, and three events. Unsupported XYZ, interpolation, stale segments,
  and Qwen spatial writes are all zero.
- All seven S06 roadmap completion criteria passed; none was skipped or
  weakened.
- 273 automated tests passed. Ruff passed. Strict mypy passed across 104
  source/script files. The lock resolved 128 packages and the installed
  106-package environment required no changes. `git diff --check` passed.

## Physical Setup and Observations

- No camera, lens, marker, mount, room, or calibration change was made in S06.
- The accepted 13 mm-equivalent ultrawide selection, 1080p/30 FPS action
  capture, action-specific pose correction, marker coordinates, and
  video-estimated zones remain the governing physical provenance.
- The RTSP smoke republishes a downscaled derivative locally and does not
  modify or replace the accepted synchronized source recording.
- Moving a camera or changing its selected lens invalidates calibration for a
  new S07 capture; retained S01-S06 session evidence remains reproducible.

## Problems and Limitations

- The recording and exports demonstrate retained offline results, not
  production live capacity or measured end-to-end throughput.
- Replay queue/latency evidence uses deterministic virtual time and makes no
  M1 throughput, memory-pressure, or sustainable-rate claim.
- The physical backpack detector remains unreliable during carrying. The
  accepted visibility evidence is specific to this recording.
- The backpack has no measured XYZ through the 6.803-second carry gap. Qwen
  text confirms semantics but never fills or smooths that gap.
- RTSP evidence covers one unauthenticated localhost stream; jitter, packet
  loss, TLS, multiple cameras, clock drift, and long-duration behavior remain
  untested.
- Rerun H.264 playback requires a local FFmpeg executable. Visual QA used the
  localhost web viewer after an unrelated native macOS surface-size failure.
- Zones are video-estimated, Qwen evidence strength is qualitative, and no
  surveyed dynamic ground truth or production accuracy claim exists.
- No completion-gate criterion was weakened.

## Decisions Made

- D041 - MediaMTX localhost fixture and bounded RTSP reconnect identity.

No new model, spatial method, production service, or Python runtime dependency
was introduced.

## Prerequisites for the Next Stage

- Consume the closed S06 orchestration manifest, `.rrd`, replay, RTSP smoke,
  and dedicated exports by exact retained hash.
- Keep both synchronized action videos, action pose, S02 static scene, S04
  occlusion-aware presentation, S05 semantic timeline, and Qwen v4 results
  available unchanged.
- Preserve capture-time authority, null XYZ for unavailable states,
  disconnected measured trajectories, source-transition versus semantic-review
  time, Qwen's non-spatial boundary, bounded queues, and one heavy-MPS permit.
- Decide whether the accepted `action_take_01` is the final demonstration
  recording. If a new final capture is chosen, recheck synchronization,
  capture-specific pose, fixed lens/settings, and all physical movement before
  dependent processing.
- Produce the reproducible final run, final Rerun recording, short demo video,
  capture/calibration guide, reproduction commands, and concise technical
  report required by S07.
- Separate demonstrated offline/virtual-time evidence from any projected live
  capacity and state what would need measurement for production deployment.

## Version Control

- GitHub repository:
  `https://github.com/hiverlabzhengjie/3d-spatial-reconstruction-rd`
- Stage-close commit: `1ae09564934d322cf67e7b42e499431960c9b277`
- Stage-close URL:
  `https://github.com/hiverlabzhengjie/3d-spatial-reconstruction-rd/commit/1ae09564934d322cf67e7b42e499431960c9b277`
- Annotated tag: `stage-06-rerun-rtsp`
- Tag URL:
  `https://github.com/hiverlabzhengjie/3d-spatial-reconstruction-rd/tree/stage-06-rerun-rtsp`
- Remote push verified: Yes. On 2026-08-05, remote `main` resolved to the
  stage-close commit and the annotated tag dereferenced exactly to it. A
  subsequent provenance-only documentation commit records these values; the
  stage-close tag remains fixed at the verified implementation commit.
- Model/vendor revisions remain those closed in S05: Qwen
  `89644892e4d85e24eaac8bacfd4f463576704203`, YOLO checkpoint SHA-256
  `a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c`,
  and DA3 revision `b2359bdf726fb44ef62acca04d629dcf158053e7`.

## Exact Next Action

Stop. Begin S07 only when explicitly requested, starting by confirming whether
the accepted `action_take_01` will be the final demonstration recording.
