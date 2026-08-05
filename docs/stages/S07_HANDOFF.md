# S07 Handoff - Final Capture, Refinement, and Reporting

**Stage:** S07 - Final Capture, Refinement, and Reporting

**Status:** Complete with known limitations

**Started:** 2026-08-05

**Closed:** 2026-08-05

## Stage Goal

Produce the final demonstration and consolidate the project's reproducibility,
physical capture guidance, findings, failures, measured offline performance,
scheduling evidence, and future production requirements.

## Entry Inputs

- Closed S06 implementation commit
  `1ae09564934d322cf67e7b42e499431960c9b277` and tag
  `stage-06-rerun-rtsp`, with later provenance commit
  `6e4bef08425f3cdd37775781c91a9b4337086bd4` on local and remote `main`.
- User-selected `action_take_01`, two synchronized 1,047-frame MP4s, accepted
  action-specific camera poses, S02 static scene, S03 perception, S04
  localization/presentation, S05 interaction/Qwen results, and S06
  orchestration/export/replay/RTSP evidence.
- Fresh independent verification that all five retained S06 layers and the
  seven-criterion S06 gate remained passed without weakening.
- No camera, lens, calibration, raw recording, accepted model output, or vendor
  source changed.

## Work Completed

- Locked the final recording and seven required S06 evidence roles in stable
  manifest
  `9bb49beb13262f8108f4b37dd0974de6cc23f7fa456214e0c14e6c540a31ba08`.
- Added typed retained-output assembly contracts with exact commands, log
  hashes, measured wall times, and explicit no-model/no-live-capacity scope.
- Diagnosed 250-frame H.264 keyframe gaps as the camera-view seeking failure
  and produced hash-bound presentation proxies with a 30-frame maximum gap,
  leaving the original synchronized MP4s unchanged.
- Regenerated and browser-verified the preferred 44.0 MB interactive Rerun with
  both camera views, metric scene, current timestamped XYZ/provenance, and
  capture-time-progressive measured trails.
- Preserved all spatial boundaries: no missing/occluded XYZ, no interpolation,
  no smoothing, no cross-anchor path joining, no carry-gap filling, and no
  Qwen spatial authority.
- Recorded D044 after the user explicitly accepted the refined interactive
  Rerun as the final demonstration in place of a separate rendered MP4.
- Added the physical capture/calibration guide, exact final-demonstration
  reproduction guide, and final technical report.
- Consolidated isolated DA3, YOLO, Qwen, orchestration, RTSP, and assembly
  evidence without misrepresenting it as one measured end-to-end live run.
- Ran fresh S07 entry, proxy, and final-Rerun verifiers and passed all five
  roadmap criteria in one close audit without weakening a criterion.

## Changed Files

- `src/spatial_reconstruction/finalization/`;
- S07 final-run, proxy, assembly, verification, and close-audit commands in
  `scripts/s07/`;
- refined presentation behavior in
  `scripts/s06/export_integrated_rerun.py` and
  `src/spatial_reconstruction/orchestration/rerun_presentation.py`;
- finalization and presentation tests in `tests/`;
- `docs/CAPTURE_CALIBRATION_GUIDE.md`;
- `docs/REPRODUCING_FINAL_DEMONSTRATION.md`;
- `docs/FINAL_TECHNICAL_REPORT.md`;
- `docs/DECISIONS.md`, `docs/STATUS.md`, the S07 stage record, and this
  handoff.

Raw captures, synchronized videos, generated Rerun/proxy artifacts, model
outputs, weights, virtual environments, caches, and vendor source remain
excluded from Git.

## Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Final-run contract | `artifacts/s07/final_run_contract_20260805/` | Hash-bound selection of `action_take_01` and accepted S06 evidence |
| Initial assembly | `artifacts/s07/final_run_20260805/` | Immutable first 51.9 MB assembly retained as presentation history |
| Seekable proxies | `artifacts/s07/rerun_video_proxies_20260805/` | Two verified presentation-only H.264 camera assets and manifest |
| Preferred final Rerun | `artifacts/s07/final_run_v2_20260805/` | Refined `.rrd`, export/assembly summaries, verification, and six visual-QA images |
| Close audit | `artifacts/s07/stage_close_audit_20260805/` | Fresh three-layer verification and five-criterion S07 gate audit |

Key identities:

- final-run manifest:
  `9bb49beb13262f8108f4b37dd0974de6cc23f7fa456214e0c14e6c540a31ba08`;
- proxy manifest SHA-256:
  `58452a2bf280b4450a57748313f8bb977fddfe03e241c0049a0abcebc77d5809`;
- preferred final Rerun SHA-256:
  `bcf84af987069151339427d57d7642cffd0e92b6c0ff05bbdbddb7c6143b64ca`.

## Verification

### Commands

```text
.venv/bin/python scripts/s07/verify_final_run_manifest.py \
  --summary artifacts/s07/final_run_contract_20260805/summary.json \
  --output artifacts/s07/stage_close_audit_20260805/entry.json

.venv/bin/python scripts/s07/verify_rerun_video_proxies.py \
  --manifest artifacts/s07/rerun_video_proxies_20260805/proxy_manifest.json \
  --output artifacts/s07/stage_close_audit_20260805/proxies.json

.venv/bin/python scripts/s07/verify_measured_final_assembly.py \
  --summary artifacts/s07/final_run_v2_20260805/summary.json \
  --visual-qa-dir artifacts/s07/final_run_v2_20260805/visual_qa \
  --visual-qa-passed \
  --output artifacts/s07/stage_close_audit_20260805/rerun.json

.venv/bin/python scripts/s07/verify_stage07_gate.py \
  --entry artifacts/s07/stage_close_audit_20260805/entry.json \
  --proxies artifacts/s07/stage_close_audit_20260805/proxies.json \
  --rerun artifacts/s07/stage_close_audit_20260805/rerun.json \
  --output artifacts/s07/stage_close_audit_20260805/gate.json

.venv/bin/pytest -q
.venv/bin/ruff check .
MYPYPATH=src .venv/bin/mypy --strict src scripts
uv --cache-dir .cache/uv lock --check
uv --cache-dir .cache/uv sync --check
git diff --check
```

### Results

- All three S07 verification layers passed and share the final-run manifest
  identity.
- The refined RRD parses, exposes all 20 required paths, and retains 1,047
  frame references per camera, 16 person measurements, 17 backpack
  measurements, 23 same-anchor segments, 320 presentation records, 160
  interaction records, and three events.
- All six visual-QA views passed. Both camera views remained visible at early,
  middle, and late seek points; trails were absent at the start, partial
  mid-run, and accumulated with intended gaps at the end.
- The close audit passed all five S07 criteria: reproducibility, intended
  movement/events, honest limitations, offline/live separation and production
  requirements, and current continuity records.
- The complete project suite passed: 279 tests, Ruff, strict mypy across 114
  source/script files, 128-package lock resolution, unchanged 106-package
  environment, and `git diff --check`.

## Physical Setup and Observations

- S07 reused the accepted `action_take_01`; no recapture or recalibration was
  performed.
- Both iPhone 16 Pro Max cameras retained 1920x1080/30 FPS, the selected
  13 mm-equivalent ultrawide lens, fixed mounts, action pose version
  `s01_capture_20260729:action_take_01:v1`, and unchanged M40-M42 world anchors.
- D043 proxies are local presentation derivatives and cannot validate or
  replace the original sources, synchronization, or calibration.
- Any new recording after camera/lens movement must follow the new capture and
  calibration guide and receive its own source/synchronization/pose identity.

## Problems and Limitations

- The backpack detector is unreliable during carrying and depends on a guarded
  `backpack` plus `handbag` policy specific to this single-bag prototype.
- The 6.803-second carry interval has no measured backpack XYZ. Semantic carry
  state and Qwen description do not fill it.
- Person observations include footpoints and explicit lower/upper-body surface
  fallbacks. One 0.377 m camera disagreement is rejected.
- No surveyed dynamic trajectory exists; absolute localization accuracy is not
  claimed.
- Zone locations are video-estimated and marker coordinates retain stated
  physical measurement uncertainty.
- Qwen is approximately five seconds per accepted six-image review and remains
  asynchronous and qualitative.
- Isolated model timings, virtual orchestration, localhost RTSP, and measured
  retained-output assembly do not demonstrate a sustainable live service.
- RTSP production networking, multiple cameras, privacy/security operations,
  long-duration reliability, end-to-end SLOs, and p95/p99 latency remain
  outside the prototype.
- A separate rendered demonstration MP4 was not produced under D044; the user
  accepted the richer interactive Rerun as the final demonstration.
- No completion-gate criterion was weakened.

## Decisions Made

- D042 - Retain `action_take_01` as the final demonstration recording.
- D043 - Use seekable Rerun video proxies and progressive measured trails.
- D044 - Treat the refined interactive Rerun as the final demonstration
  artifact instead of producing a redundant rendered MP4.

No new model, spatial-localization method, production service, or Python
runtime dependency was introduced.

## Prerequisites for the Next Stage

There is no next roadmap stage; S07 closes the approved S00-S07 programme.
Any follow-on work must begin with a new explicit scope and decision. It must
preserve the final source/artifact hashes and should not silently modify this
completed baseline.

For production-oriented follow-on work, first define service-level objectives,
privacy/security requirements, representative workload, dynamic ground truth,
and the hardware/test environment required to measure complete end-to-end and
p95/p99 behavior.

## Version Control

- GitHub repository:
  `https://github.com/hiverlabzhengjie/3d-spatial-reconstruction-rd`
- Stage-close commit:
  `dfb08de49de14701f4e8c9e21ba8813aad4b8db8`
- Stage-close URL:
  `https://github.com/hiverlabzhengjie/3d-spatial-reconstruction-rd/commit/dfb08de49de14701f4e8c9e21ba8813aad4b8db8`
- Annotated tag: `stage-07-final-demonstration`
- Tag URL:
  `https://github.com/hiverlabzhengjie/3d-spatial-reconstruction-rd/tree/stage-07-final-demonstration`
- Remote push verified: Yes. On 2026-08-05, remote `main` resolved to the
  stage-close commit and the annotated tag dereferenced exactly to it. This
  later provenance-only documentation update records those values; the tag
  remains fixed at the verified close commit.
- Model revisions remain unchanged: DA3
  `b2359bdf726fb44ef62acca04d629dcf158053e7`, Qwen
  `89644892e4d85e24eaac8bacfd4f463576704203`, and YOLO checkpoint SHA-256
  `a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c`.

## Exact Next Action

Stop. The approved eight-stage roadmap is complete. Begin follow-on work only
after the user defines a new objective and its scope.
