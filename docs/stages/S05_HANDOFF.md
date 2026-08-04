# S05 Handoff - Interaction State and Qwen Events

**Stage:** S05 - Interaction State and Qwen Events

**Status:** Complete with known detector, localization, and worker-timeout limitations

**Started:** 2026-08-03

**Closed:** 2026-08-04

## Stage Goal

Turn the verified person/backpack observations and accepted zones into an
honest pickup-carry-place sequence, represent occlusion without invented
object locations, and add bounded asynchronous Qwen semantic review that
cannot change spatial facts.

## Entry Inputs

- Closed S04 dense localization chain at commit
  `dd8a29a4111c7282351adf6a5926d1b699a18b7f` and tag
  `stage-04-da3-localization`.
- Preferred D033/D034 dense observations and presentation timeline.
- Accepted action recording `action_take_01`, synchronization manifest,
  capture-specific camera poses, and video-estimated pickup/drop-off zones.
- S03 five-FPS person/backpack timelines and D028's guarded one-bag
  `backpack` plus `handbag` policy.
- Exact cached `Qwen/Qwen3-VL-2B-Instruct` revision
  `89644892e4d85e24eaac8bacfd4f463576704203`.
- No new physical capture or calibration was required.

## Work Completed

- Implemented typed deterministic interaction contracts, capture-ordered
  transition logic, stable identities, zone/proximity authority checks, and
  explicit Qwen isolation.
- Diagnosed and corrected the original mutually exclusive carry/occluded
  model. D037 now separates interaction phase, optical visibility, and
  localization availability.
- Added a versioned, source-preserving visibility overlay. It uses affirmative
  synchronized-video review and never converts an ordinary detector miss into
  occlusion or supplies XYZ.
- Rebuilt D034 non-destructively so 33 unmeasured backpack ticks with explicit
  partial-occlusion evidence become `occluded` presentation records with null
  raw/presentation XYZ and no zone or trajectory authority.
- Produced the accepted 160-tick orthogonal semantic timeline. Frames
  `468-660` simultaneously report `carry`, `partially_occluded`, and
  `unavailable`, with sequence-continuity semantic authority and no current
  spatial authority.
- Implemented bounded, deduplicated Qwen jobs and results with immutable video,
  frame, candidate, prompt, model, queue, attempt, and timing provenance.
- Verified throttle, drop/coalescing, ordering, cancellation, retry, timeout,
  invalid-output, schema-tamper, and spatial-field rejection behavior.
- Retained failed Qwen v1/v2 experiments as diagnostic evidence. V1 truncated
  at 96 tokens; v2 produced verbose unquoted objects at 160 tokens. Neither
  was promoted from `unknown`.
- Added D040's hash-bound assistant JSON prefill. The adapter continues the
  assistant message and reconstructs retained raw output without post-hoc
  prose repair.
- Separated authoritative event-transition identity from semantic review
  time. Carry remains anchored at frame 468 but is reviewed around sustained
  carrying at frame 567, using paired frames 507, 567, and 627.
- Executed the accepted v4 plan once on serialized Apple MPS. Pickup, carry,
  and place returned schema-valid first-attempt matches with no retry or
  normalization.

## Accepted Result

- Interaction phase counts: 43 `at_pickup`, one `pickup`, 33 `carry`, 54
  `place`, and 29 `unknown` ticks.
- Visibility counts: 47 `visible`, 33 `partially_occluded`, and 80 `unknown`.
- Localization counts: 17 `measured`, 58 `stale`, and 85 `unavailable`.
- All 33 carry ticks are partially occluded and localization-unavailable with
  null backpack XYZ and zero spatial authority.
- Pickup transition: frame 462 / `15.406667 s`, measured spatial authority.
- Carry transition: frame 468 / `15.606667 s`, sequence-continuity semantic
  authority only; semantic review centre frame 567 / `18.900000 s`.
- Place transition: frame 666 / `22.210000 s`, measured spatial authority.
- Qwen results: direct schema-valid `pickup`, `carry`, and `place` matches,
  qualitative `strong` evidence, `48-52` output tokens, `4.76-5.13 s` per
  request, zero retry, zero normalization, and zero spatial writes.
- The measured backpack trajectory remains disconnected across its retained
  `6.803 s` localization gap. Semantic carry confirmation does not fill it.

## Changed Files

- `configs/s05_backpack_visibility_evidence.json`;
- `src/spatial_reconstruction/interaction/` and
  `src/spatial_reconstruction/perception/visibility.py`;
- project-owned Qwen adapter prefill support in
  `src/spatial_reconstruction/models/qwen_adapter.py`;
- evidence-aware S03/S04 derivation and presentation code in `scripts/s03/`,
  `scripts/s04/`, and `src/spatial_reconstruction/localization/`;
- S05 builders, runners, and independent verifiers in `scripts/s05/`;
- interaction, visibility, temporal, and Qwen tests in `tests/`;
- `docs/DECISIONS.md`, `docs/MODEL_SCHEDULING.md`, `docs/STATUS.md`, the S03/S04
  continuity records, the S05 stage record, and this handoff.

Raw captures, synchronized videos, generated model artifacts, cached weights,
the environment, and vendor source remain excluded from Git.

## Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Visibility overlay | `artifacts/s05/backpack_visibility_20260803/` | 160 reviewed visibility records; 33 affirmative partial-occlusion ticks |
| Occlusion-aware D034 | `artifacts/s04/temporal_presentation_occlusion_aware_20260803_v2/` | Null-XYZ occluded presentation while preserving measured states/segments |
| Orthogonal S05 timeline | `artifacts/s05/semantic_interaction_v2_20260803/` | Phase, visibility, localization, candidate events, diagnostics, and verification |
| Qwen v1 plan | `artifacts/s05/qwen_event_job_plan_20260803/` | Original bounded queue/schema experiment retained as history |
| Qwen v1 execution | `artifacts/s05/qwen_event_execution_v2_20260804/` | Six deterministic invalid outputs and safe unknown fallbacks |
| Qwen v2/v3 diagnostics | `artifacts/s05/qwen_event_job_plan_v2_20260804/`, `artifacts/s05/qwen_event_execution_v3_20260804/`, `artifacts/s05/qwen_event_job_plan_v3_20260804/`, `artifacts/s05/qwen_event_execution_v4_20260804/` | Retained schema and evidence-window failure diagnosis |
| Accepted Qwen v4 plan | `artifacts/s05/qwen_event_job_plan_v4_20260804/` | Verified jobs, source transitions, review centres, frames, prompts, and queue plan |
| Accepted Qwen execution | `artifacts/s05/qwen_event_execution_v5_20260804/` | Valid pickup/carry/place results, raw tokens, timings, frames, contact sheet, and verification |
| Close audit | `artifacts/s05/stage_close_audit_20260804/` | Fresh independent verification of all five accepted layers |

## Verification

### Commands

```text
.venv/bin/python scripts/s05/verify_visibility_evidence.py \
  --summary artifacts/s05/backpack_visibility_20260803/summary.json \
  --output <new-visibility-verification.json> --visual-qa-passed

.venv/bin/python scripts/s04/verify_temporal_presentation.py \
  --summary artifacts/s04/temporal_presentation_occlusion_aware_20260803_v2/summary.json \
  --output <new-temporal-verification.json> --visual-qa-passed

.venv/bin/python scripts/s05/verify_semantic_interaction_timeline.py \
  --summary artifacts/s05/semantic_interaction_v2_20260803/summary.json \
  --output <new-semantic-verification.json> --visual-qa-passed

.venv/bin/python scripts/s05/verify_qwen_event_job_plan.py \
  --summary artifacts/s05/qwen_event_job_plan_v4_20260804/summary.json \
  --output <new-qwen-plan-verification.json>

.venv/bin/python scripts/s05/verify_qwen_event_execution.py \
  --summary artifacts/s05/qwen_event_execution_v5_20260804/summary.json \
  --output <new-qwen-execution-verification.json>

.venv/bin/pytest -q
.venv/bin/ruff check .
MYPYPATH=src .venv/bin/mypy --strict src scripts
uv --cache-dir .cache/uv lock --check
uv --cache-dir .cache/uv sync --check
git diff --check
```

### Results

- All five accepted layers independently regenerated or revalidated from
  retained source hashes.
- Visibility audit: 160 records, 33 review-backed partial occlusions, zero
  automatic detector-miss occlusions, and zero supplied XYZ.
- Presentation audit: 320 records, 33 occluded backpack ticks, zero inferred
  positions, zero raw XYZ on non-measured states, zero unsupported occlusions,
  and no interpolation, extrapolation, or smoothing.
- Semantic audit: 160 records and pickup/carry/place candidates; all 33 carry
  ticks have partial occlusion, unavailable localization, null XYZ, and zero
  spatial authority.
- Qwen plan audit: three stable jobs, six frames each, source transitions
  `462/468/666`, review centres `462/567/666`, capacity three, and zero
  forbidden spatial fields.
- Qwen execution audit: exact revision and MPS/float16 verified; three direct
  candidate matches, zero normalization, raw tokens retained, and no spatial
  write interface.
- 259 automated tests passed. Ruff passed. Strict mypy passed across 86
  source/script files. The lock resolved 128 packages and the installed
  106-package environment required no changes. `git diff --check` passed.
- Visual QA passed for the synchronized carry interval, three-axis timeline,
  world/timeline occlusion presentation, and final Qwen contact sheet.
- No raw recording, synchronized source video, model weight, cache, or vendor
  file was modified.

## Completion Gate

- Representative recording produces a sensible pickup-carry-place sequence:
  **passed**.
- Qwen delay, timeout, invalid output, or failure cannot block or mutate
  perception, depth, geometry, or spatial state: **passed** within S05's
  separate-worker boundary. S06 retains the explicit supervisor-level hard
  timeout task.
- Qwen output conforms to the event schema or safely becomes `unknown`:
  **passed**.
- Qwen never changes coordinates, identities, capture timestamps, or zone
  membership: **passed**.
- Occlusion is represented without invented object locations: **passed**.

No roadmap completion criterion was skipped or weakened.

## Physical Setup and Observations

- No camera, lens, marker, mount, room, or calibration change was made in S05.
- S05 read only the accepted synchronized action recording and retained
  derived artifacts; raw and synchronized videos were not modified.
- The 13 mm-equivalent ultrawide selection, 1080p/30 FPS capture, action pose,
  marker coordinates, and video-estimated zones remain the governing physical
  provenance.
- Moving a camera or changing its selected lens invalidates calibration for a
  new capture, but does not invalidate this retained session's evidence.

## Problems and Limitations

- The baseline detector remains unreliable for the physical backpack during
  carrying. The accepted visibility overlay is a versioned affirmative review
  for this specific recording, not automatic general-purpose occlusion
  inference.
- Carry has semantic sequence and Qwen evidence but no measured XYZ inside the
  `6.803 s` gap. No continuous backpack trajectory is claimed.
- Qwen evidence strength is qualitative and uncalibrated. Valid semantic text
  cannot revise deterministic phase provenance or spatial facts.
- Qwen v1/v2 failed structured generation and v3 exposed a carry-window
  selection error. These failures are retained rather than hidden.
- An asynchronous thread timeout does not preempt an active MPS call. The
  Qwen runner is the isolatable process boundary; integrated supervisor-level
  hard termination/restart remains an explicit S06 requirement.
- Candidate “clips” are retained as bounded source intervals plus six exact
  synchronized evidence frames rather than separately encoded MP4 subclips.
- The prototype remains single-person, single-backpack, single-room, recorded
  MP4 research. No production, privacy, or calibrated-accuracy claim is made.

## Decisions Made

- D036 - initial measured-only state machine; superseded by D037.
- D037 - orthogonal interaction phase, visibility, and localization.
- D038 - bounded schema-only Qwen event review.
- D039 - invalid Qwen prose remains diagnostic and inference images are
  bounded.
- D040 - hash-bound JSON prefill and a separate sustained-carry review centre.

No new model, external service, dependency, triangulation method, motion prior,
or custom detector was introduced.

## Prerequisites for the Next Stage

- Consume the accepted S02 static scene, S01 calibration/synchronization,
  S03 perception timelines, S04 occlusion-aware presentation, S05 orthogonal
  interaction timeline, and accepted Qwen v4 results by exact artifact hash.
- Preserve source-transition time separately from Qwen semantic-review time.
- Preserve measured/stale/occluded/missing display semantics, disconnected
  trajectories, person anchor kinds, and null XYZ during carry.
- Integrate the existing file/RTSP source abstraction, bounded worker queues,
  and default single-MPS accelerator permit into Rerun orchestration.
- Add supervisor-level Qwen process timeout/restart and verify that geometry
  continues through delayed, invalid, failed, and terminated semantic work.
- Produce synchronized Rerun video/3D/event/diagnostic timelines and a local
  RTSP reconnect smoke test without implying production deployment.
- No new physical capture or user action is required to begin S06. A new
  capture would require rechecking synchronization and capture-specific pose.

## Version Control

- GitHub repository:
  `https://github.com/hiverlabzhengjie/3d-spatial-reconstruction-rd`
- Stage-close commit: `<pending stage-close commit>`
- Stage-close URL: `<pending remote push>`
- Annotated tag: `stage-05-interaction-events` (pending)
- Tag URL: `<pending remote push>`
- Remote push verified: No (pending stage-close publication)
- Qwen revision:
  `89644892e4d85e24eaac8bacfd4f463576704203`; YOLO checkpoint SHA-256:
  `a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c`;
  DA3 revision: `b2359bdf726fb44ef62acca04d629dcf158053e7`.

## Exact Next Action

After the stage-close commit, tag, push, and remote verification are recorded,
stop. Begin S06 only when explicitly requested, starting with the integrated
Rerun/file orchestration contract and supervisor-level worker lifecycle.
