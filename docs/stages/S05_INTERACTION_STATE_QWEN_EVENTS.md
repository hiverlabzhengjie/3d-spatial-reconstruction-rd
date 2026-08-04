# S05 - Interaction State and Qwen Events

## Status

Complete on 2026-08-04 with known detector, localization, and worker-timeout
limitations. D037's independent phase/visibility/localization state and D040's
schema-valid Qwen pickup-carry-place review passed the S05 completion gate.

## Entry Prerequisites

Verified on 2026-08-03 before implementation:

- S04 is closed at commit `dd8a29a4111c7282351adf6a5926d1b699a18b7f`
  with tag `stage-04-da3-localization`.
- The preferred dense D033/D034 corrected and presentation artifacts and their
  passed verification records are present.
- The accepted video-estimated pickup and drop-off zones and authoritative
  action capture timestamps are present.
- The exact `Qwen/Qwen3-VL-2B-Instruct` model is cached, although this work
  package does not invoke it.
- The repository was clean and local `main` matched `origin/main` at
  `e2608e3bbdd5f0d6aeb0cf1a364c5874841bf925`.
- No new physical capture or user action is required.

## Work Package 1 - Typed Interaction State Machine

Implemented policy `s05_interaction_state_v1` under D036.

### Contracts

- `InteractionZone` preserves the accepted metric zone identity, role, centre,
  radius, and coordinate-source provenance.
- `InteractionEvidence` joins one person and backpack D034 record at the exact
  same source frame and capture timestamp.
- `InteractionStateRecord` preserves source record IDs, original anchor kinds,
  transition reason, phase memory, and whether the transition has current
  spatial authority.
- Stable record IDs make persistent state outputs reproducible.

### State and Authority Rules

- States: `unknown`, `at_pickup`, `pickup`, `carry`, `place`, and `occluded`.
- Only paired current measured records can establish zone, departure, or
  person/backpack proximity facts.
- Zone and proximity checks use XY while preserving the source anchor kind.
- Stale, missing, inferred, and occluded records contribute no spatial facts.
- Ordinary unavailable evidence becomes `unknown`; only explicit source
  occlusion becomes `occluded`.
- Last authoritative phase may persist as non-spatial memory across an unknown
  or occluded tick, but the gap is never localized or labelled as carry.
- Placement requires a previously confirmed pickup; a recording that starts
  with the backpack at drop-off remains sequence-unproven.
- Qwen influence on spatial state is fixed to `false` in the policy and every
  state record.

### Initial Prototype Thresholds

- Pickup/drop-off membership: accepted `0.30 m` circular zone radius.
- Minimum pickup-centre departure: `0.30 m`.
- Maximum current person/backpack XY distance for pickup/carry: `1.0 m`.

These thresholds are provisional, inspectable trigger rules. They require
retained-video and artifact review before acceptance and are not accuracy or
probability claims.

## Verification to Date

Focused tests cover:

- a complete synthetic pickup, carry, unknown-gap, and place sequence;
- stale backpack coordinates being denied zone and transition authority;
- explicit occlusion without XYZ;
- rejection of an invented pickup before an initial drop-off observation;
- preservation of upper/lower-body anchor semantics;
- rejection of out-of-order capture time; and
- rejection of any policy that grants Qwen spatial-state authority; and
- persistent interaction-state schema round trips.

Verification results for this work package:

- `243` automated tests passed.
- Ruff passed across project source, tests, and scripts.
- Strict mypy passed across `56` source and S04 script files; the new S05
  package and focused test also pass strict typing.
- The lockfile resolves `128` packages, and the locked environment check found
  `106` packages with no changes required.
- `git diff --check` passed.

The next work package will build and independently verify a persistent state
timeline from the retained dense D034 artifact, then visually review candidate
event windows before Qwen integration begins.

## Work Package 2 - Retained Interaction Timeline and Candidate Windows

Completed on 2026-08-03 without model inference or changes to S04 evidence.
This v1 result is retained as diagnostic history and superseded for downstream
use by Work Package 3 / D037.

### Accepted Result

- Built and independently regenerated all `160` paired interaction ticks from
  the preferred dense D034 presentation artifact.
- State counts are eight `at_pickup`, one `pickup`, eight `place`, and 143
  `unknown`; there are zero `carry` and zero `occluded` ticks.
- The pickup transition is frame `462` at `15.406667 s`.
- The first place transition is frame `666` at `22.210 s`.
- Candidate generation emits one pickup and one place job window and
  deduplicates later `remains_placed` measurements after unknown ticks.
- All 143 unknown records have zero spatial transition authority. Invented XYZ,
  stale/missing zone facts, Qwen spatial influence, claimed occlusion, and
  known-gap filling are all zero.
- No threshold tuning was required.

The retained spatial evidence does not contain a separate current measured
carry tick. This does not mean carrying was absent: both synchronized videos
visibly show the backpack being carried at `18.808 s`, between the pickup and
place boundaries. The localization state remains unknown there because the
known S04 bag-evidence gap is not filled. Later Qwen work may describe this
visible interval but may not convert it into measured spatial state.

### Visual QA

The accepted contact sheet contains both cameras at the start, transition,
and end of each four-second candidate window, plus the carry-interval midpoint.

- Pickup window `13.406667-17.406667 s`: the backpack starts in the blue bed
  zone, is lifted at the measured transition, and is carried away by the end.
- Carry review `18.808333 s`: both views show the person carrying the backpack;
  the label explicitly preserves `unknown` spatial state.
- Place window `20.210-24.210 s`: the person approaches the white floor zone,
  the measured backpack enters it, and placement is visible by the end.

The state timeline and contact sheet passed visual review. The thresholds and
candidate boundaries are suitable for the next semantic work package.

### Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Run summary | `artifacts/s05/interaction_state_20260803_v2/summary.json` | Typed policy, sources, records, candidates, hashes, and limitations |
| State records | `artifacts/s05/interaction_state_20260803_v2/interaction_state_records.json` | All 160 capture-ordered state results |
| Candidate windows | `artifacts/s05/interaction_state_20260803_v2/interaction_event_candidates.json` | Deduplicated pickup/place review windows |
| Review CSV | `artifacts/s05/interaction_state_20260803_v2/interaction_state_review.csv` | State, reason, authority, zone, distance, and anchor columns |
| State timeline | `artifacts/s05/interaction_state_20260803_v2/interaction_state_timeline.png` | State authority and threshold diagnostics |
| Candidate contact sheet | `artifacts/s05/interaction_state_20260803_v2/candidate_event_contact_sheet.jpg` | Both cameras across pickup, carry review, and place |
| Verification | `artifacts/s05/interaction_state_20260803_v2/verification.json` | Independent regeneration, hashes, semantics, and visual-QA result |

The earlier `artifacts/s05/interaction_state_20260803/` run is retained as
diagnostic history. It has the same state/candidate result but a less complete
centre-frame-only contact sheet and is not the accepted downstream source.

### Reproduction Commands

```text
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python \
  scripts/s05/build_interaction_timeline.py \
  --output-dir artifacts/s05/interaction_state_<new-run-id>

.venv/bin/python scripts/s05/verify_interaction_timeline.py \
  --summary artifacts/s05/interaction_state_<new-run-id>/summary.json \
  --output artifacts/s05/interaction_state_<new-run-id>/verification.json \
  --visual-qa-passed
```

Both commands refuse to overwrite their outputs. The builder validates the
passed D034 verification, accepted zone metadata, synchronization manifest,
and both synchronized video hashes. The verifier reloads every source and
artifact hash and regenerates all records and candidates.

Project-wide verification after the accepted run:

- `243` automated tests passed.
- Ruff passed across project source, tests, and scripts.
- Strict mypy passed across `58` source and S04/S05 script files.
- The lockfile resolved `128` packages and the locked environment checked
  `106` packages with no changes required.
- `git diff --check` passed.

## Work Package 3 - D037 Visibility and Semantic-state Correction

Completed on 2026-08-03 as a non-destructive correction after reviewing S03,
S04, and both synchronized videos.

### Root causes corrected

1. S03's detector timeline correctly avoided guessing occlusion, but did not
   define how affirmative later review could be attached. Its derivation now
   declares an explicit-evidence policy while retained artifacts remain
   unchanged.
2. S04 hard-coded every tick as not occluded. Its builder/verifier now accept
   an optional hash-bound visibility overlay and use it only for backpack
   ticks without a current measurement.
3. S05 v1 put `carry` and `occluded` in one mutually exclusive enum and mapped
   every non-measured backpack tick to `unknown`. D037 replaces that model
   with independent interaction-phase, visibility, and localization axes.

### Verified result

- The visibility overlay covers all 160 ticks: 47 `visible`, 33
  `partially_occluded`, and 80 `unknown`. All 33 reviewed ticks span frames
  `468-660`; no detector miss was automatically converted to occlusion and no
  visibility record supplies XYZ.
- The corrected D034 artifact has 33 measured, 118 stale, 136 missing, 33
  occluded, and zero inferred target records. Every occluded record is a
  backpack tick with null XYZ and zero spatial authority. All 23 exact
  measured segments and the `6.803 s` localization gap remain unchanged.
- S05 v2 has 33 `carry` ticks from frame `468` / `15.606667 s` through frame
  `660` / `22.010 s`. Every one simultaneously reports
  `partially_occluded`, `unavailable`, null backpack XYZ, sequence-continuity
  phase authority, and no current spatial authority.
- Pickup remains frame `462` with measured spatial authority; place remains
  frame `666` with measured spatial authority. Carry begins at frame `468`
  with semantic sequence authority only.
- Across all ticks, phase counts are 43 `at_pickup` (8 measured and 35 bounded
  sequence continuity), one measured `pickup`, 33 continuity `carry`, 54
  `place` (8 measured and 46 sequence continuity), and 29 `unknown`.
- Candidate order is now pickup, carry, place. The carry candidate is
  explicitly non-spatial; pickup and place retain current measured authority.
- Independent verification found zero carry XYZ, zero carry spatial authority,
  zero invented XYZ, and zero Qwen spatial influence.
- Final project verification passed 246 automated tests, Ruff across source,
  tests, and scripts, strict mypy across 65 source/script files, the 128-package
  lock check, the 106-package environment check, and `git diff --check`.

### Corrected artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Visibility overlay | `artifacts/s05/backpack_visibility_20260803/` | Explicit reviewed visibility plus original per-camera detector states |
| Occlusion-aware D034 | `artifacts/s04/temporal_presentation_occlusion_aware_20260803_v2/` | Null-XYZ occlusion presentation driven only by explicit evidence |
| Orthogonal S05 v2 | `artifacts/s05/semantic_interaction_v2_20260803/` | Phase, visibility, localization, candidates, diagnostic, and verification |

### Reproduction commands

```text
.venv/bin/python scripts/s05/build_visibility_evidence.py \
  --output-dir artifacts/s05/backpack_visibility_<new-run-id>

.venv/bin/python scripts/s05/verify_visibility_evidence.py \
  --summary artifacts/s05/backpack_visibility_<new-run-id>/summary.json \
  --output artifacts/s05/backpack_visibility_<new-run-id>/verification.json \
  --visual-qa-passed

MPLCONFIGDIR=.cache/matplotlib .venv/bin/python \
  scripts/s04/build_temporal_presentation.py \
  --corrected-summary artifacts/s04/corrected_tracking_dense_20260803/summary.json \
  --corrected-verification artifacts/s04/corrected_tracking_dense_20260803/verification.json \
  --visibility-summary artifacts/s05/backpack_visibility_<new-run-id>/summary.json \
  --output-dir artifacts/s04/temporal_presentation_occlusion_aware_<new-run-id>

.venv/bin/python scripts/s04/verify_temporal_presentation.py \
  --summary artifacts/s04/temporal_presentation_occlusion_aware_<new-run-id>/summary.json \
  --output artifacts/s04/temporal_presentation_occlusion_aware_<new-run-id>/verification.json \
  --visual-qa-passed

MPLCONFIGDIR=.cache/matplotlib .venv/bin/python \
  scripts/s05/build_semantic_interaction_timeline.py \
  --output-dir artifacts/s05/semantic_interaction_v2_<new-run-id>

.venv/bin/python scripts/s05/verify_semantic_interaction_timeline.py \
  --summary artifacts/s05/semantic_interaction_v2_<new-run-id>/summary.json \
  --output artifacts/s05/semantic_interaction_v2_<new-run-id>/verification.json \
  --visual-qa-passed
```

## Work Package 4 - Bounded Qwen Event Jobs and Results

Completed on 2026-08-03 without loading or invoking the model.

### Contracts and queue

- Policy `s05_qwen_event_review_v1` fixes the exact passed Qwen revision,
  deterministic decoding, `96` maximum new tokens, `45 s` timeout, two total
  attempts, capacity three, and offline throttle-and-drain behavior.
- Every job retains candidate/state identity, capture frame/time, clip bounds,
  capture session, synchronization manifest, source videos and hashes, prompt
  identity/hash, model revision, attempt, priority, and stable job/dedup IDs.
- Each event uses six time-major inputs: before Camera A/B, transition Camera
  A/B, and after Camera A/B. Transition frame identities must exactly match
  the D037 candidate.
- Deduplication coalesces duplicate pending, in-flight, or completed logical
  events. A retry is accepted only after a non-completed terminal disposition,
  must increment the attempt by one, and cannot exceed attempt two.
- The asynchronous worker converts invalid JSON/schema, timeout, and processor
  failure into explicit terminal outcomes with schema-valid `unknown`
  interpretations. Geometry and state processing are outside this queue and
  cannot be blocked or mutated by it.
- Completed interpretations contain only event label, candidate match,
  qualitative evidence strength, concise summary, visible evidence,
  uncertainty, and `spatial_claims_present=false`. Extra coordinate fields or
  a true spatial-claims flag fail schema validation.

### Retained plan

The plan independently regenerates three unique jobs:

| Event | Transition | Clip | Ordered inputs | Authority entering Qwen |
|---|---:|---:|---:|---|
| Pickup | frame `462`, `15.406667 s` | `13.406667-17.406667 s` | 6 | Measured spatial source; Qwen annotation only |
| Carry | frame `468`, `15.606667 s` | `13.606667-17.606667 s` | 6 | Sequence-continuity source; no spatial authority |
| Place | frame `666`, `22.210000 s` | `20.210000-24.210000 s` | 6 | Measured spatial source; Qwen annotation only |

The capacity-three queue accepts all three unique jobs. Re-submitting the
pickup job is coalesced against its existing deduplication key. There are zero
drops, throttles, results, model calls, and forbidden spatial job fields in
this pre-execution artifact.

Final verification passed 256 project tests, Ruff, strict mypy across 68
source/script files, the 128-package lock check, the 106-package environment
check, artifact regeneration and hashes, and `git diff --check`.

### Generated artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Plan summary | `artifacts/s05/qwen_event_job_plan_20260803/summary.json` | Sources, policy, jobs, submissions, diagnostics, hashes, and limitations |
| Jobs | `artifacts/s05/qwen_event_job_plan_20260803/qwen_event_jobs.json` | Three immutable attempt-one jobs and all frame provenance |
| Prompt manifest | `artifacts/s05/qwen_event_job_plan_20260803/qwen_event_prompt_manifest.json` | Event-specific strict JSON prompts and hashes |
| Review CSV | `artifacts/s05/qwen_event_job_plan_20260803/qwen_event_job_review.csv` | Compact job, frame-sequence, prompt, model, timeout, and priority review |
| Verification | `artifacts/s05/qwen_event_job_plan_20260803/verification.json` | Independent regeneration, hashes, deduplication, and spatial-boundary checks |

### Reproduction commands

```text
.venv/bin/python scripts/s05/build_qwen_event_job_plan.py \
  --output-dir artifacts/s05/qwen_event_job_plan_<new-run-id>

.venv/bin/python scripts/s05/verify_qwen_event_job_plan.py \
  --summary artifacts/s05/qwen_event_job_plan_<new-run-id>/summary.json \
  --output artifacts/s05/qwen_event_job_plan_<new-run-id>/verification.json
```

## Work Package 5 - First Qwen Event Execution and Failure Evidence

Completed on 2026-08-04 with the exact cached model on Apple MPS.

- The first full-resolution attempt was interrupted before any semantic result
  was persisted because six `1920x1080` frames exposed that the asynchronous
  thread timeout cannot terminate an in-progress accelerator call. Its 18
  extracted frames and contact sheet remain in
  `artifacts/s05/qwen_event_execution_20260804/` as explicitly incomplete
  diagnostic evidence.
- The corrected execution bounded every actual inference image to a maximum
  dimension of `768` pixels, retained all 18 resized JPEGs, loaded the exact
  approved model once, and ran serially on `mps` with `float16` precision.
- The six bounded requests each used `2219` input tokens and completed in
  `8.23-8.72 s`, below the `45 s` attempt limit.
- All six generations—attempt one and the single allowed repair attempt for
  pickup, carry, and place—reached exactly `96` output tokens and returned
  truncated fenced JSON. The paired attempts were deterministic.
- Raw text, token IDs, token counts, input tensor shapes, timings, errors, and
  exact job/frame provenance are retained. Invalid-output results preserve
  those diagnostics while their typed interpretations safely remain
  `unknown`.
- The raw diagnostic text visibly identified pickup, carry, and place, and the
  reviewed contact sheet supports those descriptions. Because the responses
  were neither complete JSON nor schema-valid, none is accepted as an event
  fact.
- Independent verification passed for exact model revision, MPS/float16 use,
  one model load, all 18 frame artifacts and hashes, all six attempts, three
  final unknown results, raw-token retention, a decodable contact sheet, and
  absence of a spatial write interface.
- Final software checks passed all 256 project tests, Ruff, strict mypy across
  86 source/script files, the 128-package lock check, the 106-package
  environment check, and `git diff --check`.

### Generated artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Execution summary | `artifacts/s05/qwen_event_execution_v2_20260804/summary.json` | Model/runtime identity, outcomes, timings, hashes, and final labels |
| Frame manifest | `artifacts/s05/qwen_event_execution_v2_20260804/frame_manifest.json` | Exact 18-image inference evidence and source-frame provenance |
| Raw attempts | `artifacts/s05/qwen_event_execution_v2_20260804/qwen_attempt_results.json` | Six raw responses, diagnostics, errors, and unknown fallbacks |
| Final results | `artifacts/s05/qwen_event_execution_v2_20260804/qwen_final_results.json` | Latest bounded result for pickup, carry, and place |
| Contact sheet | `artifacts/s05/qwen_event_execution_v2_20260804/qwen_event_contact_sheet.jpg` | Visual review of all ordered event inputs |
| Verification | `artifacts/s05/qwen_event_execution_v2_20260804/verification.json` | Independent integrity, coverage, and authority checks |

### Reproduction commands

```text
.venv/bin/python scripts/s05/run_qwen_event_jobs.py \
  --output-dir artifacts/s05/qwen_event_execution_v2_<new-run-id>

.venv/bin/python scripts/s05/verify_qwen_event_execution.py \
  --summary artifacts/s05/qwen_event_execution_v2_<new-run-id>/summary.json \
  --output artifacts/s05/qwen_event_execution_v2_<new-run-id>/verification.json
```

## Work Package 6 - Schema-valid Qwen Pickup-Carry-Place Review

Completed on 2026-08-04 under D040.

### Revision evidence

- Policy v2 reduced the model-facing response to five semantic fields and
  raised the output bound from `96` to `160` tokens. The model still emitted
  deterministic unquoted, verbose, truncated objects, so all six retained v2
  attempts remained `invalid_output` and `unknown`.
- Policy v3 added an assistant prefill of `{"event_label":"`, included that
  prefill in prompt identity, and reconstructed the raw response without
  post-hoc prose repair. Pickup and place then produced valid first-attempt
  JSON. The carry candidate validly returned `pickup` because its original
  frame-468 review window emphasized lifting rather than sustained carrying.
- Policy v4 preserves frame `468` / `15.606667 s` as the immutable carry-onset
  candidate but centres semantic review at frame `567` / `18.900000 s`, the
  frame-aligned midpoint between carry onset and measured placement. Its carry
  evidence frames are `507`, `567`, and `627` from both cameras. Pickup and
  place review centres remain frames `462` and `666`.

### Accepted result

- The exact approved Qwen revision loaded once on MPS with float16. All three
  768-pixel-bounded requests completed on attempt one in `4.76-5.13 s` and
  used `48-52` output tokens, below the `160`-token and `45 s` bounds.
- Pickup, carry, and place each returned complete schema-valid JSON, matched
  its candidate, reported qualitative `strong` evidence, and required no
  retry or response normalization.
- The carry summary is “a woman carries a backpack”; its retained visible
  evidence describes the person carrying the black backpack. Visual review of
  both cameras across frames `507`, `567`, and `627` supports that statement
  while the deterministic spatial record remains partially occluded,
  localization-unavailable, null-XYZ, and non-spatial.
- `matches_candidate` is derived from the model label and expected candidate;
  `spatial_claims_present=false` is application-owned. Neither is entrusted to
  free-form model output.
- Independent verification passed exact model/runtime identity, all hashes and
  18 frame artifacts, the distinct source/review frame identities, three
  direct completed matches, zero normalization, raw-token retention, and the
  absence of a spatial write interface.
- The v1-v3 plans and results remain backward-compatible diagnostic history;
  none was overwritten or promoted.
- Final verification passed all 259 project tests, Ruff, strict mypy across 86
  source/script files, the 128-package lock check, the 106-package environment
  check, and `git diff --check`.

### Accepted artifacts

| Artifact | Location | Purpose |
|---|---|---|
| v4 plan | `artifacts/s05/qwen_event_job_plan_v4_20260804/` | Verified source candidates, separate review centres, prompts, jobs, and queue plan |
| v4 execution | `artifacts/s05/qwen_event_execution_v5_20260804/summary.json` | Accepted model/runtime summary and artifact hashes |
| Raw results | `artifacts/s05/qwen_event_execution_v5_20260804/qwen_attempt_results.json` | Three schema-valid raw responses with tokens, shapes, and timings |
| Final results | `artifacts/s05/qwen_event_execution_v5_20260804/qwen_final_results.json` | Accepted pickup, carry, and place interpretations |
| Contact sheet | `artifacts/s05/qwen_event_execution_v5_20260804/qwen_event_contact_sheet.jpg` | Both cameras across all three corrected review windows |
| Strong verification | `artifacts/s05/qwen_event_execution_v5_20260804/verification_v2.json` | Direct-match, normalization, integrity, and authority verification |

### Reproduction commands

```text
.venv/bin/python scripts/s05/build_qwen_event_job_plan.py \
  --output-dir artifacts/s05/qwen_event_job_plan_v4_<new-run-id>

.venv/bin/python scripts/s05/verify_qwen_event_job_plan.py \
  --summary artifacts/s05/qwen_event_job_plan_v4_<new-run-id>/summary.json \
  --output artifacts/s05/qwen_event_job_plan_v4_<new-run-id>/verification.json

.venv/bin/python scripts/s05/run_qwen_event_jobs.py \
  --plan-summary artifacts/s05/qwen_event_job_plan_v4_<new-run-id>/summary.json \
  --plan-verification artifacts/s05/qwen_event_job_plan_v4_<new-run-id>/verification.json \
  --output-dir artifacts/s05/qwen_event_execution_v5_<new-run-id>

.venv/bin/python scripts/s05/verify_qwen_event_execution.py \
  --summary artifacts/s05/qwen_event_execution_v5_<new-run-id>/summary.json \
  --output artifacts/s05/qwen_event_execution_v5_<new-run-id>/verification.json
```

### Exact Next Action

Publish and verify the S05 stage-close commit and annotated tag, record their
provenance in `docs/stages/S05_HANDOFF.md`, then stop. Begin S06 only when
explicitly requested.
