# S04 Handoff - DA3-Depth 3D Localization and Fusion

**Stage:** S04 - DA3-Depth 3D Localization and Fusion

**Status:** Complete with known sparse-observation limitations

**Started:** 2026-08-01

**Closed:** 2026-08-03

## Stage Goal

Convert the retained person and backpack masks into honest per-camera and
cross-camera world-space observations using pose-conditioned DA3 metric depth
from the same synchronized action content. Preserve exact source/depth
identity, freshness, missing-data state, anchor meaning, and the separation
between measured and presentation trajectories.

## Entry Inputs

- Completed S01 synchronization/calibration and S03 perception timelines.
- Accepted `action_take_01`: 1,047 complete synchronized frame pairs with no
  missing camera frames and at most `6.667 ms` inter-camera offset.
- Action pose `s01_capture_20260729:action_take_01:v1`, camera intrinsics,
  room bounds, and video-estimated pickup/drop-off zones.
- Exact DA3 revision `b2359bdf726fb44ef62acca04d629dcf158053e7` and
  unmodified vendor fingerprint
  `683cad1fec1186cd2a22f2b6d083b73d4c83c7ab1140f45ba24876612bc51d43`.
- S03 source-sized masks and explicit per-camera person/backpack states under
  D028's one-bag `backpack` plus `handbag` policy.

## Work Completed

- Added typed, source-preserving contracts for action-depth jobs, exact mask
  alignment, confidence-filtered visible surfaces, target anchors,
  cross-camera states, and temporal presentation.
- Ran pose-conditioned DA3 at process resolution 504 on the preferred dense
  profile of 17 synchronized dynamic pairs.
- Formalized D025 action-pair marker scaling. Every pair uses one shared scale
  derived from current M40-M42 observations in both cameras; raw depth and
  confidence remain unchanged and no camera-specific fallback is allowed.
- Selected D030's candidate-relative `p20` confidence rule inside each current
  target mask, preventing globally high confidence from retaining locally weak
  foreground samples.
- Preserved visible-surface measurements separately from semantic anchors.
  Person footpoints require margin-valid lower-body evidence; cropped views
  remain explicit upper-body observations, and synchronized mate views can
  supply a stronger same-time footpoint.
- Applied D032/D033 fusion only to same-job, same-kind, current observations
  within the `0.35 m` disagreement gate. Single-view states remain labelled
  single-camera and disagreement states emit no combined XYZ.
- Applied D034 presentation semantics: exact measurements alone have raw XYZ;
  stale positions are display-only for at most one second; missing remains
  without coordinates; no interpolation, extrapolation, or smoothing occurs.
- Added D035's mask-aware dense profile while retaining the sparse run as a
  comparison baseline and leaving the known backpack evidence gap unfilled.
- Produced updated, visually accepted measured-segment and timeline
  diagnostics that distinguish original/dense keyframes, anchor kinds,
  rejected disagreement, stale display, missing data, and disconnected gaps.

## Accepted Result

- 17 raw DA3 predictions; every view contains `141120` finite positive depth
  and finite confidence values with exact source provenance.
- 17 D025 shared scales from `5-6` accepted marker observations per pair;
  scale range `1.093693-1.170350`, maximum within-pair marker deviation
  `1.259%` against the `5%` limit.
- 44 exactly aligned current masks, 44 corrected surfaces, 44 anchors, and 34
  target-pair observations.
- Person: 16 usable measurements (three fused, 13 single-camera), with one
  honest frame-828 disagreement at `0.377 m` and no combined XYZ.
- Backpack: 17 single-camera visible-cluster measurements; no unsupported
  claim of two-camera backpack fusion.
- Final 320-slot presentation: 33 measured, 123 stale, 164 missing, zero
  inferred and zero claimed-occluded states.
- 23 exact same-kind measured segments: eight person and 15 backpack. The
  `6.803 s` backpack gap remains disconnected.
- Backpack measured endpoints move `2.545 m` in 3D, from `0.128 m` in XY of
  the pickup-zone centre to `0.166 m` from the drop-off-zone centre.
- Dense person display coverage increases from `47/160` to `76/160` ticks;
  backpack coverage increases from `47/160` to `80/160`. These are temporal
  evidence gains, not absolute XYZ-accuracy measurements.

### Post-close D037 clarification

S04 remains closed. On 2026-08-03, S05 supplied a separate affirmative
synchronized-video visibility overlay for the carry interval. The D034
builder/verifier gained an optional evidence input and produced a new
non-destructive run with 33 backpack `occluded` ticks at frames `468-660`.
Those ticks have no XYZ or spatial authority. The original dense D034 artifact
and every measured observation/segment remain unchanged diagnostic history.

## Changed Files

- `configs/s04_action_keyframes.json` and
  `configs/s04_action_keyframes_dense.json`;
- `src/spatial_reconstruction/localization/`;
- `scripts/s04/`;
- S04 localization tests in `tests/`;
- `docs/DECISIONS.md`, `docs/MODEL_SCHEDULING.md`, and `docs/STATUS.md`;
- `docs/stages/S04_DA3_DEPTH_LOCALIZATION_FUSION.md`; and
- this handoff.

Raw captures, vendor source, model weights, generated artifacts, environments,
and caches remain excluded from Git.

## Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Dense keyframe profile | `configs/s04_action_keyframes_dense.json` | Preferred 17-pair selection and reasons |
| Raw dynamic DA3 | `artifacts/s04/action_depth_dense_20260803/` | Depth/confidence, source identity, pose, runtime, and previews |
| D025 marker scaling | `artifacts/s04/action_depth_scale_dense_20260803/` | Shared scales, corrected arrays, marker sheet, and verification |
| Exact mask alignment | `artifacts/s04/mask_alignment_dense_20260803/` | Current source-mask mappings and 44-mask contact sheet |
| Corrected D030-D033 | `artifacts/s04/corrected_tracking_dense_20260803/` | Surfaces, semantic anchors, pair states, and world diagnostics |
| Final D034/D035 presentation | `artifacts/s04/temporal_presentation_dense_final_20260803_v2/` | States, segments, CSVs, updated diagnostics, verification, and density comparison |
| Sparse presentation baseline | `artifacts/s04/temporal_presentation_20260803_v3/` | Verified eight-pair comparison baseline |
| D037 occlusion-aware presentation | `artifacts/s04/temporal_presentation_occlusion_aware_20260803_v2/` | Explicit-evidence occlusion states with null XYZ; measured evidence unchanged |

## Verification

### Commands

```text
.venv/bin/python scripts/s04/verify_action_depth_preflight.py \
  --summary artifacts/s04/action_depth_dense_20260803/summary.json \
  --output <new-verification.json>

.venv/bin/python scripts/s04/verify_action_depth_scale.py \
  --summary artifacts/s04/action_depth_scale_dense_20260803/summary.json \
  --output <new-verification.json> --visual-qa-passed

.venv/bin/python scripts/s04/verify_mask_alignment.py \
  --summary artifacts/s04/mask_alignment_dense_20260803/summary.json \
  --output <new-verification.json> --visual-qa-passed

.venv/bin/python scripts/s04/verify_corrected_tracking.py \
  --summary artifacts/s04/corrected_tracking_dense_20260803/summary.json \
  --output <new-verification.json> --visual-qa-passed

.venv/bin/python scripts/s04/verify_temporal_presentation.py \
  --summary artifacts/s04/temporal_presentation_dense_final_20260803_v2/summary.json \
  --output <new-verification.json> --visual-qa-passed

.venv/bin/python scripts/s04/compare_temporal_density.py \
  --baseline-summary artifacts/s04/temporal_presentation_20260803_v3/summary.json \
  --baseline-verification artifacts/s04/temporal_presentation_20260803_v3/verification.json \
  --dense-summary artifacts/s04/temporal_presentation_dense_final_20260803_v2/summary.json \
  --dense-verification artifacts/s04/temporal_presentation_dense_final_20260803_v2/verification.json \
  --output <new-comparison.json>

.venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
MYPYPATH=src .venv/bin/mypy --strict src scripts/s04
uv lock --check
uv sync --locked --dry-run
git diff --check
```

Artifact builders and verifiers refuse to overwrite existing outputs. Fresh
DA3 generation requires the exact cached model and Apple MPS; independent
verification of retained artifacts does not rerun model inference.

### Results

- `235` automated tests passed.
- Ruff passed across project source, tests, and scripts.
- Strict mypy passed across `54` source and S04 script files.
- The lockfile resolved 128 packages; the locked environment checked 106
  packages and would make no changes.
- Fresh independent verification regenerated all 17 predictions and scales,
  44 alignments/surfaces/anchors, 34 pair states, 320 presentation records,
  23 segments, and the sparse/dense comparison from retained hashes.
- Transform inversion, camera/world round trips, reprojection,
  back-projection, invalid/non-positive/low-confidence filtering, static-depth
  rejection, exact-frame joins, reversed worker completion, missing cameras,
  missing targets, hidden feet, cross-camera disagreement, stale authority,
  and persistent schema behaviour passed.
- Visual QA passed for marker scaling, exact mask alignment, margin-aware
  person semantics, corrected world observations, the updated timeline, and
  the updated measured-segment preview.
- `git diff --check` passed. No raw recording or DA3 vendor file was modified.

## Completion Gate

- Person and backpack locations qualitatively plausible: **passed**.
- Backpack moves from pickup side toward drop-off side: **passed**.
- Foreground cannot silently reuse empty-room depth: **passed**.
- Fusion uses only current, valid, temporally compatible dynamic evidence:
  **passed**.
- Out-of-order completion cannot misjoin or reorder frames: **passed**.
- Missing observations generate no fabricated XYZ: **passed**.
- Coordinate and back-projection tests: **passed**.

No completion-gate criterion was skipped or weakened.

## Physical Setup and Observations

- No physical camera, lens, marker, mount, or room change was made in S04.
- Only accepted synchronized derived recordings were read; raw and derived
  recordings were not modified.
- The 13 mm-equivalent ultrawide selection, 1080p/30 FPS capture, action pose,
  floor-marker coordinates, and video-estimated zones remain the governing
  physical provenance.
- Any camera or selected-lens movement invalidates pose calibration for a new
  capture, but does not invalidate this retained action-session evidence.

## Problems and Limitations

- The baseline detector does not provide continuous backpack masks. The main
  two-camera backpack absence remains missing rather than inferred.
- All accepted backpack observations are single-camera; the capture provides
  no same-frame two-camera bag-mask pair for geometric fusion.
- One dense person pair is rejected because comparable footpoints disagree by
  `0.377 m`, above the prototype `0.35 m` gate.
- Lower/upper-body fallbacks are visible-surface observations, not anatomical
  centres or invented floor contact, and stay separate from footpoint paths.
- D030 confidence, D032 disagreement, D033 reliability, and D034 freshness
  thresholds are transparent prototype policies, not calibrated probabilities
  or production accuracy guarantees.
- No surveyed dynamic ground-truth trajectory exists, so stage accuracy is
  established qualitatively and through internal geometry/provenance tests,
  not absolute trajectory error.

## Decisions Made

- D030 - selected candidate-relative p20 confidence filtering.
- D031 - preserve raw per-camera visible surfaces before anchor derivation.
- D032 - margin-aware person anchors and same-kind disagreement eligibility.
- D033 - inspectable reliability weighting and honest pair states.
- D034 - conservative presentation without inferred motion.
- D035 - mask-aware dense dynamic DA3 keyframes.
- D025 was amended to formalize isolated action-pair marker scaling.

No non-baseline model, triangulation method, motion prior, or new dependency
was introduced.

## Prerequisites for the Next Stage

- Consume the verified dense D033/D034 records, not the earlier unscaled
  diagnostic products.
- Preserve measured/stale/missing authority: only current measured positions
  may establish zone or interaction spatial facts.
- Preserve person anchor kinds and do not convert body-surface observations
  into footpoints.
- Treat the frame-828 person disagreement and backpack evidence gap as unknown
  spatial input, not as coordinates to fill.
- Reuse the accepted pickup/drop-off zone metadata and authoritative capture
  timestamps.
- Exact `Qwen/Qwen3-VL-2B-Instruct` is cached for asynchronous semantic work;
  Qwen may describe events but may not modify coordinates, identities,
  timestamps, or zone membership.
- No additional physical capture or user action is required to begin S05.

## Version Control

- GitHub repository:
  `https://github.com/hiverlabzhengjie/3d-spatial-reconstruction-rd`
- Stage-close commit: `dd8a29a4111c7282351adf6a5926d1b699a18b7f`
- Stage-close URL:
  `https://github.com/hiverlabzhengjie/3d-spatial-reconstruction-rd/commit/dd8a29a4111c7282351adf6a5926d1b699a18b7f`
- Annotated tag: `stage-04-da3-localization`
- Tag URL:
  `https://github.com/hiverlabzhengjie/3d-spatial-reconstruction-rd/tree/stage-04-da3-localization`
- Remote push verified: Yes. On 2026-08-03, the stage-close commit was pushed
  to remote `main`, and the annotated tag dereferenced exactly to it. Remote
  `main` subsequently advanced only through close-provenance documentation;
  the stage-close commit remains its ancestor and the tag remains exact.
- Model/vendor revisions: DA3
  `b2359bdf726fb44ef62acca04d629dcf158053e7`; YOLO checkpoint SHA-256
  `a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c`;
  DA3 vendor fingerprint
  `683cad1fec1186cd2a22f2b6d083b73d4c83c7ab1140f45ba24876612bc51d43`.

## Exact Next Action

Begin S05 by defining the typed pickup-carry-place interaction state machine
over the verified measured/stale/missing person and backpack observations and
the accepted zones. Do not begin Qwen integration until the state machine's
spatial authority and unknown-state rules are tested.
