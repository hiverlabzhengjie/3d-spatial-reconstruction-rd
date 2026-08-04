# S04 DA3-Depth 3D Localization and Fusion Record

**Stage:** S04 - DA3-Depth 3D Localization and Fusion

**Status:** Complete with known sparse-observation limitations

**Started:** 2026-08-01

## Stage Goal

Convert retained person and backpack masks into honest per-camera and fused
world-space XYZ observations using pose-conditioned DA3 metric depth from the
same synchronized action content. Preserve exact source/depth identity,
freshness, missing-data state, anchor meaning, and the separation between raw
measurements and presentation trajectories.

## Entry Prerequisites

Verified on 2026-08-01 before changing the active stage:

- S03 is complete and its stage-close commit `6764aef` plus provenance update
  `4dcd575` are present on local and remote `main`.
- The worktree was clean before S04 activation.
- The accepted synchronized action pair contains `1,047` complete bundles,
  zero missing camera frames, and no more than `6.667 ms` inter-camera timing
  difference.
- The synchronized Camera A/B source hashes remain
  `1e7064fa2d4911dcf2ac82803dd95fa5b9ece332906589c0f8627232bb526136`
  and `da5bd4eeaeac0da78cc71f14a43326d5d60c5c216f2609c85553cba720e40d5a`.
- Action pose version `s01_capture_20260729:action_take_01:v1`, the camera
  intrinsics/poses, room bounds, and video-estimated zones are present.
- S03 retained `320` bounded perception results, source-sized raw masks, and
  `640` explicit per-target states with D028 vendor-label provenance.
- The exact `yolov8n-seg.pt` checkpoint hash is
  `a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c`.
- DA3 revision `b2359bdf726fb44ef62acca04d629dcf158053e7` is cached locally.
- The unmodified 161-file DA3 vendor fingerprint is
  `683cad1fec1186cd2a22f2b6d083b73d4c83c7ab1140f45ba24876612bc51d43`.
- The native Python 3.11 environment and lockfile are current.
- Baseline verification passed: 162 tests, Ruff, strict mypy, lockfile and
  environment checks, vendor/source fingerprints, and whitespace checks.

No physical-input, calibration, software, or artifact blocker is known at S04
entry.

## Active Boundaries

- Use DA3 metric depth and confidence from synchronized action frames that
  contain the dynamic entities. Empty-room depth is not a substitute.
- Use only D025's separately amended S04 action-pair profile for corrected
  dynamic depth. Do not import its S02 centre-patch rule, D026's static
  confidence percentile, or D027's door supplement.
- Join masks and depth by immutable source identity and an explicit tested
  freshness rule, never by worker completion order or an unqualified latest
  result.
- Back-project only finite, positive, confidence-valid in-mask depth.
- Treat a visible surface aggregate, a derived track anchor, a fused
  observation, and a smoothed presentation position as different states with
  different provenance.
- Missing, untracked, ambiguous, failed, mismatched, or stale inputs must not
  produce fabricated XYZ coordinates.
- Camera-local S03 track IDs do not become global identities merely because
  they are fused geometrically.
- Preserve explicit `T_world_from_camera` and `T_camera_from_world` names and
  the right-handed metre-scale world frame with Z up.

## Known Entry Limitations

- Backpack detections are sparse and fragmented across cameras.
- The approximately `17.2-22.0 s` two-camera backpack gap must remain without
  a raw measured position unless new valid same-content masks and depth exist.
- Camera A person tracking is fragmented; Camera B supplies the strongest
  representative person track.
- D028's `backpack` plus `handbag` policy is specific to the one-bag
  demonstration and does not establish general re-identification.

## First Work Step - Raw Action-Depth Preflight

1. Select a deterministic, action-spanning subset of complete synchronized
   bundles where retained S03 person or backpack masks exist.
2. Preserve the corresponding bundle, frame, mask, calibration, model, and
   synchronization identities in a DA3 action-keyframe job contract.
3. Run exact pose-conditioned two-view DA3 metric inference at the accepted
   process resolution on those action frames without S02-derived corrections.
4. Retain raw depth/confidence and representative diagnostics before choosing
   mask-depth confidence, anchor, freshness, or fusion thresholds.
5. Stop and report if action-frame depth or physical inputs are inadequate;
   do not compensate with static depth or invented coordinates.

### Result

Completed on 2026-08-01 using exact pose-conditioned two-view DA3 inference on
native Apple MPS at process resolution `504` and float16 precision.

- Added a deterministic action-depth job contract whose stable identity binds
  the synchronized bundle, exact camera frames, observed S03 mask evidence,
  phase, DA3 model/revision, process resolution, and attempt.
- Selected eight capture-ordered frames at source indices `204`, `330`, `408`,
  `462`, `666`, `708`, `780`, and `858`, spanning `6.803-28.612 s` and the
  pickup-side stationary, pickup/lift, early carry, post-gap reappearance,
  drop-off approach, and placement phases.
- Every selected job has an observed backpack mask in the configured camera
  and an observed person mask in at least one camera. Missing views remain
  absent from mask evidence rather than being promoted to observations.
- The accepted approximately `17.2-22.0 s` two-camera backpack absence is not
  sampled, filled, or assigned an XYZ position.
- Ran exact model revision `b2359bdf726fb44ef62acca04d629dcf158053e7`
  with the accepted action camera poses and the verified 161-file DA3 vendor
  fingerprint.
- Retained eight raw two-camera predictions, sixteen undistorted input
  keyframes, eight depth/confidence previews, the complete job selection, and
  native runtime/memory observations.
- Each of the sixteen depth planes contains `141,120` finite positive samples;
  every confidence plane contains `141,120` finite samples.
- Model load took `14.509 s`; the first action pair took `1.882 s`, and the
  seven warm pairs took `1.330-1.401 s` each.
- Final recorded MPS allocation was approximately `6.76 GB`, driver allocation
  approximately `8.82 GB`, against a reported recommended maximum of
  approximately `55.66 GB`.
- Raw NPZ policy fields and the strict run contract confirm a unit depth scale,
  no S02 marker-scale correction, no S02 static confidence policy, no D027
  door supplement, and no mask localization yet.
- Visual review of representative pickup, carry, reappearance, and placement
  previews shows dynamic person and bag silhouettes in the raw depth maps.
  This supports continuing to mask alignment and aggregation but is not yet a
  quantitative XYZ or metric-accuracy claim.

### Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Deterministic job selection | `artifacts/s04/action_depth_preflight_20260801/selection.json` | Exact bundle/frame/mask/model job identities |
| Raw DA3 action depth | `artifacts/s04/action_depth_preflight_20260801/predictions/` | Eight raw depth/confidence predictions with source provenance |
| Undistorted action keyframes | `artifacts/s04/action_depth_preflight_20260801/keyframes/` | Exact two-camera DA3 inputs |
| Depth/confidence previews | `artifacts/s04/action_depth_preflight_20260801/previews/` | Visual dynamic-depth QA |
| Run summary | `artifacts/s04/action_depth_preflight_20260801/summary.json` | Schema-validated model, policy, runtime, and artifact record |
| Verification | `artifacts/s04/action_depth_preflight_20260801/verification.json` | Hash, array, order, schema, and no-S02-policy checks |

### Reproduction Commands

```text
.venv/bin/python scripts/s04/run_action_depth_preflight.py \
  --output-dir artifacts/s04/action_depth_preflight_<new-run-id>

.venv/bin/python scripts/s04/verify_action_depth_preflight.py \
  --summary artifacts/s04/action_depth_preflight_<new-run-id>/summary.json \
  --output artifacts/s04/action_depth_preflight_<new-run-id>/verification.json

.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/mypy --strict src/spatial_reconstruction \
  scripts/s04/run_action_depth_preflight.py \
  scripts/s04/verify_action_depth_preflight.py
```

The model command requires native Apple MPS and refuses to overwrite an
existing output directory. The verifier refuses to overwrite existing
verification evidence.

### Verification

- `166` automated tests passed.
- Ruff passed across project source, tests, and scripts after the final import
  cleanup.
- Strict mypy passed across `33` source and stage-script files.
- Artifact verification passed for all eight predictions and sixteen
  keyframes: hashes, array metadata, job/bundle/frame identities, capture
  order, schema round trip, finite values, metric flag, unit scale, and absent
  S02 corrections all agree.
- No raw recording was modified and no CPU fallback was used.

No new method, model, correction, or architecture decision was introduced.

## Exact Source-Mask to DA3-Grid Alignment

Completed on 2026-08-01 without performing back-projection, XYZ aggregation,
anchor derivation, or cross-camera fusion.

- Traced the unmodified DA3 `upper_bound_resize` preprocessing and reproduced
  its exact geometry for the accepted `1920x1080` action inputs:
  `1920x1080 -> 504x284 -> 504x280`.
- No batch-shape crop, padding, or other spatial adjustment applies because
  both synchronized camera views produce the same processed dimensions.
- Undistorted the source-sized S03 binary masks with the accepted camera
  matrix and distortion coefficients using nearest-neighbour remapping, then
  applied both DA3 resize steps with nearest-neighbour interpolation. This
  preserves discrete mask labels while following the same spatial mapping as
  the processed RGB and depth.
- Independently reproduced all `16` retained processed RGB inputs. Every check
  passed with a maximum absolute channel difference of `1`; the minimum exact
  channel fraction was `0.729882`.
- Independently reproduced the processed camera intrinsics with a maximum
  absolute error of `1.235706e-05`, below the accepted `1e-4` tolerance.
- Aligned `20` observed S03 person/backpack masks into eight immutable
  `280x504` mask bundles. All stored masks are binary `uint8`; processed mask
  areas range from `402` to `11,839` pixels.
- Preserved source frame, synchronized bundle, DA3 job, S03 perception job,
  detection, vendor label, camera-local track, target, transform, artifact,
  and hash provenance for every aligned mask.
- Visual review of the eight-row contact sheet confirms that the observed
  person and backpack masks remain attached to their intended image regions.
  Source-unavailable observations remain absent rather than being synthesized.
- The verifier confirms all eight action-depth jobs are covered, all hashes and
  NPZ metadata agree, nearest-neighbour mask mapping is recorded, and neither
  localization nor an S02 depth correction was applied.

### Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Aligned binary masks | `artifacts/s04/mask_alignment_20260801/aligned_masks/` | Exact observed S03 masks on each retained DA3 depth grid |
| Per-job overlays | `artifacts/s04/mask_alignment_20260801/overlays/` | Full-resolution two-camera alignment diagnostics |
| Contact sheet | `artifacts/s04/mask_alignment_20260801/mask_alignment_contact_sheet.png` | Visual QA across all eight action phases |
| Run summary | `artifacts/s04/mask_alignment_20260801/summary.json` | Schema-validated transforms, identity, area, RGB, and intrinsic checks |
| Verification | `artifacts/s04/mask_alignment_20260801/verification.json` | Hash, schema, binary-mask, metadata, coverage, and visual-QA record |

### Reproduction Commands

```text
.venv/bin/python scripts/s04/align_action_masks.py \
  --output-dir artifacts/s04/mask_alignment_<new-run-id>

.venv/bin/python scripts/s04/verify_mask_alignment.py \
  --summary artifacts/s04/mask_alignment_<new-run-id>/summary.json \
  --output artifacts/s04/mask_alignment_<new-run-id>/verification.json \
  --visual-qa-passed

.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/mypy --strict src/spatial_reconstruction scripts/s04
uv --cache-dir .cache/uv lock --check
uv --cache-dir .cache/uv sync --check
git diff --check
```

Both artifact-producing commands refuse to overwrite their outputs. The
alignment step verifies all source hashes and performs no model inference. The
verification command requires explicit acknowledgement that the generated
contact sheet was visually inspected.

### Verification

- `169` automated tests passed.
- Ruff passed across project source, tests, and scripts.
- Strict mypy passed across `31` source and S04 script files.
- Lockfile and environment checks would make no changes.
- Artifact verification passed for all eight jobs, eight mask bundles, `20`
  masks, `16` RGB/intrinsic checks, eight overlays, and the contact sheet.
- `git diff --check` passed; no raw capture or DA3 vendor file was modified.

No new method, model, correction, or architecture decision was introduced.

## Raw In-Mask Depth/Confidence Diagnostics and D030 Policy

Completed on 2026-08-01 over the same eight exact synchronized action-depth
jobs and `20` verified aligned masks. This action produced no back-projection,
XYZ, spatial anchor, fusion, smoothing, or temporal filling.

### Diagnostic Comparison

- Added deterministic, typed comparison of four candidate regions:
  - the whole observed mask;
  - a two-pixel eroded interior;
  - an adaptive median/MAD depth interval followed by the largest
    eight-connected component;
  - the lower `35%` of a person's observed mask bounding extent.
- Retained `72` schema-validated records: `20` whole masks, `20` eroded
  interiors, `20` connected depth clusters, and `12` person lower-body
  candidates.
- Every real candidate contained finite positive depth and finite confidence.
  The diagnostic also retains source frame, bundle, camera, target, track,
  aligned-mask, raw-prediction, transform-policy, and artifact hashes.
- Compared same-action full-frame confidence thresholds at percentiles `20`,
  `40`, `60`, and `80`. These were a diagnostic sweep, not imported S02
  policy.
- Rejected full-frame confidence as the dynamic-object reference. At the
  full-frame `20th` percentile, at least one valid person and one valid
  backpack candidate retained zero samples. Median retention was only
  `0.06953` for person lower-body candidates and `0.01962` for backpack
  connected clusters.
- The person lower-body candidate reduced median relative depth MAD to
  `0.03479`, compared with `0.06982` for the whole mask, `0.06234` for the
  eroded interior, and `0.06066` for the generic connected cluster.
- The backpack connected cluster reduced median relative depth MAD to
  `0.01517`, compared with `0.02327` for the whole mask and `0.01586` for the
  eroded interior, while explicitly removing disconnected/boundary leakage.
- Visual review covered the all-mask contact sheet, target/strategy comparison
  chart, and representative stationary, carry, reappearance, approach, and
  placement diagnostics. Candidate outlines remain attached to the intended
  object surfaces and the raw distributions support the numeric findings.

### Selected Rule - D030

Adopted D030 and typed policy `s04_dynamic_visible_surface_v1`:

- require finite positive current-frame depth and finite confidence;
- compute confidence rank within the selected valid object candidate, not the
  full action frame and not D026's S02 static distribution;
- retain samples at or above the candidate-relative `20th` percentile;
- for the person, use the visible lower-body candidate, require at least `256`
  retained samples, and aggregate ray depth with the median;
- for the backpack, use the largest adaptive connected depth cluster, require
  at least `128` retained samples, and aggregate ray depth with the median;
- label both as visible-surface measurements. They are not body/object centres
  and the person result is not automatically a ground-contact point;
- if the candidate is absent, invalid, or undersampled, produce an unavailable
  observation with no whole-mask, stale-frame, static-depth, or invented-data
  fallback.

Candidate-relative `p20` retained approximately `80%` of every selected
candidate. The minimum retained counts were `382` for a person and `203` for a
backpack. The filtered median relative depth MAD was `0.02937` for person and
`0.01516` for backpack; maximum median shifts from the unfiltered candidates
were `2.91%` and `0.90%`, respectively. Higher candidate percentiles reduced
support substantially and increased maximum aggregate shifts, so they were
not selected.

### Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Diagnostic summary | `artifacts/s04/mask_depth_diagnostics_20260801/summary.json` | Strict 72-record comparison and source provenance |
| Comparison table | `artifacts/s04/mask_depth_diagnostics_20260801/mask_depth_comparison.csv` | Reviewable per-mask/per-strategy statistics |
| Strategy comparison | `artifacts/s04/mask_depth_diagnostics_20260801/strategy_comparison.png` | Target-wise depth spread and full-frame confidence evidence |
| All-mask contact sheet | `artifacts/s04/mask_depth_diagnostics_20260801/mask_depth_contact_sheet.png` | Coverage and mask identity review |
| Per-mask diagnostics | `artifacts/s04/mask_depth_diagnostics_20260801/per_mask/` | RGB, candidate, depth, confidence, histogram, and sweep QA |
| Policy selection | `artifacts/s04/mask_depth_diagnostics_20260801/policy_selection.json` | D030 rule plus candidate-relative percentile evidence |
| Verification | `artifacts/s04/mask_depth_diagnostics_20260801/verification.json` | Hash, schema, coverage, policy, minimum-count, and visual-QA checks |

### Reproduction Commands

```text
.venv/bin/python scripts/s04/analyze_mask_depth_samples.py \
  --output-dir artifacts/s04/mask_depth_diagnostics_<new-run-id>

.venv/bin/python scripts/s04/select_mask_depth_policy.py \
  --diagnostics-summary \
    artifacts/s04/mask_depth_diagnostics_<new-run-id>/summary.json \
  --output \
    artifacts/s04/mask_depth_diagnostics_<new-run-id>/policy_selection.json

.venv/bin/python scripts/s04/verify_mask_depth_diagnostics.py \
  --summary artifacts/s04/mask_depth_diagnostics_<new-run-id>/summary.json \
  --policy-selection \
    artifacts/s04/mask_depth_diagnostics_<new-run-id>/policy_selection.json \
  --output artifacts/s04/mask_depth_diagnostics_<new-run-id>/verification.json \
  --visual-qa-passed

.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/mypy --strict src/spatial_reconstruction scripts/s04
uv --cache-dir .cache/uv lock --check
uv --cache-dir .cache/uv sync --check
git diff --check
```

All artifact-producing commands refuse to overwrite their output. The three
S04 commands perform no model inference and use only verified raw predictions,
aligned masks, and immutable provenance from the preceding actions.

### Verification

- `175` automated tests passed.
- Ruff passed across project source, tests, and scripts.
- Strict mypy passed across `35` source and S04 script files.
- Lockfile/environment checks would make no changes and `git diff --check`
  passed.
- Artifact verification passed for all eight action-depth jobs, `20` masks,
  `72` strategy records, `20` per-mask diagnostics, summary visualizations,
  D030 policy evidence, schema round trips, and hashes.
- No raw capture or DA3 vendor file was modified; no S02 confidence policy,
  back-projection, or XYZ was used.

## Exact-Frame Raw Per-Camera Visible-Surface Localization

Completed on 2026-08-02 for all `20` D030-selected observed masks without
deriving target anchors, fusing cameras, filling time, or smoothing a
presentation trajectory.

- Added a typed exact-frame join that requires identical action-depth job,
  synchronized bundle, frame, camera, and capture timestamp identity. Its
  timestamp tolerance is zero and worker completion order is forbidden.
- Applied D030 independently to `12` person masks and `8` backpack masks.
  Every selected pixel was back-projected with its own metric depth and the
  returned processed intrinsics, then transformed using the accepted explicit
  `T_world_from_camera`.
- Adopted D031: preserve every raw camera/world sample and use the
  component-wise median of camera-frame XYZ as the robust visible-surface
  aggregate, then transform that one aggregate into the world frame.
- Retained exact action-depth, bundle, frame, camera, timestamp, S03
  perception job, camera-local track, mask, policy, calibration, transform,
  raw-prediction, sample-cloud, image-diagnostic, and hash provenance.
- The minimum retained count remained `382` for person and `203` for backpack;
  no real selected observation became unavailable under D030.
- The independent verifier regenerated the full D030 selection and geometry
  from every exact source artifact. Maximum sample reprojection error was
  `2.842171e-14 px`, maximum world/camera round-trip error was
  `3.477764e-15 m`, and maximum DA3-returned-pose error against the accepted
  pose was `1.395500e-07`.
- All `20` raw aggregates and every retained sample lie inside the approximate
  room bounds. Bounds were used only for diagnostics; no point was clipped,
  snapped, or otherwise altered.
- Visual review confirms that selected samples remain on the intended person
  lower-body or backpack visible surfaces. The raw motion progresses from the
  pickup side toward the drop-off side where observations exist.
- The four same-frame two-camera person pairs differ by `0.384-0.679 m` at the
  raw aggregate level. This view-dependent result is retained as evidence for
  explicit anchor derivation and disagreement handling; it is not averaged.
  The retained subset has no same-frame two-camera backpack pair.
- Invalid and undersampled candidates return unavailable with empty sample
  arrays and no XYZ. Tests also reject mismatched frame/camera/time identity,
  completion-order joins, camera-contract mismatch, and relabelling action
  depth as another/static frame.

### Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Raw sample clouds | `artifacts/s04/visible_surfaces_20260802/sample_clouds/` | Selected pixel/depth/confidence and camera/world XYZ samples with explicit transforms |
| Per-observation diagnostics | `artifacts/s04/visible_surfaces_20260802/image_diagnostics/` | RGB selected-sample and raw aggregate review |
| Contact sheet | `artifacts/s04/visible_surfaces_20260802/visible_surface_contact_sheet.png` | Visual QA across all `20` observations |
| World preview | `artifacts/s04/visible_surfaces_20260802/visible_surface_world_preview.png` | 3D and top-down raw per-camera motion review |
| Run summary | `artifacts/s04/visible_surfaces_20260802/summary.json` | Strict observation schemas, source identity, transforms, aggregates, and hashes |
| Verification | `artifacts/s04/visible_surfaces_20260802/verification.json` | Source regeneration, geometry, schema, provenance, bounds, and visual-QA checks |

### Reproduction Commands

```text
.venv/bin/python scripts/s04/localize_visible_surfaces.py \
  --output-dir artifacts/s04/visible_surfaces_<new-run-id>

.venv/bin/python scripts/s04/verify_visible_surfaces.py \
  --summary artifacts/s04/visible_surfaces_<new-run-id>/summary.json \
  --output artifacts/s04/visible_surfaces_<new-run-id>/verification.json \
  --visual-qa-passed

.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/mypy --strict src/spatial_reconstruction scripts/s04
uv --cache-dir .cache/uv lock --check
uv --cache-dir .cache/uv sync --check
git diff --check
```

Both artifact-producing commands refuse to overwrite their outputs. They use
only verified existing action-depth, mask, policy, calibration, and scene
artifacts and perform no model inference. The verifier requires explicit
acknowledgement that both generated visual diagnostics were inspected.

### Verification

- `184` automated tests passed.
- Ruff passed across project source, tests, and scripts.
- Strict mypy passed across `38` source and S04 script files.
- Lockfile/environment checks would make no changes and `git diff --check`
  passed.
- Artifact verification passed for all eight action-depth jobs, `20` aligned
  masks, `20` sample clouds, `20` image diagnostics, both summary previews,
  schemas, exact joins, source regeneration, transforms, hashes, and the
  explicit absence of anchors, fusion, filling, smoothing, and S02 correction.
- No raw capture or DA3 vendor file was modified.

## Target-Anchor Candidate Evaluation and D032 Policy

Completed on 2026-08-02 from the verified D031 sample clouds. This action
selected target-anchor semantics and a pre-fusion disagreement gate without
performing fusion, temporal filling, or presentation smoothing.

- Re-ran the strict D031 verifier as an entry prerequisite. All `20` raw
  observations regenerated from their exact current-frame masks, depth,
  confidence, intrinsics, and transforms before anchor work began.
- Compared `104` typed candidates with immutable raw-observation and
  sample-cloud provenance:
  - six person candidates across `12` observations: D031 reference, lowest-Z
    decile, lowest-Z quintile, bottom image-space quintile, bottom-ray floor
    intersection, and validated ground contact;
  - four backpack candidates across `8` observations: D031 reference,
    world-frame component median, trimmed 10th-to-90th percentile bounds
    centre, and trimmed mean.
- Selected D032 policy `s04_target_anchor_v1`:
  - person tracking uses the measured world-frame median of the lowest-Z
    quintile, with at least `32` samples required and at least `77` present;
  - backpack tracking uses the world-frame component median of the visible
    cluster;
  - person ground contact remains a separate floor-projected state available
    only when measured lower-quintile support is at most `0.35 m` high.
- The person tracking anchor remains available for all `12` raw person
  observations. Validated ground contact is observed for `6` and unavailable
  without XYZ for `6` elevated/hidden-foot cases.
- Direct bottom-image ray/floor intersection was evaluated using the existing
  surveyed `world Z=0` plane. It adds no fitted plane or dependency but is not
  selected: one result leaves room bounds and paired disagreement reaches
  `1.110 m`.
- The selected backpack anchor gives a `0.126 m` maximum separation across the
  three stationary pickup observations, versus `0.136 m` for D031, and a
  `0.045 m` placed-pair separation, versus `0.056 m` for D031.
- Persisted the complete `32`-state grid for eight action jobs, two cameras,
  and two targets. It contains `20` selected observed anchors and `12`
  source-unavailable states with no XYZ.
- Classified all `16` job/target camera comparisons under a prototype
  `0.35 m` gate:
  - one paired-eligible person state at `0.231 m`;
  - three paired-disagreement person states at `0.474`, `0.510`, and
    `0.759 m`;
  - twelve single-camera states, including all eight backpack comparisons;
  - zero both-camera-unavailable states in this retained subset, while the
    typed contract and synthetic tests cover that failure path.
- Visual review confirms pickup-to-drop-off motion remains plausible, elevated
  ground-contact evidence is marked unavailable, and camera disagreement is
  exposed rather than averaged.

### Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Candidate comparison CSV | `artifacts/s04/anchor_evaluation_20260802_v2/anchor_candidate_comparison.csv` | Reviewable per-observation method, support, availability, XYZ, and selection record |
| Candidate comparison chart | `artifacts/s04/anchor_evaluation_20260802_v2/anchor_candidate_comparison.png` | Availability, disagreement, backpack repeatability, and ground-evidence comparison |
| Selected-anchor preview | `artifacts/s04/anchor_evaluation_20260802_v2/selected_anchor_world_preview.png` | Top-down raw, selected, ground-contact, zone, and missing-state review |
| Run summary | `artifacts/s04/anchor_evaluation_20260802_v2/summary.json` | Strict candidates, complete selected-state grid, D032 policy, and comparison provenance |
| Verification | `artifacts/s04/anchor_evaluation_20260802_v2/verification.json` | Source regeneration, policy, missing-state, disagreement, schema, hash, and visual-QA checks |

### Reproduction Commands

```text
.venv/bin/python scripts/s04/evaluate_target_anchors.py \
  --output-dir artifacts/s04/anchor_evaluation_<new-run-id>

.venv/bin/python scripts/s04/verify_target_anchor_evaluation.py \
  --summary artifacts/s04/anchor_evaluation_<new-run-id>/summary.json \
  --output artifacts/s04/anchor_evaluation_<new-run-id>/verification.json \
  --visual-qa-passed

.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/mypy --strict src/spatial_reconstruction scripts/s04
uv --cache-dir .cache/uv lock --check
uv --cache-dir .cache/uv sync --check
git diff --check
```

Both artifact-producing commands refuse to overwrite their outputs and perform
no model inference. The evaluator requires a passed D031 verification whose
summary hash matches the current raw visible-surface summary. The verifier
requires explicit acknowledgement that both generated visual diagnostics were
inspected.

### Verification

- `195` automated tests passed.
- Ruff passed across project source, tests, and scripts.
- Strict mypy passed across `41` source and S04 script files.
- Lockfile/environment checks would make no changes and `git diff --check`
  passed.
- Artifact verification regenerated all `104` candidates from their source
  clouds, covered all `32` selected states and `16` camera comparisons, checked
  hashes and schema round trips, and confirmed all missing/ground-unavailable
  states have no placeholder XYZ.
- No camera fusion, temporal filling, presentation smoothing, raw-capture
  modification, DA3 vendor modification, or new model inference occurred.

## D032-Gated Cross-Camera Observation Layer and D033

Completed on 2026-08-02 from the verified D032 selected anchors. This action
combines only exact same-job eligible sources and keeps missing, disagreement,
single-source, and fused provenance distinct.

- Re-ran the strict target-anchor verifier as an entry prerequisite. All `104`
  candidates, `32` selected states, and `16` pre-fusion comparisons regenerated
  before cross-camera combination began.
- Adopted D033 policy `s04_cross_camera_observation_v1` with an inspectable
  reliability score:

  ```text
  sqrt(anchor_support_count) * retained_DA3_confidence_median
  / (1 + retained_depth_relative_MAD)
  ```

- Support enters through its square root, confidence uses the D030-retained
  median, and relative depth MAD applies a gentle bounded penalty. The score is
  retained as prototype evidence and is not called a probability.
- Normalized scores become contribution weights only for a D032
  paired-eligible observation. Paired disagreements retain their source scores
  but have no weights and no combined XYZ. A single valid camera receives
  weight `1.0` as an explicitly labelled passthrough, not fusion.
- Produced the complete `16`-record job/target layer:
  - one fused person observation at frame `204`;
  - twelve single-camera observations, including all eight backpack records;
  - three person disagreement states at frames `408`, `708`, and `780` with no
    combined XYZ;
  - zero both-camera-unavailable states in the retained subset, while the typed
    contract and synthetic tests cover that path.
- The frame `204` Camera A/B person reliability scores are `114.7145` and
  `61.1843`. Their normalized weights are `0.652162` and `0.347838`, producing
  `(0.065857, 2.442960, 0.161857) m` from anchors separated by `0.230854 m`.
- The maximum two-camera source-time difference is `5 ms`. Capture order and
  immutable job, bundle, camera-frame, target, raw-observation, anchor-
  candidate, score, weight, state, and output identities are retained.
- All `13` emitted fused-or-single-source XYZ observations lie inside the
  approximate room bounds. The three disagreement records retain their two
  anchors visibly but deliberately emit no output point.
- Visual review confirms plausible pickup-to-drop-off backpack motion, the
  fused point lies between its two anchors, single-camera points remain
  unchanged, and disagreement lines are shown without invented centres.

### Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Observation CSV | `artifacts/s04/cross_camera_observations_20260802/cross_camera_observations.csv` | Reviewable state, source scores/weights, disagreement, XYZ, and provenance summary |
| Reliability diagnostic | `artifacts/s04/cross_camera_observations_20260802/reliability_and_state_diagnostic.png` | State counts, fused weights, D032 gate, and source-score review |
| World preview | `artifacts/s04/cross_camera_observations_20260802/cross_camera_world_preview.png` | Fused/single XYZ plus disagreement anchors without combined positions |
| Run summary | `artifacts/s04/cross_camera_observations_20260802/summary.json` | Strict source evidence, weights, states, exact identities, and output coordinates |
| Verification | `artifacts/s04/cross_camera_observations_20260802/verification.json` | Regeneration, schema, hashes, state behavior, capture order, bounds, and visual-QA checks |

### Reproduction Commands

```text
.venv/bin/python scripts/s04/build_cross_camera_observations.py \
  --output-dir artifacts/s04/cross_camera_observations_<new-run-id>

.venv/bin/python scripts/s04/verify_cross_camera_observations.py \
  --summary artifacts/s04/cross_camera_observations_<new-run-id>/summary.json \
  --output artifacts/s04/cross_camera_observations_<new-run-id>/verification.json \
  --visual-qa-passed

.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/mypy --strict src/spatial_reconstruction scripts/s04
uv --cache-dir .cache/uv lock --check
uv --cache-dir .cache/uv sync --check
git diff --check
```

Both artifact-producing commands refuse to overwrite their outputs and perform
no model inference. The builder requires a passed D032 verification whose
summary hash matches the current anchor evaluation. The verifier requires
explicit acknowledgement that both generated visual diagnostics were
inspected.

### Verification

- `207` automated tests passed.
- Ruff passed across project source, tests, and scripts.
- Strict mypy passed across `44` source and S04 script files.
- Lockfile/environment checks would make no changes and `git diff --check`
  passed.
- Artifact verification regenerated all `16` outputs and every source score
  from the selected anchors and D031 statistics, checked capture order, exact
  frame timing, schemas, hashes, CSV coverage, room bounds, and state-specific
  XYZ/weight behavior.
- No stale carry-forward, temporal interpolation, presentation smoothing,
  model inference, raw-capture modification, or DA3 vendor modification
  occurred.

## D025 Action-Pair Marker Scaling Correction

Completed on 2026-08-03 before starting the temporal policy. This corrective
action performs no model inference and does not modify or overwrite any raw
DA3 depth, confidence, keyframe, pose, identity, or existing D030-D033 artifact.

- Amended D025 with the isolated policy
  `d025_action_pair_marker_scale_v1`. The action profile detects only M40-M42
  on each exact undistorted keyframe and retains M43's D022 exclusion.
- Replaced the S02 single centre-patch observation with a more robust dynamic
  measurement: the median of calibrated per-pixel expected-camera-Z/raw-depth
  ratios inside the protected inner `60%` of each physical `180 mm` marker.
- The calibrated marker centre must agree with the detected centre within
  `5 px`; every marker must supply at least `16` finite-positive samples.
- Acceptance requires evidence from both cameras, at least two markers per
  camera, at least five marker observations total, and every marker ratio
  within `5%` of exactly one shared pair median. Failure produces no corrected
  depth for that pair; camera-specific, partial, stale, and silent unit-scale
  fallbacks are forbidden.
- All eight selected pairs passed with `44` observations. Shared scales were
  `1.093553`, `1.130678`, `1.143375`, `1.129954`, `1.153414`, `1.137148`,
  `1.134167`, and `1.133183`; their median is `1.133675`.
- Maximum within-pair marker relative deviation is `1.2202%`, well below the
  `5%` rejection gate. Visual QA confirms every accepted outline remains on
  its intended marker and calibrated centre overlays agree with detections.
- The independent verifier confirms all original prediction hashes and
  confidence arrays are unchanged, every corrected float32 array is exactly
  `raw * shared pair scale`, and all job/frame ordering and hashes agree.
- Existing D030-D033 results remain retained as explicitly unscaled baseline
  evidence. This correction alone does not revalidate their XYZ or fusion
  output; those dependent products must be rebuilt next.

### Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Corrected depth | `artifacts/s04/action_depth_scale_20260803/corrected_predictions/` | Separate per-pair float32 corrected depth with raw references and hashes |
| Marker observations | `artifacts/s04/action_depth_scale_20260803/marker_scale_observations.csv` | Reviewable detected/projection/sample/ratio evidence |
| Pair diagnostics | `artifacts/s04/action_depth_scale_20260803/diagnostics/` | Accepted marker and calibrated-centre overlays |
| Contact sheet | `artifacts/s04/action_depth_scale_20260803/marker_scale_contact_sheet.png` | Visual QA across all eight action pairs |
| Run summary | `artifacts/s04/action_depth_scale_20260803/summary.json` | Policy, provenance, scales, corrected refs, and limitations |
| Verification | `artifacts/s04/action_depth_scale_20260803/verification.json` | Independent gate, immutable-input, exact-correction, ordering, and visual-QA checks |

### Reproduction Commands

```text
.venv/bin/python scripts/s04/scale_action_depth.py \
  --output-dir artifacts/s04/action_depth_scale_<new-run-id>

.venv/bin/python scripts/s04/verify_action_depth_scale.py \
  --summary artifacts/s04/action_depth_scale_<new-run-id>/summary.json \
  --output artifacts/s04/action_depth_scale_<new-run-id>/verification.json \
  --visual-qa-passed

.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/mypy --strict src/spatial_reconstruction scripts/s04
uv --cache-dir .cache/uv lock --check
uv --cache-dir .cache/uv sync --check
git diff --check
```

Both artifact-producing commands refuse to overwrite their outputs. The first
uses only verified retained keyframes, raw predictions, and accepted
calibration; the second requires explicit visual-QA acknowledgement.

### Verification

- `214` automated tests passed, including synthetic scale recovery plus
  missing-camera, insufficient-total-evidence, reprojection-error, marker-
  disagreement, and forbidden-camera-fallback cases.
- Ruff passed across project source, tests, and scripts.
- Strict mypy passed across `47` source and S04 script files.
- Lockfile/environment checks would make no changes and `git diff --check`
  passed.
- Artifact verification passed for all eight pairs, `44` marker observations,
  eight corrected predictions, eight pair diagnostics, the contact sheet,
  exact correction arithmetic, raw hashes, confidence hashes, and capture
  order.
- No raw capture, raw DA3 output, existing D030-D033 artifact, or DA3 vendor
  file was modified.

## Corrected Margin-Aware D030-D033 Rebuild

Completed on 2026-08-03 from the verified D025 corrected action-pair depth.
This corrective rebuild retains the earlier unscaled artifacts as labelled
baseline evidence and performs no new model inference, temporal filling, or
presentation smoothing.

### Margin-Aware Person Validity and D030-D031 Surfaces

- Added typed policy `s04_corrected_margin_aware_tracking_v1` and regenerated
  all `20` per-camera surfaces directly from D025's separate corrected arrays,
  exact aligned masks, unchanged DA3 confidence, processed intrinsics, and
  accepted action camera poses.
- Kept D030's candidate-relative `20th` confidence percentile. It remains an
  object-candidate rank filter rather than a full-frame or S02 threshold.
- Recorded left, right, top, and bottom distances for every person mask using
  a two-pixel processed-image margin band. Only bottom contact invalidates
  footpoint candidacy; top/side contact remains diagnostic because it does not
  by itself prove the lower body is hidden.
- Classified three views as bottom-truncated: frame `330` Camera B, frame `408`
  Camera B, and frame `462` Camera B. Their connected corrected-depth clusters
  are retained as measured upper-body surfaces; their mask bottoms are not
  interpreted as feet.
- Non-bottom-truncated person views retain the lower `35%` candidate; backpack
  views retain the adaptive connected depth cluster. Minimum retained support
  remains `256` person and `128` backpack samples.
- Preserved each corrected sample cloud, camera/world aggregate, source role,
  D025 pair scale, raw/corrected prediction identities, mask identity,
  diagnostic image, and immutable job/frame provenance.

### Consistent Footpoints, Synchronized-Mate Selection, and D032-D033

- A per-camera person footpoint now requires all of the following:
  - no contact with the two-pixel bottom margin;
  - at least `32` samples in the lowest world-Z quintile; and
  - median low-Z support within `-0.10-0.35 m` of surveyed `world Z=0`.
- When valid, only the measured low-Z XY is vertically projected to `Z=0` and
  labelled `person_footpoint`. Seven of twelve per-camera person views pass.
- A non-truncated view outside the floor-height band remains a measured
  lower-body surface. A bottom-truncated view remains a measured upper-body
  surface. Neither fallback is projected to the floor or described as an
  anatomical centre.
- Same-frame pair selection uses explicit semantic priority: footpoint, then
  lower-body surface, then upper-body surface. This lets the synchronized mate
  supply a clearer footpoint when one camera sees only a cropped/elevated body.
- Frame `708` therefore uses Camera B's valid footpoint while preserving Camera
  A's elevated lower-body surface as secondary evidence. Frame `408` uses
  Camera A's lower-body surface while preserving Camera B's cropped upper-body
  surface. No mixed anchor kinds can enter fusion.
- Matching kinds retain D032's `0.35 m` gate and D033's inspectable reliability
  score. Frames `204` and `780` fuse comparable footpoints. The complete person
  result is two fused and six single-camera outputs, with no forced
  disagreement or unavailable state in the eight retained pairs.
- Preferred person semantics remain explicit: frames `204`, `666`, `708`,
  `780`, and `858` report footpoints; frame `408` reports a lower-body surface;
  frames `330` and `462` report upper-body surfaces. Later presentation must
  not draw the three body-surface fallbacks as if they were ground-track
  samples.
- All eight backpack outputs remain single-camera corrected visible-cluster
  observations. Their motion remains coherent from the pickup side toward the
  drop-off side where masks exist; the accepted two-camera detection gap is
  neither sampled nor filled.

### Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Corrected run summary | `artifacts/s04/corrected_tracking_20260803/summary.json` | Strict D030-D033 policy, sources, records, hashes, and limitations |
| D030 sampling summary | `artifacts/s04/corrected_tracking_20260803/d030_sampling_summary.json` | Margin validity, candidates, confidence filtering, and roles |
| D031 surface summary | `artifacts/s04/corrected_tracking_20260803/d031_visible_surface_summary.json` | Corrected per-camera clouds and exact geometry provenance |
| D032 anchor summary | `artifacts/s04/corrected_tracking_20260803/d032_anchor_summary.json` | Footpoint gates and typed body/backpack fallbacks |
| D033 observation summary | `artifacts/s04/corrected_tracking_20260803/d033_observation_summary.json` | Same-frame semantic priority, scores, weights, states, and XYZ |
| Observation CSV | `artifacts/s04/corrected_tracking_20260803/corrected_d030_d033_observations.csv` | Reviewable corrected surface, anchor, and pair evidence |
| Sample clouds | `artifacts/s04/corrected_tracking_20260803/sample_clouds/` | Twenty corrected selected-pixel clouds |
| Image diagnostics | `artifacts/s04/corrected_tracking_20260803/image_diagnostics/` | Per-view candidate, margin, and anchor overlays |
| Margin contact sheet | `artifacts/s04/corrected_tracking_20260803/person_margin_contact_sheet.png` | Visual QA of all twelve person views and anchor roles |
| World preview | `artifacts/s04/corrected_tracking_20260803/corrected_tracking_world_preview.png` | Distinct footpoint, lower-body, upper-body, and backpack motion |
| Verification | `artifacts/s04/corrected_tracking_20260803/verification.json` | Independent source regeneration, geometry, schema, and visual-QA checks |

### Reproduction Commands

```text
.venv/bin/python scripts/s04/rebuild_corrected_tracking.py \
  --output-dir artifacts/s04/corrected_tracking_<new-run-id>

.venv/bin/python scripts/s04/verify_corrected_tracking.py \
  --summary artifacts/s04/corrected_tracking_<new-run-id>/summary.json \
  --output artifacts/s04/corrected_tracking_<new-run-id>/verification.json \
  --visual-qa-passed

.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/mypy --strict src/spatial_reconstruction scripts/s04
uv --cache-dir .cache/uv lock --check
uv --cache-dir .cache/uv sync --check
git diff --check
```

Both artifact-producing commands refuse to overwrite their outputs. The
builder requires passed D025 scale verification and validates all upstream
hashes. The independent verifier regenerates corrected samples, anchors, pair
selection, scores, weights, and coordinates, and requires explicit visual-QA
acknowledgement.

### Verification

- The independent verifier regenerated all `20` surfaces, `20` anchors, and
  `16` exact-job target observations from D025 corrected depth.
- Raw DA3 depth and confidence remained unchanged. Maximum reprojection error
  was `2.842171e-14 px`; maximum world/camera round-trip error was
  `3.477764e-15 m`.
- Three bottom-truncated views and seven per-camera footpoints regenerated
  exactly. Person pair states are two fused, six single-camera, zero
  disagreement, and zero unavailable in the retained subset.
- Visual review accepted the twelve-view margin contact sheet and the world
  preview: selected samples remain attached to the intended person surfaces,
  footpoints lie on `Z=0`, body fallbacks remain elevated and visually
  distinct, and backpack movement is coherent where observed.
- `223` automated tests passed. Ruff passed across project source, tests, and
  scripts; strict mypy passed across `50` source and S04 script files. Lockfile
  and environment checks would make no changes, and `git diff --check` passed.
- No raw capture, raw DA3 array, existing unscaled artifact, pose, mask, or DA3
  vendor file was modified.

No new model, dependency, fitted floor, triangulation, or external method was
introduced. The upper-body fallback is an exact current-frame corrected-depth
measurement only; no learned or assumed torso-to-foot offset is used.

## D034 Temporal Presentation Policy

Completed on 2026-08-03 from the independently verified corrected D030-D033
observation layer and the retained S03 five-FPS capture-time grid. This action
adds no model inference and does not alter any raw or corrected spatial
measurement.

- Adopted D034 and typed policy `s04_temporal_presentation_v1`:
  - exact corrected observations are `measured` and retain identical raw and
    presentation XYZ plus their source observation, camera, timestamp, and
    anchor-kind provenance;
  - a last coordinate may remain visibly held for no more than `1.0 s`, but it
    is labelled `stale`, has no raw XYZ, and cannot update zones, events, or a
    measured trajectory;
  - after the stale horizon, the presentation state is `missing` with no raw or
    displayed coordinate;
  - interpolation, motion extrapolation, inferred positions, smoothing, and
    anchor-kind conversion are forbidden; and
  - occlusion requires separate affirmative evidence. S03 explicitly did not
    infer occlusion, so no missing detection is relabelled occluded.
- Built `320` deterministic records: two targets across all `160` S03
  capture-time ticks. The result contains `16` measured states, `78` stale
  presentation holds, `226` missing states, zero inferred positions, and zero
  claimed occlusions.
- Every stale record preserves the exact last measurement and anchor kind but
  has no raw XYZ and no spatial authority. The maximum regenerated age is
  exactly `1.0 s`.
- Selected a `3.0 s` measured-segment gate from the observed adjacent-gap
  separation: retained local gaps reach `2.602 s`, while the next gap is
  `4.202 s` and the known backpack gap is `6.803 s`.
- Connected only adjacent exact observations of the same anchor kind. The
  result contains three person footpoint segments and five backpack visible-
  cluster segments. It contains no upper/lower-body segment, mixed-semantic
  segment, stale endpoint, or interpolated point.
- The frame `462` to `666` backpack gap remains disconnected. Segment records
  contain only exact endpoints and do not claim the straight line as a sampled
  physical path.
- Visual review confirms the measured/stale/missing timeline is legible, stale
  holds end before later measurements, the two long gaps sit above the segment
  gate, body-surface person fallbacks remain height-distinct, and the known
  backpack gap has no connecting line.

### Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Presentation records | `artifacts/s04/temporal_presentation_20260803_v3/temporal_presentation_records.json` | All 320 typed measured, stale, missing, occluded, or inferred-state slots |
| Measured segments | `artifacts/s04/temporal_presentation_20260803_v3/measured_trajectory_segments.json` | Exact compatible endpoint connections with time/distance provenance |
| Review CSV | `artifacts/s04/temporal_presentation_20260803_v3/temporal_presentation_review.csv` | Capture-ordered states, source perception, coordinates, age, and authority |
| Timeline diagnostic | `artifacts/s04/temporal_presentation_20260803_v3/temporal_state_timeline.png` | Measured/stale/missing states and adjacent-gap decisions |
| World preview | `artifacts/s04/temporal_presentation_20260803_v3/measured_segment_world_preview.png` | Exact endpoints, accepted segments, and anchor heights |
| Run summary | `artifacts/s04/temporal_presentation_20260803_v3/summary.json` | D034 policy, source hashes, records, segments, counts, and limitations |
| Verification | `artifacts/s04/temporal_presentation_20260803_v3/verification.json` | Independent regeneration, integrity, failure-state, and visual-QA evidence |

### Reproduction Commands

```text
.venv/bin/python scripts/s04/build_temporal_presentation.py \
  --output-dir artifacts/s04/temporal_presentation_<new-run-id>

.venv/bin/python scripts/s04/verify_temporal_presentation.py \
  --summary artifacts/s04/temporal_presentation_<new-run-id>/summary.json \
  --output artifacts/s04/temporal_presentation_<new-run-id>/verification.json \
  --visual-qa-passed

.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/mypy --strict src/spatial_reconstruction scripts/s04
uv --cache-dir .cache/uv lock --check
uv --cache-dir .cache/uv sync --check
git diff --check
```

Both artifact commands refuse to overwrite outputs. The builder requires a
matching passed corrected-tracking verification and the unchanged S03
timeline policy. The independent verifier reloads all source hashes and
regenerates every presentation record and measured segment.

### Verification

- All `320` records and eight segments regenerated exactly from the corrected
  D033 observations and S03 camera timelines.
- Non-measured raw XYZ count, inferred-position count, claimed-occlusion count,
  stale zone-update count, stale trajectory-extension count, mixed-semantic
  segment count, and known-gap bridge count are all zero.
- Maximum stale age is `1.0 s`; maximum accepted measured-segment gap is
  `2.602 s`; the protected backpack gap is `6.803 s`.
- Artifact hashes, persistent schemas, CSV coverage, capture ordering, source
  synchronization, and both visual diagnostics passed.
- `235` automated tests passed. Ruff passed across project source, tests, and
  scripts; strict mypy passed across `53` source and S04 script files. Lockfile
  and environment checks would make no changes, and `git diff --check` passed.
- No interpolation, motion extrapolation, presentation smoothing, raw-capture
  modification, corrected-observation modification, or DA3 vendor
  modification occurred.

No new method, model, dependency, motion prior, or floor operation was
introduced.

## Post-close D037 Occlusion-evidence Correction

Added on 2026-08-03 without reopening S04 or modifying any closed artifact.
The original D034 result correctly declined to infer occlusion from missing
detector output, but its builder passed `confirmed_occluded=False` at every
tick and therefore had no way to consume later affirmative evidence.

The builder and verifier now accept an optional, hash-bound S05 backpack-
visibility summary. A current corrected measurement still wins. Otherwise a
review-backed partial/full occlusion produces the existing D034 `occluded`
state with null raw/presentation XYZ and no zone or trajectory authority.
Detector absence by itself still cannot create that state.

The non-destructive corrected run at
`artifacts/s04/temporal_presentation_occlusion_aware_20260803_v2/` independently
regenerates all 320 records and 23 measured segments. Counts are 33 measured,
118 stale, 136 missing, 33 occluded, and zero inferred. All 33 occluded rows
belong to the backpack at frames `468-660`; all have null XYZ and zero spatial
authority. The original 33 measurements, measured segments, and disconnected
`6.803 s` backpack gap are unchanged.

## D035 Dense Dynamic-Keyframe Refinement

Completed on 2026-08-03 as a separate, non-destructive run from the verified
eight-pair baseline. The original action boundaries remain present, and nine
current-mask pairs were added at roughly one-second transport cadence where
S03 evidence permits. No pair was added inside the accepted frame `462-666`
two-camera backpack absence interval.

### Result

- Ran pose-conditioned metric DA3 on `17` synchronized pairs using the same
  model revision, `504` process resolution, accepted action pose, and raw
  preservation rules.
- Passed raw-depth verification for all 17 predictions: every view contains
  `141120` finite positive depth and finite confidence samples and no S02
  correction leaked into raw artifacts.
- Passed D025 on every pair. Each uses one shared scale from `5-6` accepted
  M40-M42 observations; scales span `1.093693-1.170350` and the maximum marker
  deviation is `1.259%` against the `5%` gate.
- Visually accepted exact nearest-neighbour alignment for all `44` retained
  person/backpack masks; all 34 RGB reproduction checks stay within one
  channel value and processed-intrinsic error stays below `1.24e-05`.
- Regenerated `44` D030/D031 surfaces, `44` D032 anchors, and `34` D033 pair
  observations. Person results are three fused, 13 single-camera, one
  disagreement, and zero unavailable. Frame `828` remains non-authoritative
  because two footpoints disagree by `0.377 m`, exceeding D032's `0.35 m`
  gate.
- Regenerated all `320` D034 presentation slots with `33` measured, `123`
  stale, `164` missing, zero inferred, and zero claimed-occluded states.
- Produced `23` exact same-kind segments: eight person and 15 backpack. Two
  upper-body person segments remain separately styled body-surface evidence;
  they do not enter the footpoint path. The known `6.803 s` backpack gap
  remains disconnected.
- Compared with the sparse baseline, person measured-plus-stale coverage rises
  from `47/160` to `76/160` ticks and backpack coverage from `47/160` to
  `80/160`. This demonstrates temporal evidence coverage, not absolute XYZ
  accuracy; no dynamic ground-truth trajectory exists.

### Generated Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Dense selection | `configs/s04_action_keyframes_dense.json` | Capture-ordered 17-pair profile and evidence reasons |
| Raw DA3 run | `artifacts/s04/action_depth_dense_20260803/summary.json` | Raw depth/confidence, exact frames, poses, model and runtime provenance |
| D025 dense scale | `artifacts/s04/action_depth_scale_dense_20260803/summary.json` | Shared pair scales, marker observations, corrected arrays, and diagnostics |
| Dense mask alignment | `artifacts/s04/mask_alignment_dense_20260803/summary.json` | All 44 exact source-mask mappings and overlays |
| Dense D030-D033 | `artifacts/s04/corrected_tracking_dense_20260803/summary.json` | Margin-aware surfaces, anchors, pair states, and XYZ |
| Dense D034 | `artifacts/s04/temporal_presentation_dense_final_20260803_v2/summary.json` | Final presentation states, exact same-kind segments, and updated diagnostics |
| Sparse/dense comparison | `artifacts/s04/temporal_presentation_dense_final_20260803_v2/density_comparison.json` | Independently regenerated coverage deltas and accuracy limitation |

### Reproduction Commands

```bash
.venv/bin/python scripts/s04/run_action_depth_preflight.py --selection-config configs/s04_action_keyframes_dense.json --output-dir <new-raw-output>
.venv/bin/python scripts/s04/verify_action_depth_preflight.py --summary <new-raw-output>/summary.json --output <new-raw-output>/verification.json
.venv/bin/python scripts/s04/scale_action_depth.py --raw-summary <new-raw-output>/summary.json --output-dir <new-scale-output>
.venv/bin/python scripts/s04/verify_action_depth_scale.py --summary <new-scale-output>/summary.json --output <new-scale-output>/verification.json --visual-qa-passed
.venv/bin/python scripts/s04/align_action_masks.py --action-depth-summary <new-raw-output>/summary.json --output-dir <new-alignment-output>
.venv/bin/python scripts/s04/verify_mask_alignment.py --summary <new-alignment-output>/summary.json --output <new-alignment-output>/verification.json --visual-qa-passed
.venv/bin/python scripts/s04/rebuild_corrected_tracking.py --action-depth-summary <new-raw-output>/summary.json --depth-scale-summary <new-scale-output>/summary.json --depth-scale-verification <new-scale-output>/verification.json --mask-alignment-summary <new-alignment-output>/summary.json --output-dir <new-corrected-output>
.venv/bin/python scripts/s04/verify_corrected_tracking.py --summary <new-corrected-output>/summary.json --output <new-corrected-output>/verification.json --visual-qa-passed
.venv/bin/python scripts/s04/build_temporal_presentation.py --corrected-summary <new-corrected-output>/summary.json --corrected-verification <new-corrected-output>/verification.json --output-dir <new-temporal-output>
.venv/bin/python scripts/s04/verify_temporal_presentation.py --summary <new-temporal-output>/summary.json --output <new-temporal-output>/verification.json --visual-qa-passed
```

The dense cadence is an evidence-quality profile, not a MacBook throughput
ceiling. Future live hardware may run DA3 more frequently if it preserves the
same exact-frame joins, D025 gate, honest disagreement, and missing-data
semantics.

### Verification

- Fresh independent verification regenerated all 17 raw predictions, all
  D025 scales and corrected arrays, all 44 aligned masks, all 44 surfaces and
  anchors, all 34 pair states, all 320 temporal records, all 23 segments, and
  the sparse/dense comparison from retained sources and hashes.
- Raw/confidence immutability, capture order, one-shared-scale enforcement,
  exact RGB/mask alignment, coordinate round trips, reprojection, segment
  semantics, stale authority, and missing-data behavior passed.
- Visual QA passed for the 17-pair marker sheet, 44-mask alignment sheet,
  person margin sheet, corrected world view, temporal-state timeline, and
  measured-segment world view.
- `235` automated tests passed. Ruff passed across source, tests, and scripts;
  strict mypy passed across `54` source/S04 script files. Lockfile and dry-run
  environment checks would make no changes, and `git diff --check` passed.
- No raw capture, DA3 vendor source, sparse baseline, inferred position,
  camera pose, timestamp, or track identity was modified.

## Completion-Gate Audit

Completed on 2026-08-03 over the preferred dense D025-D035 chain.

- Qualitative 3D plausibility passed. The backpack begins `0.128 m` in XY
  from the pickup-zone centre and ends `0.166 m` from the drop-off-zone
  centre; its measured endpoint displacement is `2.545 m`. Person anchors
  remain inside the declared room bounds and preserve footpoint, lower-body,
  and upper-body semantics.
- Foreground/static-depth separation passed. Action observations use only
  exact synchronized dynamic DA3 predictions; synthetic tests reject reuse of
  farther empty-room depth at the same pixel.
- Current-observation fusion passed. Only exact same-job, temporally compatible,
  valid, same-kind anchors can fuse. Single-camera observations remain labelled
  single-camera, and frame `828` person remains disagreement without XYZ.
- Capture ordering passed. Immutable frame/depth/mask joins and deliberate
  reverse worker completion restore authoritative capture-time order.
- Missing-data honesty passed. All non-measured raw XYZ counts, inferred
  positions, claimed occlusions, stale spatial updates, and protected-gap
  bridges are zero.
- Coordinate and projection verification passed, including transform
  inversion, camera/world round trips, reprojection, back-projection,
  invalid/non-positive/low-confidence filtering, and schema round trips.
- Fresh independent artifact verification regenerated all `17` raw
  predictions, `17` shared scales, `44` aligned masks, `44` corrected
  surfaces/anchors, `34` target-pair observations, `320` presentation states,
  and `23` measured segments.
- The updated final measured-segment preview and timeline diagnostic passed
  visual QA. They distinguish original from dense keyframes, separate person
  anchor kinds, show the rejected disagreement, and leave the `6.803 s`
  backpack gap disconnected.
- The complete suite passed: `235` tests, Ruff, strict mypy over `54` source
  and S04 script files, lockfile check, locked-environment dry run, and
  `git diff --check`.

No completion criterion was weakened. The gate is qualitative because no
surveyed dynamic ground-truth trajectory exists; temporal coverage is not
claimed as absolute XYZ accuracy.

## Audited Verification Coverage

- Transform inversion, coordinate round-trip, reprojection, and
  back-projection tests.
- Invalid, non-positive, low-confidence, mismatched, and stale depth handling.
- Synthetic foreground-versus-farther-static-surface rejection.
- Exact-frame joins under deliberately out-of-order worker completion.
- Single-camera availability, cross-camera disagreement, and both-camera
  failure.
- Hidden-feet, person-backpack occlusion, and absent-mask behavior.
- Persistent schema validation and separation of raw, anchor, fused, and
  presentation provenance.
- Qualitative 3D review showing plausible person motion and backpack movement
  from the pickup side toward the drop-off side where observations exist.

## Exact Next Action

Begin S05 by defining the typed pickup-carry-place interaction state machine
over the verified measured/stale/missing person and backpack observations and
the accepted pickup/drop-off zones. Do not allow Qwen to modify spatial facts.
