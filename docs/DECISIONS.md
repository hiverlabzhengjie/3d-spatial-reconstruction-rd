# Decision Log

This file records decisions that affect project scope, architecture, physical
setup, or interpretation of results. Add new entries; do not silently rewrite
past decisions.

## D001 - Exploratory proof of concept

**Date:** 2026-07-27
**Status:** Active

The project will implement one coherent end-to-end methodology and evaluate it
qualitatively. It will not run a rigorous controlled comparison of alternative
reconstruction or localization methods.

## D002 - Single-scene scope

**Date:** 2026-07-27
**Status:** Active

The validated environment is one living room observed by two fixed,
overlapping cameras. Interfaces may support more cameras later, but multi-room,
multi-floor, BIM, and GIS work is excluded.

## D003 - DA3 is the geometry backbone

**Date:** 2026-07-27
**Status:** Active

Use `DA3NESTED-GIANT-LARGE-1.1` in multi-view, pose-conditioned mode for metric
depth and static scene geometry.

If DA3's predicted/aligned poses are unstable, retain its metric depth and
back-project with the externally calibrated OpenCV camera poses. Do not
introduce another depth or reconstruction pipeline in this version.

## D004 - DA3-depth localization only

**Date:** 2026-07-27
**Status:** Active

Person and backpack XYZ observations will be derived from DA3 metric depth and
segmentation-mask pixels. Triangulation, floor-plane intersection, stereo
matching, SfM, MVS, and COLMAP comparisons are excluded.

## D005 - Backpack movement scenario

**Date:** 2026-07-27
**Status:** Active

The demonstration uses one person moving one backpack from a pickup zone to a
drop-off zone. A bottle is the approved fallback if the backpack cannot be
detected reliably using a standard YOLO model.

## D006 - YOLO perception

**Date:** 2026-07-27
**Status:** Active

Use Ultralytics `yolov8n-seg.pt` for person/backpack masks and ByteTrack for
camera-local tracklets. No custom detector training is planned.

## D007 - Qwen has a bounded semantic role

**Date:** 2026-07-27
**Status:** Active

Use `Qwen/Qwen3-VL-2B-Instruct` only on triggered short clips to classify or
summarize pickup, carry, and placement actions.

Qwen may not change spatial coordinates, track identity, timestamps, or zone
membership. Invalid structured output receives one repair attempt and then
becomes `unknown`.

## D008 - MP4 first, RTSP compatibility only

**Date:** 2026-07-27
**Status:** Active

Recorded MP4 files are the required development and demonstration workflow.
The input abstraction must support RTSP and receive a local smoke test, but a
production or near-live CCTV deployment is excluded.

## D009 - Rerun instead of a custom frontend

**Date:** 2026-07-27
**Status:** Active

Use Rerun for synchronized video, 3D points, camera frustums, trajectories,
zones, events, diagnostics, and shareable recordings. Do not build a custom web
viewer in this project.

## D010 - Honest missing-data behaviour

**Date:** 2026-07-27
**Status:** Active

The system must not fabricate a precise backpack position while it is occluded.
It may retain an interaction hypothesis such as "probably carried," but the
last observed position must be marked stale.

## D011 - Local, non-commercial research

**Date:** 2026-07-27
**Status:** Active

All processing remains local on the M1 Max. This is non-commercial research,
and model/library licences must be reviewed before external or commercial use.

## D012 - Qwen model identity correction

**Date:** 2026-07-27
**Status:** Active

The approved vision-language model is Qwen3-VL 2B Instruct, with the model
identifier `Qwen/Qwen3-VL-2B-Instruct`. Any earlier reference to a
prior-generation Qwen model was a documentation error and was never an
approved architectural choice.

## D013 - Stage numbering begins at S00

**Date:** 2026-07-27
**Status:** Active

The approved eight stages are numbered S00 through S07. S00 is the project
setup and MPS model gate. Stage task titles, status records, implementation
briefs, and completion handoffs must use this numbering.

## D014 - Native macOS model runtime

**Date:** 2026-07-27
**Status:** Active

Use the project-owned native Python 3.11 environment for Apple MPS model
inference. Docker may support isolated services or protocol tests later, but it
is not the model-inference runtime because its Linux containers do not expose
Apple MPS. VS Code uses the project-local `.venv`.

Ollama remains available to the operator but does not replace the exact
approved DA3, YOLO, or `Qwen/Qwen3-VL-2B-Instruct` integrations.

## D015 - Controlled use of supporting methodologies and tools

**Date:** 2026-07-27
**Status:** Active

DA3, YOLO, Qwen3-VL, OpenCV, and Rerun remain the streamlined baseline and
their current roadmap gates remain required. Other useful methodologies and
tools—including COLMAP, SfM, MVS, stereo, triangulation, floor-plane methods,
or additional models—may be introduced when they provide a concrete benefit
and do not add disproportionate complexity.

Before using a non-baseline method, the implementing task must highlight it to
the user and state:

1. the problem it solves;
2. why the existing baseline is insufficient for that problem;
3. its dependencies and operational cost;
4. the output it is allowed to influence; and
5. how it will be tested, isolated, and removed if it does not help.

The applicable stage record and this decision log must be updated. Prefer
replaceable adapters, optional dependency groups, and separate processes for
tools with conflicting runtimes. Broad method-comparison programmes and
survey-grade claims still require a separate explicit decision.

This decision supersedes the categorical exclusions in D001, D003, and D004
where they conflict with this controlled-use policy. It does not by itself add
another method to the active pipeline.

## D016 - Project-level version and experiment history

**Date:** 2026-07-27
**Status:** Active

Use one Git repository at the project root to preserve code, configuration,
tests, decisions, and stage handoffs. Establish an initial research-plan
checkpoint, then create at least one dedicated descriptive commit when each
S00-S07 stage closes. Optional annotated tags may identify important
stage/experiment versions.

Do not commit raw recordings, model weights, generated model outputs, virtual
environments, caches, or the unmodified DA3 vendor checkout. Record those
inputs through capture manifests, exact model revisions, artifact records, and
vendor fingerprints. Each stage handoff records its close commit and optional
tag.

## D017 - Public GitHub remote and stage publishing

**Date:** 2026-07-27
**Status:** Active

Use `https://github.com/hiverlabzhengjie/3d-spatial-reconstruction-rd` as the
public `origin` remote. After a stage passes its gate and its handoff is
complete, push the dedicated stage-close commit to `origin/main`, push any
annotated stage tag explicitly, and verify the remote commit/tag before
stopping.

The current streamlined workflow allows direct stage-close pushes to `main`
for the single operator. If concurrent engineering begins, adopt short-lived
branches, reviews, and pull requests through a later explicit decision. Public
GitHub history must never contain raw captures, credentials, private notes,
model weights, generated artifacts, local caches, or the DA3 vendor snapshot.

## D018 - Dynamic localization requires action-frame depth

**Date:** 2026-07-27
**Status:** Active

A 2D pixel identifies a camera ray and becomes a unique camera-space point only
when paired with a depth. The empty-room/static reconstruction records the
background surface along that ray. It must not be reused as the depth source
for a person, backpack, or other dynamic foreground entity detected in a later
action frame.

S04 dynamic XYZ observations must use DA3 depth and confidence associated with
the same camera and synchronized action-frame content as the segmentation mask,
subject only to an explicit, tested, and recorded temporal tolerance. Each raw
observation must retain depth-frame identity, timestamp, and freshness
provenance. If the required current depth is unavailable, invalid, or too old,
the raw XYZ is unavailable; the system must emit a missing, occluded, or stale
state rather than back-project against empty-room geometry.

Two overlapping cameras provide independent current observations, better
visibility, confidence fusion, and disagreement checks. They do not
automatically correct a stale/static depth lookup. Fusion may use only
temporally compatible observations that independently pass depth and
confidence checks. If both cameras lack valid current depth, no precise XYZ may
be fabricated by intersecting the rays with the static scene.

The static reconstruction may still provide visualization context, room
bounds, zones, and conservative plausibility or occlusion checks. Those checks
must not overwrite the dynamic depth measurement. A conflict such as a dynamic
observation appearing behind a known opaque surface is diagnostic evidence to
reject, down-weight, or investigate the observation, not permission to snap it
to that surface.

Elevated, downward-looking CCTV placement is beneficial and should be used
deliberately during S01 capture planning. It can improve floor visibility,
reduce some furniture occlusions, and make lower-body or ground-contact
localization more observable. It reduces risk but does not resolve the
ray-depth ambiguity: feet can remain hidden, the person can occlude the
backpack, and visible mask pixels still require current depth.

The spatial meaning of the reported XYZ must also be explicit. Back-projecting
an in-mask pixel estimates a visible surface point, not automatically a person
centre. S04 must evaluate robust mask-depth aggregation and define suitable
anchors, such as a lower-body/ground-contact estimate for a person and a robust
in-mask depth cluster for a backpack. Raw surface measurements, derived track
anchors, and smoothed/inferred presentation positions must remain
distinguishable.

DA3 may run on offline action keyframes rather than every detector frame.
Real-time 2D detection therefore does not imply a current measured 3D
observation. Reusing keyframe depth after a dynamic entity has moved is
forbidden as a raw measurement. Any later temporal propagation must be
explicitly modelled, labelled as inferred or stale as appropriate, and kept
separate from raw XYZ.

Before the S04 completion gate passes, tests and diagnostics must cover:

1. a foreground entity occluding a farther static surface;
2. rejection or explicit flagging of mismatched and stale depth frames;
3. valid single-camera operation when the other view is unavailable;
4. cross-camera disagreement and both-camera failure;
5. hidden feet and person-backpack occlusion; and
6. preservation of raw, derived-anchor, and presentation-state provenance.

## D019 - Two-view DA3 post-alignment stays in the project adapter

**Date:** 2026-07-28
**Status:** Active

The DA3 vendor API applies Umeyama Sim(3) alignment after inference whenever
camera poses are supplied. Exactly two camera centres are geometrically
insufficient to determine a full Sim(3), so the vendor post-processing raises
a degenerate-covariance error even though its pose-conditioned MPS forward pass
has completed successfully.

For exactly two supplied views, the project-owned DA3 adapter bypasses only
that post-inference Umeyama step. It preserves the nested model's already
metric-scaled depth and returns the supplied, preprocessed OpenCV intrinsics and
`T_camera_from_world` poses. It rejects this path if the model output is not
metric. Runs with another view count continue to delegate to the vendor
alignment implementation.

This is a bounded API compatibility path, not a new reconstruction method and
not a metric-accuracy claim. S00 uses synthetic cameras only to verify the
pose-conditioned interface and MPS execution. S01 and S02 must use calibrated
cameras and physical scene evidence before making geometric claims. The vendor
source remains unmodified.

## D020 - Timestamped multi-rate model workers

**Date:** 2026-07-28
**Status:** Active

Adopt DA3, YOLO/ByteTrack, and Qwen as three logically independent workers that
accept timestamped jobs and return timestamped, provenance-preserving results.
They operate at different rates: YOLO/ByteTrack at the highest practical
perception rate, DA3 for the static scene and selected synchronized action
keyframes, and Qwen only for triggered candidate-event clips.

Logical asynchrony is separate from hardware parallelism. On the current
single M1 Max, one project-owned accelerator controller serializes heavy MPS
inference by default. CPU ingestion, decoding, synchronization, queue handling,
validation, state updates, and artifact writing may continue concurrently.
Simultaneous heavy MPS inference requires later measured evidence that peak
memory, latency, throughput, and correctness remain acceptable.

Every job and result must preserve immutable source identity sufficient to
associate capture session, camera, frame, and capture timestamp. Synchronization
provenance and processing timing remain separate. Downstream joins use source
identity and an explicit temporal tolerance, never worker completion order or
an unqualified "latest result."

Queues must be bounded and expose their overload behavior. Deterministic
recorded-MP4 execution may throttle input or process model-sized batches to
avoid repeated model loading. A future live policy may coalesce or drop
superseded work to protect freshness, but it must record what was omitted and
must preserve event-relevant inputs according to an explicit priority policy.
No queue may accumulate unbounded latency.

Qwen remains asynchronous, triggered, and unable to block or modify geometry,
identity, timestamps, or zones. Its timeout or failure becomes an explicit
`unknown` semantic result while perception and geometry continue. Missing or
late DA3 output likewise produces missing/stale spatial state rather than reuse
of mismatched depth.

The worker contracts and behavior are implemented incrementally:

1. S01 establishes deterministic synchronized frame identity;
2. S03 establishes the bounded perception worker interface;
3. S04 establishes DA3 action-keyframe jobs and strict mask/depth joins;
4. S05 establishes the triggered Qwen queue and non-blocking failure path;
5. S06 integrates orchestration, accelerator arbitration, diagnostics, replay,
   and RTSP compatibility; and
6. S07 records measured offline capacity and the engineering changes required
   for future live production.

This post-S00 decision does not reopen or weaken S00. S00 independently proved
the model adapters, MPS operation, runtime provenance, separate-process
execution, and asynchronous Qwen boundary needed by this architecture.
`docs/MODEL_SCHEDULING.md` is the detailed implementation contract.

## D021 - Shared intrinsic estimate for the matched phone pair

**Date:** 2026-07-30
**Status:** Active

Camera A and Camera B are both iPhone 16 Pro Max devices using the same
13 mm-equivalent ultrawide lens, 1920x1080 recording mode, nominal 30 FPS,
disabled enhanced stabilization and automatic lens switching, and the same
focus/exposure/white-balance locking procedure. For this exploratory prototype,
the accepted Camera A ChArUco capture supplies one shared numerical intrinsic
and distortion estimate for both cameras.

Persistent calibration outputs must still contain separate `camera_a` and
`camera_b` records. Each record will state that its numeric parameters came
from the shared capture `camera_ab_intrinsics.mp4`, rather than implying that
Camera B was independently calibrated. The raw source hash and common recording
configuration will be retained.

This is a bounded accuracy tradeoff, not a claim that two physical lenses are
identical. Camera B must pass fixed-world-pose marker reprojection and frustum
plausibility checks. Materially worse Camera B reprojection, systematic
edge residuals, or failure of the S01 pose gate requires a separate Camera B
intrinsic capture and calibration.

## D022 - Fixed camera poses use common markers M40-M42

**Date:** 2026-07-30
**Status:** Active

The fixed Camera A and Camera B poses use M40, M41, and M42 as their common
world-pose anchors. These are the three complete, non-collinear 180 mm markers
detected repeatedly in both synchronized cameras. Their aggregate reprojection
errors are `1.527 px` for Camera A and `1.481 px` for Camera B, and both pose
stability and frustum-plausibility checks pass.

M43 is not used to fit either pose. Camera B shows only a clipped portion that
cannot produce an ArUco detection. In Camera A, M43 is detected but its recorded
centre `(1.10, 3.70, 0.00) m` produces `100.044 px` RMS reprojection error when
checked against the independently fitted M40-M42 pose. This is inconsistent
with the stated `+/-0.05 m` centre-measurement uncertainty.

The project will not replace M43's measured coordinates with a value inferred
from the calibration image. Its failed check remains in the diagnostic output.
The M40-M42 pose is accepted for this exploratory prototype because it retains
twelve well-distributed planar corners and passes both cameras' numerical and
visual checks. If later static geometry exposes a material world-alignment
problem, M43 must be remeasured/repositioned and the fixed-pose calibration
repeated rather than silently adjusted.

## D023 - Versioned capture-specific pose correction

**Date:** 2026-07-30
**Status:** Active

Retain the fixed-world-pose calibration as the physical reference, but allow
each later recording to store a versioned capture-specific
`T_world_from_camera` and `T_camera_from_world` correction solved from
stationary M40-M42 observations. This addresses small effective image-pose
changes caused by unavoidable mount settling or a phone camera's
per-recording stabilization state without changing the surveyed world frame,
marker coordinates, or shared intrinsic estimate.

A capture-specific correction is accepted only when:

1. M40-M42 are complete, stationary, and jointly visible in a stable interval;
2. aggregate and sampled marker reprojection remain within the established
   `5 px` thresholds;
3. the capture's per-frame pose stability, camera height, downward optical
   axis, and floor-intersection checks pass;
4. camera-centre displacement from the physical reference is no more than
   `0.05 m`; and
5. rotation difference from the physical reference is no more than
   `1.0 degree`.

Exceeding either reference-difference boundary is treated as physical camera
movement or invalid calibration, not as an automatically accepted correction.
Every downstream frame bundle must identify the capture-specific pose version
it uses. No correction may be inferred from an occluded or moving marker.

For a future live CCTV feed, the analogous policy is periodic marker-based
drift monitoring and timestamped pose-version updates while the markers are
reliably visible. It is not unconstrained per-frame pose fitting, and it does
not weaken the requirement to invalidate calibration after material physical
camera movement. Production monitoring remains outside the present prototype.

## D024 - Video-estimated pickup and drop-off zones

**Date:** 2026-07-30
**Status:** Active

Estimate the two zone centres from the synchronized empty-room video instead
of requiring a separate physical survey. The visible blue and white ropes are
treated as thin circle boundaries, not as filled colour regions. Their supplied
horizontal radius remains fixed at approximately `0.30 m`.

This activates two narrowly scoped supporting geometry operations:

1. intersect each camera's white-rope centre ray with the known world `Z=0`
   floor plane, then compare/fuse the two estimates; and
2. triangulate an initial blue-rope centre above the bed, then refine its
   `(X, Y, Z)` against the projected `0.30 m` horizontal ring boundary in both
   cameras.

The existing baseline cannot recover the bed zone's depth from one annotated
pixel because a pixel defines only a ray, and the user has declined a physical
bed-height/zone survey. The implementation uses only existing OpenCV and NumPy
geometry with a small semi-automatic annotation/validation step; it adds no
model or runtime dependency.

These estimates may influence only persistent pickup/drop-off zone metadata,
zone-membership checks, event-state transitions, and zone visualization. They
may not change camera calibration, marker coordinates, DA3 depth, person or
backpack XYZ observations, track identity, or timestamps. In particular, this
decision does not introduce triangulation as the S04 person/backpack
localization method.

Acceptance requires:

- synthetic ray/plane and two-view triangulation tests;
- positive-depth and declared-room plausibility;
- low cross-camera reprojection/ring-boundary residuals;
- independent Camera A/B agreement for the floor zone;
- annotated overlays in both cameras; and
- user visual/sanity validation of the resulting estimates.

If the views disagree materially or the rope cannot be annotated reliably, the
estimate remains unaccepted and the fallback is physical measurement or a
clearer zone capture. Inferred values remain explicitly labelled
video-estimated rather than surveyed.

## D025 - Marker-anchored scalar correction for derived DA3 depth

**Date:** 2026-07-31
**Amended:** 2026-08-03
**Status:** Active

The first calibrated S02 run used raw pose-conditioned DA3 metric depth at
process resolution `504`. Its scene structure was visually coherent, but both
cameras placed the known floor approximately `0.23 m` above world `Z=0`.
At the projected centres of accepted markers M40-M42, raw DA3 depth
underestimated the expected calibrated camera-Z depths by approximately
`14-16%` in both views.

For S02 static geometry only, derive one shared scalar depth correction per
synchronized DA3 pair from the median of:

```text
expected calibrated camera-Z depth / raw DA3 depth
```

at the projected M40-M42 centres in both cameras. A shared scalar preserves the
relative scale between the two views. Accept it only when all six marker ratios
are finite, positive, and remain within `5%` relative deviation of their
median. M43 remains excluded under D022.

This is a bounded supporting calibration operation under D015. It uses the
existing OpenCV/NumPy stack and adds no model, dependency, or independent
reconstruction pipeline. It may influence only the derived S02 static point
cloud and geometry preview. It may not alter raw DA3 depth/confidence,
intrinsics, camera poses, marker coordinates, frame identity, or timestamps.
Raw and corrected depth plus all scale observations must remain separately
inspectable.

The correction is isolated behind an explicit processing option, has synthetic
recovery and disagreement-rejection tests, and can be removed by disabling the
option. If the six ratios disagree beyond the limit, the run must stop instead
of applying a partial or camera-specific correction. This decision makes no
survey-grade accuracy claim; marker centres retain their stated
`+/-0.05 m` uncertainty.

The 2026-08-03 amendment adds a separate S04 action-pair profile after the
retained dynamic frames showed the same marker-consistent DA3 depth
underestimate. This profile supersedes only the earlier prohibition on using
D025 for S04 derived depth; it does not import the S02 centre-patch sampling
rule or any other S02 policy.

For each exact synchronized S04 action pair:

- preserve the original DA3 depth and confidence arrays byte-for-byte and
  write corrected depth as a separate derived artifact;
- detect only accepted pose-anchor markers M40-M42 on each exact undistorted
  keyframe; M43 remains excluded under D022;
- require each detection centre to remain within `5 px` of its calibrated
  projection;
- sample per-pixel `expected camera-Z / raw DA3 depth` ratios inside the
  protected inner `60%` of the known `180 mm` floor-marker square, using the
  calibrated ray intersection with world `Z=0`; require at least `16` finite,
  positive samples per marker and use their median as that marker's ratio;
- derive exactly one shared pair scale as the median of the accepted marker
  ratios across both views; require at least two accepted markers per camera,
  at least five observations across the pair, and no marker ratio more than
  `5%` from the shared median; and
- if any gate fails, mark corrected action depth unavailable for that pair.
  Do not apply a partial scale, camera-specific scale, stale scale, or silent
  unit-scale fallback.

The S04 correction may influence only later dynamic products rebuilt from the
separate corrected arrays. It does not retroactively validate or modify the
existing raw D030-D033 evidence, whose unscaled artifacts remain retained for
comparison. The action-pair profile has synthetic scale-recovery plus missing-
camera, insufficient-evidence, reprojection, and disagreement-rejection tests.
It remains a prototype calibration correction rather than a survey-grade
accuracy claim.

## D026 - Lower static-scene confidence percentile for room completeness

**Date:** 2026-07-31
**Status:** Active

The accepted S02 reconstruction at DA3's `40th` confidence percentile was
geometrically coherent, but user inspection showed that the door, primary
wall, tall white cabinet, and table-top lamp were weak or absent. Diagnostic
counts showed that expanding the declared X/Y room bounds would recover only
`21` finite-depth samples outside those limits and would therefore not restore
the missing surfaces. The relevant in-bounds surfaces instead had confidence
values below the per-pair `40th` percentile.

For the derived S02 static reconstruction, use the `20th` confidence percentile
while retaining the existing processing bounds of `(-0.5, -0.5, 0.0)` to
`(3.0, 4.5, 3.0) m`. Keep the floor clip at world `Z >= 0`; an evaluated
`Z >= -0.1 m` variant introduced a visible below-floor sheet and is rejected.
The user selected the `20th`-percentile, `Z >= 0` comparison as the preferred
completeness/noise trade-off.

This change affects only confidence filtering of derived S02 static point
clouds. It does not alter raw DA3 depth/confidence, D025 marker scaling,
calibration, camera poses, source frames, timestamps, or later dynamic-object
localization. Preserve the prior `40th`-percentile artifacts as a baseline,
write the revised run to a new artifact directory, and re-run finite-value,
bounds, cross-camera overlap, schema, hash, and visual Rerun verification
before re-closing S02.

## D027 - Bounded low-confidence supplement for the static room door

**Date:** 2026-07-31
**Status:** Active

User inspection of the D026 Rerun scene showed that the room door behind M40
was still effectively absent even though it is visible in both accepted source
views. Raw-prediction diagnostics confirmed valid metric depth inside the
existing room bounds. The door-region median confidence was approximately
`4.5-4.9`, while the D026 per-pair `20th`-percentile thresholds were
approximately `5.1-5.8`. Camera B retained no door-volume samples at that
threshold and Camera A retained only sparse fragments.

Keep the global D026 `20th`-percentile filter unchanged. For the one-time
derived S02 static reconstruction only, additionally retain finite points at
or above the per-pair `5th` confidence percentile when their reconstructed
world positions fall inside the door inclusion volume
`(-0.35, -0.40, 0.00)` to `(0.90, -0.12, 2.10) m`. The inclusion volume is a
video-estimated processing region rather than surveyed door geometry and
remains wholly inside the existing room bounds.

This bounded supplement may affect only derived S02 static point clouds and
their previews/Rerun export. It must not alter raw DA3 outputs, D025 scaling,
camera calibration, source identity, the global room bounds, S03 perception,
or S04 dynamic localization. Persist the supplemental threshold, volume, and
retained-point counts; test that unpaired or unbounded supplemental policies
fail; and re-run overlap, finite-value, room-bound, artifact-integrity, and
visual Rerun checks before accepting the result.

### Reusable technique pattern

D027 establishes a reusable technique for one-time static reconstruction when
a known, important feature has valid depth but is removed by a scene-wide
confidence cutoff:

1. Confirm the feature is visible in accepted source views and has finite,
   positive model depth.
2. Prove that confidence filtering, rather than camera coverage or room
   bounds, is the cause of the omission.
3. Keep the accepted global confidence threshold unchanged.
4. Define the smallest practical inclusion region in the calibrated world
   frame and allow a lower confidence percentile only inside that region.
5. Continue applying finite-depth and global room-bound checks to every
   retained point.
6. Persist the global and regional thresholds, inclusion bounds, and added
   per-camera counts.
7. Re-run cross-camera overlap, finite/bounds, artifact-integrity, and visual
   Rerun checks; also confirm that the feature survives visualization
   sampling.

Prefer a world-space inclusion region over a camera-specific pixel mask when
calibrated multi-view depth is available. A world-space region expresses one
physical feature consistently across cameras and frames, while a pixel mask
would require separate view-specific annotations and can drift with image
processing. Do not use regional threshold relaxation to invent missing depth,
expand the global room envelope, conceal a calibration error, or silently
change dynamic-localization policy. Each new region requires its own recorded
rationale and verification evidence.

## D028 - Guarded backpack and handbag perception policy

**Date:** 2026-07-31
**Status:** Active

The S03 dense diagnostic ran the exact `yolov8n-seg.pt` checkpoint on 33
synchronized times in both accepted action cameras. At the configured `0.25`
threshold, the physical backpack received the literal `backpack` label in only
two of 66 camera frames, both near placement. The same object received useful
`handbag` masks in thirteen Camera A frames during its stationary/pickup phase
and three Camera B frames near placement. Lowering the diagnostic floor to
`0.10` produced only five literal `backpack` detections and introduced more
false positives, so threshold reduction does not solve the class instability.

Retain the existing backpack action recording and treat vendor classes
`backpack` and `handbag` as candidates for one canonical physical backpack
track. This is a configured interpretation policy around the unchanged
standard YOLO checkpoint, not custom training or replacement detection. The
vendor class, confidence, mask, box, and raw arrays remain unchanged and
inspectable.

The policy is guarded as follows:

1. `backpack` and `handbag` are the only allowed bag candidate labels;
   `suitcase` is explicitly excluded because it repeatedly masks most of the
   bed in Camera A.
2. The production detection floor remains `0.25`; the `0.10` run is diagnostic
   evidence only.
3. An alias candidate is not by itself a confirmed object identity. ByteTrack
   must assign camera-local identity through capture-time spatial continuity,
   starting from the known single-bag scenario.
4. Vendor class changes may occur within one continuous camera-local bag track,
   but raw labels are never overwritten.
5. Conflicting simultaneous bag candidates, track discontinuity, or loss of a
   valid candidate becomes ambiguous, missing, or occluded state rather than a
   fabricated continuation.
6. This policy may influence only S03 target selection/tracking and downstream
   use of the resulting backpack masks. It may not change timestamps, source
   identities, camera calibration, DA3 depth, spatial coordinates, zones, or
   Qwen semantics.

The approved bottle remains a fallback if the guarded alias policy still fails
the S03 tracking gate. The fallback is not activated by this decision.

## D029 - Five-FPS perception cadence and representative object boundary

**Date:** 2026-07-31
**Status:** Active

Use a nominal S03 YOLO/ByteTrack cadence of `5 FPS` per camera for the accepted
two-camera recorded workflow. Do not run the proposed continuous `10 FPS`
comparison. The native 5 FPS smoke consumed approximately `30.01 s` of summed
inference wall time for about `31.81 s` of two-camera capture interval: roughly
94% of one serialized real-time compute budget before decoding, queueing,
artifact writing, or other model work. Doubling the cadence would not be a
credible future-live default on the single M1 Max.

The observed backpack fragmentation is dominated by absent detections,
occlusion, viewpoint changes, and vendor-label instability rather than clear
evidence of temporal undersampling. Camera B already retained one person track
for 152 of 160 processed frames at 5 FPS, while backpack candidates were absent
from 123 Camera A frames and 146 Camera B frames. Higher continuous sampling
may add isolated detections but does not directly solve that failure mode.

The backpack remains the approved representative movable object for this proof
of concept, not a claim about the eventual application's permanent object
taxonomy or primary value. The project value is the end-to-end handling of
synchronized perception, metric spatial observations, calibrated scene
context, visibility/occlusion, state transitions, semantic events, provenance,
and Digital Twin-style presentation. S03 must still meet its recorded gate
honestly, but it will not optimize backpack classification beyond the bounded
D028 policy, switch to custom training, or spend the project budget pursuing
an uninterrupted detector label.

Recorded-MP4 execution uses deterministic capture-order throttling with no
silent source drops. A future live execution policy may coalesce or drop
superseded ordinary perception work to protect freshness, but every such
disposition must be explicit and measured. Short event-prioritized bursts may
be evaluated later only if evidence shows a concrete benefit; continuous
10 FPS inference is not the default.

## D030 - Candidate-relative dynamic depth confidence and visible-surface rules

**Date:** 2026-08-01
**Amended:** 2026-08-03
**Status:** Active

S04 diagnostics compared whole masks, two-pixel eroded interiors, adaptive
connected depth clusters, and person lower-body regions over `20` observed
masks from eight exact synchronized action pairs. The full-frame DA3
confidence distribution is dominated by the static scene and is not a safe
dynamic-object validity reference: at the same-action-frame `20th` percentile,
at least one person and one backpack candidate retained zero samples. Median
retention was approximately `6.95%` for person lower-body candidates and
`1.96%` for backpack connected clusters.

Use policy `s04_dynamic_visible_surface_v1` for subsequent S04 dynamic sample
selection:

1. Require finite positive DA3 depth and finite confidence inside the exact
   current-frame candidate mask.
2. Compute the confidence threshold from the valid samples of that object
   candidate, not from the full image or an S02 static-scene policy. Retain
   samples at or above the candidate's `20th` confidence percentile.
3. For a person, use the lower `35%` of the observed mask bounding extent and
   require at least `256` retained samples. Aggregate ray depth with the
   median. This is a visible lower-body surface measurement, not a body centre
   or guaranteed ground-contact point.
4. For the backpack, erode the mask by two processed-grid pixels, form an
   adaptive interval around its median depth using the greater of `0.15 m` or
   `2.5 * 1.4826 * MAD`, retain the largest eight-connected component, and
   require at least `128` confidence-valid samples. Aggregate ray depth with
   the median. This is a visible backpack surface measurement, not an object
   centre.
5. If the target candidate is absent, invalid, or below its minimum retained
   sample count, emit an unavailable observation. Do not fall back to the
   whole mask, another timestamp, static depth, S02 confidence, or fabricated
   coordinates.

On the retained evidence, candidate-relative `p20` kept approximately `80%`
of each selected candidate: at least `382` person and `203` backpack samples.
It reduced median relative depth MAD from `0.03479` to `0.02937` for person
lower-body samples and held backpack connected-cluster median relative MAD
essentially stable at approximately `0.01516`. Median depth shifted by at most
`2.91%` for person and `0.90%` for backpack. Higher candidate percentiles
reduced dispersion further but discarded substantially more support and
increased maximum median shifts; they are not selected for this bounded proof
of concept.

The 2026-08-03 corrected-depth rebuild keeps candidate-relative `p20` but
amends person candidate validity under policy
`s04_corrected_margin_aware_tracking_v1`. Record contact with every processed-
image margin using a two-pixel band. Bottom-margin contact means the observed
mask is vertically truncated and therefore cannot support the lower-`35%`
footpoint path. In that case, select the adaptive connected depth cluster and
label it only as a measured upper-body surface. Top or side contact remains
diagnostic but does not automatically invalidate lower-body evidence because
it does not by itself prove hidden feet. Backpack sampling remains the
candidate-relative `p20` connected-cluster rule. All candidates use D025's
separate corrected action-pair depth while preserving the raw arrays.

This decision does not authorize back-projection, XYZ, ground snapping,
cross-camera fusion, smoothing, temporal filling, or changes to raw DA3
outputs. The numeric thresholds are prototype policy derived from the accepted
one-person/one-backpack action evidence, not calibrated confidence
probabilities or a general production guarantee.

## D031 - Preserve raw per-camera visible-surface clouds and camera-frame medians

**Date:** 2026-08-02
**Amended:** 2026-08-03
**Status:** Active

For each exact current-frame mask/depth join accepted by D030, back-project
every retained pixel with its own metric depth and the processed camera
intrinsics. Transform every sample using the accepted explicit
`T_world_from_camera`. Preserve both the full per-camera sample cloud and one
robust aggregate computed as the component-wise median of camera-frame XYZ,
then transform that aggregate once into the world frame.

The camera-frame median is selected because it is robust to residual sample
outliers, retains D030's median ray-depth meaning in camera Z, and avoids
mixing coordinate frames during aggregation. It is a raw visible-surface
summary only. It must not be labelled a person centre, ground contact,
backpack centre, fused observation, or presentation position.

Require an exact immutable join across action-depth job, synchronized bundle,
frame, camera, and capture timestamp. The tolerance for this retained
same-frame action evidence is zero, and worker completion order cannot
participate in the join. Invalid or undersampled candidates produce an
unavailable result with no placeholder XYZ. Do not clip or snap samples to
room bounds, reuse S02 depth, fill another timestamp, derive an anchor, fuse
cameras, or smooth presentation state in this raw layer.

On the retained S04 evidence, all `20` observations regenerated exactly from
their source masks and raw DA3 arrays. Maximum sample reprojection error was
`2.842171e-14 px`, maximum world/camera round-trip error was
`3.477764e-15 m`, and maximum returned-pose error was `1.395500e-07`. All
samples were inside the approximate diagnostic room bounds. The four paired
person views still differed by `0.384-0.679 m`, demonstrating why the later
anchor/fusion layer must not directly average view-dependent raw surfaces.

The 2026-08-03 rebuild applies the same exact-frame geometry to D025 corrected
depth and persists the surface role alongside every cloud: person lower body,
person upper body, or backpack visible cluster. It regenerated all `20`
surfaces, including three bottom-truncated person views, with maximum
reprojection error `2.842171e-14 px` and maximum world/camera round-trip error
`3.477764e-15 m`. The former unscaled clouds remain labelled baseline evidence
and are not silently relabelled or overwritten.

This decision authorizes raw per-camera visible-surface XYZ only. Target-anchor
semantics, cross-camera disagreement thresholds, fusion weights, temporal
state, and presentation smoothing require later S04 evidence and decisions.

## D032 - Target anchor semantics and pre-fusion disagreement gate

**Date:** 2026-08-02
**Amended:** 2026-08-03
**Status:** Active

Use policy `s04_target_anchor_v1` between D031 raw visible surfaces and any
later cross-camera observation.

For person tracking, select the component-wise world-frame median of the
lowest world-Z quintile of D031's validated lower-body sample cloud. Require at
least `32` samples; the retained evidence supplies at least `77`. This is a
measured lower-body surface anchor, not an anatomical centre and not guaranteed
ground contact. It preserves more support than the lowest decile while giving
a more consistent vertical meaning than the full raw surface or the bottom
image-space quintile.

Keep person ground contact as a separate derived state. Reuse the selected
lower-quintile XY and project it to the already surveyed `world Z=0` floor only
when the measured lower-quintile median height is at most `0.35 m`. If that
evidence gate fails, ground contact is unavailable with no XYZ. Six of twelve
retained person observations pass; six remain unavailable. This projection
never changes D031 raw XYZ or the measured person tracking anchor.

A lightweight floor method was also evaluated: intersect validated bottom-
quintile pixel rays directly with the existing horizontal world floor. It adds
only deterministic ray-plane geometry and no fitted plane, model, dependency,
or extra capture. Do not select it: one of twelve results leaves the room
bounds, paired disagreement reaches `1.110 m`, and it is not consistently more
coherent than the measured lower-body alternatives.

For the backpack, select the component-wise world-frame median of its D031
visible cluster. This remains a visible-cluster centre, not the hidden physical
centroid. On the limited repeatability evidence, it reduces the maximum
separation across three stationary pickup observations from `0.136 m` for the
D031 reference to `0.126 m`, and reduces the two placed observations from
`0.056 m` to `0.045 m`.

Before fusion, compare exact same-job selected anchors in world space. A pair
is eligible only when its Euclidean separation is at most `0.35 m`. The
bounded person evidence contains one pair at `0.231 m`; the next closest is
`0.474 m`, followed by `0.510 m` and `0.759 m`. The threshold lies inside that
observed gap and is a prototype disagreement gate, not a production-calibrated
accuracy guarantee. Above the gate, preserve a paired-disagreement state and
do not fuse. With one valid camera, preserve a single-camera observation and
its provenance. With neither camera, emit unavailable without XYZ.

The 2026-08-03 rebuild makes the person tracking anchor consistently mean a
footpoint whenever the image actually supports one. A per-camera footpoint is
available only when:

1. the person mask does not touch the two-pixel bottom margin;
2. the lowest world-Z quintile contains at least `32` corrected-depth samples;
3. its median measured height lies within `-0.10 m` to `0.35 m` of the surveyed
   `world Z=0` floor.

When all gates pass, retain the measured low-Z XY and derive a vertical
projection to `Z=0`, explicitly labelled `person_footpoint`. Seven of twelve
per-camera views pass. A non-truncated view that fails the floor-height gate
retains only a measured `person_lower_body_surface`; a bottom-truncated view
retains only a measured `person_upper_body_surface`. Neither fallback is a
footpoint, and an upper-body observation is never projected to the floor.

Resolve synchronized camera pairs by semantic priority: footpoint, then lower-
body surface, then upper-body surface. Thus a valid footpoint in either view
wins over a cropped or elevated body surface in its mate, while the alternate
view remains preserved as secondary evidence. Only two anchors of the same
kind may enter the existing `0.35 m` disagreement gate or fusion. Mixed
semantics are never averaged. This uses the synchronized mate as a visibility
fallback without inventing an anatomical offset or adding a new model.

This decision selects anchor semantics and fusion eligibility only. It does
not select confidence weights, perform camera fusion, fill time, smooth a
trajectory, or alter raw DA3/model artifacts.

## D033 - Inspectable reliability weighting and honest cross-camera states

**Date:** 2026-08-02
**Amended:** 2026-08-03
**Status:** Active

Use policy `s04_cross_camera_observation_v1` after D032 selected-anchor and
disagreement classification. For each observed camera anchor, compute:

```text
reliability = sqrt(anchor_support_count)
              * retained_DA3_confidence_median
              / (1 + retained_depth_relative_MAD)
```

The square-root support term rewards broader evidence without allowing mask
area to dominate linearly. The median DA3 confidence retains D030's
candidate-relative model evidence. The `1 + relative MAD` denominator applies
a bounded dispersion penalty rather than unstable inverse-variance weighting.
The result is an inspectable prototype score, not a probability or calibrated
accuracy estimate.

Normalize the two scores into contribution weights only when D032 marks an
exact same-job pair eligible. Fuse its selected world anchors with the weighted
mean. When two sources exceed D032's `0.35 m` gate, retain both source anchors
and reliability scores for diagnosis but assign no contribution weights and
emit no combined XYZ. When exactly one camera is valid, pass its selected
anchor through with contribution weight `1.0`, label the result single-camera,
and do not call it fusion. When neither source is valid, emit unavailable
without XYZ.

On the retained evidence, only the frame `204` person pair is fused. Camera A
scores `114.7145` and Camera B `61.1843`, giving weights `0.652162` and
`0.347838`; the combined point is `(0.065857, 2.442960, 0.161857) m`. Twelve
job/target states are single-camera and three are paired disagreements without
XYZ. All `13` emitted coordinates remain inside the approximate room bounds,
and exact-pair source times differ by at most `5 ms`.

The 2026-08-03 corrected rebuild retains the same inspectable reliability
formula but applies it only after D032's semantic-priority selection. It
produces two fused and six single-camera person outputs, with no forced
disagreement or unavailable state in the eight retained pairs. Frames `204`
and `780` fuse comparable footpoints. At frame `708`, Camera B's valid
footpoint is selected while Camera A's elevated lower-body surface is retained
as non-contributing secondary evidence. At frame `408`, Camera A's lower-body
surface is preferred over Camera B's bottom-truncated upper-body surface.
Frames `330` and `462` retain measured upper-body fallbacks only. Therefore
the preferred person layer has outputs at all eight sampled times, but only
frames `204`, `666`, `708`, `780`, and `858` are footpoint observations;
fallback kinds must remain visibly and semantically separate in any later
trajectory.

All eight backpack records remain honest single-camera visible-cluster
observations. No temporal interpolation, stale carry-forward, mixed-semantic
fusion, or presentation smoothing is introduced.

This decision performs only same-job cross-camera combination. It does not
carry a position forward, interpolate time, fill a disagreement, smooth a
trajectory, change track identity, or alter raw/anchor artifacts.

## D034 - Conservative temporal presentation without inferred motion

**Date:** 2026-08-03
**Status:** Active

Use policy `s04_temporal_presentation_v1` over D033's verified corrected
observation layer and the authoritative S03 five-FPS capture-time grid. The
policy separates what was measured from what may remain briefly visible for
operator continuity; presentation state never upgrades stale or missing data
into a new spatial fact.

At an exact corrected D033 keyframe, emit `measured` with identical raw and
presentation XYZ, the original anchor kind, observation identity, source
cameras, and capture timestamp. Only this current measured state may update
zone membership or extend a measured trajectory.

After a measurement, its last presentation coordinate may remain visible for
at most `1.0 s` as `stale`. The stale record has no raw XYZ, retains the exact
source measurement timestamp and anchor kind, and may not update zones, event
spatial facts, or measured trajectories. The one-second horizon is a
conservative presentation interval shorter than the smallest retained
inter-measurement gap (`1.4 s`) and half the provisional two-second DA3
schedule. It is not a motion or accuracy guarantee. Once it expires, emit
`missing` with no raw or presentation XYZ.

D035 later replaces the sparse DA3 schedule with denser local observations as
close as `0.6 s`. Keep `1.0 s` as a maximum display-only age, not as a claim
that it is shorter than every dense interval; a current exact measurement
always supersedes the stale hold at its capture tick.

Do not interpolate or extrapolate motion, smooth positions, infer coordinates,
or convert anchor kinds. In particular, a stale upper- or lower-body surface
remains that surface and never becomes a footpoint. A missing S03 detection is
not automatically an occlusion: `occluded` requires separate explicit
evidence. The retained S03 timeline contains no such evidence, so this D034
artifact claims zero occluded states.

For trajectory visualization, connect only adjacent exact measurements of the
same target and anchor kind when their capture-time gap is at most `3.0 s`.
The retained adjacent gaps separate naturally: local gaps reach
`2.602 s`, followed by `4.202 s` and the known `6.803 s` backpack gap. The
three-second threshold lies inside that evidence gap. A segment stores only
its two measured endpoints; it creates no intermediate samples and may not use
stale positions. Thus the long backpack absence remains visibly disconnected,
and person upper/lower-body fallbacks cannot connect into the footpoint track.

These thresholds are bounded prototype presentation choices, not production
freshness or accuracy targets. This decision adds no model, dependency,
triangulation, motion prior, or non-baseline method and does not alter D025-
D033 outputs.

## D035 - Mask-aware dense dynamic DA3 keyframes

**Date:** 2026-08-03
**Status:** Active

Use the dense S04 action profile in `configs/s04_action_keyframes_dense.json`
as the preferred dynamic localization evidence. Retain the original eight
action-boundary pairs and add nine capture-ordered pairs where the S03
five-FPS timeline has a current backpack mask in at least one named camera and
a current person mask in at least one camera. Target roughly one-second local
spacing where evidence permits; do not add DA3-derived XYZ inside the accepted
frame `462-666` two-camera backpack absence interval.

Every selected pair remains subject to the complete D025 action-pair marker
gate. Dense sampling does not reuse a stale scale, invent a camera-specific
fallback, weaken D030's candidate-relative `p20` rule, change D032 anchor
semantics, or bypass D033 disagreement handling. Coverage checks must be
derived from exact source identities rather than hard-coded counts so the same
contracts can verify both the retained eight-pair baseline and the 17-pair
dense profile.

The accepted dense run contains `17` synchronized DA3 predictions, `44`
exact mask/depth surfaces and anchors, and `34` target-pair observations.
D025 accepted `5-6` marker observations per pair; shared scales range from
`1.093693-1.170350`, with a maximum within-pair relative marker deviation of
`1.259%`, below the `5%` gate. D033 provides `33` usable measurements. The
remaining frame `828` person pair is explicitly `disagreement`: two footpoints
are `0.377 m` apart, beyond the `0.35 m` gate, so neither view is arbitrarily
made authoritative.

D034 may connect adjacent exact person observations when their anchor kinds
match, including a separately styled upper-body-to-upper-body segment. It may
never connect an upper/lower-body fallback into a footpoint segment, convert
the fallback to floor contact, use a stale endpoint, or create intermediate
samples. This makes the verifier consistent with D034's original same-kind
rule; the sparse baseline happened to contain no adjacent body-surface pair.

Against the verified eight-pair baseline, the dense profile raises person
measured-plus-stale display coverage from `47/160` (`29.375%`) to `76/160`
(`47.5%`) ticks and backpack coverage from `47/160` to `80/160` (`50.0%`).
Measured segments increase from `8` to `23`; the `6.803 s` backpack hole
remains disconnected, and inferred positions remain zero. These figures show
better temporal evidence coverage and trajectory detail, not absolute XYZ
accuracy: the capture has no dynamic ground-truth trajectory.

This cadence is not constrained by the prototype MacBook's throughput. Future
live hardware may schedule DA3 more frequently, including on separate
accelerators, provided exact-frame joins, D025 gating, honest disagreement,
and missing-data rules remain unchanged and production throughput is measured
under its own deployment conditions.

## D036 - Measured-only pickup-carry-place interaction state

**Date:** 2026-08-03
**Status:** Superseded by D037

Use typed policy `s05_interaction_state_v1` as the deterministic spatial-state
authority for S05. The state set is `unknown`, `at_pickup`, `pickup`, `carry`,
`place`, and `occluded`. State records remain capture-time ordered and retain
the exact S04 person/backpack record identities and original anchor kinds.

Only paired current `measured` D034 records with `may_update_zone_membership`
may establish interaction spatial facts. Horizontal XY distance is used for
the two accepted circular zones and for person/backpack proximity because the
zone boundaries are horizontal and the person's footpoint/lower-body/
upper-body and backpack visible-cluster Z coordinates have deliberately
different semantics. The original anchor kind remains explicit; XY comparison
does not convert a body surface into a footpoint or object centre.

The initial bounded prototype policy uses the accepted `0.30 m` zone radii, a
`0.30 m` minimum pickup-centre departure distance, and a `1.0 m` maximum
person/backpack XY distance for pickup/carry evidence. These are inspectable
event-trigger thresholds, not calibrated probabilities or accuracy claims.
They must be reviewed against the retained synchronized recording before the
state-machine artifact is accepted.

An authoritative backpack measurement inside the pickup zone establishes
`at_pickup`. A later measured departure outside that zone becomes `pickup`
only when current measured person/backpack proximity also passes. A subsequent
measured outside-zone observation with current proximity becomes `carry`. A
current backpack measurement inside the drop-off zone becomes `place` only
after pickup has already been confirmed. An initial drop-off observation may
not invent a prior pickup.

Stale, missing, and inferred records cannot supply zone membership, movement,
or proximity. They produce `unknown`. `Occluded` requires a source record that
is already explicitly `occluded`; it has no coordinate. Unknown and occluded
ticks may preserve the last authoritative phase as non-spatial machine memory,
so a later current measurement can be evaluated in sequence, but the gap
itself remains unknown and no path or position is filled.

Qwen is outside this transition authority. Its future schema may describe a
candidate clip, but it cannot change state-machine coordinates, identities,
capture timestamps, zone membership, transition facts, or phase memory.

The retained dense D034 timeline and synchronized videos were subsequently
used to validate the initial thresholds without changing them. The accepted
run contains 160 interaction ticks: eight `at_pickup`, one authoritative
`pickup`, eight `place`, and 143 `unknown`. Pickup occurs at frame `462`
(`15.406667 s`) when the backpack is `0.338 m` from the pickup centre and the
current person/backpack XY distance is `0.822 m`; both pass their respective
`0.30 m` and `1.0 m` gates. Place first occurs at frame `666` (`22.210 s`)
inside the accepted drop-off zone after pickup has been confirmed.

There is no separate current measured `carry` state in this retained spatial
timeline. The synchronized video visibly shows the person carrying the
backpack at the interval midpoint, but the corresponding S04 backpack
localization gap remains `unknown`. This is accepted honest missing-data
behavior, not a reason to widen thresholds or fabricate a coordinate. The
bounded pickup and place windows, plus the explicitly non-authoritative carry
interval review frame, are suitable inputs for later semantic review.

## D037 - Orthogonal interaction phase, visibility, and localization

**Date:** 2026-08-03
**Status:** Active

Replace D036's single mutually exclusive state axis with three independent
axes under policies `s05_backpack_visibility_overlay_v1` and
`s05_semantic_interaction_v2`:

- interaction phase: `unknown`, `at_pickup`, `pickup`, `carry`, or `place`;
- reviewed optical visibility: `visible`, `partially_occluded`,
  `fully_occluded`, `out_of_view`, or `unknown`; and
- backpack localization availability: `measured`, `stale`, or `unavailable`.

The retained S03 detector timeline remains an immutable record of detector
presence and still performs no occlusion inference. A missing detector result
does not imply occlusion. A separate, versioned overlay may label visibility
only from affirmative evidence. For the reviewed frame `468-660` carry
interval, synchronized video establishes `partially_occluded`: the backpack is
repeatedly overlapped by the person's arm/body and remains partly visible in
some views despite unreliable bag detections. This label supplies no XYZ.

D034 may consume that explicit overlay. A current corrected measurement still
takes precedence; otherwise an explicitly partially/fully occluded backpack
tick becomes `occluded` with null raw and presentation XYZ, no anchor, no zone
authority, and no measured-trajectory authority. Existing S03/S04 artifacts
remain unchanged as diagnostic history; the occlusion-aware S04 result is a
new derived artifact.

After a measured pickup, S05 may retain a bounded semantic `carry` phase by
sequence continuity until a measured place, even while localization is
unavailable. The retained maximum unlocalized-carry horizon is `10.0 s`,
which contains the verified `6.803 s` pickup-to-place localization gap without
making a production timing or probability claim. Sequence continuity has
semantic authority only: it cannot supply XYZ, proximity, zones, movement,
track points, or current spatial authority.

Thus the accepted carry ticks simultaneously report `phase=carry`,
`visibility=partially_occluded`, and `localization=unavailable`, with null
backpack XYZ. Qwen may later describe or confirm a separate event review, but
cannot change coordinates, track identity, capture timestamps, zones, or the
provenance of these deterministic records.

## D038 - Bounded schema-only Qwen event review

**Date:** 2026-08-03
**Status:** Active

Use policy `s05_qwen_event_review_v1` to turn the three verified D037 event
candidates into independent Qwen review jobs. Each job uses the exact approved
`Qwen/Qwen3-VL-2B-Instruct` revision
`89644892e4d85e24eaac8bacfd4f463576704203`, deterministic decoding, a
`96`-token output bound, a `45 s` attempt timeout, and at most two attempts.

Each event receives six ordered synchronized frames: before, transition, and
after, with Camera A followed by Camera B at every time. Frame and video
references retain source indices, capture timestamps, SHA-256 hashes, capture
session, and synchronization-manifest provenance. These inputs are semantic
review evidence only and contain no spatial-write interface.

The recorded-MP4 queue has capacity three and uses throttle-and-drain rather
than dropping. Deduplication keys bind candidate identity, prompt hash, model
revision, and policy while excluding processing time. A duplicate pending,
in-flight, or completed logical event is coalesced. A retry is accepted only
after a failed, timed-out, invalid, cancelled, or dropped attempt, must be the
next sequential attempt, and may not exceed attempt two. Future live use may
select the same contract's explicit drop-oldest policy, but every drop remains
observable and retryable within the same bound.

Qwen must return one strict JSON object with `event_label`,
`matches_candidate`, qualitative `evidence_strength`, concise `summary`, up to
four `visible_evidence` items, `uncertainty`, and
`spatial_claims_present=false`. Labels are `pickup`, `carry`, `place`, or
`unknown`; evidence strength is qualitative, not a calibrated probability.
Invalid JSON/schema, timeout, or processor failure becomes a typed terminal
result containing an `unknown` interpretation. Raw invalid text may be
retained for diagnosis, but it cannot become an event fact.

Neither jobs nor results expose coordinate, track-identity, capture-time,
zone-membership, or spatial-authority write fields. Qwen annotations remain
separate from the deterministic D037 state records. This decision adds no new
model, dependency, or external method.

## D039 - Invalid Qwen prose remains diagnostic; bound inference images

**Date:** 2026-08-04
**Status:** Active

The first D038 execution establishes two operational boundaries. Actual Qwen
event images are downscaled without enlargement to a maximum dimension of
`768` pixels and retained byte-for-byte with their source-frame provenance.
This keeps the six-image request within practical local MPS cost while leaving
the synchronized source videos unchanged.

An invalid model response retains its raw text, token IDs/counts, input tensor
shapes, timing, and validation error, but its semantic interpretation remains
`unknown`. This applies even when the prose appears visually correct. In the
first execution, all pickup, carry, and place attempts reached the D038
`96`-token ceiling, emitted truncated fenced JSON, and repeated identically on
repair. Their visible descriptions are useful failure evidence but are not
event facts and cannot affect D037 phase, visibility, localization, or any
spatial authority.

The full-resolution diagnostic also confirmed that an `asyncio` thread timeout
does not preempt an in-progress MPS call. S05 therefore treats the whole Qwen
runner as the isolatable worker-process boundary; S06 must implement and test
supervisor-level termination/restart for a hard timeout. No concurrent heavy
MPS inference is authorized.

## D040 - Prefilled JSON and a separate carry review centre

**Date:** 2026-08-04
**Status:** Active

Supersede D038's model-facing v1 response format for accepted S05 execution
with `s05_qwen_event_review_v4`, while retaining v1-v3 artifacts as immutable
diagnostic history. The generation bound is `160` tokens. The model supplies
only `event_label`, qualitative `evidence_strength`, `summary`, one concise
`visible_evidence` string, and `uncertainty`. The application derives
`matches_candidate` from the expected event and returned label and fixes
`spatial_claims_present=false`; Qwen does not control either boundary.

V3/V4 include the exact assistant prefill `{"event_label":"` in the prompt
contract and its hash. Generation continues that assistant message, and the
adapter reconstructs the retained raw response from the immutable prefill plus
generated tokens. This is input-side structured generation, not post-hoc
repair: malformed or incomplete continuations still become `invalid_output`
and `unknown`. A single complete `json` code fence is the only permitted,
explicitly recorded normalization, though the accepted run required none.

Separate an event's authoritative transition identity from its semantic review
centre. Pickup and place use their transition frames. Carry preserves frame
`468` / `15.606667 s` as the sequence-continuity onset, but Qwen review is
centred at frame `567` / `18.900000 s`, the frame-aligned midpoint before the
measured place event. Its three paired review times use frames `507`, `567`,
and `627`. This prevents the pickup motion at carry onset from dominating the
sustained-carry review without moving the transition, inventing localization,
or granting Qwen spatial authority.

The accepted v4 execution returned direct schema-valid pickup, carry, and
place matches on attempt one, with no retry or normalization. Evidence strength
remains a qualitative model description, not probability or spatial truth.
