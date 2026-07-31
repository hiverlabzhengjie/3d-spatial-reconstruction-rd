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

## D025 - Marker-anchored scalar correction for S02 static depth

**Date:** 2026-07-31
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
intrinsics, camera poses, marker coordinates, frame identity, timestamps, or
S04 dynamic localization. Raw and corrected depth plus all scale observations
must remain separately inspectable.

The correction is isolated behind an explicit processing option, has synthetic
recovery and disagreement-rejection tests, and can be removed by disabling the
option. If the six ratios disagree beyond the limit, the run must stop instead
of applying a partial or camera-specific correction. This decision makes no
survey-grade accuracy claim; marker centres retain their stated
`+/-0.05 m` uncertainty.

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
