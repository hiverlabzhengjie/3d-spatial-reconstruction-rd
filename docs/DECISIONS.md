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
