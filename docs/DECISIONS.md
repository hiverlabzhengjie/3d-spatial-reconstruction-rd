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
