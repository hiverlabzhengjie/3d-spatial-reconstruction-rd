# Project Status

**Last updated:** 2026-07-31
**Overall phase:** Implementation
**Current stage:** S02 - DA3 Static Room Geometry
**Stage state:** Complete - completion gate passed; stage-close provenance pending

## Completed

- Reviewed the original CCTV spatial-reconstruction proposal.
- Clarified that its main Digital Twin value is live spatial event context over
  a calibrated 3D scene.
- Refined the work into a focused, exploratory two-camera prototype.
- Selected DA3 Nested 1.1 as the geometry and depth backbone.
- Selected YOLOv8 segmentation/tracking, Qwen3-VL-2B semantic interpretation,
  OpenCV calibration, and Rerun presentation.
- Chosen a one-person, one-backpack pickup-carry-place demonstration.
- Reduced the roadmap to eight sequential stages.
- Created the initial project continuity pack.
- Prepared `docs/stages/S00_IMPLEMENTATION_BRIEF.md` without starting
  technical implementation.
- Completed S00 WP1:
  - installed `uv` and created the native arm64 Python 3.11 `.venv`;
  - added `pyproject.toml`, `uv.lock`, minimal package metadata, VS Code
    settings, and the initial licence record;
  - verified that the locked environment is reproducible and Apple MPS is
    available to a native process;
  - recorded exact results in `docs/stages/S00_WP1_ENVIRONMENT.md`.
- Relocated the unmodified DA3 checkout to `Depth-Anything-3-main/` and removed
  the obsolete `Glossary/` folder at the user's request.
- Adopted D015: useful supporting methods such as COLMAP may be introduced when
  their benefit and complexity are highlighted to the user and recorded.
- Completed S00 WP2:
  - added validated `configs/default.yaml` and project configuration loading;
  - added immutable typed contracts for frames, cameras, depth, detections,
    spatial observations, and model-run diagnostics;
  - enforced explicit camera transform names and honest missing, occluded, and
    stale states;
  - passed 25 automated tests plus Ruff and strict mypy checks;
  - recorded exact results in `docs/stages/S00_WP2_CONTRACTS.md`.
- Initialized project-level Git history under D016 with an initial
  research-plan checkpoint and vendor-source fingerprint.
- Created the public GitHub repository
  `hiverlabzhengjie/3d-spatial-reconstruction-rd`, configured it as `origin`,
  and adopted D017 for verified stage-close pushes.
- Adopted D018 to require action-frame depth for dynamic localization:
  empty-room depth cannot place a foreground person or backpack, two-camera
  overlap is redundancy rather than a stale-depth correction, and elevated
  downward-looking cameras reduce but do not eliminate occlusion and
  ray-depth ambiguity.
- Completed S00 WP3:
  - added explicit rigid-transform inversion and camera/world point utilities;
  - added OpenCV projection, metric depth back-projection, and
    depth/confidence filtering without placeholder XYZ values;
  - passed 49 project tests plus Ruff and strict mypy checks;
  - recorded exact results in `docs/stages/S00_WP3_GEOMETRY.md`.
- Completed S00 WP4:
  - added explicit device selection with honest MPS, CPU fallback, and failure
    provenance;
  - added synchronized cold/warm timing, current/peak process RSS, and
    supported MPS memory observations;
  - added a reusable no-model JSON diagnostic and verified it in restricted
    failure, explicit CPU-fallback, and native MPS modes;
  - passed 66 project tests plus Ruff and strict mypy checks;
  - recorded exact results in
    `docs/stages/S00_WP4_RUNTIME_DIAGNOSTICS.md`.
- Completed S00 WP5:
  - added a project-owned DA3 adapter that isolates MPS autocast and optional
    COLMAP import compatibility without modifying vendor source;
  - adopted D019 for the vendor's degenerate exact-two-view Umeyama
    post-alignment boundary;
  - ran the exact DA3 Nested 1.1 checkpoint on native MPS with supplied
    synthetic camera conditions at 336, 420, and 504;
  - selected 504 and a provisional two-second keyframe interval with
    substantial observed MPS memory headroom;
  - passed 83 project tests plus Ruff and strict mypy checks;
  - recorded exact evidence in `docs/stages/S00_WP5_DA3_MPS.md`.
- Completed S00 WP6:
  - added a project-owned, replaceable `yolov8n-seg.pt` adapter with
    source-sized mask normalization and strict output validation;
  - ran one cold and two warm predictions on the user-supplied representative
    image using actual Apple MPS with no CPU fallback;
  - preserved the source image byte-for-byte and retained normalized
    detections, raw arrays, an annotated preview, timings, and memory evidence;
  - observed two `bed` detections and honestly recorded zero `person` and
    `backpack` detections in the unstaged scene;
  - passed 92 project tests plus Ruff and strict mypy checks;
  - recorded exact evidence in `docs/stages/S00_WP6_YOLO_MPS.md`.
- Completed S00 WP7:
  - added an asynchronous, project-owned adapter for exactly
    `Qwen/Qwen3-VL-2B-Instruct` with no spatial-state write interface;
  - extracted four uniformly ordered frames from the unchanged vendor
    `robot_unitree.mp4` fixture and retained prompt/input provenance;
  - ran deterministic cold and warm generation on actual Apple MPS with no CPU
    fallback, producing identical non-empty 41-token descriptions;
  - recorded exact revision, prompt, raw token IDs, timings, memory, and the
    MP4 header-versus-decodable-frame discrepancy;
  - passed 106 project tests plus Ruff and strict mypy checks;
  - recorded exact evidence in `docs/stages/S00_WP7_QWEN_MPS.md`.
- Completed S00 WP8:
  - re-ran DA3, YOLO, and Qwen as three separate native Apple MPS processes;
  - confirmed all three exact model identities, outputs, timings, memory
    records, raw-input preservation, and absence of CPU fallback;
  - re-ran 106 automated tests plus Ruff, strict mypy, lockfile, environment,
    and whitespace checks;
  - restored and re-verified the exact 161-file DA3 vendor fingerprint after
    removing three transient Finder metadata files;
  - reviewed representative DA3, YOLO, and Qwen diagnostics;
  - confirmed the public stage-close scope excludes raw inputs, vendor source,
    model weights, caches, environments, and generated artifacts;
  - recorded exact evidence in `docs/stages/S00_WP8_VERIFICATION.md`.
- Adopted D020 after S00 close:
  - DA3, YOLO/ByteTrack, and Qwen will be independent timestamped workers
    operating at different rates;
  - the single-M1 default will serialize heavy MPS inference while allowing
    CPU pipeline work to continue;
  - deterministic offline batching and future freshness-oriented live
    scheduling will share bounded queues and the same provenance contracts;
  - implementation is assigned incrementally to S01 and S03-S07 without
    reopening S00.
- Began S01 physical-input preparation:
  - confirmed two labelled iPhone 16 Pro Max cameras using fixed
    1080p/30 FPS recording and their 13 mm-equivalent ultrawide lenses;
  - accepted the two stable temporary mounts for the prototype;
  - generated and verified the canonical A4 ChArUco target
    `s01-charuco-6x8-30mm-5x5-100-v1`;
  - printed, dimension-checked, and mounted the ChArUco target on rigid matte
    backing;
  - reviewed the two-camera floor-marker placement trial and accepted the
    corrected four-position layout;
  - confirmed both replacement trial clips decode at 1920x1080 and
    approximately 30 FPS;
  - generated and verified the canonical 180 mm floor-marker set
    `s01-floor-markers-40-43-180mm-5x5-100-v1`;
  - accepted manually measured floor-marker centres with stated
    `+/-0.05 m` accuracy: M40 `(0.00, 0.00, 0.00)`, M41
    `(1.23, 0.45, 0.00)`, M42 `(0.00, 2.20, 0.00)`, and M43
    `(1.10, 3.70, 0.00)` metres;
  - accepted the shared Camera A/B intrinsic-calibration capture at 1920x1080
    and approximately 30 FPS: 25 distinct steady candidates covered the image,
    and a provisional standard OpenCV calibration produced `0.280 px` RMS
    reprojection error;
  - adopted D021 to use the shared numeric intrinsic estimate for both matched
    iPhone 16 Pro Max cameras, subject to Camera B world-marker reprojection
    validation and an independent-calibration fallback;
  - accepted the raw fixed-world-pose calibration pair and preserved both
    recordings unchanged under the S01 capture session;
  - synchronized the derived Camera A/B world-pose clips from their start and
    end claps, correcting the measured `293 ppm` relative clock drift; the
    residual disagreement is `0.021 ms` at the start and `12.271 ms` at the
    end, below one 30 FPS frame;
  - confirmed all four floor-marker IDs in Camera A and complete IDs M40-M42
    in Camera B. Camera B's clipped M43 is accepted for pose solving because
    the three complete non-collinear markers provide twelve image corners;
  - estimated fixed camera poses from M40-M42 over deterministic samples of
    the synchronized recordings:
    - Camera A centre is approximately `(0.131, 3.999, 2.151) m`, with
      `1.527 px` aggregate reprojection RMS;
    - Camera B centre is approximately `(2.176, 3.670, 2.201) m`, with
      `1.481 px` aggregate reprojection RMS;
  - verified that both cameras' sampled-corner error, per-frame pose stability,
    height, downward optical axis, and floor-intersection checks pass;
  - accepted Camera B's shared intrinsic estimate under D021 because its
    world-marker reprojection is comparable to Camera A's;
  - adopted D022 to use common markers M40-M42 and retain M43 as an excluded
    failed diagnostic: Camera A's M43 observation disagrees with its recorded
    centre by `100.044 px` RMS and its coordinate was not rewritten from the
    imagery;
  - synchronized the empty-room recording pair with `0.667 ms` start and
    `0.583 ms` end clap residuals, equal 1,220-frame outputs, and corrected
    `12.389 ppm` relative drift;
  - reviewed the empty-room content and restricted candidate static imagery to
    synchronized time `22.0-38.0 s`: the operator is still visible during the
    setup portion at 10-18 seconds;
  - confirmed that the blue bed pickup circle, white floor drop-off circle,
    and M40-M42 are visible in both empty-room views;
  - found that direct fixed-pose carryover exceeds the existing `5 px`
    reprojection threshold (`10.032 px` p95 for Camera A and `6.853 px` for
    Camera B), although an empty-room refit remains internally strong at
    `1.294 px` and `1.137 px` RMS with only small centre/orientation changes;
  - adopted D023 to retain the original world-pose calibration as the physical
    reference while allowing versioned per-recording corrections from
    stationary M40-M42 under strict reprojection and reference-difference
    limits;
  - accepted empty-room pose version
    `s01_capture_20260729:empty_room:v1`:
    - Camera A centre `(0.129, 4.002, 2.160) m`, `1.403 px` marker RMS,
      `0.009 m` and `0.574 deg` from the reference;
    - Camera B centre `(2.183, 3.672, 2.206) m`, `1.146 px` marker RMS,
      `0.010 m` and `0.342 deg` from the reference;
  - verified that the corrected empty-room poses pass all marker, stability,
    frustum, and D023 reference-difference checks.
  - adopted D024 to estimate the two approximately `0.30 m` radius zones from
    the synchronized empty-room views, treating each visible blue/white rope
    as a thin boundary centreline rather than a filled colour region;
  - generated provisional video-estimated zone metadata using empty-room pose
    version `s01_capture_20260729:empty_room:v1`:
    - blue bed pickup-zone centre `(1.736, 2.815, 0.599) m`, with Camera A/B
      ring-boundary RMS of `8.845 px` and `4.632 px`;
    - white floor drop-off-zone centre `(0.338, 0.592, 0.000) m`, with Camera
      A/B ring-boundary RMS of `9.873 px` and `9.447 px`;
  - verified the declared room bounds, positive projections, the `10 px`
    per-camera boundary threshold, and `0.102 m` independent Camera A/B
    floor-centre disagreement against the `0.15 m` limit;
  - retained annotated Camera A/B overlays and labelled both results
    `video_estimated`;
  - received user visual/sanity validation on `2026-07-31` that both estimated
    positions and the approximately `0.60 m` bed-zone height are physically
    sensible, accepting the two zones as S01 metadata while preserving their
    non-surveyed provenance.
  - reviewed both raw pickup-carry-place takes and selected
    `action_take_01`:
    - it contains the complete intended action in both camera views;
    - the backpack begins in the blue pickup zone and ends at the white
      drop-off zone;
    - its start/end synchronization claps are stronger and less ambiguous than
      the retained backup take;
  - synchronized the preferred dynamic pair with one second of context around
    both clap anchors and corrected the measured `123.430 ppm` relative drift:
    - both derived outputs contain `1,047` decoded frames;
    - residual clap disagreement is `2.375 ms` at the start and `5.854 ms` at
      the end, below one 30 FPS frame;
    - the retained six-time pair preview is visually synchronized;
  - accepted action pose version
    `s01_capture_20260729:action_take_01:v1` under D023:
    - Camera A centre `(0.127, 3.991, 2.134) m`, `1.417 px` marker RMS,
      `0.019 m` and `0.834 deg` from the fixed reference;
    - Camera B centre `(2.177, 3.661, 2.199) m`, `1.275 px` marker RMS,
      `0.009 m` and `0.420 deg` from the fixed reference;
  - verified the action poses pass marker reprojection, sampled-error, pose
    stability, camera-frustum, and D023 reference-difference checks.
  - added immutable `FrameIdentity` and `SynchronizedFrameBundle` contracts
    whose deterministic IDs include capture session, camera/frame source,
    source and capture timestamps, source fingerprint, synchronization
    manifest, and capture-specific pose provenance;
  - added PyAV-backed file and RTSP frame-source implementations:
    - local files are content-hash checked before decode;
    - RTSP uses the same protocol and timestamp transform while persistent
      references exclude credentials and query values;
  - added deterministic earliest-unconsumed capture-time bundling with a
    half-frame (`1/60 s`) tolerance, explicit incomplete/missing-camera state,
    and rejection of duplicate/non-monotonic frames or mixed provenance;
  - verified real same-input replay twice for both accepted downstream pairs:
    - empty room: `1,220` complete bundles, zero missing, maximum
      inter-camera time difference `3.333 ms`;
    - preferred action: `1,047` complete bundles, zero missing, maximum
      inter-camera time difference `6.667 ms`;
    - both ordered bundle-ID digests reproduce exactly and reverse simulated
      worker completion restores authoritative capture order;
  - verified immutable real 1920x1080 BGR pixel delivery and persistent schema
    round trips;
  - added explicit automated failure tests for missing cameras, duplicate and
    non-monotonic frames, mixed synchronization provenance, tampered IDs,
    duplicate/unknown worker results, and credential-safe RTSP references;
  - recorded local capture notes plus conservative approximate room/zone
    metadata. The processing bounds are `(-0.5, -0.5, 0.0)` to
    `(3.0, 4.5, 3.0) m` and are not represented as surveyed walls/ceiling.
  - closed S01 with commit
    `9d2bb08778c3a6fe014c8e300ab511d9dafa6b4a`, pushed it to `origin/main`,
    and verified the remote branch resolved to that exact commit.
- Completed S02 static room geometry:
  - added deterministic accepted-window keyframe selection, confidence/depth/
    bounds filtering, marker-scale diagnostics, metric back-projection,
    deterministic voxel fusion, colored PLY output, and strict persistent
    summary contracts;
  - ran exact DA3 Nested 1.1 pose-conditioned two-camera inference on native
    Apple MPS at synchronized times `22.010`, `30.013`, and `37.983 s`, with
    no CPU fallback;
  - preserved raw depth/confidence and immutable source/model/calibration
    provenance for all three pairs;
  - adopted D025 after the raw calibrated run exposed a consistent `14-16%`
    marker-depth underestimate in both views, applying only a bounded shared
    scalar to derived S02 static geometry while leaving raw DA3 output
    unchanged;
  - accepted scales `1.164240`, `1.157371`, and `1.157654`, with `1.606%`
    maximum observation deviation against the `5%` rejection limit;
  - retained Camera A/B clouds with `30,239` and `22,332` points and a
    `45,919`-point fused static scene;
  - measured bidirectional `0.10 m` shared-surface overlap of `69.295%` and
    `86.638%`, with all accepted points finite and inside the declared bounds;
  - visually verified recognizable living-room geometry plus both calibrated
    cameras, markers, bounds, and zones in the accepted Rerun recording;
  - produced schema-validated run, Rerun, and verification evidence, including
    a retained Rerun viewer capture;
  - passed `141` automated tests, Ruff, strict mypy, lockfile/environment,
    Rerun-structure, artifact-hash, and whitespace checks;
  - recorded exact evidence in
    `docs/stages/S02_DA3_STATIC_GEOMETRY.md` and
    `docs/stages/S02_HANDOFF.md`.

## Current Blockers and Unknowns

- No S00 completion-gate blocker remains.
- The fixed-world-pose, empty-room, and preferred dynamic action pairs are
  synchronized. The backup action pair remains unchanged and available but is
  not selected for baseline processing.
- The two iPhone 16 Pro Max cameras, fixed 1080p/30 FPS recording setup,
  selected 13 mm-equivalent ultrawide lenses, and stable mounts are confirmed.
- The canonical A4 ChArUco board is physically ready.
- Floor-marker identities, size, axis orientation, and manually measured
  coordinates are defined with `+/-0.05 m` uncertainty; their placement and
  both camera mounts must remain unchanged through the remaining captures.
- The shared Camera A/B intrinsic-calibration capture and both fixed camera
  poses are accepted. Camera B passed its required marker-reprojection check.
- M43 is excluded from pose fitting under D022; its recorded centre should be
  remeasured if later geometry reveals a material world-alignment problem.
- The empty-room pair is synchronized and its D023 capture-specific pose
  version is accepted for the declared stable interval.
- Pickup and drop-off zone estimates are accepted as video-estimated metadata.
- The preferred dynamic pair and its D023 capture-specific pose correction are
  accepted.
- No S01 completion-gate blocker remains.
- No S02 completion-gate blocker remains.

## Available Software Inputs

- `Depth-Anything-3-main/`
- DA3 example images and video inside its vendor checkout
- Native arm64 Python 3.11 `.venv` resolved by `uv.lock`
- Installed S00 model libraries
- Exact DA3 Nested 1.1, YOLOv8n-seg, and Qwen3-VL 2B revisions cached locally
- Optional PyCOLMAP dependency locked but not installed in the main `.venv`
- MacBook Pro with M1 Max, 32-core GPU, and 64 GB unified memory

## Available or Planned Physical Inputs

- Two fixed phone cameras: confirmed
- Stable mounts/tripods: confirmed for the prototype
- ChArUco board: printed, dimension-checked, mounted, and ready
- Printed floor markers: printed, positioned, and visible in world-pose capture
- Tape measurements: marker centres recorded with stated `+/-0.05 m` accuracy
- Target backpack: planned
- Empty-room recording: captured, synchronized, and restricted to a stable
  `22.0-38.0 s` candidate interval
- Pickup-carry-place recording: preferred take synchronized and accepted;
  backup raw take retained unchanged
- Representative room-view image: supplied and used for WP6

## Exact Next Action

Complete and verify the S02 stage-close Git provenance, then stop. Do not begin
S03 person/backpack perception until the user explicitly requests it.
