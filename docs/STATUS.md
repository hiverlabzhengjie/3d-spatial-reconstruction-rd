# Project Status

**Last updated:** 2026-08-05
**Overall phase:** Complete
**Current stage:** S07 - Final Capture, Refinement, and Reporting
**Stage state:** Complete with known limitations

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
  - accepted scales `1.164252`, `1.157365`, and `1.157682`, with `1.606%`
    maximum observation deviation against the `5%` rejection limit;
  - after user review, diagnosed the weak door, primary wall, tall cabinet,
    and table-top lamp as a confidence-filtering issue rather than an X/Y
    bound issue: expanding X/Y would have recovered only `21` finite samples;
  - adopted D026 after the user selected the `20th`-percentile, `Z >= 0`
    comparison, retaining the original room bounds and rejecting a
    below-floor `Z >= -0.1 m` sheet;
  - retained revised Camera A/B clouds with `43,978` and `38,874` points and a
    `71,613`-point fused static scene; the prior `40th`-percentile result is
    preserved as a baseline;
  - measured revised bidirectional `0.10 m` shared-surface overlap of `72.152%`
    and `85.000%`, with all accepted points finite and inside the unchanged
    declared bounds;
  - after a second user review, confirmed the door behind M40 had valid
    in-bounds depth but confidence below p20 in both cameras;
  - adopted D027 for the one-time static scene: p5 samples may supplement only
    the video-estimated door volume `(-0.35, -0.40, 0.00)` to
    `(0.90, -0.12, 2.10) m`; S03/S04 remain unaffected;
  - retained `18,930` Camera A and `9,426` Camera B supplemental samples,
    producing revised Camera A/B clouds of `52,006` and `43,561` points and an
    `81,709`-point fused scene;
  - verified `10,126` fused points in the door volume, including `3,719` in
    the Rerun visualization sample, with revised bidirectional overlap of
    `73.363%` and `86.297%`;
  - documented D027 as a reusable one-time static-reconstruction technique:
    preserve the global threshold, selectively lower confidence only inside a
    calibrated world-space feature region, and require bounded provenance and
    geometry/Rerun verification;
  - visually verified recognizable living-room geometry plus both calibrated
    cameras, markers, bounds, and zones in the accepted Rerun recording;
  - produced schema-validated run, Rerun, and verification evidence, including
    a retained Rerun viewer capture;
  - passed `143` automated tests, Ruff, strict mypy, lockfile/environment,
    Rerun-structure, artifact-hash, and whitespace checks;
  - recorded exact evidence in
    `docs/stages/S02_DA3_STATIC_GEOMETRY.md` and
    `docs/stages/S02_HANDOFF.md`;
  - closed S02 with commit
    `4084c34e9c1d26d6dae0294fa0321ec238824704`, pushed it to `origin/main`,
    and verified the remote branch resolved to that exact commit.
  - re-verified and published the user-approved D026 completeness revision with
    commit `e163b4e72c90ac798e84df264162b93541922a3c`; `origin/main` resolved to
    that exact revision before the final provenance-only update.
  - re-verified and published the D027 door-inclusive revision with commit
    `9226a85911cba0e032bdf76b5d32bb9828ff1997`; `origin/main` resolved to
    that exact revision before this final provenance-only update.
- Completed S03:
  - ran the exact `yolov8n-seg.pt` checkpoint on representative synchronized
    action frames and retained raw masks, boxes, classes, confidences, timings,
    frame identities, and annotated previews;
  - adopted D028 after dense evidence showed that guarded `backpack` plus
    `handbag` candidates are usable for the one physical demonstration bag,
    while `suitcase` is an excluded bed-region false positive;
  - added camera-local persistent ByteTrack support and explicit tentative
    detections without assigned track IDs;
  - adopted D029 to keep nominal perception at 5 FPS per camera, reject a
    continuous 10 FPS experiment on the single M1 Max, and treat the backpack
    as a representative rather than permanent application object class;
  - tracked the person through the representative sequence, including one
    Camera B track with `152/160` observations;
  - retained a `32`-observation Camera A stationary/pickup backpack track and
    an eight-observation Camera B placement track across preserved `handbag`
    and `backpack` vendor labels;
  - ran two independent capacity-eight bounded queues over `160` synchronized
    5 FPS bundles per camera, recording `152` explicit throttle events per
    queue, zero drops/failures, stable job IDs, and exact capture order;
  - derived `640` immutable person/backpack target-state records with nested
    confidence/vendor/track provenance, mask area, border visibility, and
    explicit observed, untracked, ambiguous, missing, or failed state;
  - recorded rather than filled the main two-camera backpack absence from
    approximately `17.2-22.0 s`; image-plane absence is not automatically
    labelled occlusion and no coordinates are invented;
  - passed the S03 completion gate without activating the bottle fallback or
    weakening the accepted 5 FPS policy;
  - passed `162` automated tests, Ruff, strict mypy, lockfile/environment, raw
    artifact, source-integrity, and whitespace checks;
  - recorded exact evidence in
    `docs/stages/S03_PERSON_BACKPACK_PERCEPTION.md` and
    `docs/stages/S03_HANDOFF.md`;
  - closed S03 with commit
    `6764aef05556151963b116e40436c8abbd80abd4`, pushed it to `origin/main`,
    and verified the remote branch resolved to that exact commit before the
    final provenance-only documentation update.
- Began S04 raw action-depth work:
  - added deterministic action-keyframe configuration and typed DA3 job/run
    contracts that bind exact synchronized frames to observed S03 masks;
  - selected eight capture-ordered action pairs from `6.803-28.612 s`, spanning
    stationary pickup-side evidence through placement while leaving the known
    two-camera backpack gap unfilled;
  - ran exact pose-conditioned DA3 Nested 1.1 on all eight pairs at resolution
    `504` using native MPS float16 with no CPU fallback;
  - preserved sixteen raw metric depth/confidence planes, exact source/model/
    calibration provenance, undistorted keyframes, and diagnostic previews;
  - explicitly applied none of D025's static scale, D026's static confidence
    policy, or D027's door supplement;
  - verified all raw arrays, hashes, identities, capture order, schema round
    trips, metric/unit-scale flags, and absence of S02 corrections;
  - visually confirmed that representative raw depth maps contain coherent
    dynamic person and bag silhouettes without yet claiming accepted XYZ;
  - passed `166` automated tests, Ruff, and strict mypy checks;
  - recorded exact evidence in
    `docs/stages/S04_DA3_DEPTH_LOCALIZATION_FUSION.md`.
- Completed the S04 exact source-mask to DA3-grid alignment action:
  - traced and reproduced DA3's unmodified `upper_bound_resize` geometry as
    `1920x1080 -> 504x284 -> 504x280`, with no crop or padding;
  - undistorted S03 binary masks with the accepted camera models and used
    nearest-neighbour interpolation for mask remapping and both resize steps;
  - independently reproduced all `16` processed RGB images within one channel
    value and processed intrinsics within `1.235706e-05`;
  - retained `20` observed person/backpack masks in eight binary `280x504`
    bundles with exact frame, bundle, DA3-job, perception-job, detection,
    track, target, transform, artifact, and hash provenance;
  - visually accepted the eight-phase contact sheet while preserving every
    source-unavailable observation as missing;
  - verified hashes, schemas, metadata, binary values, nearest-neighbour
    mapping, complete action-depth job coverage, and the absence of both XYZ
    localization and S02 corrections;
  - passed `169` automated tests, Ruff, strict mypy, lockfile/environment, and
    whitespace checks;
  - recorded exact evidence in
    `docs/stages/S04_DA3_DEPTH_LOCALIZATION_FUSION.md`.
- Completed S04 raw in-mask depth/confidence diagnostics and rule selection:
  - compared whole-mask, two-pixel eroded, adaptive connected-depth-cluster,
    and person lower-body candidates over all `20` verified action masks;
  - retained `72` typed diagnostic records plus per-mask RGB/depth/confidence,
    histogram, candidate-outline, and confidence-sweep artifacts;
  - rejected full-frame confidence as a dynamic-object validity basis because
    even its `20th` percentile retained zero samples for some valid person and
    backpack candidates;
  - adopted D030 and policy `s04_dynamic_visible_surface_v1`: candidate-relative
    confidence `p20`, finite positive depth, finite confidence, target-specific
    visible-surface candidates, median ray depth, explicit sample minimums, and
    unavailable-without-fallback behavior;
  - selected the lower `35%` mask extent and a `256`-sample minimum for person,
    and the largest adaptive connected depth cluster with a `128`-sample
    minimum for backpack;
  - verified minimum retained counts of `382` person and `203` backpack
    samples, with maximum filtered-median shifts of `2.91%` and `0.90%`;
  - preserved visible-surface semantics without calling the result a person
    centre, backpack centre, ground contact, anchor, fused position, or XYZ;
  - passed `175` automated tests, Ruff, strict mypy, lockfile/environment,
    artifact-integrity, visual-QA, and whitespace checks;
  - recorded D030 in `docs/DECISIONS.md` and exact evidence in
    `docs/stages/S04_DA3_DEPTH_LOCALIZATION_FUSION.md`.
- Completed S04 exact-frame raw per-camera visible-surface localization:
  - added typed exact mask/depth joins that require identical action-depth job,
    bundle, frame, camera, and timestamp identity with zero tolerance and no
    worker-completion-order association;
  - applied D030 independently to all `20` retained masks, back-projected every
    selected sample with processed intrinsics, and transformed it with the
    accepted explicit `T_world_from_camera`;
  - retained `12` person and `8` backpack sample clouds plus component-wise
    camera-frame median aggregates, exact upstream hashes, distributions,
    transforms, and diagnostic images;
  - adopted D031 to keep these visible-surface clouds and robust aggregates as
    raw per-camera measurements, separate from later target anchors and fusion;
  - verified complete regeneration from exact current-frame sources, maximum
    reprojection error `2.842171e-14 px`, maximum camera/world round-trip error
    `3.477764e-15 m`, and maximum returned-pose error `1.395500e-07`;
  - confirmed all `20` aggregates and all retained samples lie inside the
    approximate room bounds without clipping or snapping;
  - visually confirmed plausible pickup-to-drop-off motion while preserving
    substantial view-dependent raw person-surface disagreement
    (`0.384-0.679 m` over four paired views) for the next anchor/fusion action;
  - passed `184` automated tests, Ruff, strict mypy, lockfile/environment,
    artifact-integrity, schema, visual-QA, and whitespace checks;
  - recorded D031 in `docs/DECISIONS.md` and exact evidence in
    `docs/stages/S04_DA3_DEPTH_LOCALIZATION_FUSION.md`.
- Completed S04 target-anchor candidate evaluation and pre-fusion policy:
  - re-verified the D031 prerequisite directly from all `20` raw source clouds
    before deriving any anchor;
  - compared `104` typed candidates: six person methods across `12`
    observations and four backpack methods across `8` observations;
  - selected the measured world-frame median of the lowest-Z sample quintile
    as the person tracking anchor, retaining all `12` observations with at
    least `77` supporting samples while explicitly not calling it ground
    contact or an anatomical centre;
  - evaluated a lightweight bottom-pixel ray intersection against the existing
    surveyed `world Z=0` floor and rejected it because one result left the room
    bounds and paired disagreement reached `1.110 m`;
  - retained a separate person ground-contact candidate only when the measured
    lower-quintile support is at most `0.35 m` above the floor: `6` observations
    passed and `6` remain unavailable without placeholder XYZ;
  - selected the world-frame component-wise median of the visible backpack
    cluster, reducing the stationary pickup maximum separation from `0.136 m`
    to `0.126 m` and the placed-pair separation from `0.056 m` to `0.045 m`;
  - persisted all `32` action-job/camera/target anchor states: `20` observed and
    `12` source-unavailable with no XYZ;
  - adopted D032's prototype `0.35 m` cross-camera eligibility gate. One person
    pair at `0.231 m` is eligible for later fusion; three pairs at
    `0.474-0.759 m` are disagreement states, and all `8` backpack comparisons
    remain honest single-camera states;
  - performed no camera fusion, temporal filling, or presentation smoothing;
  - passed `195` automated tests, Ruff, strict mypy, lockfile/environment,
    artifact regeneration, schema, visual-QA, and whitespace checks;
  - recorded D032 in `docs/DECISIONS.md` and exact evidence in
    `docs/stages/S04_DA3_DEPTH_LOCALIZATION_FUSION.md`.
- Completed the S04 D032-gated cross-camera observation layer:
  - re-verified all `104` D032 candidates, `32` selected camera/target states,
    and `16` pre-fusion comparisons before combining any anchors;
  - adopted D033 policy `s04_cross_camera_observation_v1` with the transparent
    reliability score `sqrt(anchor support) * median retained DA3 confidence /
    (1 + retained depth relative MAD)`;
  - normalized reliability scores into contribution weights only for a D032
    paired-eligible case; disagreement cases retain their scores for diagnosis
    but receive no weights and no combined XYZ;
  - produced `16` exact-job target observations: one genuinely fused person
    observation, `12` explicit single-camera passthroughs, and three paired
    disagreements without XYZ;
  - fused frame `204` person anchors with Camera A/B weights `0.652162` and
    `0.347838`, yielding world XYZ `(0.065857, 2.442960, 0.161857) m` from a
    `0.230854 m` eligible separation;
  - preserved all eight backpack observations as single-camera rather than
    implying two-camera fusion where no paired mask exists;
  - verified capture order, a maximum source-time difference of `5 ms`, exact
    source regeneration, stable schemas/hashes, and all `13` emitted XYZ
    values inside room bounds;
  - performed no stale carry-forward, temporal interpolation, or presentation
    smoothing;
  - passed `207` automated tests, Ruff, strict mypy, lockfile/environment,
    artifact regeneration, visual-QA, and whitespace checks;
  - recorded D033 in `docs/DECISIONS.md` and exact evidence in
    `docs/stages/S04_DA3_DEPTH_LOCALIZATION_FUSION.md`.
- Formalized and verified D025 action-pair marker scaling before continuing
  S04 temporal work:
  - amended D025 with an isolated dynamic profile that preserves raw DA3 depth
    and confidence and writes only separate corrected-depth artifacts;
  - detected M40-M42 on all exact undistorted action keyframes and sampled the
    protected inner `60%` of each known marker square using calibrated
    per-pixel floor-ray camera-Z rather than a single centre patch;
  - required one shared scale per synchronized pair, both cameras, at least
    two accepted markers per camera, at least five observations total, at
    least `16` valid samples per marker, `5 px` reprojection agreement, and no
    more than `5%` scale deviation, with no partial or camera-specific fallback;
  - accepted all eight action pairs from `44` marker observations; scales span
    `1.093553-1.153414` with median `1.133675`, and the worst marker deviation
    is `1.2202%`;
  - visually accepted the full eight-pair marker overlay and independently
    verified raw prediction hashes, unchanged confidence, exact
    `corrected = raw * shared scale`, capture order, and artifact hashes;
  - retained the existing D030-D033 outputs as an unscaled diagnostic baseline;
    they are not retroactively relabelled as corrected or revalidated.
- Rebuilt D030-D033 from D025 corrected action-pair depth with margin-aware
  person validity and consistent footpoint semantics:
  - retained candidate-relative `p20`, exact mask/depth/frame joins, unchanged
    raw DA3 depth/confidence, and the existing inspectable D033 reliability
    formula under policy `s04_corrected_margin_aware_tracking_v1`;
  - recorded a two-pixel processed-image margin assessment for every person
    mask and classified the three bottom-truncated views at frames `330`, `408`
    Camera B, and `462` as upper-body-only evidence rather than treating their
    image bottoms as feet;
  - regenerated all `20` corrected per-camera surfaces, `20` anchors, and `16`
    exact-pair observations directly from the verified D025 artifacts;
  - derived seven per-camera footpoints only from non-truncated masks whose
    lowest-Z quintile has at least `32` samples and median height within
    `-0.10-0.35 m`, then vertically projected only that measured XY to the
    surveyed `Z=0` floor;
  - retained elevated lower-body and bottom-truncated upper-body measurements
    as explicit fallback surface kinds, never as inferred feet;
  - preferred a synchronized mate's valid footpoint over a weaker body-surface
    view, resolving frame `708` from Camera B while preserving Camera A as
    secondary evidence; frame `408` uses Camera A's lower-body surface rather
    than Camera B's cropped torso;
  - allowed fusion only between matching anchor kinds within `0.35 m`, yielding
    two fused and six single-camera person outputs with zero forced mixed-
    semantic fusion, disagreement holes, or unavailable retained pairs;
  - preserved honest semantics: five retained frame pairs report person
    footpoints, frame `408` reports a lower-body surface, and frames `330` and
    `462` report upper-body surfaces that later presentation must style and
    connect separately;
  - independently regenerated every surface, anchor, and pair with maximum
    reprojection error `2.842171e-14 px` and maximum world/camera round-trip
    error `3.477764e-15 m`, and visually accepted both diagnostic views;
  - passed `223` automated tests, Ruff, strict mypy across `50` files,
    lockfile/environment, artifact regeneration, visual-QA, and whitespace
    checks;
  - performed no model inference, temporal filling, stale carry-forward,
    presentation smoothing, raw-capture modification, or vendor modification.
- Defined and verified the D034 temporal presentation policy:
  - added typed policy `s04_temporal_presentation_v1` over the corrected D033
    layer and authoritative S03 five-FPS capture-time grid;
  - kept exact corrected observations as the only `measured` states with raw
    XYZ and spatial authority;
  - allowed a presentation-only stale hold for at most `1.0 s`, preserving the
    source timestamp and anchor kind while forbidding raw XYZ, zone updates,
    event spatial updates, or trajectory extension;
  - emitted missing-without-XYZ after the stale horizon and performed no
    interpolation, extrapolation, smoothing, or inferred positioning;
  - required affirmative occlusion evidence and therefore produced zero
    claimed occlusions from S03's non-inferential missing detections;
  - built `320` capture-ordered states: `16` measured, `78` stale, `226`
    missing, zero inferred, and zero occluded;
  - selected a `3.0 s` same-kind measured-segment gate inside the observed gap
    between local intervals up to `2.602 s` and the next `4.202 s` interval;
  - produced three person footpoint and five backpack visible-cluster segments
    using exact adjacent endpoints only, with no body-surface, mixed-semantic,
    stale, or interpolated segment;
  - preserved the `6.803 s` frame `462-666` backpack gap as disconnected;
  - independently regenerated every record and segment, verified all hashes,
    schemas, state authority, CSV coverage, and visually accepted the timeline
    and world diagnostics;
  - passed `235` automated tests, Ruff, strict mypy across `53` files,
    lockfile/environment, artifact regeneration, visual-QA, and whitespace
    checks.
- Refined S04 with D035's mask-aware dense dynamic DA3 profile:
  - retained the original eight action boundaries and added nine observed-mask
    pairs, producing `17` synchronized DA3 keyframes at approximately
    `0.6-1.8 s` local spacing where evidence permits;
  - kept the frame `462-666` two-camera backpack absence interval unsampled
    and unfilled;
  - passed D025 for all 17 pairs using `5-6` current marker observations per
    pair, shared scales of `1.093693-1.170350`, and maximum marker deviation
    `1.259%` against the `5%` gate;
  - regenerated `44` exact D030 surfaces, `44` D032 anchors, and `34` D033
    target-pair records from current corrected depth and masks;
  - accepted `33` pair measurements and preserved frame `828` person as one
    explicit `0.377 m` disagreement beyond the `0.35 m` gate;
  - produced `33` measured, `123` stale, and `164` missing D034 states with
    zero inferred or claimed-occluded states;
  - increased person measured-plus-stale display coverage from `47/160` to
    `76/160` ticks and backpack coverage from `47/160` to `80/160`;
  - increased exact same-kind measured segments from `8` to `23`, while the
    known `6.803 s` backpack gap remained disconnected;
  - retained the sparse run as the verified comparison baseline and recorded
    that coverage improvement is not an absolute XYZ-accuracy measurement;
  - independently regenerated the dense raw, D025, alignment, D030-D033,
    D034, and sparse/dense comparison artifacts; passed `235` automated tests,
    Ruff, strict mypy across `54` source/script files, lockfile and dry-run
    environment checks, visual QA, and whitespace checks.
- Closed the S04 completion gate over the preferred dense D025-D035 chain:
  - regenerated and visually accepted the final measured-segment preview and
    temporal-state timeline with all 17 dense keyframe pairs;
  - independently verified every retained artifact and all missing, stale,
    disagreement, exact-join, coordinate, and static-depth rejection rules;
  - confirmed the backpack's measured endpoints move `2.545 m`, from within
    `0.128 m` XY of the pickup-zone centre to within `0.166 m` of the
    drop-off-zone centre;
  - passed all seven S04 roadmap gates without interpolation, fabricated XYZ,
    stale/static-depth reuse, mixed-semantic fusion, or accuracy overclaim;
  - re-ran `235` tests, Ruff, strict mypy, lockfile/environment, artifact,
    visual-QA, and whitespace checks successfully.
  - created stage-close commit
    `dd8a29a4111c7282351adf6a5926d1b699a18b7f`, added annotated tag
    `stage-04-da3-localization`, pushed both to the public remote, and verified
    that the close commit is on remote `main` and the dereferenced tag resolves
    exactly to it; later `main` updates are provenance-only documentation.

## Completed S05

- Began S05 after re-verifying the completed S04 handoff, preferred dense
  D033/D034 artifacts, accepted pickup/drop-off zones, exact Qwen model cache,
  and clean `main == origin/main` repository state.
- Adopted D036 and implemented typed policy `s05_interaction_state_v1`:
  - the deterministic states are `unknown`, `at_pickup`, `pickup`, `carry`,
    `place`, and explicitly evidenced `occluded`;
  - only paired current `measured` S04 records may establish zone membership,
    pickup departure, or person/backpack proximity;
  - stale, missing, inferred, and occluded records carry no transition XYZ;
  - unknown/occluded ticks may retain the last authoritative phase as memory
    but do not claim a current spatial state; and
  - Qwen is structurally prohibited from changing interaction spatial state.
- Added focused tests for pickup-carry-place transitions, an unknown backpack
  gap, stale-coordinate rejection, explicit occlusion, unproven sequences,
  person anchor-kind preservation, capture ordering, persistent schema round
  trips, and Qwen isolation; all `243` project tests, Ruff, strict mypy,
  lockfile/environment, and whitespace checks pass.
- Applied D036 to all `160` paired ticks from the verified dense D034 timeline
  and independently regenerated the result:
  - eight ticks are `at_pickup`, one is authoritative `pickup`, eight are
    `place`, and 143 are honestly `unknown`;
  - pickup is detected at frame `462` / `15.406667 s` and first placement at
    frame `666` / `22.210 s`;
  - one deduplicated pickup candidate and one deduplicated place candidate
    were produced; repeated measured placement ticks do not create duplicates;
  - no tick contains invented XYZ, Qwen-controlled spatial state, unsupported
    occlusion, or an unknown state with spatial authority; and
  - the known backpack gap remains unfilled, so no separate measured carry
    state is claimed.
- Visually accepted both synchronized cameras across the start, transition,
  and end of each four-second candidate window. The pickup window shows the
  bag leaving the blue bed zone, the midpoint shows visible carrying while the
  spatial state remains unknown, and the place window shows approach and
  placement at the white floor zone. The initial thresholds required no
  tuning.
- Reviewed the S03-S05 occlusion/carry gap and superseded D036 with D037:
  - S03 remains an immutable detector-presence timeline; its derivation now
    explicitly declares that missing detections require separate visibility
    evidence and never imply occlusion or XYZ;
  - a versioned visibility overlay covers all 160 ticks and records 47
    detector-supported `visible`, 33 review-backed `partially_occluded`, and
    80 `unknown` ticks;
  - the 33 partial-occlusion records span frames `468-660`, retain each
    camera's original detector state, and supply zero coordinates;
  - an occlusion-aware D034 run independently regenerates 33 backpack
    `occluded` records while preserving all 33 measured records, all 23 exact
    measured segments, and the disconnected `6.803 s` backpack gap;
  - S05 v2 separates interaction phase, visibility, and localization rather
    than placing carry and occlusion in one enum;
  - frames `468-660` now coexist as `carry`, `partially_occluded`, and
    `unavailable`, with null backpack XYZ and zero spatial authority; and
  - the event candidates are pickup at frame `462`, carry at frame `468`, and
    place at frame `666`.
- Adopted D038 and verified the pre-inference Qwen event-worker layer:
  - created one stable pickup, carry, and place job from the verified D037
    candidates, each with six ordered before/transition/after frames across
    Camera A and Camera B;
  - bound every frame to synchronized source-video hashes, frame indices,
    capture timestamps, capture session, and synchronization manifest;
  - fixed the exact passed MPS model revision, deterministic decoding,
    `96`-token bound, `45 s` timeout, and at most one repair attempt;
  - implemented a capacity-three offline throttle queue with logical-event
    deduplication, bounded sequential retries, explicit cancellation and
    future-live drop accounting;
  - implemented strict event JSON plus typed completed, failed, timed-out, and
    invalid-output results, with safe `unknown` fallback for every
    non-completed outcome;
  - verified duplicate coalescing, throttling, drop-oldest accounting,
    capture ordering, timeout/failure isolation, invalid-output repair, schema
    tamper detection, and rejection of spatial output fields; and
  - generated and independently regenerated the three-job plan with zero
    inference results and zero forbidden spatial job fields; and
  - passed 256 project tests, Ruff, strict mypy across 68 source/script files,
    lock/environment checks, artifact regeneration, and whitespace checks.
- Executed the three verified Qwen jobs under D039:
  - bounded the 18 actual inference frames to `768` pixels maximum dimension
    after an interrupted full-resolution diagnostic exposed non-preemptive
    thread-timeout behavior;
  - loaded the exact approved revision once on MPS in float16 and processed
    pickup, carry, and place serially with one bounded repair each;
  - retained six deterministic 96-token raw responses with token/tensor
    diagnostics; each was truncated/malformed JSON and therefore became a
    typed `invalid_output` with final `unknown` interpretation;
  - measured `2219` input tokens and `8.23-8.72 s` per bounded request, below
    the `45 s` attempt timeout;
  - observed that the raw diagnostic prose visibly names pickup, carry, and
    place, including the previously disputed carry interval, but promoted none
    of it to a schema-valid event fact or spatial state; and
  - independently verified model/runtime identity, 18 frame artifacts, all
    attempts and final results, raw-token retention, contact-sheet decoding,
    hashes, and the absence of a spatial write interface; and
  - passed all 256 project tests, Ruff, strict mypy across 86 source/script
    files, lock/environment checks, and whitespace checks.
- Completed the D040 Qwen response and evidence-window correction:
  - retained v2 as an invalid-output experiment after its compact `160`-token
    prompt still produced deterministic unquoted, verbose objects;
  - added an identity-bound assistant JSON prefill in v3, producing valid
    pickup and place responses while honestly exposing that the carry-onset
    window still looked like pickup;
  - preserved frame `468` as the carry transition but centred v4 semantic
    evidence at frame `567` / `18.900 s`, with both cameras at frames `507`,
    `567`, and `627` across the sustained carry interval;
  - obtained schema-valid first-attempt pickup, carry, and place matches with
    qualitative strong evidence, `48-52` output tokens, `4.76-5.13 s`
    processing, zero retry, and zero response normalization;
  - kept `matches_candidate` and the no-spatial-claims boundary
    application-owned rather than model-controlled;
  - visually accepted the corrected contact sheet and independently verified
    exact model/runtime identity, source/review frames, all artifact hashes,
    three direct matches, raw diagnostics, and no spatial write interface; and
  - preserved all v1-v3 artifacts as backward-compatible diagnostic history;
    and
  - passed all 259 project tests, Ruff, strict mypy across 86 source/script
    files, lock/environment checks, and whitespace checks.
- Closed the S05 completion gate without weakening any criterion:
  - freshly regenerated the visibility overlay, occlusion-aware presentation,
    orthogonal interaction timeline, Qwen v4 plan, and Qwen execution from
    retained source hashes;
  - confirmed a sensible pickup-carry-place sequence, bounded Qwen
    failure/unknown behavior, schema-valid accepted output, zero Qwen spatial
    writes, and occlusion without invented locations; and
  - recorded the complete evidence, limitations, reproduction commands, and
    S06 prerequisites in `docs/stages/S05_HANDOFF.md`.
- Published the S05 stage close:
  - created commit `6cdcd12de055f0ffe357d1fd2e8fdcd6c077faab` with message
    `stage(S05): complete interaction state and Qwen events`;
  - created annotated tag `stage-05-interaction-events`;
  - pushed the commit to public `origin/main` and pushed the tag; and
  - verified remote `main` and the dereferenced tag both resolved exactly to
    the stage-close commit before the provenance-only documentation update.

## Completed S06

- Re-verified the S06 entry prerequisites before changing the project:
  - local `main` was clean and matched `origin/main` at `00b250b`;
  - the accepted S02 static scene, S04 occlusion-aware presentation, S05
    semantic interaction, Qwen v4 plan, and Qwen v4 execution passed fresh
    independent verification;
  - all `259` pre-S06 tests, Ruff, strict mypy, lockfile, environment, and
    whitespace checks passed; and
  - no new physical capture or calibration was required.
- Completed S06 Work Package 1:
  - added policy `s06_integrated_offline_orchestration_v1` with file sources,
    capture time as the sole authoritative Rerun timeline, bounded queue
    capacities, throttle-and-drain offline overload behavior, and exactly one
    heavy-MPS permit;
  - bound both synchronized action videos plus seven accepted S01-S05 inputs
    by exact content hash in one stable orchestration manifest;
  - added a subprocess supervisor with hard timeout, terminate/kill fallback,
    bounded restart, explicit lifecycle outcomes, and degraded-state records;
  - verified that a non-preemptible synthetic Qwen process is killed at the
    hard boundary while an independent geometry worker still completes;
  - independently regenerated manifest
    `ecb59e8b7142db940f5935817fa6323c5c34735a98cb37fc3f3f73a49ffd09c0`;
  - passed all `264` project tests, Ruff, strict mypy across `91`
    source/script files, lockfile/environment, artifact verification, and
    whitespace checks; and
  - performed no model inference, Rerun assembly, RTSP test, raw-capture
    modification, or vendor modification in this work package.
- Refined the WP1 manifest before Rerun assembly:
  - promoted the action-specific camera calibration and scene/zone metadata to
    first-class hash-bound inputs rather than relying on transitive references;
  - retained the original seven-artifact manifest as diagnostic history; and
  - accepted the nine-artifact v2 manifest
    `87a1c225049f167d6b5f87632d953d2d242ac7479eb20a33bfc24393f359a8f7`.
- Completed S06 Work Package 2:
  - assembled both complete synchronized H.264 camera videos into one Rerun
    recording with `1,047` frame references per camera on `capture_time`;
  - logged `328` labelled YOLO/ByteTrack boxes and `298` combined person/
    backpack segmentation overlays at the retained five-FPS evidence times;
  - logged the accepted static point cloud, action-camera frustums, pickup and
    drop-off zones, `320` presentation states, and `23` disconnected exact
    measured trajectory segments;
  - visually distinguished measured person footpoint, lower-body, upper-body,
    backpack, stale, missing, and occluded semantics without adding XYZ;
  - logged all `160` orthogonal interaction records and separate pickup,
    carry, and place transition/Qwen review events; carry remains transition
    frame `468` and review frame `567`;
  - installed Homebrew FFmpeg `8.1.2_1`, which Rerun requires to decode the
    retained H.264 assets in the viewer;
  - recorded the native viewer's unrelated `80,000 px` window-surface failure
    and completed visual QA through the equivalent localhost Rerun web viewer;
  - removed cluttering static trajectory labels after the first visual review
    while preserving every disconnected colored segment; and
  - accepted `digital_twin_stage06_v2.rrd`, `51,928,057` bytes with SHA-256
    `0ec24e52ee4ab592bb02d9c2c30bbca5f455129466421f8b2ee2bb612f8d1fe9`;
  - passed `266` project tests, Ruff, strict mypy across `94` source files,
    lock/environment consistency, artifact verification, visual QA, and
    whitespace checks.
- Completed S06 Work Package 3:
  - added a typed deterministic virtual-time replay bound to accepted manifest
    `87a1c225049f167d6b5f87632d953d2d242ac7479eb20a33bfc24393f359a8f7`;
  - exercised `34` immutable logical jobs and `35` attempts across five
    bounded queues, with `112` explicit accept, throttle, drain, retry, pop,
    terminal, and duplicate-coalescing events;
  - recorded nine throttle-and-drain submissions, one duplicate coalescing,
    zero drops, two explicit final degraded results, and one successful Qwen
    restart without duplicate result persistence;
  - proved identical capture-ordered output digest
    `746c0f1175982dbd61a13514c7c4398f3de8ff65960ab39335692ea03a5ead9b`
    under two deliberately different worker completion schedules;
  - proved all heavy-MPS attempt intervals non-overlapping with maximum
    occupancy one across nine intervals and geometry completion independent of
    the Qwen retry;
  - completed a graceful-shutdown exercise with one in-flight completion,
    four explicit pending cancellations, zero final backlog/in-flight work,
    and the accelerator permit released;
  - independently regenerated the accepted v2 replay report exactly from the
    hash-bound source manifest; and
  - passed `269` project tests, Ruff, strict mypy across `97` source files,
    lock/environment consistency, artifact verification, and whitespace
    checks.
- Completed S06 Work Package 4:
  - adopted D041 and installed MediaMTX `1.19.3` as an isolated localhost-only
    RTSP test fixture, with no new Python runtime or production dependency;
  - added bounded reconnect policy `s06_rtsp_bounded_reconnect_v1`, explicit
    per-attempt outcomes, bounded exhaustion, and credential-safe persistent
    stream references;
  - preserved contiguous global frame indices, unique immutable frame IDs, and
    strictly increasing capture timestamps across reconnect, including an
    explicit `1.747282 s` observed outage gap rather than silent compression;
  - restricted the accepted fixture to RTSP/TCP on `127.0.0.1:18554` and
    disabled every unrelated MediaMTX protocol/service listener;
  - decoded `45` 640x360 frames: `27` before the deliberate FFmpeg publisher
    outage and `18` after recovery, across four of the allowed eight connection
    attempts (`stream_ended`, two failed opens, then `target_reached`);
  - validated the final RTSP identity directly as a standard perception-worker
    job without model inference;
  - independently verified the accepted `rtsp_smoke_v4_20260805` hashes,
    process logs, source-manifest binding, reconnect history, identities,
    timestamps, and worker compatibility; and
  - passed `272` project tests, Ruff, strict mypy across `100` source files,
    lock/environment consistency, artifact verification, and whitespace
    checks.
- Completed S06 Work Package 5 and the stage completion audit:
  - added dedicated typed JSONL exports for all `320` accepted presentation
    records, all `23` exact measured trajectory segments, and the three
    pickup-carry-place events;
  - preserved the carry transition at frame `468` separately from Qwen review
    frame `567`, with phase/spatial authority and Qwen provenance explicit;
  - independently regenerated every export from the hash-bound S04/S05/Qwen
    sources with zero non-measured raw XYZ, unavailable-state presentation
    XYZ, interpolated/stale segments, or Qwen spatial writes;
  - reran all five accepted S06 verifiers into
    `artifacts/s06/stage_close_audit_20260805/`;
  - passed all seven roadmap completion criteria in one reproducible gate
    audit without weakening a criterion; and
  - passed `273` project tests, Ruff, strict mypy across `104` source/script
    files, lock/environment consistency, artifact verification, and
    whitespace checks.

## Completed S07

- The user selected the accepted `action_take_01` as the final demonstration
  recording; D042 records that no recapture or recalibration is required for
  this retained pair.
- Re-verified the S07 entry prerequisites before implementation:
  - local `main` was clean and matched `origin/main` at `6e4bef0`;
  - all five S06 verification layers passed fresh independent checks;
  - the unified seven-criterion S06 gate passed without weakening; and
  - the accepted Rerun retained SHA-256
    `0ec24e52ee4ab592bb02d9c2c30bbca5f455129466421f8b2ee2bb612f8d1fe9`.
- Completed S07 Work Package 1:
  - added typed final-recording selection, final-artifact, and reproducible-run
    policy contracts;
  - bound both 1,047-frame synchronized `action_take_01` videos and seven
    accepted S06 evidence layers by exact content hash;
  - accepted final-run manifest
    `9bb49beb13262f8108f4b37dd0974de6cc23f7fa456214e0c14e6c540a31ba08`;
  - independently verified the recording selection, source hashes, S06 gate,
    Rerun hash, and all missing-data/non-spatial policy boundaries;
  - performed no model inference, Rerun regeneration, demo-video generation,
    recapture, recalibration, or upstream-artifact modification; and
  - passed all 276 project tests, Ruff, strict mypy across 108 source/script
    files, lockfile/environment, artifact verification, and whitespace checks.
- Completed S07 Work Package 2:
  - added typed measured-step and final-assembly contracts with exact execution
    ordering and recomputed timing/throughput validation;
  - diagnosed the original camera MP4s' five-keyframe, 250-frame maximum-gap
    structure as the interactive seek/black-frame problem and added
    hash-bound, presentation-only H.264 proxies with 35 keyframes and a
    30-frame maximum gap, without modifying the synchronized source videos;
  - regenerated the preferred refined
    `artifacts/s07/final_run_v2_20260805/digital_twin_stage07_final.rrd`
    without model inference, size `44,022,273` bytes and SHA-256
    `bcf84af987069151339427d57d7642cffd0e92b6c0ff05bbdbddb7c6143b64ca`;
  - added timestamped XYZ/frame/camera/anchor provenance for every presentation
    record and capture-time-progressive measured dots and same-anchor segments;
  - preserved missing/occluded null XYZ, display-only stale values,
    disconnected gaps, and distinct person anchor semantics; no smoothing,
    interpolation, cross-anchor joining, or fabricated coordinates were added;
  - measured `1.298 s` for S07 entry verification and `2.461 s` for Rerun
    export, totalling `3.759 s` for the refined retained-output assembly;
  - recorded an assembly-only factor of `0.1076x` capture duration, or `9.291`
    seconds of recorded content assembled per wall-clock second, explicitly
    not a live or fresh-model throughput claim;
  - independently parsed all 20 required Rerun entity paths and verified all
    stable S06 semantic counts, event identities, capture-time authority, and
    missing/stale trajectory boundaries;
  - browser-tested Camera A and Camera B at early, middle, and late seek points
    without an observed blackout, and verified no trails at capture start,
    partial trails mid-run, full disconnected trails at the end, and explicit
    coordinate/provenance presentation;
  - retained the initial 51.9 MB assembly as immutable historical evidence and
    designated the refined v2 recording as the preferred interactive file; and
  - passed all 279 project tests, Ruff, strict mypy across 113 source/script
    files, lockfile/environment, artifact verification, and whitespace checks.
- Completed S07 Work Package 3 and the stage completion audit:
  - recorded D044 after the user accepted the refined interactive Rerun as the
    final demonstration instead of a separate rendered MP4;
  - added a durable capture/calibration guide, exact final-demonstration
    reproduction guide, and concise technical report;
  - consolidated DA3, YOLO, Qwen, queue/replay, RTSP, and assembly measurements
    while explicitly separating isolated/offline evidence from projected live
    capacity;
  - documented detector fragmentation, mixed person anchors, the rejected
    disagreement, the 6.803-second null-XYZ carry gap, Qwen failures, dynamic
    ground-truth absence, and RTSP/production limits;
  - reran the entry, seekable-proxy, and refined-Rerun verifiers into
    `artifacts/s07/stage_close_audit_20260805/`;
  - passed all five S07 ROADMAP completion criteria without weakening any
    criterion and created `docs/stages/S07_HANDOFF.md`; and
  - passed all 279 project tests, Ruff, strict mypy across 114 source/script
    files, lock/environment consistency, artifact verification, and whitespace
    checks.

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
- S03 dense sampling confirmed that the physical backpack is not reliably
  labelled `backpack` by the baseline checkpoint. D028's guarded `backpack`
  plus `handbag` candidate policy is accepted for the one physical
  demonstration bag; `suitcase` remains excluded.
- D028 bag tracks remain fragmented across viewpoint/absence phases. The main
  approximately `17.2-22.0 s` two-camera gap is persisted as missing rather
  than filled. This is an accepted proof-of-concept limitation, not a blocker
  for the S03 representative-segment gate.
- D029 cancels the proposed continuous 10 FPS comparison. The backpack is the
  representative movable object for this proof of concept, while the core value
  remains synchronized, metric, provenance-preserving spatial/event context.
  S03 keeps honest backpack evidence without optimizing a future application's
  object taxonomy prematurely.
- The integrated D028 bounded replay completed all 160 jobs per camera in exact
  capture order through independent capacity-eight queues. Each queue recorded
  152 throttle-and-drain events, zero drops, zero failures, stable replay job
  IDs, explicit empty-candidate frames, and complete raw artifacts.
- No S03 completion-gate blocker remains.
- S04 entry prerequisites were re-verified on 2026-08-01:
  - the project worktree was clean and `main` matched `origin/main` at
    `4dcd575`;
  - both synchronized action-video hashes, the synchronization manifest, the
    accepted action pose, scene metadata, S03 bounded replay, and target
    timelines were present and internally consistent;
  - the cached YOLO checkpoint, exact DA3 model revision, and 161-file DA3
    vendor fingerprint matched their recorded identities;
  - 162 automated tests, Ruff, strict mypy, lockfile/environment checks, and
    `git diff --check` passed.
- No S04 entry blocker is known. The sparse/fragmented backpack masks and the
  approximately `17.2-22.0 s` two-camera gap remain accepted limitations that
  S04 must represent without fabricated XYZ observations.
- D025 corrected action-pair depth and the margin-aware D030-D033 rebuild are
  verified for the preferred dense profile's `44` observed masks. The earlier
  20-mask corrected run remains a verified sparse comparison baseline; the
  accepted unscaled products remain diagnostic history and are not used by
  preferred later corrected products.
- Three person views are bottom-truncated and cannot provide feet. Corrected
  pair selection uses a synchronized mate's footpoint when available and
  otherwise preserves a lower- or upper-body surface with its true semantics.
- The preferred dense person layer has `16` usable outputs from 17 pairs:
  three fused, 13 single-camera, one explicit disagreement, and zero
  unavailable localization inputs. Ten outputs are footpoints, two are
  lower-body surfaces, and four are upper-body surfaces. Same-kind body
  surfaces may be shown as a separately styled measured segment but must never
  be joined into or converted into the footpoint track.
- All `17` corrected backpack records remain single-camera visible-cluster
  observations because the retained evidence has no same-frame two-camera
  backpack mask pair. The approximately `17.2-22.0 s` gap remains missing.
- D032's `0.35 m` comparable-anchor gate and D033 reliability score remain
  bounded prototype choices, not calibrated probabilities or production
  accuracy guarantees.
- D034 now supplies an honest presentation state at all 160 five-FPS ticks per
  target. Stale coordinates are display-only for at most one second and cannot
  affect zones, events, or measured tracks; after that the coordinate is
  absent. The original dense artifact claims no occlusion. The versioned D037
  correction consumes explicit visibility evidence and labels 33 backpack
  ticks occluded without coordinates or spatial authority.
- The dense measured trajectory is intentionally segmented. It contains eight
  person segments and 15 backpack segments. Person body-surface segments
  remain visually and semantically separate from footpoints; the frame `828`
  disagreement and `6.803 s` backpack gap remain non-authoritative and
  disconnected. These are qualitative prototype measurements, not a
  continuously observed or ground-truth physical path.
- No S04 completion-gate blocker remains. Its sparse backpack observations,
  one rejected person disagreement, and absence of surveyed dynamic
  ground-truth remain documented limitations for S05 rather than fabricated
  spatial facts.
- No S05 completion-gate blocker remains. Detector unreliability, the
  unlocalized carry interval, qualitative Qwen evidence, and hard process
  supervision assigned to S06 remain explicit accepted limitations.
- No S06 completion-gate blocker remains. Work Packages 1-5 establish the
  hash-bound orchestration, hard worker-supervision boundary, integrated
  file-backed Rerun presentation, deterministic bounded-queue replay,
  serialized-MPS policy, explicit degraded/shutdown evidence, local RTSP/TCP
  reconnect compatibility, and dedicated track/trajectory/event exports. The
  fresh stage audit passed all seven criteria without weakening them.
- No S07 completion-gate blocker remains. Work Packages 1-3 provide the
  hash-bound final run, refined and verified interactive demonstration,
  measured retained-output assembly, capture/calibration and reproduction
  guides, technical report, and five-criterion close audit. Under D044 the
  user accepted the interactive Rerun as the final demonstration; no separate
  rendered MP4 is claimed. The approved S00-S07 roadmap is complete.

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
- Target backpack: confirmed and used in the accepted action recording
- Empty-room recording: captured, synchronized, and restricted to a stable
  `22.0-38.0 s` candidate interval
- Pickup-carry-place recording: preferred take synchronized and accepted;
  backup raw take retained unchanged
- Representative room-view image: supplied and used for WP6

## Exact Next Action

Stop. The approved S00-S07 roadmap is complete. Begin follow-on work only from
a new user-approved objective and scope.
