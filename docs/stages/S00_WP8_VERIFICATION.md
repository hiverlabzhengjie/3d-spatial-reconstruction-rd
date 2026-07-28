# S00 WP8 - Integrated Verification and Stage-Close Evidence

**Date:** 2026-07-28

**Status:** Complete

**S00 completion gate:** Passed without a device or model fallback

## Purpose and Boundary

WP8 re-runs and reviews the complete S00 gate before stage close. DA3, YOLO,
and Qwen were executed as three separate native processes so their memory did
not overlap. The automated suite and reproducibility checks were also run from
the locked project environment.

No S01 capture, calibration, synchronization, or physical-scene work was
started.

## Automated Verification

Commands:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/mypy src scripts/diagnose_runtime.py \
  scripts/smoke/da3_two_view.py scripts/smoke/yolo_seg.py \
  scripts/smoke/qwen_multiframe.py
uv --cache-dir /private/tmp/spatial-reconstruction-uv-cache lock --check
uv --cache-dir /private/tmp/spatial-reconstruction-uv-cache sync --check
git diff --check
```

Results:

- pytest: 106 passed in 1.04 seconds;
- Ruff: passed;
- strict mypy: passed across 17 source/script files;
- `uv.lock`: current;
- `.venv`: consistent with the lock, 102 installed packages checked;
- whitespace/error-marker check: passed;
- lockfile SHA-256:
  `4a8dcaf5430c9a9b68ffa965289b40b02e4553403c6e99e7fc9dc7a76c3d1873`.

The tests include schemas and serialization, transform inversion and coordinate
round trips, projection/back-projection, invalid depth/confidence filtering,
missing camera/object behavior, runtime failure provenance, DA3 adapter
boundaries, YOLO empty/invalid results, and Qwen ordered/empty/bounded output.

## Heavy Gates

### Gate A - DA3 pose-conditioned two-view metric depth

Command:

```bash
.venv/bin/python scripts/smoke/da3_two_view.py \
  --revision b2359bdf726fb44ef62acca04d629dcf158053e7 \
  --output-dir artifacts/s00/wp8/da3_gate_20260728
```

Result:

- exact model and revision passed on Apple MPS using float16 autocast;
- two vendor views plus supplied synthetic OpenCV intrinsics and
  `T_camera_from_world` poses were recorded;
- 336, 420, and 504 completed;
- selected output shapes: depth/confidence `[2, 280, 504]`;
- finite, strictly positive depth fraction: 1.0;
- selected repeated 504 pair mean: 1.374738 seconds;
- provisional interval remains two seconds, or 60 frames at 30 FPS;
- final MPS allocation: 6,759,876,608 bytes;
- final MPS driver allocation: 8,831,549,440 bytes;
- no CPU fallback and no gate warning.

The vendor's optional `gsplat` warning is expected because Gaussian rendering
is disabled and outside this gate. The depth and confidence previews were
inspected and show corresponding two-view structure. They are not a
metric-accuracy claim.

Key artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `summary.json` | `887478dcef909d8251ef23a205ae180125a73524f48e2f692eafd7375dd04336` |
| `manifest.json` | `84659f46d7fb7a97cb1e3f14617f61d7661f870f14e3421ce48e8d1cfb5849b5` |
| `prediction.npz` | `c6e949b63b630a71c4d187ae8bdb7153f283c606fdbf695a5fb2c02504c11c87` |
| `depth_preview.png` | `14e9ad909be206504536dba2deed76d382c06677c910d34ed66fe24af72c819c` |
| `confidence_preview.png` | `588f75ed7240af71f2cca98ce346cd17cbf6d22033e99cfea2e2f19fd2b3272f` |

### Gate B - YOLOv8n-seg representative image

Command:

```bash
.venv/bin/python scripts/smoke/yolo_seg.py \
  --image <path-to-representative-room-image> \
  --output-dir artifacts/s00/wp8/yolo_gate_20260728
```

Result:

- exact checkpoint fingerprint
  `a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c`;
- Apple MPS, float32, no CPU fallback;
- cold inference: 1.376747 seconds;
- warm inference: 0.066178 and 0.065715 seconds;
- two source-sized `bed` masks, with confidences approximately 0.90 and 0.49;
- zero `person` and `backpack`, honestly expected for the unstaged image;
- input SHA-256 unchanged before and after;
- preview inspected; the second result remains a partial/duplicate bed
  diagnostic rather than a target detection.

Key artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `summary.json` | `e3467aa3bf8420f2b6ae8717e933ceeb19b4bbbb77adc43adaab6c69d0898247` |
| `manifest.json` | `4eb2447ce477555e8d03a355a45557f31cc287838745391a3713792a8d42e6a5` |
| `detections.json` | `d507ba347a9f8e708a8cc75e4760d1d13168193edaa730a2f6d461a4481210b8` |
| `masks.npz` | `3eab45bc8be929ada48ab9bf45089c3295f3e422cd061a9c9cd2db1440ad547e` |
| `annotated_preview.jpg` | `f9d6b1fa56c18e78eaa940c7064b0f081db52a0c5490396e1f348e199809f675` |

### Gate C - Qwen3-VL ordered multi-frame text

Command:

```bash
.venv/bin/python scripts/smoke/qwen_multiframe.py \
  --output-dir artifacts/s00/wp8/qwen_gate_20260728
```

Result:

- exact `Qwen/Qwen3-VL-2B-Instruct` revision
  `89644892e4d85e24eaac8bacfd4f463576704203`;
- four uniformly ordered frames accepted;
- Apple MPS, float16, no CPU fallback;
- deterministic 64-token bound;
- cold generation: 6.575227 seconds;
- warm generation: 6.614412 seconds;
- cold and warm output: identical, non-empty 41-token text;
- output accurately describes standing, airborne rotation, and landing in the
  inspected frames;
- asynchronous adapter enabled and spatial write interface absent;
- input video SHA-256 unchanged before and after.

Key artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `summary.json` | `bd4e47c75cfe1825710e716f2bf1c2d53f511e7004bf1cc2222b289944c3c01d` |
| `manifest.json` | `98dd398f1b8fb3f173b525a9ac95b803a70dc3239a494d99543b682e17f9bbc2` |
| `raw_responses.json` | `439727b09b5e398c7ef9e94174307388ce291783cc3f3b3d7a6fba02e059a4c8` |
| `cold_response.txt` | `71729c08e5f5f39028f93ddad66a4effedbc3192e47d84dcd01c64db9c376ecd` |
| `warm_response.txt` | `71729c08e5f5f39028f93ddad66a4effedbc3192e47d84dcd01c64db9c376ecd` |

## Gate Review

| Roadmap completion statement | WP8 result |
|---|---|
| DA3 two-view inference runs or has a documented MPS workaround | Passed on MPS through the bounded project-owned adapter |
| YOLOv8n-seg runs on a representative image | Passed on the user-supplied image |
| Qwen3-VL-2B accepts a small multi-frame input and returns text | Passed with four frames and bounded non-empty text |
| The project test command runs successfully | Passed, 106 tests |

The gate was not weakened. The DA3 two-view post-alignment handling is the
approved D019 compatibility boundary, not a replacement model or methodology.

## Vendor and Public-Repository Audit

- Three transient Finder `.DS_Store` files were found inside the ignored DA3
  checkout during the pre-close audit and removed.
- The DA3 vendor checkout then returned to exactly 161 files, 25,297,499 bytes,
  and aggregate SHA-256
  `683cad1fec1186cd2a22f2b6d083b73d4c83c7ab1140f45ba24876612bc51d43`.
- The WP8 DA3 manifest independently recorded the same expected and actual
  fingerprint.
- Raw user media, vendor source, model weights, caches, virtual environments,
  and generated artifacts remain ignored and outside the public commit.
- The tracked `.env` contains only the non-secret local Matplotlib cache path;
  it contains no credential.
- No optional supporting methodology was activated.

## Completion Decision

S00 passes all four completion gates and may close. The remaining missing
physical inputs belong to S01 and do not weaken S00.

## Exact Next Action

After explicit user approval to begin S01, inventory the two phone/lens
configurations and confirm rigid mounts plus the printed ChArUco board's exact
dimensions before recording any calibration session.
