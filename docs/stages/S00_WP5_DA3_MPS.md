# S00 WP5 DA3 Adapter and MPS Gate Record

**Work package:** WP5 - DA3 adapter and independent smoke check

**Date:** 2026-07-28

**Result:** Complete

## Outcome

- Added a project-owned adapter for the unmodified DA3 vendor snapshot.
- Loaded exactly `depth-anything/DA3NESTED-GIANT-LARGE-1.1` at revision
  `b2359bdf726fb44ef62acca04d629dcf158053e7`.
- Ran pose-conditioned, metric, two-view inference on native Apple MPS.
- Passed the 336, 420, and 504 resolution ladder using float16 autocast.
- Selected 504 as the provisional S00 processing resolution.
- Recommended a provisional two-second keyframe interval for 30 FPS video.
- Preserved compact arrays, camera/input provenance, previews, timings, and
  memory observations under the ignored local artifact directory.

No user-supplied image or video was required. WP5 used only the two approved
vendor SOH images. No model other than the approved DA3 checkpoint ran.

## Project-Owned Compatibility Boundaries

### MPS autocast

The vendor `forward` selects a dtype through a CUDA capability query regardless
of the actual image device. The adapter replaces only that instance-level
autocast boundary:

- MPS `auto` resolves directly to float16;
- float32 remains an explicit bounded fallback;
- CUDA capability functions are called only for a CUDA device; and
- vendor source files are not edited.

### Optional COLMAP exporter

The vendor API imports its optional COLMAP exporter eagerly, which otherwise
loads PyCOLMAP into the MPS process. The adapter installs a narrow disabled
export stub before importing the vendor API. A COLMAP export request raises an
actionable error and must use the already-approved isolated optional process if
a later stage justifies it. COLMAP was not run in WP5.

The vendor also logged that optional `gsplat` rendering was unavailable. This
was expected: `infer_gs=False`, Gaussian output is outside this gate, and no
optional Gaussian dependency was installed.

### Exact two-view post-alignment

The first native attempt completed the MPS forward pass at 336 but then failed
inside the vendor's Umeyama post-alignment with:

```text
GeometryException: Degenerate covariance rank, Umeyama alignment is not possible
```

Two camera centres cannot determine a full Sim(3). D019 records the bounded
adapter path: for exactly two views, preserve the nested model's already metric
depth and return the supplied processed intrinsics and
`T_camera_from_world`. Non-metric output is rejected, and other view counts
continue to use the vendor implementation.

This keeps the approved two-view methodology. The supplied cameras are
synthetic API fixtures and do not establish reconstruction accuracy.

## Inputs and Provenance

| Input | Recorded value |
|---|---|
| Vendor aggregate SHA-256 | `683cad1fec1186cd2a22f2b6d083b73d4c83c7ab1140f45ba24876612bc51d43` |
| `000.png` SHA-256 | `ea78c3b872b1e8b27de48cadf1d4a692cd42ddf5f72fcab78e2be2937935fb79` |
| `010.png` SHA-256 | `c91a69d4e050e75e3760fbacda18a452b9abcf3065e6d6bd940b4a99d48f7982` |
| Source dimensions | 1208 by 680 for both images |
| Synthetic baseline | 0.20 metres along world X |
| Camera convention | OpenCV `T_camera_from_world` |
| Gaussian inference | Disabled |
| Pose estimation only / ray pose | Disabled |

The vendor fingerprint matched both before and after inference. Model weights
remain in the local Hugging Face cache and are excluded from Git.

## Successful Native MPS Result

Artifact directory:

```text
artifacts/s00/da3/wp5_20260728_native_2/
```

The 504 output has:

- depth shape `[2, 280, 504]`, float32;
- confidence shape `[2, 280, 504]`, float32;
- two finite 4-by-4 `T_camera_from_world` matrices;
- two finite 3-by-3 intrinsic matrices;
- 100% finite and strictly positive depth;
- depth range approximately 3.97 to 44.16 metres; and
- confidence range approximately 1.00 to 6.72.

DA3 confidence is not assumed to be normalized. Later filtering must use its
observed/source semantics rather than clipping it to `[0, 1]`.

Returned camera poses matched the supplied pose-conditioned inputs. The depth
and confidence previews were inspected and showed coherent, corresponding
two-view structure. This is an execution/API observation, not an accuracy
assessment.

## Timing and Memory

Successful cached-load process:

| Phase | Seconds |
|---|---:|
| Model load | 7.364 |
| 336 cold | 0.827 |
| 336 warm 1 | 0.563 |
| 336 warm 2 | 0.558 |
| 420 probe | 1.270 |
| 504 probe | 1.136 |
| 504 repeat 1 | 1.050 |
| 504 repeat 2 | 1.050 |

The selected 504 repeated-pair mean was approximately 1.050 seconds. The
simple S00 recommendation is therefore one keyframe pair every two seconds,
or 60 frames at 30 FPS. S02 must revisit this with calibrated room images and
useful temporal coverage.

At the end of repeated 504 inference:

- process RSS was approximately 1.42 GB;
- MPS tensor allocation was approximately 6.76 GB;
- MPS driver allocation was approximately 8.83 GB; and
- the recommended MPS maximum was approximately 55.66 GB.

Driver allocation was about 15.9% of the recommended maximum, leaving
conservative headroom for this isolated model process. The successful load
monitor observed a temporary process RSS peak of approximately 13.38 GB.

The first uncached attempt spent approximately 97.8 seconds downloading/loading
and observed an approximately 16.81 GB temporary RSS peak before reaching its
successful 336 MPS forward and subsequent two-view post-alignment failure.
That failure evidence remains in
`artifacts/s00/da3/wp5_20260728_native_1/`.

## Artifacts

Successful artifact SHA-256 values:

| File | SHA-256 |
|---|---|
| `prediction.npz` | `ac5cd9f372ed6af28c0ac9fee240720ad85e6b623569b1c173ce7ed565a196c9` |
| `manifest.json` | `79c64de579eb196939136fbf59c9e12d8cba9a4880564f0f8fcda6590160f27c` |
| `summary.json` | `c0b25273e0e0b790c54a7e2faaa90c03c5cd7285bd38c18a8de2a8f2a183b43c` |
| `depth_preview.png` | `7efb23ff3f32c472d07c515d921c08a035ab00dd872655d6ffce0da530e715a2` |
| `confidence_preview.png` | `588f75ed7240af71f2cca98ce346cd17cbf6d22033e99cfea2e2f19fd2b3272f` |

## Files Added

- `src/spatial_reconstruction/models/__init__.py`
- `src/spatial_reconstruction/models/da3_adapter.py`
- `src/spatial_reconstruction/models/da3_mps.py`
- `scripts/smoke/da3_two_view.py`
- `tests/test_da3_adapter.py`
- `tests/test_da3_mps.py`

## Verification

Commands:

```text
.venv/bin/pytest -q tests/test_da3_adapter.py tests/test_da3_mps.py
.venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/mypy src scripts/diagnose_runtime.py scripts/smoke/da3_two_view.py
uv lock --check
uv sync --check
.venv/bin/python scripts/smoke/da3_two_view.py \
  --output-dir artifacts/s00/da3/wp5_20260728_native_2
```

Results:

- WP5 focused tests: 17 passed;
- complete project suite: 83 passed;
- Ruff: pass;
- strict mypy: pass across 13 source/script files;
- lockfile/environment: current, 102 packages, no changes;
- restricted process failure summary: pass;
- native MPS resolution ladder: pass; and
- vendor source fingerprint after inference: unchanged.

Tests cover explicit pose conversion and round trips, synthetic camera
construction, invalid camera pairing, prediction shape/finiteness/metric
validation, normalized 4-by-4 poses, vendor fingerprint reproduction,
device-specific autocast policy, installed forward behavior, the two-view
metric alignment path, and rejection of non-metric use.

## Decisions

D019 records the bounded two-view post-alignment compatibility path. No
additional reconstruction methodology was introduced.

## Exact Next Action

Obtain one representative, unmodified living-room image from the user, then
begin S00 WP6 only: implement the YOLOv8n-seg adapter and independent MPS smoke
check.
