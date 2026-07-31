# S01 Preferred Dynamic Action Synchronization and Pose QA

**Date:** 2026-07-31

**Stage:** S01 - Capture, Synchronization, and Calibration

**Status:** Preferred dynamic take synchronized and accepted

## Selection

Both raw action takes were reviewed without modification. `action_take_01` was
selected because it contains the complete pickup-carry-place sequence in both
views, preserves useful backpack visibility, and supplies two strong matching
clap anchors. The backpack starts inside the blue bed pickup zone and ends at
the white floor drop-off zone.

The `action_take_02` files remain unchanged as a backup. They were not
synchronized or substituted into the baseline.

## Immutable raw inputs

| Source | SHA-256 |
| --- | --- |
| `camera_a_prefer.mp4` | `598448adc5e230b41d072330b716c2324d9931e8ca6ca33c37378a6907430748` |
| `camera_b_prefer.mp4` | `407f9ba9792930195e8580a799243f648c4555c47065e6fc9505a869c14679e6` |
| `camera_a_backup.mp4` | `1f843070d7b56bddee489e7d6f0d06b7771b49a64c7edad8c63099cd1a75ae06` |
| `camera_b_backup.mp4` | `6c021b9a18b3c3a46d96ad8fcc94c5af4cc101aca33406fb1dd8c5286bc15c80` |

All files decode at `1920x1080` and approximately 30 FPS with 48 kHz stereo
audio.

## Synchronization

The numerical anchors are sample-level clap onsets:

| Event | Camera A source time | Camera B source time |
| --- | ---: | ---: |
| Start | `1.741791667 s` | `12.199875000 s` |
| End | `34.655104167 s` | `45.117250000 s` |

Camera B's event span is `32.917375 s` versus `32.9133125 s` for Camera A,
corresponding to `123.430 ppm` relative drift. The derived clips retain one
second before and after the anchors.

Validation:

- Camera A output: `1,047` decoded frames;
- Camera B output: `1,047` decoded frames;
- start-clap residual: `-2.375 ms` Camera A minus Camera B;
- end-clap residual: `+5.854 ms` Camera A minus Camera B;
- both residuals are below one 30 FPS frame;
- the six-time pair preview is visually synchronized.

The complete timestamp mapping, source hashes, filter expressions, output
hashes, and content QA are retained in:

`artifacts/s01/action_take_01/synchronized/synchronization_manifest.json`

## Capture-specific pose

The synchronized action recording was independently checked against stationary
M40-M42 observations under D023. Dynamic person occlusions only remove
detections; they are not converted into marker coordinates. The remaining
detections are numerous and stable.

| Camera | Centre XYZ (m) | Anchor RMS | Sampled p95 | Fixed-reference delta |
| --- | --- | ---: | ---: | --- |
| A | `(0.127, 3.991, 2.134)` | `1.417 px` | `2.076 px` | `0.019 m`, `0.834 deg` |
| B | `(2.177, 3.661, 2.199)` | `1.275 px` | `2.185 px` | `0.009 m`, `0.420 deg` |

Both cameras pass all marker reprojection, per-frame stability, height,
downward optical-axis, floor-intersection, and D023 fixed-reference checks.
The accepted pose version is:

`s01_capture_20260729:action_take_01:v1`

Retained outputs:

- `artifacts/s01/calibration/action_take_01_pose/camera_calibration.json`
- `artifacts/s01/calibration/action_take_01_pose/camera_a_reprojection_preview.jpg`
- `artifacts/s01/calibration/action_take_01_pose/camera_b_reprojection_preview.jpg`
- `artifacts/s01/calibration/action_take_01_pose/camera_pair_reprojection_preview.jpg`

## Reproduction

The synchronization filters and encoding parameters are recorded verbatim in
the synchronization manifest. Re-run the pose correction with:

```bash
.venv/bin/python scripts/calibration/estimate_fixed_poses.py \
  --input-config artifacts/s01/calibration/action_take_01_pose_inputs.json \
  --output-dir artifacts/s01/calibration/action_take_01_pose
```

Verification commands:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/mypy src scripts/calibration/estimate_zones.py \
  scripts/calibration/estimate_fixed_poses.py
git diff --check
```

Results:

- full project suite: `117 passed`;
- Ruff: passed;
- strict mypy: passed across `18` source/script files;
- whitespace validation: passed;
- synchronization-manifest hash matches the pose output's recorded provenance;
- all four raw action SHA-256 hashes remain unchanged.

## Next S01 work

S01 is not yet complete. The next work package is the file source,
RTSP-compatible source boundary, and immutable synchronized frame-bundle
implementation. It must prove deterministic identities and capture-time
ordering under replay, plus explicit missing-camera and duplicate-frame
behaviour, before the S01 completion gate can close.
