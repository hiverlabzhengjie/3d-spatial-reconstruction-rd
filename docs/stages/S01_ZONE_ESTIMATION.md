# S01 Video-Estimated Zone Metadata

**Date:** 2026-07-31

**Stage:** S01 - Capture, Synchronization, and Calibration

**Status:** Accepted as video-estimated S01 zone metadata

## Purpose

Define the pickup and drop-off regions without a separate physical zone
survey. Under D024, the blue and white ropes are interpreted as thin circular
boundary centrelines with an approximate fixed horizontal radius of `0.30 m`.
The enclosed surfaces are not treated as painted or colour-filled areas.

These regions are metadata only. Later stages use them for backpack
zone-membership checks, pickup/carry/place event-state transitions, and Rerun
zone visualization. They do not alter camera calibration, DA3 depth, raw
person/backpack positions, track identity, or timestamps.

## Inputs and method

- Empty-room pose version:
  `s01_capture_20260729:empty_room:v1`
- Representative synchronized time: `30.0 s`
- Input annotations:
  `artifacts/s01/zones/zone_estimation_inputs.json`
- White floor zone:
  intersected each fitted image-centre ray with the known `Z=0` floor, fused
  the estimates, then refined the centre against both projected rope
  boundaries.
- Blue bed zone:
  triangulated the two fitted image-centre rays, then refined `(X, Y, Z)`
  against both projected rope boundaries.

The implementation uses project-owned OpenCV/NumPy geometry and introduces no
new dependency. The radius is held fixed, so the result remains conditioned on
the user's approximate `0.30 m` physical radius.

## Provisional results

| Zone | Video-estimated centre XYZ (m) | Radius (m) | Camera A RMS | Camera B RMS |
| --- | --- | ---: | ---: | ---: |
| Blue bed pickup | `(1.736, 2.815, 0.599)` | `0.300` | `8.845 px` | `4.632 px` |
| White floor drop-off | `(0.338, 0.592, 0.000)` | `0.300` | `9.873 px` | `9.447 px` |

The independent white-zone floor-ray estimates were
`(0.381, 0.583, 0.000) m` from Camera A and
`(0.281, 0.599, 0.000) m` from Camera B, a `0.102 m` disagreement below the
declared `0.15 m` acceptance limit. The blue-zone initial two-ray solution had
`0.014 m` perpendicular residual from each camera ray.

Both final centres lie within the declared room bounds. Both world circles
project in front of both cameras, and every per-camera rope-boundary residual
is below the declared `10 px` limit.

## Retained outputs

- `artifacts/s01/zones/estimated_zones.json`
- `artifacts/s01/zones/camera_a_zone_estimation_overlay.jpg`
- `artifacts/s01/zones/camera_b_zone_estimation_overlay.jpg`
- `artifacts/s01/zones/camera_pair_zone_estimation_overlay.jpg`

Overlay key:

- green points: manually selected visible rope centreline samples;
- yellow line: ellipse fitted to those image samples;
- coloured line: projection of the fitted `0.30 m` world-space circle.

## Verification

Commands:

```bash
.venv/bin/pytest -q tests/test_zones.py tests/test_fixed_pose.py \
  tests/test_transforms.py
.venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/mypy src scripts/calibration/estimate_zones.py \
  scripts/calibration/estimate_fixed_poses.py
.venv/bin/python scripts/calibration/estimate_zones.py
```

Results:

- targeted geometry/calibration tests: `21 passed`;
- full project suite: `117 passed`;
- Ruff: passed;
- strict mypy: passed across `18` source/script files;
- real estimate: all automated zone checks passed.

The new synthetic checks cover known floor-ray intersection, two-view
triangulation, horizontal-circle recovery, and rejection of parallel or
behind-camera plane intersections.

The user visually reviewed the overlays on `2026-07-31` and confirmed that the
estimated positions and approximately `0.60 m` bed-zone height are physically
sensible. The D024 user validation gate is therefore closed. The coordinates
remain explicitly video-estimated rather than surveyed.

## Reproduction and validation

Re-run the estimator with:

```bash
.venv/bin/python scripts/calibration/estimate_zones.py
```

The completed user validation compared the coloured projected circles with the
physical rope centrelines in both camera overlays and sanity-checked:

1. whether the white centre is plausibly about `0.34 m` along X and `0.59 m`
   along Y from M40;
2. whether the blue centre is plausibly at about `0.60 m` above the floor; and
3. whether the blue centre's `(1.74, 2.81) m` floor-plan location matches the
   circle on the bed.

If later geometry materially contradicts this check, refine the existing
annotations or obtain one clearer empty-room zone view. Do not silently label
the estimates surveyed.
