# S00 WP3 Geometry Foundation Record

**Work package:** WP3 - Coordinate foundation

**Date:** 2026-07-28

**Result:** Complete

## Outcome

- Added project-owned rigid-transform validation, inversion, and point
  transformation utilities.
- Added OpenCV pinhole projection and metric depth back-projection utilities.
- Added explicit invalid-depth and confidence filtering that returns only
  valid samples, never placeholder XYZ coordinates.
- Added deterministic synthetic tests for transform, reprojection,
  back-projection, invalid-input, missing-camera, and empty-result behaviour.

No model inference, calibration, capture processing, or DA3 vendor modification
was performed.

## Coordinate and Geometry Contract

The implementation follows the canonical conventions:

- world coordinates are right-handed, measured in metres, with Z upward;
- camera coordinates use OpenCV X-right, Y-down, Z-forward;
- transforms use explicit source/target names such as
  `T_world_from_camera` and `T_camera_from_world`;
- rigid transforms must be finite 4-by-4 homogeneous matrices with an
  orthonormal, determinant-positive rotation;
- projection accepts only finite camera points with strictly positive Z; and
- back-projection accepts only finite, strictly positive metric depth.

The utilities operate on either one point or a batch of points. Camera/world
wrappers consume the validated `CameraPose` contract from WP2.

## Filtering and Missing-Data Behaviour

`depth_confidence_valid_mask` accepts finite positive depth and finite
confidence at or above the caller's threshold. It does not assume confidence is
normalized, because the project must preserve the source model's confidence
semantics.

`backproject_valid_pixels` returns:

1. only the XYZ rows that passed the validity mask; and
2. the complete Boolean mask for traceability to the input samples.

When no samples are valid, the result is an empty `(0, 3)` array. Missing
intrinsics raise `MissingCameraError`. Neither path fabricates coordinates.

## D018 Boundary

A deterministic test demonstrates that one pixel combined with an action-frame
foreground depth produces a different 3D point from the same pixel combined
with static empty-room depth. This protects the ray-depth principle behind
D018.

WP3 does not yet validate temporal freshness or depth-frame provenance. Those
checks depend on frame-aligned model outputs and belong to S04 dynamic
localization. Empty-room depth remains prohibited as a substitute for
action-frame depth when localizing a person or backpack.

## Files

- `src/spatial_reconstruction/geometry/__init__.py`
- `src/spatial_reconstruction/geometry/transforms.py`
- `src/spatial_reconstruction/geometry/projection.py`
- `tests/test_transforms.py`
- `tests/test_projection.py`

## Verification

Commands:

```text
.venv/bin/pytest -q
.venv/bin/ruff check src tests
.venv/bin/mypy src
```

Results:

- pytest: 49 passed across the complete project test suite;
- Ruff: pass; and
- strict mypy: pass across six source files.

The new tests cover:

- rigid-transform inversion and identity composition;
- camera-to-world-to-camera round trips;
- single-point and batched point transformations;
- invalid shape, non-finite values, non-orthonormal rotation, and reflection;
- OpenCV projection/back-projection numerical round trips;
- principal-point projection on the optical axis;
- zero, negative, NaN, and infinite depth rejection;
- non-positive camera-Z rejection;
- confidence and invalid-depth filtering;
- empty valid results without placeholder XYZ rows;
- missing camera intrinsics; and
- the D018 foreground-versus-static-depth distinction.

## Decisions

No new project decision was required. WP3 implements the established coordinate,
missing-data, and D018 policies. COLMAP or another supporting methodology was
not needed for this work package.

## Exact Next Action

Begin S00 WP4 only: implement shared device selection, timing, and memory
diagnostics.
