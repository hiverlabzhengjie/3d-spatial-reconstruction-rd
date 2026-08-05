# Capture and Calibration Guide

This guide describes the physical procedure required to reproduce the accepted
two-camera living-room prototype. It does not make the existing calibration
portable to another room, lens, mount position, or recording mode.

## Equipment

- Two fixed cameras with rigid mounts and continuous power.
- Matched 1920x1080, nominal 30 FPS recording settings.
- The selected 13 mm-equivalent ultrawide lens on both accepted phones.
- Printed A4 ChArUco board `s01-charuco-6x8-30mm-5x5-100-v1`, mounted flat on
  rigid matte backing and dimension-checked after printing.
- Four 180 mm floor markers from
  `s01-floor-markers-40-43-180mm-5x5-100-v1`.
- Tape measure, painter's tape, one backpack, and two visible approximately
  0.30 m-radius zone boundaries.

## Camera setup

1. Mount both cameras in landscape orientation with overlapping views of the
   person, pickup zone, carry route, and drop-off zone.
2. Select 1920x1080 at nominal 30 FPS and the same fixed lens for all captures.
   Disable lens switching and enhanced stabilization. Lock focus, exposure, and
   white balance where the camera application permits.
3. Prevent all mount contact after pose calibration. Record the camera model,
   lens, resolution, frame rate, lock settings, lighting, and any movement in
   the capture notes.
4. Treat any physical camera movement, lens change, resolution change, or
   material stabilization-state change as calibration invalidation. Do not
   reuse a previous pose merely because the scene looks similar.

## Intrinsic calibration

1. Record at least 25 distinct steady ChArUco views covering the image centre,
   edges, corners, varied board rotations, and useful distance range.
2. Keep the printed board flat and fully visible. Avoid motion blur, glare, and
   severe grazing angles.
3. Calibrate each physical camera independently by default. The accepted
   prototype used one shared numeric estimate only because the matched phones
   and settings were identical and Camera B independently passed its marker
   reprojection gate.
4. Reject a calibration with poor spatial coverage, unstable parameters, or
   visibly systematic edge residuals. The accepted shared calibration achieved
   `0.280 px` RMS, which is evidence for this capture only.

## World frame and fixed poses

The right-handed world frame uses metres, `Z` upward, origin at M40, `X` along
the selected primary wall, and `Y` across the room. Record marker centres by
physical measurement; never infer a missing surveyed coordinate from imagery.

The accepted marker centres are:

| Marker | World centre `(X, Y, Z)` metres |
|---|---|
| M40 | `(0.00, 0.00, 0.00)` |
| M41 | `(1.23, 0.45, 0.00)` |
| M42 | `(0.00, 2.20, 0.00)` |
| M43 | `(1.10, 3.70, 0.00)` diagnostic only |

1. Record both fixed cameras while M40-M42 are complete, stationary, and
   jointly visible. Keep M43 for diagnosis; it is excluded from the accepted
   pose because its recorded location failed reprojection.
2. Solve and persist explicit `T_world_from_camera` and
   `T_camera_from_world`; never store an ambiguous field named `extrinsics`.
3. Inspect marker overlays and camera frustums. Require aggregate and sampled
   marker reprojection within 5 px, stable camera centres/orientations,
   plausible camera height, a downward optical axis, and floor intersections
   inside the declared room.
4. A later recording may use a versioned D023 capture-specific correction only
   when M40-M42 remain stationary, every reprojection/stability check passes,
   camera-centre displacement is at most 0.05 m, and rotation differs by at
   most 1 degree from the physical reference. Otherwise recalibrate.

## Synchronization and recording structure

1. Store every session separately. Never overwrite raw camera files.
2. Record a bright flash and audible clap visible to both cameras near the
   start and end. Retaining both anchors allows clock-drift correction rather
   than assuming equal phone clocks.
3. Preserve raw hashes, decoded frame counts, nominal frame rates, detected
   clap times, offset, drift correction, and residual disagreement in the
   synchronization manifest.
4. Review multiple paired timestamps from start to end. The accepted
   `action_take_01` derivatives contain 1,047 frames per camera; their residual
   clap disagreement remains below one 30 FPS frame.

## Empty-room and action captures

- Empty room: record the stable room, M40-M42, and both zone boundaries with no
  moving person in the selected reconstruction interval. The accepted stable
  interval is 22-38 seconds; earlier setup footage contains the operator.
- Action: begin with the backpack stationary in the pickup zone. Record one
  person approaching, picking it up, carrying it through the overlap, placing
  it in the drop-off zone, moving away, and leaving it stationary.
- Keep both cameras, marker locations, lens selections, room layout, and
  lighting unchanged between pose validation and the dependent action take.
- Preserve imperfect observations. Occlusion, detector absence, invalid depth,
  and cross-camera disagreement must become explicit unavailable states rather
  than invented XYZ.

## Required acceptance checks

- The synchronized pair is visually coherent from start through end.
- Replaying the same files produces identical frame-bundle identities and
  capture ordering.
- Both camera marker overlays and frustums pass the recorded pose gates.
- Transform inversion, coordinate round-trip, reprojection, back-projection,
  invalid-depth, and missing-camera tests pass.
- Raw recordings and unmodified DA3 vendor source retain their recorded hashes.

The accepted S01 commands and numerical evidence are retained in the S01 stage
records under `docs/stages/`. A new physical capture must begin from those
procedures; it must not inherit the `action_take_01` calibration identity.
