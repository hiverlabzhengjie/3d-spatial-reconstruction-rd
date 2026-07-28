# S00 WP6 - YOLOv8n-seg MPS Gate

**Date:** 2026-07-28

**Status:** Complete

**Roadmap gate:** Gate B passed

## Purpose and Boundary

WP6 proves that the exact baseline `yolov8n-seg.pt` checkpoint can run
independently on Apple MPS and that its output can be validated and retained in
project-owned contracts. Tracking, ByteTrack, cross-camera association, 3D
localization, and detector-quality evaluation remain outside this work package.

No optional methodology such as COLMAP was used. The bottle fallback was not
activated.

## Representative Input

The user supplied:

- filename: `stage_00_WP6_sample_image.jpeg`, kept outside the repository;
- intended use: a representative living-room view, approximately rather than
  exactly matching one intended camera viewpoint;
- decoded dimensions: 4032 x 3024;
- container reported by Pillow: MPO, with two embedded frames;
- frame used: embedded frame 0, after EXIF display-orientation handling;
- EXIF orientation: 1;
- SHA-256:
  `1cc9ccf0cd2c28ee91f72f4d8443e84bd9a7f2f933ceb50fc26146e93c17c0ab`.

The source hash was identical before and after inference. The source file was
not copied into the repository, overwritten, or otherwise modified.

The image contains neither a deliberately staged person nor a deliberately
staged backpack. Their absence therefore cannot be used to assess target-class
recall.

## Project-Owned Implementation

WP6 added:

- a replaceable `YOLOSegAdapter` which accepts only the approved
  `yolov8n-seg.pt` identifier;
- an image loader that selects the first embedded image frame and applies EXIF
  display orientation without changing the source;
- validation and normalization for image dimensions, boxes, integer class
  identifiers, finite confidence values, masks, and annotated output;
- source-sized binary masks plus retained raw boxes, classes, confidences, and
  masks for troubleshooting;
- an isolated `scripts/smoke/yolo_seg.py` command;
- a configured inference image size of 640 pixels;
- automated tests for successful, empty, and invalid result structures.

The smoke command invokes prediction rather than tracking and records
`tracking_enabled: false`. It performs one cold and two warm predictions in a
fresh process. It requires actual MPS and does not silently fall back to CPU.

## Exact Model and Runtime

| Item | Recorded value |
|---|---|
| Model identifier | `yolov8n-seg.pt` |
| Checkpoint SHA-256 | `a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c` |
| Ultralytics version | 8.4.107 |
| Device | Apple MPS |
| Actual parameter precision | float32 |
| Inference image size | 640 |
| Confidence threshold | 0.25 |
| CPU fallback | none |
| Tracking | disabled |

The checkpoint is held in the ignored local model cache at
`.cache/models/yolov8n-seg.pt`; it is not committed to Git.

## Native MPS Result

The final clean run passed and produced two COCO `bed` detections:

| Class | Confidence | Interpretation |
|---|---:|---|
| bed | 0.904071 | Main bed |
| bed | 0.492559 | Partial bed/bedding region at the lower left |

No `person` or `backpack` was detected. This is an honest and valid result for
the supplied unstaged scene. The second, lower-confidence bed result appears to
cover a partial bedding/bed region and is retained as a diagnostic example of
duplicate or partial-instance behavior. It does not weaken the structural S00
gate.

The normalized mask array has shape `[2, 3024, 4032]` and dtype `uint8`.
All detections have finite in-range confidences, in-bounds boxes, and masks
associated with the source image. The annotated preview was inspected.

## Timing and Memory Observations

WP4 synchronized phase timing is the authoritative end-to-end measurement:

| Phase | Time (s) | Process RSS after phase | MPS allocated | MPS driver allocated |
|---|---:|---:|---:|---:|
| Model load | 0.288200 | 563,527,680 | 0 | 393,216 |
| Cold inference | 1.428195 | 1,292,320,768 | 71,828,992 | 2,399,092,736 |
| Warm inference 1 | 0.066324 | 1,463,156,736 | 84,055,296 | 2,399,092,736 |
| Warm inference 2 | 0.068561 | 1,463,255,040 | 71,828,992 | 2,399,092,736 |

The final observed process peak was 1,463,287,808 bytes. PyTorch reported an
MPS recommended maximum working-set size of 55,662,788,608 bytes. These figures
describe this one representative run; they are not capacity guarantees for
later concurrent or video workloads.

Ultralytics also reported last-run internal per-image preprocessing, inference,
and postprocessing times in the manifest. Those internal values are retained
for diagnosis but are not substituted for synchronized end-to-end timings.

## Retained Artifacts

Final run directory:
`artifacts/s00/yolo/wp6_20260728_native_2/`

| Artifact | SHA-256 |
|---|---|
| `annotated_preview.jpg` | `f9d6b1fa56c18e78eaa940c7064b0f081db52a0c5490396e1f348e199809f675` |
| `detections.json` | `3f4e3245345c1d1621eaa45640174f0dddb5a68b9676799a66e5a616afa14381` |
| `manifest.json` | `2330a607fffce2ffe22537a87924e8ae2642f7c61d20ba80d41213218707437c` |
| `masks.npz` | `3eab45bc8be929ada48ab9bf45089c3295f3e422cd061a9c9cd2db1440ad547e` |
| `summary.json` | `89a2f838426246ee5491f7d088646637b40cde8260c894336f1521527bb8801f` |

The artifact directory is intentionally ignored by Git.

## Reproduction Commands

From the project root:

```bash
uv run python scripts/smoke/yolo_seg.py \
  --image <path-to-representative-room-image> \
  --output-dir artifacts/s00/yolo/wp6_20260728_native_2
```

The output directory must not already exist. For a repeat run, use a new
directory name or omit `--output-dir` to obtain a timestamped directory.

Automated verification:

```bash
uv run pytest -q
uv run ruff check src tests scripts
uv run mypy src scripts/diagnose_runtime.py \
  scripts/smoke/da3_two_view.py scripts/smoke/yolo_seg.py
```

## Gate-B Verification

- [x] A representative living-room image was supplied by the user.
- [x] The exact checkpoint identity and byte fingerprint were recorded.
- [x] Actual MPS segmentation inference completed with no CPU fallback.
- [x] Image size, detections, boxes, confidences, and masks were validated.
- [x] Cold and warm timings plus process and MPS memory were recorded.
- [x] Normalized output, retained raw arrays, summary, manifest, and annotated
      preview exist.
- [x] The source image remained unchanged.
- [x] Empty target-class behavior and invalid result structures are tested and
      represented honestly.
- [x] Tracking remained disabled and no later-stage perception logic was added.

## Known Limits

- This single unstaged image proves execution and output structure, not person
  or backpack recall.
- The viewpoint is representative but is not the final surveyed camera pose.
- The two bed-class results show that partial or duplicate instance behavior
  must be handled deliberately in later perception work.
- S03 must evaluate the backpack under the real capture conditions before any
  approved bottle fallback is considered.

## Exact Next Action

Begin WP7 only: implement the independent adapter for exactly
`Qwen/Qwen3-VL-2B-Instruct` and run its bounded multi-frame MPS smoke check on
four to eight ordered frames from the existing vendor
`robot_unitree.mp4`. No user video is required for that S00 compatibility gate.
