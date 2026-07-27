# Model and Library Licence Record

**Reviewed:** 2026-07-27
**Project use:** Local, non-commercial university research

This is the initial S00 record. Before external distribution or commercial
reuse, verify every model revision and all locked transitive dependencies
individually and obtain legal advice where appropriate.

| Component | Intended use | Licence or constraint | S00 treatment |
|---|---|---|---|
| `depth-anything/DA3NESTED-GIANT-LARGE-1.1` weights | Metric depth and geometry | CC BY-NC 4.0 per the vendor model table | Local non-commercial research only |
| Depth Anything 3 source checkout | Unmodified vendor dependency | Apache-2.0 in the checkout | Preserve source and notices; project changes stay outside it |
| `yolov8n-seg.pt` and Ultralytics | Detection and segmentation | Ultralytics AGPL-3.0 or Enterprise terms apply | Local research prototype; review obligations before distribution |
| `Qwen/Qwen3-VL-2B-Instruct` | Triggered semantic interpretation | Apache-2.0 on the model card | Local processing with the exact approved model |
| PyTorch and torchvision | Apple MPS inference | BSD-style licence | Locked Python dependency |
| Hugging Face Transformers and Hub | Model loading and processing | Apache-2.0 | Locked Python dependency |
| OpenCV | Calibration and image utilities | Apache-2.0 | Locked Python dependency |
| COLMAP/PyCOLMAP | Optional supporting geometry or validation work | BSD-3-Clause; verify the selected build and notices before activation | Locked as an optional extra; not installed in the main MPS environment |
| Pydantic | Configuration and schemas | MIT | Locked Python dependency |

The lockfile is a reproducibility record, not a licence audit. Optional
commercial, hosted, or redistributed use remains outside the approved project
scope.
