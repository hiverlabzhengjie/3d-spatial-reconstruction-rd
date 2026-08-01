"""Summarize target and bag-like YOLO classes across an S03 diagnostic run."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

BAG_LIKE_CLASSES = ("backpack", "handbag", "suitcase")
ASSESSED_THRESHOLDS = (0.10, 0.15, 0.20, 0.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_summary_path = args.run_summary.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    summary = _read_object(run_summary_path)
    rows = _load_detections(summary, project_root=Path.cwd().resolve())

    threshold_results: dict[str, Any] = {}
    for threshold in ASSESSED_THRESHOLDS:
        camera_results: dict[str, Any] = {}
        for camera_id in ("camera_a", "camera_b"):
            class_results: dict[str, Any] = {}
            for class_name in ("person", *BAG_LIKE_CLASSES):
                selected = [
                    row
                    for row in rows
                    if row["camera_id"] == camera_id
                    and row["class_name"] == class_name
                    and row["confidence"] >= threshold
                ]
                class_results[class_name] = {
                    "detection_count": len(selected),
                    "frame_count": len({row["frame_id"] for row in selected}),
                    "times_seconds": sorted(
                        {round(float(row["time_seconds"]), 6) for row in selected}
                    ),
                    "confidence_min": (
                        min(float(row["confidence"]) for row in selected)
                        if selected
                        else None
                    ),
                    "confidence_max": (
                        max(float(row["confidence"]) for row in selected)
                        if selected
                        else None
                    ),
                }
            camera_results[camera_id] = class_results
        threshold_results[f"{threshold:.2f}"] = camera_results

    bag_details: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["class_name"] not in BAG_LIKE_CLASSES:
            continue
        box = row["box"]
        area_fraction = (
            (float(box["x_max"]) - float(box["x_min"]))
            * (float(box["y_max"]) - float(box["y_min"]))
            / (1920 * 1080)
        )
        bag_details[str(row["class_name"])].append(
            {
                "time_seconds": row["time_seconds"],
                "camera_id": row["camera_id"],
                "frame_id": row["frame_id"],
                "confidence": row["confidence"],
                "box_area_fraction": area_fraction,
            }
        )

    result = {
        "schema_version": 1,
        "stage": "S03",
        "purpose": "backpack_class_viability_analysis",
        "source_run_summary": str(run_summary_path.relative_to(Path.cwd().resolve())),
        "model": summary["model"],
        "sample_count": len(summary["samples"]),
        "sampled_bundle_count": len(summary["selection"]["selected_bundles"]),
        "assessed_thresholds": ASSESSED_THRESHOLDS,
        "bag_like_classes": BAG_LIKE_CLASSES,
        "threshold_results": threshold_results,
        "bag_like_detection_details": dict(sorted(bag_details.items())),
        "interpretation_boundary": (
            "Counts preserve vendor class labels. This analysis does not relabel "
            "handbag or suitcase detections as the target backpack."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def _load_detections(
    summary: dict[str, Any], *, project_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in summary["samples"]:
        detection_path = project_root / sample["artifacts"]["detections"]
        payload = _read_object(detection_path)
        for detection in payload["detections"]:
            rows.append(
                {
                    "time_seconds": sample["capture_timestamp_seconds"],
                    "camera_id": sample["camera_id"],
                    "frame_id": sample["frame_id"],
                    "class_name": detection["class_name"],
                    "confidence": detection["confidence"],
                    "box": detection["box"],
                }
            )
    return rows


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
