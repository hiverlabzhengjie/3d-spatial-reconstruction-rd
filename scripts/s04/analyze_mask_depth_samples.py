"""Compare raw in-mask DA3 depth/confidence strategies without producing XYZ."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from spatial_reconstruction.contracts import PerceptionTarget
from spatial_reconstruction.localization import (
    ActionDepthRunSummary,
    AlignedMaskRecord,
    CandidateMask,
    MaskAlignmentRunSummary,
    MaskDepthDiagnosticConfig,
    MaskDepthDiagnosticRecord,
    MaskDepthDiagnosticRunSummary,
    MaskDepthStrategy,
    build_mask_depth_candidates,
    compute_mask_depth_diagnostics,
)

UInt8Array = NDArray[np.uint8]
Float32Array = NDArray[np.float32]
CAMERA_INDEX = {"camera_a": 0, "camera_b": 1}
STRATEGY_COLORS = {
    MaskDepthStrategy.WHOLE_MASK: "#ffd43b",
    MaskDepthStrategy.ERODED_INTERIOR: "#00d4ff",
    MaskDepthStrategy.CONNECTED_DEPTH_CLUSTER: "#ff2fa2",
    MaskDepthStrategy.PERSON_LOWER_BODY: "#39ff88",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--mask-alignment-summary",
        type=Path,
        default=Path("artifacts/s04/mask_alignment_20260801/summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--erosion-radius-pixels", type=int, default=2)
    parser.add_argument("--person-lower-body-fraction", type=float, default=0.35)
    parser.add_argument("--cluster-minimum-half-width-m", type=float, default=0.15)
    parser.add_argument("--cluster-mad-scale", type=float, default=2.5)
    parser.add_argument(
        "--confidence-percentiles",
        type=float,
        nargs="+",
        default=(20.0, 40.0, 60.0, 80.0),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    alignment_path = _resolve(project_root, args.mask_alignment_summary)
    output_dir = _resolve(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    diagnostics_dir = output_dir / "per_mask"
    diagnostics_dir.mkdir()

    alignment = MaskAlignmentRunSummary.model_validate_json(
        alignment_path.read_text(encoding="utf-8")
    )
    action_path = _resolve(
        project_root, Path(alignment.source_action_depth_summary_ref)
    )
    _require_hash(action_path, alignment.source_action_depth_summary_sha256)
    action = ActionDepthRunSummary.model_validate_json(
        action_path.read_text(encoding="utf-8")
    )
    predictions = {record.job.job_id: record for record in action.predictions}
    config = MaskDepthDiagnosticConfig(
        erosion_radius_pixels=args.erosion_radius_pixels,
        person_lower_body_fraction=args.person_lower_body_fraction,
        cluster_minimum_half_width_m=args.cluster_minimum_half_width_m,
        cluster_mad_scale=args.cluster_mad_scale,
        confidence_percentiles=tuple(args.confidence_percentiles),
    )

    raw_cache: dict[str, tuple[Float32Array, Float32Array, UInt8Array]] = {}
    mask_cache: dict[str, UInt8Array] = {}
    records: list[MaskDepthDiagnosticRecord] = []
    diagnostic_artifacts: list[dict[str, Any]] = []
    contact_items: list[tuple[AlignedMaskRecord, UInt8Array, UInt8Array]] = []

    for mask_index, aligned_record in enumerate(alignment.aligned_masks):
        prediction = predictions.get(aligned_record.action_depth_job_id)
        if prediction is None:
            raise ValueError("aligned mask refers to unknown action-depth job")
        raw = raw_cache.get(prediction.raw_prediction_ref)
        if raw is None:
            raw_path = _resolve(project_root, Path(prediction.raw_prediction_ref))
            _require_hash(raw_path, prediction.raw_prediction_sha256)
            with np.load(raw_path, allow_pickle=False) as arrays:
                if bool(arrays["s02_corrections_applied"].item()):
                    raise ValueError("action depth unexpectedly contains S02 correction")
                raw = (
                    cast(Float32Array, np.asarray(arrays["depth_m"]).copy()),
                    cast(Float32Array, np.asarray(arrays["confidence"]).copy()),
                    cast(UInt8Array, np.asarray(arrays["processed_images_rgb"]).copy()),
                )
            raw_cache[prediction.raw_prediction_ref] = raw
        depth_all, confidence_all, images_all = raw

        masks = mask_cache.get(aligned_record.aligned_mask_artifact_ref)
        if masks is None:
            mask_path = _resolve(
                project_root, Path(aligned_record.aligned_mask_artifact_ref)
            )
            _require_hash(mask_path, aligned_record.aligned_mask_artifact_sha256)
            with np.load(mask_path, allow_pickle=False) as arrays:
                if bool(arrays["localization_performed"].item()):
                    raise ValueError("aligned mask unexpectedly records localization")
                masks = cast(UInt8Array, np.asarray(arrays["masks"]).copy())
            mask_cache[aligned_record.aligned_mask_artifact_ref] = masks

        camera_index = CAMERA_INDEX[aligned_record.camera_id]
        source_mask = masks[aligned_record.aligned_mask_index]
        depth = depth_all[camera_index]
        confidence = confidence_all[camera_index]
        image = images_all[camera_index]
        if source_mask.shape != depth.shape or image.shape[:2] != depth.shape:
            raise ValueError("aligned mask, depth, confidence, and RGB grids differ")
        candidates = build_mask_depth_candidates(
            source_mask,
            depth,
            target=aligned_record.target,
            config=config,
        )
        phase_id = prediction.job.phase_id
        payloads = compute_mask_depth_diagnostics(
            mask=source_mask,
            depth_m=depth,
            confidence=confidence,
            target=aligned_record.target,
            config=config,
        )
        identity = {
            "action_depth_job_id": aligned_record.action_depth_job_id,
            "bundle_id": aligned_record.bundle_id,
            "camera_id": aligned_record.camera_id,
            "frame_id": aligned_record.frame_id,
            "source_frame_index": aligned_record.source_frame_index,
            "phase_id": phase_id,
            "target": aligned_record.target.value,
            "perception_job_id": aligned_record.perception_job_id,
            "camera_local_track_id": aligned_record.camera_local_track_id,
            "aligned_mask_artifact_ref": aligned_record.aligned_mask_artifact_ref,
            "aligned_mask_artifact_sha256": (
                aligned_record.aligned_mask_artifact_sha256
            ),
            "aligned_mask_index": aligned_record.aligned_mask_index,
            "raw_prediction_ref": prediction.raw_prediction_ref,
            "raw_prediction_sha256": prediction.raw_prediction_sha256,
        }
        mask_records = [
            MaskDepthDiagnosticRecord.model_validate(identity | payload)
            for payload in payloads
        ]
        records.extend(mask_records)

        diagnostic_path = diagnostics_dir / (
            f"{mask_index:02d}_{aligned_record.source_frame_index:04d}_"
            f"{aligned_record.camera_id}_{aligned_record.target.value}.png"
        )
        _save_mask_diagnostic(
            image=image,
            source_mask=source_mask,
            depth=depth,
            confidence=confidence,
            candidates=candidates,
            records=mask_records,
            phase_id=phase_id,
            camera_id=aligned_record.camera_id,
            target=aligned_record.target,
            path=diagnostic_path,
        )
        diagnostic_artifacts.append(
            {
                "action_depth_job_id": aligned_record.action_depth_job_id,
                "camera_id": aligned_record.camera_id,
                "target": aligned_record.target.value,
                "diagnostic_ref": _relative(diagnostic_path, project_root),
                "diagnostic_sha256": _sha256(diagnostic_path),
            }
        )
        contact_items.append((aligned_record, image, source_mask))

    csv_path = output_dir / "mask_depth_comparison.csv"
    _write_comparison_csv(records, csv_path)
    strategy_path = output_dir / "strategy_comparison.png"
    _save_strategy_comparison(records, strategy_path)
    contact_path = output_dir / "mask_depth_contact_sheet.png"
    _save_contact_sheet(contact_items, contact_path)

    summary = MaskDepthDiagnosticRunSummary(
        schema_version=1,
        status="completed_pending_policy_selection",
        stage="S04",
        created_at_utc=datetime.now(UTC),
        source_mask_alignment_summary_ref=_relative(alignment_path, project_root),
        source_mask_alignment_summary_sha256=_sha256(alignment_path),
        source_action_depth_summary_ref=_relative(action_path, project_root),
        source_action_depth_summary_sha256=_sha256(action_path),
        configuration=config,
        records=tuple(records),
        comparison_csv_ref=_relative(csv_path, project_root),
        comparison_csv_sha256=_sha256(csv_path),
        strategy_comparison_ref=_relative(strategy_path, project_root),
        strategy_comparison_sha256=_sha256(strategy_path),
        contact_sheet_ref=_relative(contact_path, project_root),
        contact_sheet_sha256=_sha256(contact_path),
        per_mask_diagnostics=tuple(diagnostic_artifacts),
        limitations=(
            "Diagnostics use eight selected synchronized action pairs, not every S03 frame.",
            "Camera-space ray depth is not directly comparable across camera viewpoints.",
            "No back-projection, XYZ, anchor, fusion, smoothing, or temporal filling occurs.",
        ),
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary.status,
                "mask_count": len(alignment.aligned_masks),
                "diagnostic_record_count": len(records),
                "strategy_counts": {
                    strategy.value: sum(record.strategy is strategy for record in records)
                    for strategy in MaskDepthStrategy
                },
                "summary": _relative(summary_path, project_root),
            },
            indent=2,
        )
    )
    return 0


def _save_mask_diagnostic(
    *,
    image: UInt8Array,
    source_mask: UInt8Array,
    depth: Float32Array,
    confidence: Float32Array,
    candidates: tuple[CandidateMask, ...],
    records: list[MaskDepthDiagnosticRecord],
    phase_id: str,
    camera_id: str,
    target: PerceptionTarget,
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    axes[0, 0].imshow(image)
    axes[0, 0].contour(source_mask, levels=[0.5], colors=["#ffffff"], linewidths=1)
    axes[0, 0].set_title("processed RGB + source mask")

    axes[0, 1].imshow(image)
    for candidate in candidates:
        axes[0, 1].contour(
            candidate.mask,
            levels=[0.5],
            colors=[STRATEGY_COLORS[candidate.strategy]],
            linewidths=1,
        )
    axes[0, 1].set_title("candidate outlines")

    masked_depth = np.where(source_mask > 0, depth, np.nan)
    depth_values = depth[source_mask > 0]
    depth_image = axes[0, 2].imshow(
        masked_depth,
        cmap="magma_r",
        vmin=float(np.percentile(depth_values, 2)),
        vmax=float(np.percentile(depth_values, 98)),
    )
    figure.colorbar(depth_image, ax=axes[0, 2], fraction=0.046)
    axes[0, 2].set_title("whole-mask depth (m)")

    masked_confidence = np.where(source_mask > 0, confidence, np.nan)
    confidence_values = confidence[source_mask > 0]
    confidence_image = axes[1, 0].imshow(
        masked_confidence,
        cmap="viridis",
        vmin=float(np.percentile(confidence_values, 2)),
        vmax=float(np.percentile(confidence_values, 98)),
    )
    figure.colorbar(confidence_image, ax=axes[1, 0], fraction=0.046)
    axes[1, 0].set_title("whole-mask confidence")

    for candidate in candidates:
        values = depth[candidate.mask]
        axes[1, 1].hist(
            values[np.isfinite(values) & (values > 0)],
            bins=35,
            histtype="step",
            linewidth=1.4,
            color=STRATEGY_COLORS[candidate.strategy],
            label=candidate.strategy.value,
        )
    axes[1, 1].set_title("candidate depth distributions")
    axes[1, 1].set_xlabel("ray depth (m)")
    axes[1, 1].legend(fontsize=7)

    for record in records:
        axes[1, 2].plot(
            [item.full_frame_percentile for item in record.confidence_sweep],
            [item.retained_fraction for item in record.confidence_sweep],
            marker="o",
            color=STRATEGY_COLORS[record.strategy],
            label=record.strategy.value,
        )
    axes[1, 2].set_ylim(-0.03, 1.03)
    axes[1, 2].set_xlabel("same-frame confidence percentile")
    axes[1, 2].set_ylabel("candidate retained fraction")
    axes[1, 2].set_title("confidence sweep")
    axes[1, 2].legend(fontsize=7)

    for axis in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]):
        axis.axis("off")
    figure.suptitle(f"{phase_id} · {camera_id} · {target.value}")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _write_comparison_csv(
    records: list[MaskDepthDiagnosticRecord], path: Path
) -> None:
    percentile_names = (20, 40, 60, 80)
    fieldnames = [
        "source_frame_index",
        "phase_id",
        "camera_id",
        "target",
        "strategy",
        "source_mask_pixel_count",
        "candidate_pixel_count",
        "candidate_fraction",
        "depth_median_m",
        "depth_mad_m",
        "depth_iqr_m",
        "depth_p05_m",
        "depth_p95_m",
        "confidence_median",
        *(f"retained_at_p{value}" for value in percentile_names),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            retention = {
                round(item.full_frame_percentile): item.retained_fraction
                for item in record.confidence_sweep
            }
            writer.writerow(
                {
                    "source_frame_index": record.source_frame_index,
                    "phase_id": record.phase_id,
                    "camera_id": record.camera_id,
                    "target": record.target.value,
                    "strategy": record.strategy.value,
                    "source_mask_pixel_count": record.source_mask_pixel_count,
                    "candidate_pixel_count": record.candidate_pixel_count,
                    "candidate_fraction": record.candidate_fraction_of_source_mask,
                    "depth_median_m": record.depth_m.median,
                    "depth_mad_m": record.depth_m.median_absolute_deviation,
                    "depth_iqr_m": record.depth_m.interquartile_range,
                    "depth_p05_m": record.depth_m.p05,
                    "depth_p95_m": record.depth_m.p95,
                    "confidence_median": record.confidence.median,
                    **{
                        f"retained_at_p{value}": retention[value]
                        for value in percentile_names
                    },
                }
            )


def _save_strategy_comparison(
    records: list[MaskDepthDiagnosticRecord], path: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    strategies = list(MaskDepthStrategy)
    for column, target in enumerate((PerceptionTarget.PERSON, PerceptionTarget.BACKPACK)):
        target_records = [record for record in records if record.target is target]
        available = [
            strategy
            for strategy in strategies
            if any(record.strategy is strategy for record in target_records)
        ]
        relative_mad = [
            [
                record.depth_m.median_absolute_deviation / record.depth_m.median
                for record in target_records
                if record.strategy is strategy
            ]
            for strategy in available
        ]
        retention_p60 = [
            [
                next(
                    item.retained_fraction
                    for item in record.confidence_sweep
                    if item.full_frame_percentile == 60
                )
                for record in target_records
                if record.strategy is strategy
            ]
            for strategy in available
        ]
        axes[0, column].boxplot(relative_mad, tick_labels=[item.value for item in available])
        axes[0, column].set_title(f"{target.value}: relative depth MAD")
        axes[0, column].tick_params(axis="x", rotation=20)
        axes[0, column].set_ylabel("MAD / median depth")
        axes[1, column].boxplot(
            retention_p60,
            tick_labels=[item.value for item in available],
        )
        axes[1, column].set_title(
            f"{target.value}: retention at same-frame confidence p60"
        )
        axes[1, column].tick_params(axis="x", rotation=20)
        axes[1, column].set_ylabel("retained fraction")
        axes[1, column].set_ylim(0, 1.02)
    figure.suptitle("S04 raw in-mask DA3 sampling comparison (no XYZ)")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_contact_sheet(
    items: list[tuple[AlignedMaskRecord, UInt8Array, UInt8Array]], path: Path
) -> None:
    columns = 4
    rows = (len(items) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(16, rows * 2.4), constrained_layout=True)
    flat_axes = np.asarray(axes).reshape(-1)
    for axis, (record, image, mask) in zip(flat_axes, items, strict=False):
        axis.imshow(image)
        color = "#00dcff" if record.target is PerceptionTarget.PERSON else "#ff28a0"
        axis.contour(mask, levels=[0.5], colors=[color], linewidths=1.2)
        axis.set_title(
            f"f{record.source_frame_index} · {record.camera_id}\n{record.target.value}",
            fontsize=8,
        )
        axis.axis("off")
    for axis in flat_axes[len(items) :]:
        axis.axis("off")
    figure.suptitle("S04 masks evaluated for raw depth/confidence diagnostics")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _require_hash(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"artifact hash changed for {path}: {actual} != {expected}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _relative(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
