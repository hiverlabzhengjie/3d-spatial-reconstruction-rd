"""Emit a no-model runtime diagnostic using the shared S00 WP4 utilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from spatial_reconstruction.config import load_project_config
from spatial_reconstruction.contracts import (
    MemoryObservation,
    ModelRunObservation,
    RunOutcome,
    TimingObservation,
)
from spatial_reconstruction.runtime import (
    DeviceName,
    DeviceUnavailableError,
    PeakRSSMonitor,
    PhaseTimer,
    SystemMemorySource,
    build_run_observation,
    sample_memory,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Explicitly permit and record CPU fallback when MPS is unavailable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path; the summary is always printed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_project_config()
    requested = config.runtime.preferred_device

    try:
        selection = select_device(
            requested,
            allow_cpu_fallback=args.allow_cpu_fallback,
        )
    except DeviceUnavailableError as exc:
        summary = build_run_observation(
            model_id="diagnostic/no-model",
            model_revision=None,
            selection=None,
            requested_device=requested,
            device_selection_error=str(exc),
            precision="unknown",
            input_description="WP4 no-model runtime diagnostic",
            timings=(),
            memory=(),
            outcome=RunOutcome.FAILED,
            warnings=(str(exc),),
        )
        _emit(summary, args.output)
        return 1

    source = SystemMemorySource()
    timings: list[TimingObservation] = []
    memory: list[MemoryObservation] = [
        sample_memory("before_operation", device=selection.actual, source=source)
    ]

    try:
        with PeakRSSMonitor(source=source) as monitor:
            _run_tensor_sum("cold_operation", selection.actual, timings)
            memory.append(
                sample_memory(
                    "after_cold_operation",
                    device=selection.actual,
                    source=source,
                    process_peak_rss_bytes=monitor.peak_rss_bytes,
                )
            )
            _run_tensor_sum("warm_operation", selection.actual, timings)
        memory.append(
            sample_memory(
                "after_warm_operation",
                device=selection.actual,
                source=source,
                process_peak_rss_bytes=monitor.peak_rss_bytes,
            )
        )
    except Exception as exc:
        summary = build_run_observation(
            model_id="diagnostic/no-model",
            model_revision=None,
            selection=selection,
            precision="float32",
            input_description="two 16x16 tensor sums",
            timings=timings,
            memory=memory,
            outcome=RunOutcome.FAILED,
            warnings=(f"{type(exc).__name__}: {exc}",),
        )
        _emit(summary, args.output)
        return 1

    summary = build_run_observation(
        model_id="diagnostic/no-model",
        model_revision=None,
        selection=selection,
        precision="float32",
        input_description="two 16x16 tensor sums",
        timings=timings,
        memory=memory,
        outcome=RunOutcome.PASSED,
    )
    _emit(summary, args.output)
    return 0


def _run_tensor_sum(
    phase: str,
    device: DeviceName,
    timings: list[TimingObservation],
) -> None:
    timer = PhaseTimer(phase=phase, device=device)
    with timer:
        result = torch.ones((16, 16), dtype=torch.float32, device=device).sum()
        if float(result.item()) != 256.0:
            raise RuntimeError("unexpected tensor diagnostic result")
    if timer.observation is None:
        raise RuntimeError("runtime timer did not emit an observation")
    timings.append(timer.observation)


def _emit(summary: ModelRunObservation, output: Path | None) -> None:
    payload = summary.model_dump_json(indent=2)
    print(payload)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{payload}\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
