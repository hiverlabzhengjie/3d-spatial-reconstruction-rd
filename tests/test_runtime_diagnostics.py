from __future__ import annotations

from collections.abc import Iterator

import pytest

from spatial_reconstruction.contracts import RunOutcome
from spatial_reconstruction.runtime import (
    DeviceCapabilities,
    DeviceSelection,
    DeviceUnavailableError,
    PeakRSSMonitor,
    PhaseTimer,
    build_run_observation,
    sample_memory,
    select_device,
    synchronize_device,
)


class FakeMemorySource:
    def __init__(self, rss_values: Iterator[int] | None = None) -> None:
        self._rss_values = rss_values or iter((1_000,))

    def process_rss_bytes(self) -> int:
        return next(self._rss_values)

    def mps_allocated_bytes(self) -> int | None:
        return 200

    def mps_driver_allocated_bytes(self) -> int | None:
        return 300

    def mps_recommended_max_bytes(self) -> int | None:
        return 400


MPS_READY = DeviceCapabilities(
    mps_built=True,
    mps_available=True,
    cuda_available=False,
)


def test_select_device_uses_available_mps() -> None:
    selection = select_device(
        "mps",
        allow_cpu_fallback=False,
        capabilities=MPS_READY,
    )

    assert selection == DeviceSelection(requested="mps", actual="mps")
    assert selection.used_fallback is False


@pytest.mark.parametrize(
    ("capabilities", "message"),
    [
        (
            DeviceCapabilities(
                mps_built=False,
                mps_available=False,
                cuda_available=False,
            ),
            "not built with MPS",
        ),
        (
            DeviceCapabilities(
                mps_built=True,
                mps_available=False,
                cuda_available=False,
            ),
            "built but unavailable",
        ),
    ],
)
def test_select_device_reports_actionable_mps_failure(
    capabilities: DeviceCapabilities,
    message: str,
) -> None:
    with pytest.raises(DeviceUnavailableError, match=message):
        select_device(
            "mps",
            allow_cpu_fallback=False,
            capabilities=capabilities,
        )


def test_select_device_marks_explicit_cpu_fallback() -> None:
    selection = select_device(
        "mps",
        allow_cpu_fallback=True,
        capabilities=DeviceCapabilities(
            mps_built=True,
            mps_available=False,
            cuda_available=False,
        ),
    )

    assert selection.actual == "cpu"
    assert selection.used_fallback is True
    assert selection.fallback is not None
    assert selection.warnings == (selection.fallback,)


def test_explicit_cpu_selection_is_not_a_fallback() -> None:
    selection = select_device(
        "cpu",
        allow_cpu_fallback=False,
        capabilities=MPS_READY,
    )

    assert selection == DeviceSelection(requested="cpu", actual="cpu")


def test_synchronize_device_uses_only_selected_accelerator() -> None:
    calls: list[str] = []
    synchronizers = {
        "mps": lambda: calls.append("mps"),
        "cuda": lambda: calls.append("cuda"),
    }

    synchronize_device("cpu", synchronizers=synchronizers)
    synchronize_device("mps", synchronizers=synchronizers)

    assert calls == ["mps"]


def test_missing_accelerator_synchronizer_is_actionable() -> None:
    with pytest.raises(RuntimeError, match="no synchronizer"):
        synchronize_device("mps", synchronizers={})


def test_phase_timer_synchronizes_before_and_after_timing() -> None:
    calls: list[str] = []
    clock_values = iter((10.0, 10.25))
    timer = PhaseTimer(
        phase="cold_inference",
        device="mps",
        synchronizers={"mps": lambda: calls.append("sync")},
        clock=lambda: next(clock_values),
    )

    with timer:
        calls.append("operation")

    assert calls == ["sync", "operation", "sync"]
    assert timer.observation is not None
    assert timer.observation.phase == "cold_inference"
    assert timer.observation.seconds == pytest.approx(0.25)


def test_phase_timer_records_duration_when_operation_raises() -> None:
    clock_values = iter((1.0, 1.1))
    timer = PhaseTimer(
        phase="failed_inference",
        device="cpu",
        clock=lambda: next(clock_values),
    )

    with pytest.raises(ValueError, match="model failure"):
        with timer:
            raise ValueError("model failure")

    assert timer.observation is not None
    assert timer.observation.seconds == pytest.approx(0.1)


def test_memory_sample_includes_mps_and_peak_counters() -> None:
    observation = sample_memory(
        "after_inference",
        device="mps",
        source=FakeMemorySource(),
        process_peak_rss_bytes=1_500,
    )

    assert observation.process_rss_bytes == 1_000
    assert observation.process_peak_rss_bytes == 1_500
    assert observation.mps_allocated_bytes == 200
    assert observation.mps_driver_allocated_bytes == 300
    assert observation.mps_recommended_max_bytes == 400


def test_cpu_memory_sample_does_not_claim_mps_counters() -> None:
    observation = sample_memory(
        "cpu_phase",
        device="cpu",
        source=FakeMemorySource(),
    )

    assert observation.process_rss_bytes == 1_000
    assert observation.mps_allocated_bytes is None
    assert observation.mps_driver_allocated_bytes is None
    assert observation.mps_recommended_max_bytes is None


def test_memory_sample_raises_reported_peak_to_current_rss() -> None:
    observation = sample_memory(
        "adjusted_peak",
        device="cpu",
        source=FakeMemorySource(),
        process_peak_rss_bytes=999,
    )

    assert observation.process_rss_bytes == 1_000
    assert observation.process_peak_rss_bytes == 1_000


def test_peak_rss_monitor_retains_highest_polled_value() -> None:
    monitor = PeakRSSMonitor(
        source=FakeMemorySource(iter((100, 350, 200))),
        interval_seconds=1.0,
    )

    assert monitor.poll_once() == 100
    assert monitor.poll_once() == 350
    assert monitor.poll_once() == 350
    assert monitor.peak_rss_bytes == 350


@pytest.mark.parametrize(
    ("selection", "expected_device", "expects_fallback"),
    [
        (DeviceSelection(requested="mps", actual="mps"), "mps", False),
        (DeviceSelection(requested="cpu", actual="cpu"), "cpu", False),
        (
            DeviceSelection(
                requested="mps",
                actual="cpu",
                fallback="explicit MPS-to-CPU workaround",
                warnings=("explicit MPS-to-CPU workaround",),
            ),
            "cpu",
            True,
        ),
    ],
)
def test_no_model_summary_distinguishes_device_and_fallback(
    selection: DeviceSelection,
    expected_device: str,
    expects_fallback: bool,
) -> None:
    memory = sample_memory(
        "before_load",
        device=selection.actual,
        source=FakeMemorySource(),
    )
    timer = PhaseTimer(
        phase="diagnostic",
        device="cpu",
        clock=iter((2.0, 2.01)).__next__,
    )
    with timer:
        pass
    assert timer.observation is not None

    summary = build_run_observation(
        model_id="diagnostic/no-model",
        model_revision=None,
        selection=selection,
        precision="unknown",
        input_description="no-model WP4 diagnostic",
        timings=(timer.observation,),
        memory=(memory,),
        outcome=RunOutcome.PASSED,
    )

    assert summary.device == expected_device
    assert (summary.fallback is not None) is expects_fallback
    assert summary.model_validate_json(summary.model_dump_json()) == summary


def test_no_model_summary_records_device_selection_failure_without_actual_device() -> None:
    error = "MPS is built but unavailable to this process"
    summary = build_run_observation(
        model_id="diagnostic/no-model",
        model_revision=None,
        selection=None,
        requested_device="mps",
        device_selection_error=error,
        precision="unknown",
        input_description="no-model WP4 diagnostic",
        timings=(),
        memory=(),
        outcome=RunOutcome.FAILED,
        warnings=(error,),
    )

    assert summary.requested_device == "mps"
    assert summary.device is None
    assert summary.device_selection_error == error
    assert summary.outcome is RunOutcome.FAILED
    assert summary.model_validate_json(summary.model_dump_json()) == summary
