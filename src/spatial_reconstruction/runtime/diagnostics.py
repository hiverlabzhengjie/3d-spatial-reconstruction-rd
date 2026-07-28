"""Device selection, synchronized timing, and memory observations."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Literal, Protocol, Self

import psutil  # type: ignore[import-untyped]
import torch

from spatial_reconstruction.contracts import (
    MemoryObservation,
    ModelRunObservation,
    RunOutcome,
    TimingObservation,
)

DeviceName = Literal["mps", "cpu", "cuda"]
PrecisionName = Literal["float32", "float16", "bfloat16", "mixed", "unknown"]
Clock = Callable[[], float]
Synchronizer = Callable[[], None]


class DeviceUnavailableError(RuntimeError):
    """Raised when the requested accelerator is unavailable without fallback."""


@dataclass(frozen=True, slots=True)
class DeviceCapabilities:
    """Relevant PyTorch accelerator capabilities for deterministic selection."""

    mps_built: bool
    mps_available: bool
    cuda_available: bool

    @classmethod
    def detect(cls) -> Self:
        """Read accelerator capabilities from the active PyTorch runtime."""

        return cls(
            mps_built=torch.backends.mps.is_built(),
            mps_available=torch.backends.mps.is_available(),
            cuda_available=torch.cuda.is_available(),
        )


@dataclass(frozen=True, slots=True)
class DeviceSelection:
    """Requested and actual device with explicit fallback provenance."""

    requested: DeviceName
    actual: DeviceName
    fallback: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def used_fallback(self) -> bool:
        """Whether selection changed the requested device."""

        return self.requested != self.actual


def select_device(
    preferred_device: DeviceName,
    *,
    allow_cpu_fallback: bool,
    capabilities: DeviceCapabilities | None = None,
) -> DeviceSelection:
    """Select a PyTorch device without silently substituting CPU."""

    detected = capabilities or DeviceCapabilities.detect()
    if preferred_device == "cpu":
        return DeviceSelection(requested="cpu", actual="cpu")

    if preferred_device == "mps":
        if detected.mps_built and detected.mps_available:
            return DeviceSelection(requested="mps", actual="mps")
        if not detected.mps_built:
            reason = "PyTorch was not built with MPS support"
        else:
            reason = "MPS is built but unavailable to this process"
        return _unavailable_or_cpu_fallback(
            requested="mps",
            reason=reason,
            allow_cpu_fallback=allow_cpu_fallback,
        )

    if detected.cuda_available:
        return DeviceSelection(requested="cuda", actual="cuda")
    return _unavailable_or_cpu_fallback(
        requested="cuda",
        reason="CUDA is unavailable to this process",
        allow_cpu_fallback=allow_cpu_fallback,
    )


def _unavailable_or_cpu_fallback(
    *,
    requested: Literal["mps", "cuda"],
    reason: str,
    allow_cpu_fallback: bool,
) -> DeviceSelection:
    if not allow_cpu_fallback:
        raise DeviceUnavailableError(
            f"{reason}; requested device '{requested}' cannot be used. "
            "Enable an explicit CPU fallback only as a documented workaround."
        )

    fallback = f"{requested} unavailable; explicitly fell back to CPU: {reason}"
    return DeviceSelection(
        requested=requested,
        actual="cpu",
        fallback=fallback,
        warnings=(fallback,),
    )


def synchronize_device(
    device: DeviceName,
    *,
    synchronizers: Mapping[DeviceName, Synchronizer] | None = None,
) -> None:
    """Synchronize pending accelerator work; CPU requires no synchronization."""

    if device == "cpu":
        return
    available = (
        synchronizers
        if synchronizers is not None
        else {
            "mps": torch.mps.synchronize,
            "cuda": torch.cuda.synchronize,
        }
    )
    synchronize = available.get(device)
    if synchronize is None:
        raise RuntimeError(f"no synchronizer is configured for device '{device}'")
    synchronize()


@dataclass(slots=True)
class PhaseTimer:
    """Context manager that records one accelerator-synchronized phase."""

    phase: str
    device: DeviceName
    synchronizers: Mapping[DeviceName, Synchronizer] | None = None
    clock: Clock = time.perf_counter
    observation: TimingObservation | None = field(default=None, init=False)
    _started_at: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.phase.strip():
            raise ValueError("timing phase must not be empty")

    def __enter__(self) -> Self:
        synchronize_device(self.device, synchronizers=self.synchronizers)
        self._started_at = self.clock()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        del exc_type, exc_value, traceback
        synchronize_device(self.device, synchronizers=self.synchronizers)
        if self._started_at is None:
            raise RuntimeError("phase timer was not started")
        elapsed = self.clock() - self._started_at
        if not math.isfinite(elapsed) or elapsed < 0:
            raise RuntimeError("phase timer produced an invalid elapsed duration")
        self.observation = TimingObservation(phase=self.phase.strip(), seconds=elapsed)
        return False


class MemorySource(Protocol):
    """Injectable source for process and MPS memory counters."""

    def process_rss_bytes(self) -> int:
        """Return current resident-set size."""

    def mps_allocated_bytes(self) -> int | None:
        """Return tensor memory allocated by MPS when supported."""

    def mps_driver_allocated_bytes(self) -> int | None:
        """Return total MPS driver allocation when supported."""

    def mps_recommended_max_bytes(self) -> int | None:
        """Return the recommended MPS working-set maximum when supported."""


class SystemMemorySource:
    """Memory counters from the current process and active PyTorch runtime."""

    def __init__(self) -> None:
        self._process = psutil.Process()

    def process_rss_bytes(self) -> int:
        """Return current process RSS."""

        return int(self._process.memory_info().rss)

    def mps_allocated_bytes(self) -> int | None:
        """Return current MPS tensor allocation when the API is usable."""

        return _optional_non_negative_counter(torch.mps, "current_allocated_memory")

    def mps_driver_allocated_bytes(self) -> int | None:
        """Return current MPS driver allocation when the API is usable."""

        return _optional_non_negative_counter(torch.mps, "driver_allocated_memory")

    def mps_recommended_max_bytes(self) -> int | None:
        """Return PyTorch's recommended MPS working-set maximum when available."""

        return _optional_non_negative_counter(torch.mps, "recommended_max_memory")


def _optional_non_negative_counter(owner: object, name: str) -> int | None:
    counter = getattr(owner, name, None)
    if not callable(counter):
        return None
    try:
        value = int(counter())
    except (RuntimeError, TypeError, ValueError):
        return None
    return value if value >= 0 else None


def sample_memory(
    phase: str,
    *,
    device: DeviceName,
    source: MemorySource | None = None,
    process_peak_rss_bytes: int | None = None,
) -> MemoryObservation:
    """Capture one process/MPS memory observation for a named phase."""

    memory = source or SystemMemorySource()
    if device == "mps":
        allocated = memory.mps_allocated_bytes()
        driver_allocated = memory.mps_driver_allocated_bytes()
        recommended_max = memory.mps_recommended_max_bytes()
    else:
        allocated = None
        driver_allocated = None
        recommended_max = None

    process_rss_bytes = memory.process_rss_bytes()
    if process_peak_rss_bytes is not None:
        process_peak_rss_bytes = max(process_peak_rss_bytes, process_rss_bytes)
    return MemoryObservation(
        phase=phase,
        process_rss_bytes=process_rss_bytes,
        process_peak_rss_bytes=process_peak_rss_bytes,
        mps_allocated_bytes=allocated,
        mps_driver_allocated_bytes=driver_allocated,
        mps_recommended_max_bytes=recommended_max,
    )


class PeakRSSMonitor:
    """Poll process RSS on a short background interval and retain its peak."""

    def __init__(
        self,
        *,
        source: MemorySource | None = None,
        interval_seconds: float = 0.05,
    ) -> None:
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("RSS polling interval must be finite and positive")
        self._source = source or SystemMemorySource()
        self._interval_seconds = interval_seconds
        self._peak_rss_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def peak_rss_bytes(self) -> int:
        """Highest RSS observed so far."""

        with self._lock:
            return self._peak_rss_bytes

    def poll_once(self) -> int:
        """Read and retain one RSS sample; useful at explicit phase boundaries."""

        rss = self._source.process_rss_bytes()
        if rss < 0:
            raise RuntimeError("process RSS must not be negative")
        with self._lock:
            self._peak_rss_bytes = max(self._peak_rss_bytes, rss)
            return self._peak_rss_bytes

    def __enter__(self) -> Self:
        if self._thread is not None:
            raise RuntimeError("RSS monitor cannot be started more than once")
        self.poll_once()
        self._thread = threading.Thread(
            target=self._poll_until_stopped,
            name="peak-rss-monitor",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        del exc_type, exc_value, traceback
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.poll_once()
        return False

    def _poll_until_stopped(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self.poll_once()


def build_run_observation(
    *,
    model_id: str,
    model_revision: str | None,
    selection: DeviceSelection | None,
    requested_device: DeviceName | None = None,
    device_selection_error: str | None = None,
    precision: PrecisionName,
    input_description: str,
    timings: Sequence[TimingObservation],
    memory: Sequence[MemoryObservation],
    outcome: RunOutcome,
    warnings: Sequence[str] = (),
    artifact_refs: Sequence[str] = (),
) -> ModelRunObservation:
    """Build the shared smoke-summary contract with device provenance included."""

    if selection is None:
        if requested_device is None or device_selection_error is None:
            raise ValueError(
                "a missing device selection requires requested_device and device_selection_error"
            )
        actual_device = None
        fallback = None
        selection_warnings: tuple[str, ...] = ()
    else:
        if requested_device is not None or device_selection_error is not None:
            raise ValueError(
                "requested_device and device_selection_error apply only when selection failed"
            )
        requested_device = selection.requested
        actual_device = selection.actual
        fallback = selection.fallback
        selection_warnings = selection.warnings

    combined_warnings = selection_warnings + tuple(warnings)
    return ModelRunObservation(
        model_id=model_id,
        model_revision=model_revision,
        requested_device=requested_device,
        device=actual_device,
        device_selection_error=device_selection_error,
        precision=precision,
        input_description=input_description,
        timings=tuple(timings),
        memory=tuple(memory),
        outcome=outcome,
        warnings=combined_warnings,
        fallback=fallback,
        artifact_refs=tuple(artifact_refs),
    )
