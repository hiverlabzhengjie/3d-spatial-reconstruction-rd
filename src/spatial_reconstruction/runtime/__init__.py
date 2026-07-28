"""Shared runtime selection and diagnostic utilities."""

from spatial_reconstruction.runtime.diagnostics import (
    DeviceCapabilities,
    DeviceName,
    DeviceSelection,
    DeviceUnavailableError,
    MemorySource,
    PeakRSSMonitor,
    PhaseTimer,
    PrecisionName,
    SystemMemorySource,
    build_run_observation,
    sample_memory,
    select_device,
    synchronize_device,
)

__all__ = [
    "DeviceCapabilities",
    "DeviceName",
    "DeviceSelection",
    "DeviceUnavailableError",
    "MemorySource",
    "PeakRSSMonitor",
    "PhaseTimer",
    "PrecisionName",
    "SystemMemorySource",
    "build_run_observation",
    "sample_memory",
    "select_device",
    "synchronize_device",
]
