# S00 WP4 Runtime Diagnostics Record

**Work package:** WP4 - Shared device, timing, and memory diagnostics

**Date:** 2026-07-28

**Result:** Complete

## Outcome

- Added explicit MPS, CPU, and CUDA device selection without silent fallback.
- Added accelerator synchronization around named timing phases.
- Added current and peak process RSS observation.
- Added supported MPS allocated, driver-allocated, and recommended-maximum
  memory counters.
- Extended the shared model-run summary to distinguish requested device, actual
  device, explicit fallback, and failure before any device was selected.
- Added a reusable no-model command that emits the same JSON summary contract
  intended for the DA3, YOLO, and Qwen3-VL smoke checks.

The proposed separate timing and memory modules were consolidated into one
small `runtime/diagnostics.py` module to keep the operator-facing design
streamlined. The model adapters remain independent and replaceable.

No model weights were downloaded, and no model inference or DA3 vendor
modification was performed.

## Device Policy

`select_device` chooses MPS only when PyTorch reports both `is_built()` and
`is_available()`. An unavailable requested accelerator raises
`DeviceUnavailableError` with an actionable message unless CPU fallback was
explicitly allowed.

An explicit fallback records:

- the originally requested device;
- CPU as the actual device;
- a fallback explanation; and
- the same explanation as a warning.

A device-selection failure records the requested device and error while leaving
the actual device empty. It therefore cannot falsely claim an MPS or CPU run.

## Timing and Memory

`PhaseTimer` synchronizes the selected accelerator before starting and before
stopping its monotonic timer. CPU phases do not call an accelerator
synchronizer.

`PeakRSSMonitor` polls process RSS on a configurable short background interval.
Phase snapshots include current RSS and may include the observed peak. MPS
snapshots also query, when supported:

- current tensor allocation;
- driver allocation; and
- recommended maximum working-set memory.

Unsupported MPS counters remain `null`; they are not replaced with zero.

## Files

- `src/spatial_reconstruction/runtime/__init__.py`
- `src/spatial_reconstruction/runtime/diagnostics.py`
- `scripts/diagnose_runtime.py`
- `tests/test_runtime_diagnostics.py`
- `src/spatial_reconstruction/contracts.py` (shared summary extensions)
- `tests/test_contracts.py` (summary contract updates)

## Verification

Commands:

```text
.venv/bin/pytest -q tests/test_runtime_diagnostics.py
.venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/mypy src scripts/diagnose_runtime.py
uv lock --check
uv sync --check
.venv/bin/python scripts/diagnose_runtime.py
.venv/bin/python scripts/diagnose_runtime.py --allow-cpu-fallback
```

The no-fallback diagnostic was also run as a native process with Apple GPU
access, because the restricted implementation process cannot access MPS.

Results:

- WP4 diagnostic tests: 17 passed;
- complete project suite: 66 passed;
- Ruff: pass;
- strict mypy: pass across the package and diagnostic command;
- lockfile: current, with 102 installed packages requiring no changes;
- restricted no-fallback run: emitted a valid failed summary, reported MPS
  built but unavailable, and returned non-zero;
- restricted explicit-fallback run: passed on CPU and recorded the fallback
  and warning; and
- native run: passed on MPS without fallback.

The representative native MPS observation reported:

- cold 16-by-16 tensor operation: approximately 0.0473 seconds;
- warm operation: approximately 0.000542 seconds;
- process RSS: approximately 224 MB before and 311 MB after;
- peak process RSS: approximately 311 MB;
- MPS driver allocation after operation: 8,798,208 bytes; and
- recommended MPS maximum: 55,662,788,608 bytes.

These tiny no-model timings are only a functional diagnostic, not a model
performance estimate.

## Failure Found and Corrected

The first injected-synchronizer test exposed an `or` fallback that treated an
explicitly empty synchronizer map as absent and called the live sandboxed MPS
synchronizer. That test process terminated inside PyTorch MPS synchronization.
The implementation now distinguishes `None` from an explicit mapping, the
regression test passes, and the finalized native MPS command completes.

## Decisions

No new project decision was required. No additional methodology or optional
tool, including COLMAP, was used.

## Exact Next Action

Begin S00 WP5 only: implement the project-owned DA3 adapter and run its
independent pose-conditioned two-view MPS smoke check.
