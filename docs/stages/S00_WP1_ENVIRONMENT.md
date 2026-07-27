# S00 WP1 Environment Record

**Work package:** WP1 - Environment and reproducibility
**Date:** 2026-07-27
**Result:** Complete

## Outcome

- Installed `uv` 0.11.32 with Homebrew.
- Created project-local `.venv` using CPython 3.11.15 for macOS arm64.
- Added `pyproject.toml` and resolved 123 packages across the main and optional
  dependency sets into `uv.lock`.
- Installed 101 packages into `.venv`; no model weights were downloaded.
- Added a minimal importable `spatial_reconstruction` package.
- Configured VS Code to use `.venv/bin/python` and the DA3 vendor source path.
- Added the initial direct model/library licence record.
- Confirmed native Apple MPS availability.
- Kept `Depth-Anything-3-main/` as an unmodified project-root vendor
  dependency.

## Machine and Key Versions

| Item | Observed value |
|---|---|
| macOS | 26.5.1, build 25F80 |
| Architecture | arm64 |
| Python | CPython 3.11.15 |
| uv | 0.11.32 |
| PyTorch | 2.13.0 |
| torchvision | 0.28.0 |
| Transformers | 5.14.1 |
| Ultralytics | 8.4.107 |
| OpenCV | 4.11.0 |
| PyAV | 16.1.0 |
| Pydantic | 2.13.4 |
| psutil | 7.2.2 |

Exact transitive versions are recorded in `uv.lock`.

## Reproduction

From the project root:

```text
brew install uv
uv venv --python 3.11
uv sync --locked
source .venv/bin/activate
python -c "import spatial_reconstruction"
```

VS Code should select `.venv/bin/python` automatically from
`.vscode/settings.json`.

## Verification Performed

```text
uv lock --check
uv sync --check
.venv/bin/ruff check src
.venv/bin/mypy src
```

Results:

- Lockfile current: pass.
- Environment synchronized with the lockfile: pass.
- Project import: pass, version `0.1.0`.
- Ruff: pass.
- mypy strict mode: pass.
- PyTorch reports MPS built and available in a native process: pass.
- Transformers exposes `AutoModelForMultimodalLM` and `AutoProcessor`: pass.
- Ultralytics, OpenCV, PyAV, Pydantic, and psutil imports: pass.
- DA3 API import from `Depth-Anything-3-main/src`: pass with the optional
  COLMAP module isolated as described below.

No automated project tests or model inference were run; those belong to later
S00 work packages.

## Compatibility Observation

The vendor DA3 API imports all exporters at module import time, including its
COLMAP exporter. The macOS `pycolmap` wheel and PyTorch each bundle
`libomp.dylib`, causing a duplicate OpenMP runtime abort when both load.

Under D015, COLMAP is allowed when it provides a concrete benefit. PyCOLMAP is
therefore locked as the optional `colmap` extra but intentionally absent from
the main MPS environment. A native verification confirmed that the unmodified
DA3 API imports successfully when the future project-owned adapter isolates
the exporter module. WP5 must implement and test this narrow compatibility
boundary; it must not set the unsafe
`KMP_DUPLICATE_LIB_OK` workaround or edit vendor source.

If a later work package justifies COLMAP, run PyCOLMAP or the COLMAP CLI in a
separate process/environment and exchange schema-validated files. Highlight
the intended use to the user before activation. This preserves both methods
without loading their conflicting OpenMP runtimes into one process.

The sandboxed process could see that PyTorch was built with MPS but reported
MPS unavailable. A native process outside that sandbox reported both
`is_built()` and `is_available()` as true. Model smoke tests therefore need the
native execution permission used for actual Apple GPU access.

## Scope Notes

- Docker Desktop is not used for model inference because Linux containers do
  not expose Apple MPS.
- Ollama does not replace the exact approved model integrations.
- COLMAP and other supporting methods are permitted under D015; none was
  needed or executed in WP1.
- Open3D, Rerun, optional DA3 Gaussian tooling, model weights, and inference
  artifacts were not added in WP1.
- The removed `Glossary/` folder contained two PDFs in addition to the DA3
  checkout. The user explicitly requested its deletion; DA3 was preserved at
  the project root.

## Exact Next Action

Begin WP2 by adding `configs/default.yaml`, validated configuration loading,
and the minimal typed core contracts from the S00 implementation brief.
