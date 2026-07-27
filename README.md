# 3D Spatial Reconstruction R&D

Exploratory, DA3-centred prototype for converting two synchronized,
overlapping living-room recordings into a metric 3D scene, person and backpack
tracks, pickup-carry-place events, and a synchronized Rerun recording.

Public repository:
https://github.com/hiverlabzhengjie/3d-spatial-reconstruction-rd

The canonical project scope and current stage are documented in
`docs/PROJECT_BRIEF.md`, `docs/ROADMAP.md`, and `docs/STATUS.md`.

## Local development

This project uses native macOS Python 3.11 so PyTorch can access Apple MPS.
Docker is not used for model inference.

```text
uv sync --locked
source .venv/bin/activate
python -c "import spatial_reconstruction"
```

The DA3 source checkout is an unmodified vendor dependency at
`Depth-Anything-3-main/`.

## Development checks

```text
uv run pytest
uv run ruff check src tests
uv run mypy src tests
```

## Version control

Git tracks project-owned code, configuration, tests, decisions, and stage
records. Raw captures, model weights, generated artifacts, `.venv`, caches, and
the unmodified DA3 checkout remain local. See
`docs/VENDOR_DEPENDENCIES.md` for the vendor fingerprint.

Each completed stage receives a descriptive stage-close commit. Annotated tags
may identify important reproducible stage or experiment checkpoints. Stage
commits and tags are pushed to the public `origin` remote after verification.
