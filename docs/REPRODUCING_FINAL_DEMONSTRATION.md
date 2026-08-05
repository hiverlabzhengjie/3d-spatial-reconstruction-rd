# Reproducing the Final Demonstration

These commands reproduce and verify the accepted Stage 07 presentation from
retained local inputs. They do not rerun YOLO, DA3, or Qwen; the final assembly
is intentionally bound to the already verified S02-S06 outputs by SHA-256.

## Required local state

- Project checkout at the Stage 07 close commit, using native arm64 Python
  3.11 and the project `.venv` resolved by `uv.lock`.
- FFmpeg and FFprobe available on `PATH`; the accepted machine used Homebrew
  FFmpeg 8.1.2_1.
- The unchanged synchronized `action_take_01` Camera A/B videos and all
  hash-bound S06 artifacts referenced by
  `artifacts/s07/final_run_contract_20260805/final_run_manifest.json`.
- Rerun support from the locked Python environment.
- Apple MPS and cached model weights are not required for retained-output
  assembly. They are required only to regenerate upstream model evidence.

Generated artifact directories are ignored by Git and must be transferred or
regenerated locally; a public source checkout alone does not contain private
recordings or model outputs.

## Verify the environment and accepted entry contract

From the project root:

```bash
uv --cache-dir .cache/uv lock --check
uv --cache-dir .cache/uv sync --check

.venv/bin/python scripts/s07/verify_final_run_manifest.py \
  --summary artifacts/s07/final_run_contract_20260805/summary.json \
  --output artifacts/s07/reproduction_<new-run-id>/entry_verification.json
```

The verifier refuses hash changes, missing/duplicate roles, another recording,
wrong frame counts, a weakened S06 gate, or a different S06 orchestration
identity.

## Build and independently verify seekable presentation videos

Use a fresh output directory because artifact-producing commands refuse to
overwrite an existing result:

```bash
.venv/bin/python scripts/s07/build_rerun_video_proxies.py \
  --final-run-summary artifacts/s07/final_run_contract_20260805/summary.json \
  --output-dir artifacts/s07/rerun_video_proxies_<new-run-id>

.venv/bin/python scripts/s07/verify_rerun_video_proxies.py \
  --manifest \
    artifacts/s07/rerun_video_proxies_<new-run-id>/proxy_manifest.json \
  --output \
    artifacts/s07/rerun_video_proxies_<new-run-id>/verification.json
```

Expected invariants are two 1920x1080, 30 FPS, H.264/yuv420p videos; 1,047
frames each; maximum keyframe gap 30; exact original source hashes; and
`raw_sources_modified=false`, `spatial_outputs_modified=false`.

## Assemble the interactive Rerun

```bash
.venv/bin/python scripts/s07/run_measured_final_assembly.py \
  --final-run-summary artifacts/s07/final_run_contract_20260805/summary.json \
  --presentation-video-manifest \
    artifacts/s07/rerun_video_proxies_<new-run-id>/proxy_manifest.json \
  --output-dir artifacts/s07/final_run_<new-run-id>
```

The generated directory contains the `.rrd`, stable export summary, exact
commands, stdout/stderr hashes, and measured retained-output assembly time.
Rerun serialization is not guaranteed byte-identical between exports; verify
stable semantics and the newly generated file's own bound hash.

## Open and inspect the recording

```bash
.venv/bin/rerun \
  artifacts/s07/final_run_<new-run-id>/digital_twin_stage07_final.rrd
```

If the native window fails on the current macOS display surface, use the local
web viewer:

```bash
.venv/bin/rerun --serve-web --bind 127.0.0.1 \
  artifacts/s07/final_run_<new-run-id>/digital_twin_stage07_final.rrd
```

Review all six presentation views. In particular:

1. scrub Camera A and Camera B at early, middle, and late times and confirm
   visible decoded frames;
2. confirm the metric scene, cameras, zones, and current states share the
   `capture_time` timeline;
3. confirm no trajectory is present at capture start, partial measured trails
   appear mid-run, and complete disconnected trails appear only at the end;
4. inspect the coordinate/provenance log for timestamped measured XYZ, explicit
   anchor kinds, display-only stale state, and absent missing/occluded XYZ;
5. confirm pickup, carry, and place, including the separate carry transition
   and semantic-review times.

Save representative screenshots as:

```text
visual_qa/final_camera_a.png
visual_qa/final_camera_b.png
visual_qa/final_metric_twin.png
visual_qa/final_coordinates.png
visual_qa/final_state_timeline.png
visual_qa/final_events.png
```

Then run the independent verifier:

```bash
.venv/bin/python scripts/s07/verify_measured_final_assembly.py \
  --summary artifacts/s07/final_run_<new-run-id>/summary.json \
  --visual-qa-dir artifacts/s07/final_run_<new-run-id>/visual_qa \
  --visual-qa-passed \
  --output artifacts/s07/final_run_<new-run-id>/verification.json
```

The explicit `--visual-qa-passed` flag records a human inspection; file
existence alone does not establish correct playback.

## Verify the Stage 07 close gate

```bash
.venv/bin/python scripts/s07/verify_stage07_gate.py \
  --entry artifacts/s07/stage_close_audit_<new-run-id>/entry.json \
  --proxies artifacts/s07/stage_close_audit_<new-run-id>/proxies.json \
  --rerun artifacts/s07/stage_close_audit_<new-run-id>/rerun.json \
  --output artifacts/s07/stage_close_audit_<new-run-id>/gate.json

.venv/bin/pytest -q
.venv/bin/ruff check .
MYPYPATH=src .venv/bin/mypy --strict src scripts
uv --cache-dir .cache/uv lock --check
uv --cache-dir .cache/uv sync --check
git diff --check
```

The accepted preferred file remains:

```text
artifacts/s07/final_run_v2_20260805/digital_twin_stage07_final.rrd
SHA-256 bcf84af987069151339427d57d7642cffd0e92b6c0ff05bbdbddb7c6143b64ca
```

For a new physical capture, begin with `docs/CAPTURE_CALIBRATION_GUIDE.md` and
re-run the applicable S01-S06 stages. Do not reuse `action_take_01` source,
synchronization, pose, or model-output identities for different footage.
