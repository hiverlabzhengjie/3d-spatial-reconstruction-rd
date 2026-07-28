# S00 WP7 - Qwen3-VL Multi-Frame MPS Gate

**Date:** 2026-07-28

**Status:** Complete

**Roadmap gate:** Gate C passed

## Purpose and Boundary

WP7 proves that exactly `Qwen/Qwen3-VL-2B-Instruct` accepts multiple ordered
frames and returns bounded non-empty text on Apple MPS. The adapter is
asynchronous so later semantic work need not block geometry processing.

This is an independent model/API compatibility gate. It does not implement the
S05 event schema, trigger logic, repair attempt, state machine, or `unknown`
fallback. It cannot accept or write coordinates, track identities, timestamps,
or zone membership. No optional methodology, Ollama model, Docker runtime, or
prior-generation Qwen model was used.

## Input and Sampling

The approved vendor fallback was used because a user action sample is not
required for the S00 compatibility gate:

| Item | Recorded value |
|---|---|
| Source | `Depth-Anything-3-main/assets/examples/robot_unitree.mp4` |
| SHA-256 | `99bc274f7613a665c6135085fe01691ebfaa9033101319071f37c550ab21d1ea` |
| Size | 1,964,268 bytes |
| Dimensions | 1024 x 576 |
| Average frame rate | 49.435028 fps |
| Decodable frames | 174 |
| Selected frame indices | 0, 58, 115, 173 |
| Selected timestamps | 0.00, 1.16, 2.30, 3.46 seconds |

The MP4 stream header reports 175 frames, while PyAV decodes 174. An initial
preflight caught this discrepancy. The final sampler performs a count-only
decode pass and samples uniformly against the authoritative decodable count,
then performs a second pass to extract only the four selected frames.

The video hash was identical before and after the run. Extracted frames were
written below the ignored S00 artifact directory; the source video was not
modified.

## Project-Owned Implementation

WP7 added:

- a strict `Qwen3VLAdapter` that accepts only
  `Qwen/Qwen3-VL-2B-Instruct`;
- deterministic endpoint-inclusive uniform frame sampling for four through
  eight frames;
- source-video hash preservation and ordered frame metadata;
- one ordered multi-image message with a concise factual prompt;
- deterministic decoding with `do_sample=False` and a 64-token output bound;
- response validation for one non-empty sequence within the token bound;
- retained raw output token IDs and processor tensor shapes;
- an `async` generation interface implemented through an isolated worker
  thread;
- an isolated `scripts/smoke/qwen_multiframe.py` command;
- tests for ordering, bounds, missing/empty output, exact model identity, and
  the absence of spatial fields from the response contract.

The prompt explicitly asks for directly visible action, permits `unknown`, and
forbids inferred coordinates, identities, timestamps, or zone membership.
Frame timestamps are retained only in the external prompt manifest for input
provenance; they are not passed through the semantic response interface.

## Exact Model and Runtime

| Item | Recorded value |
|---|---|
| Model identifier | `Qwen/Qwen3-VL-2B-Instruct` |
| Hugging Face revision | `89644892e4d85e24eaac8bacfd4f463576704203` |
| Transformers | 5.14.1 |
| PyTorch | 2.13.0 |
| Device | Apple MPS |
| Actual parameter precision | float16 |
| Selected frames | 4 |
| Maximum new tokens | 64 |
| Decoding | deterministic, no sampling |
| CPU fallback | none |

The exact snapshot occupies approximately 4.0 GB in the ignored
`.cache/huggingface/` directory. The model card records Apache-2.0, already
listed in `docs/licences/MODEL_AND_LIBRARY_LICENCES.md`.

## Native MPS Result

The final cold and warm generations both produced 41 tokens and identical
decoded text:

> The robot begins in a standing position. It then performs a dynamic movement,
> leaping into the air and rotating mid-air. After the rotation, it lands back
> on the ground in a standing position.

Visual inspection of the sampled frames supports the ordered description: the
robot begins upright, is airborne and rotating in the middle of the sequence,
and is upright again in the last frame. This is useful diagnostic evidence, not
a semantic-accuracy benchmark.

Processor diagnostics for each request:

- input tokens: 2,394;
- output tokens: 41;
- `image_grid_thw`: `[4, 3]`;
- `pixel_values`: `[9216, 1536]`;
- non-empty response: yes;
- within configured token bound: yes.

## Timing and Memory Observations

WP4 synchronized phase timing is the authoritative end-to-end measurement:

| Phase | Time (s) | Process RSS after phase | Observed peak RSS | MPS allocated | MPS driver allocated |
|---|---:|---:|---:|---:|---:|
| Model load | 10.179744 | 1,538,310,144 | 8,092,827,648 | 4,255,064,832 | 4,974,854,144 |
| Cold generation | 15.147281 | 1,802,059,776 | 1,844,068,352 | 4,255,065,088 | 8,139,948,032 |
| Warm generation | 6.762316 | 1,802,682,368 | 1,802,797,056 | 4,255,065,088 | 9,213,689,856 |

PyTorch reported an MPS recommended maximum working-set size of
55,662,788,608 bytes. The high model-load process peak is a transient CPU-side
checkpoint-loading observation. These values describe one local run and are
not a concurrent-model capacity guarantee.

## Attempts and Fallbacks

1. A restricted preflight correctly failed because MPS was unavailable to that
   sandboxed process. It made no CPU fallback.
2. The first native attempt was interrupted during the unauthenticated Hub
   checkpoint transfer and left a resumable partial cache plus extracted
   frames, but no model result or completion summary.
3. The second native attempt resumed the exact snapshot, loaded it on MPS, and
   passed both generations.

No model, device, precision, frame-count, or token-count fallback was used for
the passing run.

## Retained Artifacts

Final run directory:
`artifacts/s00/qwen/wp7_20260728_native_2/`

| Artifact | SHA-256 |
|---|---|
| `cold_response.txt` | `71729c08e5f5f39028f93ddad66a4effedbc3192e47d84dcd01c64db9c376ecd` |
| `warm_response.txt` | `71729c08e5f5f39028f93ddad66a4effedbc3192e47d84dcd01c64db9c376ecd` |
| `raw_responses.json` | `439727b09b5e398c7ef9e94174307388ce291783cc3f3b3d7a6fba02e059a4c8` |
| `prompt_manifest.json` | `b9853b30e49b29aba6bca5d15dea3ab11e30383b4994a7f0035f69cb75180fb3` |
| `manifest.json` | `98dd398f1b8fb3f173b525a9ac95b803a70dc3239a494d99543b682e17f9bbc2` |
| `summary.json` | `87695b3e65972e8a974f2834c2a64d3bb77c1c53c22fd8f4eaa1b44bd9e2387e` |
| `frames/frame_00.jpg` | `17a1a695c3aa8f4081e8b1539c05bd4d252172505b0358dabd5f31c2c0ce8056` |
| `frames/frame_01.jpg` | `221737906761a7c0cb33c6d9e18090b32abc73b5f916f22c6b9a5e16d456cae0` |
| `frames/frame_02.jpg` | `5b6c3e5e18850adce49030fae18e2e4cd813c940cca73c6fbb5add974170d4bb` |
| `frames/frame_03.jpg` | `475c31346bb4987f9658883524814850b7ee80e034953633235ee91d02c310f8` |

Model weights and generated artifacts remain ignored by Git.

## Reproduction Commands

From the project root:

```bash
uv run python scripts/smoke/qwen_multiframe.py \
  --output-dir artifacts/s00/qwen/wp7_20260728_native_2
```

The output directory must not already exist. For a repeat run, use a new
directory or omit `--output-dir` to obtain a timestamped path.

Automated verification:

```bash
uv run pytest -q
uv run ruff check src tests scripts
uv run mypy src scripts/diagnose_runtime.py \
  scripts/smoke/da3_two_view.py scripts/smoke/yolo_seg.py \
  scripts/smoke/qwen_multiframe.py
```

## Gate-C Verification

- [x] The exact Qwen3-VL 2B Instruct identifier and revision are recorded.
- [x] Four temporally ordered frames were accepted in one request.
- [x] Cold and warm responses are non-empty and within the token bound.
- [x] Prompt metadata, frame metadata, raw tokens, decoded text, timings, and
      memory observations are retained.
- [x] The passing run used actual MPS with no fallback.
- [x] The adapter is asynchronous and independent of geometry processing.
- [x] The adapter response cannot write coordinates, identity, timestamps, or
      zones.
- [x] The source video remained unchanged.

## Known Limits

- The vendor robot sequence proves multi-frame execution, not living-room
  pickup-carry-place interpretation.
- Four sparse frames can omit brief or ambiguous motion details.
- Semantic event schemas, one repair attempt, `unknown` fallback, and triggered
  living-room clips remain S05 work.
- The model must remain descriptive only; later spatial facts continue to come
  from calibrated geometry, perception, and state logic.

## Exact Next Action

Begin WP8 only: run the integrated S00 verification with DA3, YOLO, and Qwen in
separate processes; review all four completion gates; assemble the S00 handoff;
then create and push the stage-close commit only if every gate remains passed.
