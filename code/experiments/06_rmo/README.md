# RMO-P0 — Routing Markov-Order Pilot

Tiny, CPU-only, exploratory follow-up. Asks whether the strong EPD-P0 `ID_PATH` signal
(R² = 0.66969 predicting Layer-12 router logits from the Layer 1–11 selection path) is
mostly trivial local routing persistence, or whether routing history from before the
immediately previous layer still adds incremental predictive information.

The scope, thresholds, and stopping rule are fixed in [protocol.md](protocol.md) and were
committed before any statistic existed.

## What this is not

Not a method experiment. No cache, no attention, no recurrence, no architecture. No new
model inference, no GPU, no new samples, no new dataset, no hyperparameter search. It
reuses already-extracted EPD-P0 data **read-only**.

## Data provenance

Every quantity comes from the closed, read-only artifact

    /home/h-li/work/expert_provenance_decomposition/artifacts/EPD_P0/merged_raw.npz

with `data_manifest.json` asserted to hash
`bbbe06fcb451b217de3b03cf75d37921a4e19f555d95d92302b8898d8a3466ba` before anything is
computed. The FIT = 512 / TEST = 256 split is EPD-P0's own stored `is_test` mask; nothing
is resampled. The underlying capture used `allenai/OLMoE-1B-7B-0125` revision
`9b0c1aa87e34a20052389dce1f0cf01da783f654`, frozen, FP16, unquantized, no gradients.

**This is a post-hoc analysis of an already-observed dataset. It is not independent
confirmation of anything.**

## Reused from EPD-P0

Re-derived here rather than imported, so this project has no dependency on EPD-P0 source
code and cannot modify it:

- the 64-d binary selection indicator `E_l[e] = 1 iff e in native Top-8 at layer l`
  (EPD-P0 `src/epd/representations.py: build_id_path`)
- the target transform: per-sample logit centring then FIT-only `StandardScaler`
  (EPD-P0 `center_logits` and `TargetScaler`)
- the compression shape `StandardScaler -> PCA(randomized) -> StandardScaler`, fit on FIT
  only (EPD-P0 `FrozenPipeline`), at 16 components with `random_state=20260920`

Source EPD-P0 commit: `995d23e` (exploratory result). That repository is untouched.

## Design

Layer 11 is held fixed at its own 16-d representation `z_recent`, fitted once on FIT and
reused by all five models. Four cumulative older-history windows are each compressed to
16 d. Every model is `Ridge(alpha=1.0)` on exactly 32 input dimensions:

| model | input | history available |
|---|---|---|
| k=1 | `[z_recent ; zeros(16)]` | L11 only |
| k=2 | `[z_recent ; z_old(2)]` | L10–11 |
| k=4 | `[z_recent ; z_old(4)]` | L8–11 |
| k=8 | `[z_recent ; z_old(8)]` | L4–11 |
| k=11 | `[z_recent ; z_old(11)]` | L1–11 |

The only thing that varies is how much earlier routing history is available.

## Layout

    README.md          this file
    protocol.md        frozen protocol
    analyze.py         the whole analysis, one run, CPU only
    test_analysis.py   tests A-J from the protocol
    artifacts/
      results.json     machine-readable result
      RESULTS.md       human-readable result

## Run

    /home/h-li/miniconda3/envs/sensorllm/bin/python -m pytest test_analysis.py -q
    /home/h-li/miniconda3/envs/sensorllm/bin/python analyze.py

The `sensorllm` env is used because it holds the sklearn 1.3.2 / numpy 1.24.4 versions
EPD-P0's own analysis ran under. `analyze.py` refuses to write results unless the tests
have passed in the same invocation chain (protocol test J).

## Result

See [artifacts/RESULTS.md](artifacts/RESULTS.md).

## Status

CLOSED. No method is derived from the outcome. Any further experiment requires explicit
user authorization.
