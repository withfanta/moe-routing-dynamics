# RMO-P0 — Routing Markov-Order Pilot: result

Exploratory. No formal verdict. **Post-hoc analysis of the already-observed EPD-P0
dataset — not independent confirmation of anything.**

## Question

Once the Layer-11 expert-selection pattern is known, does earlier expert-selection
history still improve prediction of Layer-12 router logits?

## Provenance

| item | value |
|---|---|
| source | `/home/h-li/work/expert_provenance_decomposition/artifacts/EPD_P0/` (closed, read-only) |
| EPD-P0 commit | `995d23e` |
| `data_manifest.json` sha256 | `bbbe06fcb451b217de3b03cf75d37921a4e19f555d95d92302b8898d8a3466ba` (asserted) |
| `merged_raw.npz` sha256 | `39c9b609d89d7fe568e88221b35bdcba6f686e4cec7af1ba1cdd76527d030442` |
| `shard_0.npz` sha256 | `690e9ffe2e624830df18356fbd2150ed6a47ba5c4f7f259b9ce7f67b6a55675c` |
| `shard_1.npz` sha256 | `42d05fe99010e4c6903de4b209f122ae5f58bf5d178ee0ca36d6606cd32ce31b` |
| `shard_2.npz` sha256 | `f9820d78864c029167f32199ae0be4023242876b8f187b080962a6254748cd30` |
| `shard_3.npz` sha256 | `590d974a2cd8a2644b59ea0f02e977493bd84c8f39b195e782285ec0bf800c0c` |
| split | FIT = 512 / TEST = 256, EPD-P0's own `is_test` mask, not resampled |
| new model inference | **none** — no model loaded, no GPU used |
| underlying capture | `allenai/OLMoE-1B-7B-0125` @ `9b0c1aa87e34a20052389dce1f0cf01da783f654`, by EPD-P0 |
| checks | 27 passed (protocol tests A–J) |

## Design

Input is the 64-d binary native Top-8 selection indicator per layer — identities only, no
probabilities, no logits, no activation content. Target is the native Layer-12 router
logits, per-sample centred then standardized with FIT statistics only.

Layer 11 is held fixed: `z_recent` is fitted **once** on FIT
(`StandardScaler -> PCA(16, randomized, random_state=20260920) -> StandardScaler`,
PCA evr 0.6981) and reused unchanged by all five models. Each older-history window gets
its own 16-d block under the identical recipe, fit on FIT only.

| window | older layers | older raw dim | PCA evr @16 |
|---|---|---|---|
| k=2 | 10 | 64 | 0.6819 |
| k=4 | 8, 9, 10 | 192 | 0.5583 |
| k=8 | 4–10 | 448 | 0.4765 |
| k=11 | 1–10 | 640 | 0.4055 |

Every probe is `Ridge(alpha=1.0)`, multi-output, on exactly 32 input dimensions. The k=1
baseline pads with `zeros(16)` so width never varies.

## Main table

| history available | final dim | TEST R² |
|---|---|---|
| L11 only | 32 | **+0.59879** |
| L10–11 | 32 | **+0.62320** |
| L8–11 | 32 | **+0.64705** |
| L4–11 | 32 | **+0.66265** |
| L1–11 | 32 | **+0.66544** |

## Cumulative gains over L11 only

| | value |
|---|---|
| G_2 | **+0.02441** |
| G_4 | **+0.04826** |
| G_8 | **+0.06387** |
| G_11 | **+0.06665** |

## Stepwise gains

| | value |
|---|---|
| S_2 (L10–11 − L11) | **+0.02441** |
| S_4 (L8–11 − L10–11) | **+0.02385** |
| S_8 (L4–11 − L8–11) | **+0.01561** |
| S_11 (L1–11 − L4–11) | **+0.00279** |

## Exploratory pattern: SHORT-HISTORY

Layer 10 and Layers 8–10 improve prediction beyond Layer 11 alone, but older history adds
little after that, under this frozen probe.

The curve is monotone (largest drop along it: 0.000) and the stepwise gains decay
steadily: +0.024, +0.024, +0.016, +0.003. Adding Layers 1–3 on top of Layers 4–11 moves
R² by less than 0.003.

### The label is close to a boundary

The decision between SHORT-HISTORY and LONGER-HISTORY-CANDIDATE rests on how much the
late windows add over the best recent window:

    max(R2_8, R2_11) − max(R2_2, R2_4) = 0.66544 − 0.64705 = +0.01839

against the descriptive 0.02 figure — a margin of 0.0016. Neither the threshold nor the
classifier was changed after this was seen. Read directionally, the honest statement is
that the gain from history older than Layers 8–10 is small but not clearly zero, and this
probe does not separate "small" from "none" at this sample size. The 0.02 figure is not a
preregistered significance threshold, and no significance test was run.

### What this says about the EPD-P0 signal

EPD-P0's `ID_PATH` R² of 0.66969 over Layers 1–11 is largely, but not entirely, local
persistence. Layer 11 alone already reaches +0.59879 here — about 89% of the level the
full path reaches under this probe — so most of that signal is available from the
immediately preceding layer. The remaining +0.067 is concentrated in Layers 8–10.

Two limits on reading that comparison too precisely: the +0.66969 figure came from a
32-dim compression of all 11 layers, whereas these models spend 16 dims on Layer 11 and
16 on older history, so the numbers are close but not the same estimator. And PCA
explained variance falls from 0.68 (one older layer) to 0.41 (ten older layers), so the
flat tail at k=8 → k=11 partly reflects a fixed 16-d budget spread over more layers
rather than an absence of information in Layers 1–3. That confound would, if anything,
understate deep history — it cannot manufacture the observed decay.

## Scientific limit

This tests predictive dependence beyond Layer 11 under a fixed linear low-dimensional
probe. It does **not** prove causal memory, a formal Markov order, that a recurrent
router will help, that storing old paths improves NLL, or that routing should use
long-term memory. The allowed conclusion is only that expert-routing trajectories show
some predictive structure beyond the immediately preceding layer under this frozen probe,
concentrated in the nearby layers.

Consistent with XEC-P0's NOT_PROMISING actionability result: predictable is not the same
as useful.

## Governance

No architecture, new dataset, new model inference, hyperparameter search, or rescue
modification was performed. EPD-P0 and all four earlier projects remain closed,
read-only, and unmodified. No method is derived from this result.

Commits: `3b70f14` frozen exploratory protocol → `07f0854` implementation and tests →
this result. Analysis run once at `07f0854`; not modified after R² values were seen.
