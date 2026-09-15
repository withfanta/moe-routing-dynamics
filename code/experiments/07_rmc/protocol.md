# RMC-P0 — Routing Memory Cross-Model Pilot (frozen protocol)

Frozen and committed before any formal router prediction or R² value exists.

## 0. Single scientific question (user-locked)

In a pretrained sparse MoE different from OLMoE: after the immediately preceding MLP-MoE
routing state is already known, does earlier MLP expert-selection history still provide
incremental held-out information about the current MLP router state?

The object of study is **higher-order predictive routing dependence**. Not causal memory,
not routing correctness, not routing regret, not language-model improvement, not the
usefulness of recurrent routers.

## 1. Model and why

`jetmoe/jetmoe-8b`, **BASE** checkpoint (not SFT/chat). Architecturally distant from
OLMoE: 24 blocks, each holding both a Mixture of Attention heads (MoA) and a Mixture of
MLP Experts, with 8 experts and Top-2 per token, ~8B total / ~2.2B active parameters.

Confirmed from the pinned `config.json`: `num_hidden_layers` 24, `moe_num_experts` 8,
`moe_top_k` 2, `hidden_size` 2048, `ffn_hidden_size` 5632.

**Only the MLP expert router is studied. The Mixture-of-Attention router is never used.**

## 2. Pinned provenance

| item | value |
|---|---|
| checkpoint | `jetmoe/jetmoe-8b` |
| checkpoint commit SHA | `d8fd02ccf7911aa8148a63c7984ffd2e465b0352` |
| implementation | `transformers.models.jetmoe` (official, in-tree) |
| implementation version | transformers 4.45.1 |
| dataset | `Salesforce/wikitext`, `wikitext-103-raw-v1` |
| dataset revision | `b08601e04326c79dfdd32d625aee71d232d685c3` |

The official in-tree JetMoE implementation shipped with transformers 4.45.1 is used
verbatim. No model implementation is rewritten and no third-party JetMoE code is
installed. The resolved SHA is passed to every worker; floating `main` is never used
during the formal run. Package, torch, CUDA and Python versions are recorded in
`artifacts/environment.json`, and the implementation file's own sha256 is recorded in
`artifacts/model_provenance.json` so the exact executed code is pinned by content.

## 3. Hard anti-drift rule

Not permitted: OLMoE reruns; another model; another dataset; another target layer;
another history length; activation content; rejected experts; router probabilities as
historical features; attention-router history; routing regret; counterfactual routes; NLL
interventions; MLP probes; GRU; Transformer probe; nonlinear probe; alpha search; PCA
dimension search; multiple data seeds; subgroup analysis.

This is ONE cross-model replication. If the frozen result fails, the failure is recorded
and not rescued.

## 4. Data

`Salesforce/wikitext` / `wikitext-103-raw-v1`, TRAIN split only, at the pinned revision.
Tokenized with the JetMoE tokenizer. EOS inserted between original WikiText records.
Non-overlapping blocks of 129 tokens: positions 0..127 are context, position 128 is the
continuation token. The experimental token is **context position 127**; only its routing
is analyzed.

FIT = 512 blocks, TEST = 256 blocks, disjoint, sampling seed 20260920.

This is new model-specific tokenization. OLMoE token block indices are **not** reused —
no prior manifest is read. The manifest is frozen before formal statistics and its
sha256 recorded.

## 5. Compute

At most four free Tesla V100-SXM2-32GB, selected after inspecting `nvidia-smi`; physical
indices are never assumed. Four independent single-GPU extraction workers, no DDP and no
model parallelism. JetMoE-8B in FP16 is ~16 GB and should fit one 32 GB V100.

If a single V100 genuinely cannot load the pinned FP16 model, the run stops with
`TECHNICAL_MEMORY_BLOCKER`. No quantization, no silent model substitution, no
model-parallel redesign without explicit authorization.

768 samples in four deterministic shards of 192 (128 FIT + 64 TEST each).

Model: `eval()`, frozen, FP16, `torch.inference_mode()`, no parameter receives gradients.
Initial microbatch 8; on OOM only the microbatch changes, 8 → 4 → 2 → 1.

## 6. Router identification

Both submodules exist in every block and both use the same gating class, so the router is
identified structurally, never by guessing a name:

- `model.layers[i].mlp` is `JetMoeMoE` (the MLP mixture) and carries `.router`
- `model.layers[i].self_attention.experts` is `JetMoeMoA` (attention mixture) and also
  carries `.router`

Capture hooks attach to `model.layers[i].mlp.router` only. Note that the library's own
`output_router_logits=True` returns a flat tuple **interleaving attention and MLP router
logits** (`all_router_logits += (layer_outputs[-2], layer_outputs[-1])`); that path is
therefore not used, since it invites exactly the MoA/MoE confusion this section guards
against.

`JetMoeTopKGating.forward` computes `logits = self.layer(hidden_states).float()` and
selects `logits.topk(top_k)`. The captured logit vector is that raw fp32 `logits` and the
captured identities are that same topk, so identities and logits are consistent by
construction.

Test D proves the captured module is the MLP MoE and not the MoA: the hooked module must
be identical (`is`) to `layers[i].mlp.router`, must not be the MoA router, its parent must
be a `JetMoeMoE`, and its output width must be 8. If MLP-router identification cannot be
established unambiguously, the run stops with `ROUTER_IDENTIFICATION_BLOCKER`.

## 7. Capture

For every sample and every block `l = 1..24`, at the experimental token only: the native
Top-2 MLP expert IDs and the full native MLP router-logit vector (expected width 8).

Not captured: MLP expert outputs, hidden states, MoA routing, rejected-expert
activations, next-token NLL.

## 8. Target layers

Exactly two human-readable target blocks, fixed before results: **Layer 12** (middle
depth) and **Layer 20** (later depth). Both admit the same maximum tested history, k = 8
layers. No other targets.

## 9. Routing-state input

For every layer `l`, `S_l ∈ R^8` with `S_l[e] = 1` if MLP expert `e` is in the native
Top-2 for the experimental token, else 0. Every `S_l` contains exactly two ones. No rank
ordering, no router score, no probability — selected expert identity only.

## 10. Target router state

For target layer `m`, the native MLP router logits `g_m ∈ R^8`. Centre per sample,
`q_m = g_m − mean(g_m)`, then standardize each of the 8 target dimensions with FIT
statistics only, applying those unchanged to TEST.

## 11. History windows

Per target `m`, exactly k = 1, 2, 4, 8:

| k | layers used |
|---|---|
| 1 | `S_{m-1}` |
| 2 | `S_{m-2}, S_{m-1}` |
| 4 | `S_{m-4} … S_{m-1}` |
| 8 | `S_{m-8} … S_{m-1}` |

The primary scientific comparison is **k=1 versus k=4**. k=2 and k=8 are secondary shape
checks only.

## 12. Dimension matching

Every probe receives exactly 16 dimensions. The immediately preceding layer always gets a
fixed 8-d slot.

**Recent state.** `z_recent = StandardScaler(S_{m-1})` with FIT statistics only, 8 d. The
recent-layer scaler is fitted **once per target** and reused for every k.

**Older history.** For k > 1, concatenate all history strictly before `S_{m-1}`:
k=2 → `S_{m-2}` (raw 8); k=4 → `[S_{m-4}; S_{m-3}; S_{m-2}]` (raw 24);
k=8 → `[S_{m-8}; … ; S_{m-2}]` (raw 56). For each window independently, fit on FIT only:

    StandardScaler -> PCA(n_components=8, svd_solver="randomized", random_state=20260920) -> StandardScaler

giving `z_old(k)`, 8 d. For k=1, `z_old = zeros(8)`.

Final feature for every model: `[z_recent ; z_old]`, 16 d. All models therefore share an
identical recent routing state, final dimension, probe family, target, and FIT/TEST
samples. Only older routing history changes.

## 13. Probe

For each target in {12, 20} and each k in {1, 2, 4, 8}: `sklearn.linear_model.Ridge(alpha=1.0)`,
multi-output, mapping 16-d history to the 8-d centred standardized target router logits.
No alpha search, no alternative model. Eight probes total.

## 14. Metric

On TEST, `sklearn.metrics.r2_score(y_true, y_pred, multioutput="uniform_average")`, giving
`R2_m_1, R2_m_2, R2_m_4, R2_m_8` per target.

Primary: `Delta4_m = R2_m_4 − R2_m_1`.
Secondary: `Delta2_m = R2_m_2 − R2_m_1`, `Delta8_m = R2_m_8 − R2_m_1`.
Stepwise descriptive: `Step2_m = R2_m_2 − R2_m_1`, `Step4_m = R2_m_4 − R2_m_2`,
`Step8_m = R2_m_8 − R2_m_4`.

## 15. Paired bootstrap for the primary comparison

Per target independently: bootstrap the TEST sample **indices**, 10000 resamples, seed
314159. Each resample recomputes `R2(k=4) − R2(k=1)` on the same resampled indices for
both models, yielding a 95% percentile CI for `Delta4_m`. Predictions are never resampled
independently; indices are resampled jointly for both models. No bootstrap is required
for secondary metrics.

## 16. Pre-registered confirmation rule

REPLICATED only if **all six** hold:

Layer 12: (1) `R2_12_1 > 0`; (2) `Delta4_12 >= +0.02`; (3) 95% bootstrap CI lower bound
for `Delta4_12 > 0`.

Layer 20: (4) `R2_20_1 > 0`; (5) `Delta4_20 >= +0.02`; (6) 95% bootstrap CI lower bound
for `Delta4_20 > 0`.

Otherwise `NOT_REPLICATED`. There is no INCONCLUSIVE category for ordinary numerical
results. Technical blockers are separate and are reported as such.

Thresholds are frozen here and are not modified after results are seen.

## 17. What the rule means

The primary claim is deliberately narrow: routing history beyond the immediately
preceding layer contains incremental predictive information. k=8 is **not** required to
outperform k=4, so the exact memory-decay shape is not preregistered. The k=2/k=4/k=8
curve is descriptive; if replicated, diminishing returns may be compared qualitatively
with OLMoE, but that shape is never required for the verdict.

## 18. Required tests

A pinned JetMoE base checkpoint commit · B pinned JetMoE implementation commit ·
C model frozen/eval/FP16 · D captured router demonstrably the MLP MoE · E captured Top-2
IDs reproduce native MLP router selection · F each `S_l` exactly 8 dims with exactly two
ones · G MoA routing never used as input · H FIT = 512 / TEST = 256 and disjoint ·
I target layers exactly 12 and 20 · J target router logits native MLP only ·
K recent-layer preprocessing fitted once per target and reused · L older-history PCA uses
FIT only · M every final feature exactly 16 dims · N Ridge alpha = 1.0 for all eight
probes · O no TEST statistics enter preprocessing · P no formal R² generated before all
implementation checks pass.

## 19. Project structure

    README.md  protocol.md  extract.py  analyze.py  test_rmc.py
    artifacts/ environment.json model_provenance.json data_manifest.json
               shard_0.npz shard_1.npz shard_2.npz shard_3.npz results.json RESULTS.md

## 20. Preregistration order

Before formal extraction: resolve the immutable checkpoint SHA; pin the implementation;
build and freeze the data manifest; write this protocol; commit protocol + manifest. At
that commit no formal router predictions or R² values exist. Then implement
extraction/tests; run smoke tests; extract four shards; merge; run all eight probes once;
bootstrap the primary comparisons; apply the verdict mechanically; write the result; STOP.

## 21. Git

Three commits only, no speculative branch:

1. `RMC-P0 preregistration and frozen provenance`
2. `RMC-P0 implementation and tests`
3. `RMC-P0 result and closure`

## 22. Interpretation limit

If REPLICATED, the allowed conclusion is only: in two architecturally different
pretrained sparse MoEs, routing history beyond the immediately preceding layer contains
incremental held-out information about future MLP expert routing; combined with the prior
OLMoE exploratory result this supports cross-architecture short-range higher-order
routing dependence. Not claimed: formal Markov order, causal memory, long-term memory, a
recurrent router improving performance, better NLL, universal behavior across all MoEs.

If NOT_REPLICATED, the allowed conclusion is only: the OLMoE higher-order
routing-dependence observation did not replicate cleanly in JetMoE under the frozen
protocol. No rescue by changing target layers, PCA, dataset, model, or probe.

## 23. Relationship to prior work

RMO-P0 (OLMoE) was post-hoc exploratory on already-observed data and reported
SHORT-HISTORY. RMC-P0 is an independent extraction on a different model and different
tokenization, with a preregistered verdict rule. All previous projects are CLOSED and
READ-ONLY; none is read, modified, or reinterpreted here.

## 24. Governance

This experiment asks only: does predictive routing dependence beyond the immediately
preceding layer replicate in a substantially different sparse-MoE architecture? It does
not ask how to make routing memory useful. No method is derived from the outcome.
