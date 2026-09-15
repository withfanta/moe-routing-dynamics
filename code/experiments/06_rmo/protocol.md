# RMO-P0 — Routing Markov-Order Pilot (frozen exploratory protocol)

Frozen before any statistic exists. Written, committed, and only then implemented.

## 0. Scientific question (user-locked)

EPD-P0 found `ID_PATH -> Layer-12 router logits` at R² = 0.66969 using the binary
expert-selection path over Layers 1–11. That signal may be largely trivial local
persistence: adjacent MoE layers may simply route alike.

RMO-P0 asks exactly one question:

> Given the expert-selection pattern at Layer 11, does routing history from earlier
> layers still improve prediction of Layer-12 router logits?

In notation: does `E_1 … E_10` carry predictive information about the Layer-12 router
state beyond `E_11`, where `E_l` is the native Top-8 expert-selection identity pattern
at layer `l`?

This is exploratory direction-finding. It is **not** a method experiment.

## 1. Hard anti-drift rules

Not permitted, at any point:

new model inference; new samples; new dataset; new target layer; expert activations;
rejected experts; routing regret; counterfactual routes; MLP; GRU; attention; nonlinear
probes; alpha search; multiple PCA dimensions; additional seeds; subgroup analysis;
transition mining.

No GPU is used. OLMoE is not loaded. This is ONE small decomposition.

If the EPD artifact lacks Layer 1–11 selected expert IDs, FIT/TEST membership, or
Layer-12 router logits, the run stops with `DATA_NOT_AVAILABLE` and nothing is
re-extracted.

## 2. Data reuse — read-only

Source, read-only, never modified:

    /home/h-li/work/expert_provenance_decomposition/artifacts/EPD_P0/

EPD-P0 is CLOSED. Required manifest hash, asserted before any statistic:

    data_manifest.json  sha256 = bbbe06fcb451b217de3b03cf75d37921a4e19f555d95d92302b8898d8a3466ba

The merged artifact `merged_raw.npz` supplies every quantity. Its hash and the four
shard hashes are recorded in `artifacts/results.json`.

Split is EPD-P0's exactly: FIT = 512, TEST = 256, taken from the stored `is_test` mask.
No resampling.

**This is a post-hoc exploratory analysis on an already-observed dataset.** The report
must not call it independent confirmation.

## 3. Input representation

For every sample and every historical layer `l = 1..11`, the same 64-d binary
expert-selection vector EPD-P0 used:

    E_l[e] = 1  if expert e is in the native Top-8 at layer l, else 0

No router probabilities, no historical router logits, no activation content. Only
selected expert identities. Each `E_l` has exactly 8 ones.

## 4. Target

EPD-P0's target, reproduced exactly: native Layer-12 64-d router logits `g_i`.

1. Centre per sample: `q_i = g_i - mean(g_i)` over the 64 expert dimensions.
2. Standardize each target dimension with `StandardScaler` fit on FIT only.
3. Apply those FIT statistics unchanged to TEST.

## 5. Fixed Layer-11 representation

Layer 11 gets its own representation, fitted **once** on FIT only from `E_11` alone:

    StandardScaler -> PCA(n_components=16, svd_solver="randomized", random_state=20260920) -> StandardScaler

The result is `z_recent`, 16-d. The identical `z_recent` is reused in every one of the
five models. The Layer-11 pipeline is never refitted per window.

## 6. Earlier-history windows

Exactly four cumulative older-history windows, older layers only:

| name | older layers | older raw dim |
|---|---|---|
| k=2 | 10 | 64 |
| k=4 | 8, 9, 10 | 192 |
| k=8 | 4, 5, 6, 7, 8, 9, 10 | 448 |
| k=11 | 1 … 10 | 640 |

`k` names the total routing-history depth available together with Layer 11.

## 7. Fixed 16-dimension older-history budget

For each older-history window independently, fit on FIT only:

    StandardScaler -> PCA(n_components=16, svd_solver="randomized", random_state=20260920) -> StandardScaler

giving `z_old(k)`, 16-d. No PCA sweep. No target information enters PCA.

## 8. Five models — all exactly 32 dimensions

`sklearn.linear_model.Ridge(alpha=1.0)`, multi-output, for all five.

| model | input | history available |
|---|---|---|
| k=1 | `[z_recent ; zeros(16)]` | Layer 11 only |
| k=2 | `[z_recent ; z_old(2)]` | Layer 11 + Layer 10 |
| k=4 | `[z_recent ; z_old(4)]` | Layer 11 + Layers 8–10 |
| k=8 | `[z_recent ; z_old(8)]` | Layer 11 + Layers 4–10 |
| k=11 | `[z_recent ; z_old(11)]` | Layer 11 + Layers 1–10 |

Same Layer-11 representation, same total dimensionality (32), same Ridge, same alpha,
same FIT/TEST, same target. The only difference is how much earlier routing history is
available. The k=1 zero block holds dimensionality constant so the baseline is not
advantaged or penalised by input width.

## 9. Primary metric

On TEST: `R2_1, R2_2, R2_4, R2_8, R2_11` via

    sklearn.metrics.r2_score(y_true, y_pred, multioutput="uniform_average")

Cumulative gains over the Layer-11-only baseline:

    G_2 = R2_2 - R2_1;  G_4 = R2_4 - R2_1;  G_8 = R2_8 - R2_1;  G_11 = R2_11 - R2_1

Stepwise gains:

    S_2 = R2_2 - R2_1;  S_4 = R2_4 - R2_2;  S_8 = R2_8 - R2_4;  S_11 = R2_11 - R2_8

No other predictive metric.

## 10. Interpretation labels (exploratory only)

No exact formal Markov order is claimed. Exactly one of:

- **LOCAL-DOMINANT** — adding earlier history produces little or no consistent
  improvement over Layer 11 alone.
- **SHORT-HISTORY** — Layer 10 / Layers 8–10 give clear improvement, older history adds
  little after that.
- **LONGER-HISTORY-CANDIDATE** — performance continues to materially improve when adding
  Layers 4–10 and/or Layers 1–10 beyond the recent-history models.
- **NO-CLEAR-PATTERN** — the curve is irregular or the differences are tiny.

For directional discussion only, ~0.02 R² may be called a noticeable difference. This is
**not** a preregistered significance threshold. No label is forced.

## 11. Scientific limit

RMO-P0 tests predictive dependence beyond Layer 11 under a fixed linear low-dimensional
probe. It does not prove causal memory, a formal Markov order, that a recurrent router
would help, that storing old paths improves NLL, or that routing should use long-term
memory.

If old history helps, the only allowed conclusion is: expert-routing trajectories show
predictive structure beyond the immediately preceding layer under this frozen probe.

## 12. Minimal tests

- **A** source manifest hash matches EPD-P0's `bbbe06f…`
- **B** FIT = 512 and TEST = 256
- **C** selected-ID features are binary 64-d vectors (values in {0,1}, 8 ones per layer)
- **D** target is exactly Layer-12 native router logits
- **E** Layer-11 preprocessing is fitted once and reused by every model
- **F** each older-history PCA uses FIT only
- **G** every final probe input has exactly 32 dimensions
- **H** no TEST statistics enter preprocessing
- **I** Ridge alpha is exactly 1.0 for all five models
- **J** no formal statistics are computed before implementation checks pass

## 13. Run order

1. create project; 2. write this protocol; 3. verify EPD artifacts; 4. commit frozen
protocol; 5. implement `analyze.py`; 6. run minimal tests; 7. run analysis exactly once;
8. write results; 9. commit result; 10. STOP.

The analysis is not modified after R² values are seen.

## 14. Git

Three commits only, no speculative branches:

1. `RMO-P0 frozen exploratory protocol`
2. `RMO-P0 implementation and tests`
3. `RMO-P0 result and closure`

## 15. Governance

RMO-P0 asks only: "Once Layer 11 routing is already known, does earlier expert-selection
history still improve prediction of Layer 12 routing?"

It does not ask "How can we make long-term routing memory work?" No method is derived
from the answer. Any further experiment requires explicit authorization.
