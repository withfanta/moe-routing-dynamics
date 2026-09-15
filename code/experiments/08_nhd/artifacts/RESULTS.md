# NHD-P0 — Nonlinear History Decoding Pilot: result and closure

Mechanism-disambiguation experiment on the frozen RMC-P0 JetMoE artifacts. Single analysis
run, CPU only, no new model inference, no GPU. Source repository RMC-P0 verified clean at
commit `cc60edd` with all four artifact hashes matching the preregistered pins.

Question asked, and only this: **does the history gain survive a small nonlinear
recent-state predictor?** Secondarily: does older history predict held-out residual
variation left unexplained by a nonlinear recent-state predictor?

- Model: `jetmoe/jetmoe-8b` BASE, MLP MoE router only (`model.layers[i].mlp.router`).
  No attention-router (MoA) information anywhere.
- Target layers: exactly Layer 12 and Layer 20.
- R = raw binary Top-2 MLP selection state of layer m−1, 8 dims.
- H = raw binary Top-2 states of layers m−4, m−3, m−2, 24 dims. No PCA, no compression.
- `[H;R]` = 32 dims. Target = native MLP router logits at layer m, per-sample centred,
  standardized with FIT statistics only.
- FIT = 512, TEST = 256, the same samples in the same order as RMC-P0.
- Nonlinear predictor: `Linear -> GELU -> Linear`, hidden 32, AdamW (lr 1e-3, wd 1e-4),
  full batch, max 300 epochs, early stopping (patience 30) on a 384/128 validation split
  carved out of FIT. TEST never used for early stopping, architecture, width, or debugging.
- Initialization seeds: exactly 42, 123, 2026.
- MLP_RECENT_MATCHED hidden width **77**, fixed mechanically before any TEST result by
  minimising |params − MLP_HISTORY params|: 1317 vs 1320, difference 3.

All 36 implementation checks (A–S) pass. `analyze.py` gates itself on them, so no TEST
statistic was computed before they passed.

## Layer 12

LINEAR (Ridge, alpha 1.0, held-out R²)

| model | TEST R² |
|---|---|
| LINEAR_RECENT (R, 8 d) | +0.20972 |
| LINEAR_HISTORY (`[H;R]`, 32 d) | +0.37188 |
| linear gain | **+0.16216** |

NONLINEAR (held-out R² per seed)

| model | params | seed 42 | seed 123 | seed 2026 | mean |
|---|---|---|---|---|---|
| MLP_RECENT | 552 | +0.20871 | +0.21206 | +0.21257 | +0.21111 |
| MLP_RECENT_MATCHED | 1317 | +0.22051 | +0.21742 | +0.21633 | +0.21809 |
| MLP_HISTORY | 1320 | +0.38018 | +0.38677 | +0.38051 | +0.38249 |

- A = MLP_RECENT − LINEAR_RECENT = **+0.00139** (per seed −0.00101, +0.00234, +0.00285)
- B = MLP_HISTORY − MLP_RECENT = **+0.17137** (per seed +0.17147, +0.17471, +0.16794)
- B_matched = MLP_HISTORY − MLP_RECENT_MATCHED = **+0.16440** (per seed +0.15967, +0.16935, +0.16419)

RESIDUAL (Part B)

| quantity | value |
|---|---|
| residual TEST R² | **+0.20549** |
| permuted-history TEST R² (descriptive control) | −0.33325 |

## Layer 20

LINEAR (Ridge, alpha 1.0, held-out R²)

| model | TEST R² |
|---|---|
| LINEAR_RECENT (R, 8 d) | +0.13948 |
| LINEAR_HISTORY (`[H;R]`, 32 d) | +0.35382 |
| linear gain | **+0.21434** |

NONLINEAR (held-out R² per seed)

| model | params | seed 42 | seed 123 | seed 2026 | mean |
|---|---|---|---|---|---|
| MLP_RECENT | 552 | +0.14678 | +0.15214 | +0.14760 | +0.14884 |
| MLP_RECENT_MATCHED | 1317 | +0.14602 | +0.15210 | +0.14688 | +0.14834 |
| MLP_HISTORY | 1320 | +0.36819 | +0.36339 | +0.37078 | +0.36745 |

- A = MLP_RECENT − LINEAR_RECENT = **+0.00936** (per seed +0.00730, +0.01267, +0.00812)
- B = MLP_HISTORY − MLP_RECENT = **+0.21861** (per seed +0.22141, +0.21124, +0.22318)
- B_matched = MLP_HISTORY − MLP_RECENT_MATCHED = **+0.21912** (per seed +0.22217, +0.21129, +0.22390)

RESIDUAL (Part B)

| quantity | value |
|---|---|
| residual TEST R² | **+0.23556** |
| permuted-history TEST R² (descriptive control) | −0.32149 |

## Classification

**RESIDUAL-HISTORY-VALUE**

Both targets receive this label independently, so the targets agree. The criteria were
fixed in `protocol.md` section 10 before the run and were not changed after results
existed: B and B_matched both ≥ 0.02 in the mean and positive for all three seeds
(satisfied at both targets), residual R² > 0 and not near zero (+0.20549 and +0.23556,
both far above the 0.02 near-zero band). The 0.02 figure is an operational reading of
"near zero" and "consistently exceeds", not a significance threshold.

The permutation control is descriptive only and was not tuned against: destroying the
pairing between residual history and residual target drives held-out R² from about +0.21
to about −0.33 at both targets, i.e. worse than predicting the mean.

## What this does and does not establish

The nonlinear recent-state predictor did not close the history gap. Given R alone, the
small MLP performed essentially as well as Ridge on R (A = +0.00139 and +0.00936, both
inside the near-zero band), while adding older history raised held-out R² by roughly +0.17
and +0.22. The parameter-matched control rules out "more parameters" as the explanation:
MLP_RECENT_MATCHED has 1317 trainable parameters against MLP_HISTORY's 1320, and its
advantage over plain MLP_RECENT is at most +0.007 (Layer 12) and −0.001 (Layer 20).

Two limits on reading A. Its small size means the tested nonlinear predictor found little
nonlinear structure in R beyond what Ridge already extracted; this is a statement about
this predictor at this capacity on 512 FIT samples, not about nonlinearity in general. A
different or larger nonlinear model was not tried, by protocol.

**Even RESIDUAL-HISTORY-VALUE does not prove that historical information is mathematically
absent from R; it only shows that the tested nonlinear recent-state predictor does not
explain the full held-out history gain.**

B is not conditional information, and nothing here is a claim of information-theoretic
conditional independence.

## Provenance

- Source: RMC-P0 at `cc60edd`, clean; `merged.npz`, `data_manifest.json`,
  `model_provenance.json`, `results.json` all hash-matched to the preregistered pins.
- No new model inference, no GPU, no dataset access. Prior repositories (RMC-P0, RMO-P0,
  EPD-P0, XEC-P0 and the others) unmodified, 0 working-tree changes each.
- No PCA, no compression, no router probabilities, no expert activations, no attention
  router.
- No rescue modifications: hidden width, layer count, activation, dropout, learning rate,
  optimizer, history window, target layers, model, dataset, estimator family,
  residualization, and seed count are all as committed before the run.
- Environment: python 3.10.19, numpy 1.24.4, scikit-learn 1.3.2, torch 2.4.1+cu121
  (CPU, 4 threads).
- Exactly three commits.
