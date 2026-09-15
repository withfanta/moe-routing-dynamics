# NHD-P0 — Nonlinear History Decoding Pilot

A minimal mechanism-disambiguation experiment. See `protocol.md` for the frozen protocol;
this file only summarises it.

## What this asks

RMC-P0 found, in JetMoE, that recent routing plus older history predicts future MLP-router
logits better on held-out data than recent routing alone. That result is consistent with two
different stories:

1. Older history carries predictive information that the immediately preceding routing state
   does not.
2. The information is already in the recent state R, but related to the target Y
   nonlinearly. A linear probe cannot decode it, and older history H happens to supply a
   representation a linear predictor can access.

NHD-P0 tries to separate these two, and only these two:

- **Part A** asks whether the history gain survives a small nonlinear recent-state
  predictor.
- **Part B** asks whether older history predicts held-out residual variation left over
  after a nonlinear recent-state predictor.

Neither part licenses an information-theoretic claim. Even the strongest available label
here shows only that *the tested* nonlinear predictor does not explain the full gain.

## Data

No new inference and no GPU. The frozen RMC-P0 artifacts are read, never written; their
sha256 hashes are pinned in `nhd.py` and checked before anything is computed, so if that
tree ever changes this project fails loudly rather than quietly analysing different data.
The same 512 FIT / 256 TEST samples are used.

RMC-P0, RMO-P0, EPD-P0, XEC-P0 and every earlier repository are CLOSED and read-only. This
project writes only inside its own directory.

## Variables

For target layer m (only 12 and 20):

    R = S_{m-1}                              8 dims
    H = [S_{m-4}, S_{m-3}, S_{m-2}]         24 dims
    [H ; R]                                 32 dims

`S_l` is the raw binary Top-2 MLP expert-selection indicator: identities only, no
probabilities, no gate values, no rank order, no expert activations, no attention-router
information. No PCA and no compression anywhere — this is the main structural difference
from RMC-P0, where the older-history block was compressed to 8 dimensions.

Y is the native MLP router logits at m, centred within each sample and standardized with
FIT statistics only, exactly as in RMC-P0.

## Models

Two Ridge references (`LINEAR_RECENT`, `LINEAR_HISTORY`) and three small MLPs
(`MLP_RECENT`, `MLP_HISTORY`, `MLP_RECENT_MATCHED`). The last is a parameter-capacity
control: without it, an advantage for `MLP_HISTORY` could just be extra parameters. Its
width is derived mechanically from `MLP_HISTORY`'s parameter count — W = 77, giving 1317
parameters against 1320 — and was fixed before any TEST number existed. No width sweep.

Training uses FIT only, with one frozen TRAIN/VAL split (384/128, seed 20260921) for early
stopping. TEST is never consulted for architecture, epochs, width, or debugging. Three fixed
seeds: 42, 123, 2026.

## Reading the numbers

    A         = R2(MLP_RECENT)  - R2(LINEAR_RECENT)        nonlinear decoding gain
    B         = R2(MLP_HISTORY) - R2(MLP_RECENT)           nonlinear history gain
    B_matched = R2(MLP_HISTORY) - R2(MLP_RECENT_MATCHED)   capacity-controlled
    C         = residual TEST R2                           cross-fitted residual probe

B is not conditional information. The residual test is cross-fitted so that no FIT sample is
ever residualized by a model trained on it, and TEST residuals come only from models trained
on FIT.

One descriptive label is reported: `NONLINEAR-ACCESSIBILITY-COMPATIBLE`,
`RESIDUAL-HISTORY-VALUE`, `JOINT-ONLY`, or `NO-CLEAR-PATTERN`. Its criteria, including what
"near zero" means numerically, are fixed in `protocol.md` before results exist and are not
changed afterwards. No binary scientific verdict is forced, and the label is allowed to come
out `NO-CLEAR-PATTERN`.

## Run order

    python -c "import nhd; nhd.verify_source()"   # checks A, B
    python analyze.py                             # gated on test S; writes results

`analyze.py` runs `test_nhd.py` as a subprocess and refuses to compute any TEST statistic
unless every check passes.
