# NHD-P0 — Nonlinear History Decoding Pilot

Frozen protocol. Written and committed before any TEST statistic exists. Nothing in this
document is modified after results are seen.

## 1. Purpose

RMC-P0 established in JetMoE that, for held-out prediction of future MLP-router logits,
recent routing plus older history beats recent routing alone. That does **not** show older
history carries predictive information absent from the immediately preceding routing state.

An alternative explanation: the information may already be present in the recent routing
state R but related to future routing Y nonlinearly, so a linear probe cannot decode it,
while older history H supplies a representation a linear predictor can access more easily.

NHD-P0 asks only:

1. Does the history gain survive a small nonlinear recent-state predictor?
2. Secondarily, does older history predict held-out residual variation left unexplained by
   a nonlinear recent-state predictor?

This is mechanism disambiguation, not an information-theoretic claim. No statement of
conditional independence is made or implied.

## 2. Data

Reuses ONLY the frozen RMC-P0 JetMoE artifacts, read-only. No new model inference, no GPU.
Same samples, same split: FIT = 512, TEST = 256. Only MLP expert-selection states and the
target MLP router logits already extracted by RMC-P0 are used.

Source artifact hashes are verified before any analysis. If the required per-layer Top-2
expert IDs or router logits were absent, the run stops with
`REQUIRED_ARTIFACT_NOT_AVAILABLE`. JetMoE is not rerun without explicit authorization.

Verified source (RMC-P0, at its closure commit `cc60edd`):

    artifacts/merged.npz            7dafe8048772937c239bf9ab61577d9cce648847af4a6227e4ca5d86a9758726
    artifacts/data_manifest.json    371601f7ef6c491d571b1528fb15ef5dcd8cd361844786aee9fc8c4aef8afe02
    artifacts/model_provenance.json e26d8ec79271f1312baad4c7993b6bd153d69f5426abb104221cae9e25585335
    artifacts/results.json          595ee0b98ed70bd5a9758e8ede0042ac1ab7ccaef3192f8c44077584795c0851

RMC-P0, RMO-P0, EPD-P0, XEC-P0 and every earlier repository are CLOSED. This project reads
one file tree and writes only inside its own directory.

## 3. Targets

Exactly Layer 12 and Layer 20. No additional target layers.

## 4. Variables

For target layer m:

- `R = S_{m-1}`, the raw 8-dimensional binary Top-2 MLP expert-selection indicator. Dim 8.
- `H = [S_{m-4}, S_{m-3}, S_{m-2}]`. Dim 24.
- Full history input `[H ; R]`. Dim 32.

No PCA. No history compression. No router probabilities. No expert activations. No
attention-router (MoA) information. Identities only.

Target `Y` = native MLP router logits at layer m, centred within each sample exactly as in
RMC-P0, then standardized per dimension using FIT statistics only.

## 5. Part A — direct nonlinear comparison

Per target, five models:

1. `LINEAR_RECENT` — `R -> Y`, Ridge(alpha=1.0)
2. `LINEAR_HISTORY` — `[H;R] -> Y`, Ridge(alpha=1.0)

These two are descriptive raw-feature references.

3. `MLP_RECENT` — `R -> Linear(8,32) -> GELU -> Linear(32,8)`
4. `MLP_HISTORY` — `[H;R] -> Linear(32,32) -> GELU -> Linear(32,8)`
5. `MLP_RECENT_MATCHED` — `R -> Linear(8,W) -> GELU -> Linear(W,8)`

W is a parameter-capacity control, fixed BEFORE any TEST result by choosing the width whose
trainable-parameter count is closest to `MLP_HISTORY`. MLP_HISTORY has
32*32+32 + 32*8+8 = 1320 parameters. `MLP_RECENT_MATCHED` has 8W+W + 8W+8 = 17W+8, giving
W = 77 at 1317 parameters (difference 3). Computed mechanically in code; no width sweep.

## 6. MLP training

FIT only. Fixed: AdamW, lr = 1e-3, weight_decay = 1e-4, max_epochs = 300.

One deterministic split inside FIT, frozen before training: TRAIN = 384, VAL = 128,
seed 20260921. Early stopping on validation MSE, patience 30, restoring the best
validation checkpoint.

TEST is never inspected during architecture choice, epoch choice, width choice, or
debugging. Exactly three initialization seeds: 42, 123, 2026. Each seed and the mean are
reported. No seed is added after results are seen.

## 7. Primary metrics

Per target and seed, TEST R² (`multioutput="uniform_average"`).

    A = R2(MLP_RECENT) - R2(LINEAR_RECENT)          nonlinear recent-state decoding gain
    B = R2(MLP_HISTORY) - R2(MLP_RECENT)            nonlinear history gain
    B_matched = R2(MLP_HISTORY) - R2(MLP_RECENT_MATCHED)

B is not called conditional information.

## 8. Part B — cross-fitted residual test (secondary)

Cross-fitting uses only the original FIT = 512, split deterministically into Fold A = 256
and Fold B = 256, seed 20260922.

Y residualization: train the same MLP_RECENT family `f_A: R -> Y` on Fold A and predict
Fold B; `f_B` on Fold B predicting Fold A. `epsilon_Y = Y - f(R)`, every sample predicted
by a model that did not train on it.

H residualization: `g_A, g_B: R -> H` with `R -> Linear(8,32) -> GELU -> Linear(32,24)`,
same optimizer and training protocol, cross-fitted identically. `epsilon_H = H - g(R)`.

One residual probe on the 512 cross-fitted FIT residuals: `epsilon_H -> epsilon_Y`,
Ridge(alpha=1.0). No alpha search.

## 9. Residual test evaluation

For TEST residualization, `f_full: R -> Y` and `g_full: R -> H` are trained on all
FIT = 512 with the same fixed architectures and protocol, then applied to untouched TEST:

    epsilon_Y_test = Y_test - f_full(R_test)
    epsilon_H_test = H_test - g_full(R_test)

The frozen residual Ridge is evaluated `epsilon_H_test -> epsilon_Y_test`, giving
`C = residual_R2` on TEST.

Paired permutation control, descriptive only: within TEST, permute the epsilon_H sample
rows with fixed seed 271828 and evaluate the same frozen probe. Not tuned against.

## 10. Descriptive classification

Exactly one label, criteria fixed here:

- `NONLINEAR-ACCESSIBILITY-COMPATIBLE` — MLP_RECENT greatly improves over LINEAR_RECENT,
  AND nonlinear history gain B is near zero, AND residual R² is near zero.
- `RESIDUAL-HISTORY-VALUE` — MLP_RECENT improves, but MLP_HISTORY still consistently
  exceeds both MLP_RECENT and MLP_RECENT_MATCHED, AND TEST residual prediction stays
  positive.
- `JOINT-ONLY` — nonlinear history gain positive but residual prediction near zero.
- `NO-CLEAR-PATTERN` — unstable or contradictory results.

Operational reading of the words above, fixed before results: "near zero" means
|value| < 0.02; "greatly improves" means A >= 0.02; "consistently exceeds" means B and
B_matched both >= 0.02 in the mean AND positive for all three seeds; "stays positive" means
residual R² > 0. Stability requires the two targets to agree on the label; if they do not,
or if a target's seeds disagree in sign on B, the result is NO-CLEAR-PATTERN. The 0.02
figure matches the magnitude treated as noticeable in the prior projects and is NOT a
significance threshold.

No binary scientific verdict is forced. Classification criteria are not changed after TEST
results exist.

## 11. Anti-rescue rule

After TEST results exist, none of the following may change: hidden width, number of layers,
dropout, activation, learning rate, optimizer, history window, target layers. Nothing is
added: no OLMoE, no other model, no other dataset, no kernel methods, no Random Forest, no
XGBoost, no Transformer, no GRU, no mutual-information estimator, no change to
residualization, no extra seeds. Any further experiment requires explicit authorization.

## 12. Required checks

A source repository unchanged · B exact RMC-P0 artifact hashes match · C FIT = 512 and
TEST = 256 exactly · D target layers exactly 12 and 20 · E R is the raw 8-d binary Top-2
MLP routing state · F each R row has exactly two ones · G H is exactly the three older
routing states, dim 24 · H `[H;R]` is exactly dim 32 · I no PCA · J no MoA router data ·
K target construction matches RMC-P0 · L MLP architectures frozen before TEST ·
M MLP_RECENT_MATCHED width determined mechanically · N TEST never used for early stopping ·
O cross-fitting never predicts a FIT sample from a model trained on that sample ·
P epsilon_Y and epsilon_H both cross-fitted · Q residual Ridge uses cross-fitted FIT
residuals only · R TEST residuals produced only by models trained on FIT · S no TEST metric
exists before all checks pass.

## 13. Structure

    README.md  protocol.md  nhd.py  analyze.py  test_nhd.py
    artifacts/ source_verification.json results.json RESULTS.md

## 14. Git

Three commits only:

1. `NHD-P0 frozen protocol and source-artifact verification`
2. `NHD-P0 implementation and tests`
3. `NHD-P0 result and closure`

## 15. Interpretation limit

Even `RESIDUAL-HISTORY-VALUE` does not prove that historical information is mathematically
absent from R. It shows only that the tested nonlinear recent-state predictor does not
explain the full held-out history gain. Not claimed: information-theoretic conditional
independence, causal memory, a formal Markov order, that a recurrent router would help, or
better NLL.
