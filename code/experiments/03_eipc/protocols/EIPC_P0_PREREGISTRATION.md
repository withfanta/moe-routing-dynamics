# EIPC-P0 Preregistration

**Expert-Identity Preservation Cache Pilot.**

Written after the environment/provenance audit and **before** any formal EIPC
feature, router-prediction statistic, R², A, or B existed. Everything below is
frozen from this commit.

## 1. Question

Standard sparse MoE aggregates immediately, `y_l = sum_e p_{l,e} * o_{l,e}`. Future
layers keep the fused representation, but explicit provenance — *this part came from
Layer l, Expert e* — is lost.

EIPC-P0 asks exactly: does preserving the identities and individual outputs of
previously **selected** experts provide additional held-out information about future
routing states beyond (1) the same historical information after normal expert
fusion, and (2) the same individual expert outputs with expert identities destroyed?

This is an information-content experiment. It does **not** test whether an attention
cache improves LM accuracy, and no cache architecture is implemented.

EIPC-P0 is not REDV-V3 and not a rescue of any prior hypothesis. The prior projects
(`rejected_expert_delayed_value`, `delayed_rejected_evidence`) are closed and
read-only; no conclusion of theirs is imported, and EIPC-P0 executes no rejected
experts, no counterfactual routing, and no routing regret.

## 2. Model — frozen

| item | value |
|---|---|
| model id | `allenai/OLMoE-1B-7B-0125` |
| revision (pinned, resolved) | `9b0c1aa87e34a20052389dce1f0cf01da783f654` |
| instruct checkpoint | no |
| architecture | `OlmoeForCausalLM` |
| hidden_size | 2048 |
| num_hidden_layers | 16 |
| num_experts | 64 |
| num_experts_per_tok | 8 |
| norm_topk_prob | `false` |
| dtype | float16 |
| quantized | no |

Native routing under `transformers 4.45.1`: router logits → fp32 softmax over all 64
experts → Top-8 → **no** Top-k renormalization → original probabilities weight expert
outputs. Model frozen, `model.eval()`, `torch.inference_mode()`.

## 3. Packages — recorded

python 3.10 (`envs/sensorllm`), torch 2.4.1+cu121, CUDA 12.1, transformers 4.45.1,
datasets 4.5.0, numpy 1.24.4, scipy 1.10.1, scikit-learn 1.3.2. No package installed
for this experiment.

## 4. Compute — two V100 GPUs

`nvidia-smi` was run first; physical GPUs 0 and 1 were **not** free (2061 MiB and
20101 MiB free respectively, GPU 1 under active load), confirming they must not be
assumed available. All eight devices are `Tesla V100-SXM2-32GB`.

Two idle devices are used, one per worker, each loading its own independent frozen
OLMoE copy, pinned by `CUDA_VISIBLE_DEVICES`. No DDP, no model parallelism. Inside
each process `torch.cuda.get_device_name(0)` must report `Tesla V100-SXM2-32GB`.
Physical index, UUID, free VRAM, and torch-visible device are recorded in
`artifacts/EIPC_P0/environment.json`.

| worker | role |
|---|---|
| GPU A | FIT extraction (1024 samples) |
| GPU B | TEST extraction (1024 samples) |

Initial microbatch 8, reduced 8 → 4 → 2 → 1 on OOM. Scientific sample counts never
change.

## 5. Data — WikiText train only

`Salesforce/wikitext`, `wikitext-103-raw-v1`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, **train** split exclusively.

The prior REDV/DREV experiments used validation and test and explicitly never touched
train (all three prior manifests record `train_split_used = False`), so train is
fresh data and no prior manifest is needed as input.

Exact OLMoE tokenizer (`GPTNeoXTokenizerFast`), EOS (50279) between original WikiText
records, non-overlapping 129-token blocks: tokens 0..127 the context, token 128 the
next-token continuation token. The experimental token is context position **127**.

The record list is truncated to the first 40000 train records, which yields 2661048
tokens and 20628 blocks — far more than needed. The truncation is part of the frozen
sample definition and is recorded in the manifest.

Exactly **1024 FIT** and **1024 TEST** blocks, drawn without replacement, disjoint by
construction. Sampling seed **20260917**.

Frozen in `artifacts/EIPC_P0/data_manifest.json` before formal statistics exist:
sha256 `23d75e6ee9f2a76249001e3e9411254ad07efcc20b667c6b317e8d523761fc4c`, FIT
fingerprint `7a7910a0f6653d49…`, TEST fingerprint `ac0dce7ccfcba1cb…`.

## 6. History and target layers

Exactly two future routing targets.

| target | code index | historical cache | history code indices |
|---|---|---|---|
| Layer 8 | 7 | Layers 1–7 | 0–6 |
| Layer 12 | 11 | Layers 1–11 | 0–10 |

This intentionally tests a middle-depth and a later-depth target. No other target
layer is tested.

## 7. Individual expert states

At historical layer `l` and experimental token `i`, the native router selects exactly
Top-8 experts. With `e` an expert identity in 0..63, `o_{i,l,e}` the full 2048-d
output of selected expert `e`, and `p_{i,l,e}` its **original** 64-way router softmax
probability:

    c_{i,l,e} = p_{i,l,e} * o_{i,l,e}          for selected e
    c_{i,l,e} = 0                              otherwise

These are **selected experts only**; no rejected expert is executed. The ordinary
fused MoE contribution is exactly

    y_{i,l} = sum_{e=0..63} c_{i,l,e}

so the FUSED and EXPERT representations are built from exactly the same native
information. The only conceptual difference is whether expert identity is preserved
before compression.

## 8. Implementation of selected contributions

During the normal forward pass, for the experimental token, capture: expert-block
input, router logits, native Top-8 identities, native Top-8 probabilities. Analysis
code then re-executes **only those already-selected Top-8 experts** on the captured
input to obtain individual outputs. This is analysis only and does not modify model
execution.

Required numerical invariant, tested before formal extraction: the weighted sum of
the eight separately executed expert outputs reproduces the stock native MoE output
for that token within FP16 tolerance.

## 9. Fixed low-dimension projection

A single fixed random linear projection `P : R^2048 -> R^32`, generated once from
`numpy.random.default_rng(20260917)` with entries `Normal(0, 1/sqrt(32))`.

`P` is fixed, non-trainable, identical across layers, across experts, and across FIT
and TEST. Its sha256 is
`fb9e6e9b7568f6d4a5ce74a7bee1ba2eab5a48a02fb2a93f80520e4ec0480282`, recorded in
`artifacts/EIPC_P0/projection.json`. No projection-dimension sweep.

    s_{i,l,e} = P(c_{i,l,e})            dim 32

Because `P` is linear, `P(y_{i,l}) = sum_e s_{i,l,e}`. This property is unit-tested
numerically.

## 10. Representation A — FUSED history

Per historical layer, `f_{i,l} = sum_e s_{i,l,e}`, the projected ordinary fused
expert contribution; then concatenated over the history in fixed layer order.

| target | raw FUSED dimension |
|---|---|
| Layer 8 | 7 × 32 = 224 |
| Layer 12 | 11 × 32 = 352 |

Contains historical MoE information but no individual expert provenance.

## 11. Representation B — EXPERT_IDENTITY

Per historical layer, 64 expert slots; slot `e` holds `s_{i,l,e}` if expert `e` was
selected, otherwise a 32-d zero vector. **The slot index IS the expert identity.**
Each layer is therefore 64 × 32 = 2048 dimensions, and layer identity is preserved by
concatenating layers in fixed order.

| target | raw dimension |
|---|---|
| Layer 8 | 7 × 64 × 32 = 14336 |
| Layer 12 | 11 × 64 × 32 = 22528 |

Preserves layer identity, expert identity, and selected expert contribution.

## 12. Representation C — SHUFFLED_IDENTITY control

The exact same EXPERT_IDENTITY tensors, then expert identity alone is destroyed: for
every (sample, historical layer) the 64 expert slots are cyclically shifted by a
non-zero offset drawn in {1, …, 63}, deterministic RNG seed **314159**, with
independent draws for FIT and TEST (seed offset 0 for FIT, 1 for TEST).

A non-zero cyclic shift guarantees no expert remains in its original identity slot.
Preserved: the same contribution vectors, the same layer, the same number of selected
experts, the same numerical values. Destroyed: only *which expert identity produced
this vector*.

No multiple shuffle seeds, and no selection of a favorable control.

## 13. Dimension matching — FIT-only unsupervised compression

For each (target layer, representation type), fit on **FIT data only**:

1. `StandardScaler`
2. `PCA(n_components=64, svd_solver="randomized", random_state=20260917)`
3. `StandardScaler` on the resulting 64 components

The fitted pipeline is applied unchanged to TEST. No target router information enters
preprocessing. Final dimensions: FUSED 64, EXPERT_IDENTITY 64, SHUFFLED_IDENTITY 64.
No PCA-dimension sweep.

## 14. Future routing target

At target layer `m`, capture the native 64-dimensional router-logit vector `g_{i,m}`.
Router logits are used rather than Top-k IDs because they preserve the full future
routing state.

Softmax routing is invariant to adding a constant to all logits, so each target is
centred per sample:

    q_{i,m} = g_{i,m} - mean(g_{i,m})           mean over its 64 expert dimensions

Each target dimension is then standardized using FIT-set mean/std only, applied
unchanged to TEST. No TEST target information enters fitting.

## 15. Probes — exactly three per target

| probe | input | target |
|---|---|---|
| A FUSED | 64-d FUSED history | 64-d future routing `q` |
| B EXPERT_IDENTITY | 64-d expert-preserving history | same `q` |
| C SHUFFLED_IDENTITY | 64-d identity-destroyed history | same `q` |

All are `sklearn.linear_model.Ridge(alpha=1.0)`, using its native multi-output
support. Six probes total (3 representations × 2 targets). No alpha search, no MLP,
no nonlinear model, no classifier, no other probe.

## 16. Primary metric

On TEST,
`sklearn.metrics.r2_score(true_router_state, predicted_router_state, multioutput="uniform_average")`,
giving `R2_fused(m)`, `R2_identity(m)`, `R2_shuffled(m)`.

    A_m = R2_identity(m) - R2_fused(m)
    B_m = R2_identity(m) - R2_shuffled(m)

`A_m` measures whether retaining individual expert states provides more
future-routing information than early fusion. `B_m` measures whether the specific
expert identity itself matters. These are the ONLY primary comparison quantities.

## 17. Frozen decision rule

EIPC-P0 is **PROMISING** only if ALL conditions hold:

1. `R2_identity > 0` at BOTH target layers;
2. `A_m > 0` at BOTH target layers;
3. `B_m > 0` at BOTH target layers;
4. `mean(A_m) >= +0.02`;
5. `mean(B_m) >= +0.02`.

If ANY condition fails: **NOT_PROMISING**.

There is no ordinary numerical INCONCLUSIVE category. A genuine execution failure may
be recorded separately as `TECHNICAL_BLOCKER`. Thresholds are not changed after formal
results exist.

## 18. Interpretation

PROMISING means only: *individual selected-expert provenance contains incremental
held-out information about future routing states beyond early fusion and an
expert-identity-destroyed control, under this frozen OLMoE/WikiText setting.* It does
**not** prove that an expert cache architecture improves performance.

If PROMISING: stop EIPC-P0. Do not implement cache attention; a later method
experiment requires explicit user authorization.

If NOT_PROMISING: stop this expert-identity-cache premise. Do not add target layers,
increase n, change PCA, change projection dimension, change alpha, try another model
or dataset, add nonlinear probes, or build attention anyway. No automatic EIPC-P1.

## 19. Run plan

After preregistration and tests: GPU A extracts all 1024 FIT samples, GPU B all 1024
TEST samples. Each worker records, for Layers 1–11, the native Top-8 IDs, native
Top-8 probabilities, and selected-expert weighted contributions after the fixed 32-d
projection; plus native router logits for Layers 8 and 12. Full 2048-d expert outputs
are projected immediately and not retained, keeping artifacts small.

Workers write `fit_raw.npz` and `test_raw.npz` and compute **no** final R². After
both finish, `finalize_eipc.py` runs ONCE on CPU and: builds the three
representations; fits FIT-only preprocessing; fits exactly six probes; computes TEST
R² only after all probes are fitted; applies the frozen rule; writes the result.

## 20. Tests before the formal run

A. exact model ID/revision/config; B. model frozen and eval mode; C. native router
Top-8 semantics reproduced correctly; D. weighted sum of separately executed selected
experts reproduces the stock native MoE token output within FP16 tolerance; E. fixed
projection deterministic and hash-stable; F. projection linearity
`P(sum_e c_e) ≈ sum_e P(c_e)`; G. FUSED derived from exactly the same expert
contributions used by EXPERT_IDENTITY; H. expert-identity slots correspond to the
true expert IDs; I. shuffled identity uses non-zero cyclic shifts only; J. shuffled
representation contains exactly the same contribution vectors as the real identity
representation, only in different slots; K. FIT and TEST manifests disjoint;
L. PCA/scalers fit only on FIT; M. final dimensions all equal 64; N. future targets
contain only native router logits from the target layer; O. no target-layer or later
information appears in historical features; P. smoke tests do not compute formal
R²/A/B verdict statistics.

## 21. Output artifacts

Only `artifacts/EIPC_P0/`: `environment.json`, `model_provenance.json`,
`data_manifest.json`, `projection.json`, `fit_raw.npz`, `test_raw.npz`,
`results.json`, `RESULTS.md`, `run.log`.

`results.json` carries, for Target 8 and Target 12: `R2_fused`, `R2_identity`,
`R2_shuffled`, `A`, `B`; plus `mean_A`, `mean_B`, `final_verdict`. No extra analysis
affects the verdict.

## 22. Governance

Do not ask "Can we make an expert cache architecture work?" Ask only: "Before
selected expert outputs are fused and their provenance disappears, does preserving
that expert-specific provenance retain additional information about future routing
states?" If the answer is no, the cache premise ends.
