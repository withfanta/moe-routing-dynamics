# DREV-P0 Preregistration

**Delayed Rejected-Evidence Value Pilot.**

Written after the environment/provenance audit and **before** any formal DREV
statistic — no `G_5`, `G_6`, `G_8`, `G_12`, `R2`, or `D` had been computed at this
commit. Everything below is frozen from here.

## 1. Question

Does rejected near-miss expert evidence produced at an early MoE layer contain
sample-specific predictive information about routing regret several layers later,
even when immediate next-layer value is weak?

The hypothesis is specifically about **delayed** value: immediate next-layer value
may be absent while delayed cross-layer value may still exist.

This is a prerequisite phenomenon test. It implements no memory, no attention, no
recurrent routing, no Routing Surprise, no dynamic-k, no expert reconsideration,
and no new MoE architecture.

DREV-P0 is **not** REDV-V3. The prior project at
`/home/h-li/work/rejected_expert_delayed_value` is closed and read-only; its
negative result is not modified, reinterpreted, or rescued, and none of its
scientific conclusions are imported here.

## 2. Model — frozen

| item | value |
|---|---|
| model id | `allenai/OLMoE-1B-7B-0125` |
| revision (pinned, resolved) | `9b0c1aa87e34a20052389dce1f0cf01da783f654` |
| architecture | `OlmoeForCausalLM` |
| hidden_size | 2048 |
| num_hidden_layers | 16 |
| num_experts | 64 |
| num_experts_per_tok | 8 |
| norm_topk_prob | `false` |
| dtype | float16 |
| quantized | no |
| training | none |

Router semantics must match `transformers 4.45.1` exactly: router logits from a
bias-free gate, fp32 softmax over all 64 experts, Top-8 selection, **no** Top-k
renormalization, and each expert output weighted by its original 64-way
probability.

## 3. Packages — recorded

python 3.10 (`envs/sensorllm`), torch 2.4.1+cu121, CUDA 12.1, transformers 4.45.1,
datasets 4.5.0, numpy 1.24.4, scipy 1.10.1, scikit-learn 1.3.2. No package was
installed for this experiment.

## 4. Data — previously untouched blocks only

`Salesforce/wikitext`, `wikitext-103-raw-v1`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`.

Every block ever used in REDV-V1 or REDV-V2 is excluded. Audit from the two frozen
prior manifests (read-only):

| quantity | count |
|---|---|
| blocks available (validation 1957 + test 2238) | 4195 |
| used by REDV-V1 | 1024 |
| used by REDV-V2 | 2048 |
| REDV-V1 ∩ REDV-V2 | 0 |
| union previously used | 3072 |
| **remaining untouched** | **1123** |
| required by DREV-P0 | 1024 |

1123 ≥ 1024, margin 99. Train is never used. If fewer than 1024 untouched blocks
had existed, the run stops with `TECHNICAL_DATA_BLOCKER`; sample counts are not
reduced.

Sample: exactly **512 fit** and **512 test** blocks, sampling seed **20260916**,
drawn without replacement from the untouched pool. Pool order is deterministic:
validation remainder ascending, then test remainder ascending; 1024 candidates are
drawn, the first 512 becoming fit and the last 512 test.

## 5. Token construction

Exact OLMoE tokenizer (`GPTNeoXTokenizerFast`). EOS (50279) inserted between
WikiText records. Non-overlapping 129-token blocks; tokens 0..127 the context and
token 128 the next-token target. Experimental token: context position **127**. No
overlapping windows.

Frozen into `artifacts/DREV_P0/data_manifest.json` before formal statistics exist,
recording source split, original block index, token-stream offset, target, role,
and per-role fingerprints.

## 6. Source layer — exactly one

Human-readable **Layer 4**, code index **3**. Chosen because it is early enough to
leave multiple future depths available; **not** selected based on DREV results. No
other source layer is tested. Write `l = 4`.

## 7. Source rejected evidence

At Layer 4, native experts are router ranks 1–8 and near-miss rejected experts are
exactly ranks **9, 10, 11, 12**. For analysis, those four are executed on the exact
same expert input Layer 4 receives.

    r_i = sum_{j in ranks 9..12}  p_ij * E_j(x_i)

* `i` — sample index
* `j` — rejected expert identity
* `x_i` — Layer-4 expert input for the experimental token
* `E_j(x_i)` — full frozen FFN output of rejected expert j
* `p_ij` — original 64-way router softmax probability for expert j
* `r_i` — 2048-dimensional rejected-evidence vector

Ranks 9–12 are **not** renormalized. No learned aggregation weights, no attention.

## 8. Baseline source feature

    b_i = [h_i ; g_i]        2048 + 64 = 2112

`h_i` is the normal hidden state after Layer 4 for the experimental token; `g_i` is
the complete Layer-4 router-logit vector. This baseline is fixed for every horizon.

## 9. Future target layers

| target layer | code index | Δ | role |
|---|---|---|---|
| 5 | 4 | 1 | immediate |
| 6 | 5 | 2 | delayed |
| 8 | 7 | 4 | delayed |
| 12 | 11 | 8 | delayed |

All intermediate layers run natively. Only the TARGET layer's routing decision for
the experimental token is counterfactually changed; everything else stays native.

## 10. Routing regret at each future layer

At each target layer independently, the native route is ranks 1–8, and exactly four
equal-compute alternatives replace only native rank 8:

    ranks 1-7 + rank 9,  + rank 10,  + rank 11,  + rank 12

Each route executes exactly 8 experts. Every forced expert keeps its original
64-way router probability; no renormalization. After the target-layer intervention
all later layers run normally, then the final true next-token NLL is computed.

    G_m(i) = max(0, L_native(m,i) - L_best_alt(m,i))

where `L_best_alt` is the minimum NLL among the four alternatives. The experiment
produces `G_5`, `G_6`, `G_8`, `G_12` for the SAME source rejected vector `r_i`.

## 11. Compression — fixing p >> n

DREV-P0 does **not** repeat REDV's 2112/4160-dimensional Ridge setup. One frozen
unsupervised compression, fit on the 512 fit samples ONLY:

* baseline `b`: StandardScaler → PCA to exactly 64 components
* rejected `r`: StandardScaler → PCA to exactly 64 components

using `sklearn.decomposition.PCA(n_components=64, svd_solver="randomized",
random_state=20260916)`. No PCA-dimension sweep, no alternative projection, and no
target information enters the scaler or PCA.

After projection `z_b` and `z_r` are each 64-dimensional, so probe inputs are **128
dimensions** against `n_fit = 512`, instead of p ≫ n.

## 12. Matched shuffle control

Exactly ONE deterministic derangement, seed **271828**: one permutation for fit and
one independent permutation for test, drawn fit-then-test from a single seeded
generator. `perm(i) != i` for every sample.

    z_r_shuffled(i) = z_r(perm(i))

Derangement algorithm: draw `rng.permutation(n)`, then scan `i` ascending and
whenever `p[i] == i` swap `p[i]` with `p[(i + 1) % n]`. One forward pass suffices;
the seed is never changed to obtain a derangement.

The control preserves dimensionality, rejected-feature distribution, PCA
representation, and within-vector covariance. It destroys only correct sample
correspondence.

The **same** source permutation pair is used for all four horizons. No separate
favorable shuffle per horizon, and no multiple shuffle seeds.

## 13. Probes

Per horizon, exactly TWO models, both `sklearn.linear_model.Ridge(alpha=1.0)`:

| probe | input | dim | target |
|---|---|---|---|
| REAL | `[z_b ; z_r]` | 128 | `G_m` |
| SHUFFLED | `[z_b ; z_r_shuffled]` | 128 | `G_m` |

No alpha tuning, no MLP, no nonlinear predictor, no other probe. The projected
features are already centred and scaled by the frozen pipeline; the target is not
standardized.

## 14. Metrics

On the untouched test set, per target `m`: `R2_real(m)`, `R2_shuffle(m)`, and

    D_m = R2_real(m) - R2_shuffle(m)

`D_m` is the sample-specific predictive value of the correct rejected evidence at
target layer `m`. Also reported descriptively: `mean G_m`, `median G_m`,
`proportion G_m > 0`. No extra metric affects the verdict.

## 15. What counts as delayed

Δ = 1 is the immediate horizon. Delayed horizons are exactly Δ = 2, 4, 8 (target
layers 6, 8, 12). The pilot asks whether delayed horizons show information that is
absent or weaker at Δ = 1.

## 16. Frozen pilot decision rule

    mean_D_delayed = mean(D_6, D_8, D_12)

DREV-P0 is **PROMISING** only if ALL of the following hold:

1. at least TWO of the three delayed horizons have `R2_real > 0`;
2. at least TWO of the three delayed horizons have `D_m > 0`;
3. `mean_D_delayed >= +0.02`;
4. `mean_D_delayed > D_5`.

If ANY condition fails: **NOT_PROMISING**.

This is a screening result, not a universal theorem. The rule is not modified after
any formal DREV statistic exists.

## 17. Interpretation limit

If PROMISING, the only permitted conclusion is: *Layer-4 rejected evidence contains
sample-specific information about some later routing-regret horizons under this
frozen pilot setting.* This does **not** license concluding that cross-layer memory
works. No memory and no attention is built automatically.

If NOT_PROMISING, this delayed-rejected-evidence hypothesis stops: no added source
layers or horizons, no PCA-dimension change, no alpha change, no larger n, no other
model or dataset. No automatic DREV-P1.

## 18. Two-V100 execution plan

Two free Tesla V100 GPUs, two independent single-GPU workers. No DDP, no model
parallelism; OLMoE fits on one V100 and each worker loads its own frozen copy,
pinned by `CUDA_VISIBLE_DEVICES`. Inside each worker
`torch.cuda.get_device_name(0)` must report a Tesla V100.

| worker | horizons | target layers |
|---|---|---|
| GPU A | Δ = 1, Δ = 2 | 5, 6 |
| GPU B | Δ = 4, Δ = 8 | 8, 12 |

Both workers use the same manifest, source features, model revision, code, and
routing semantics, and write independent files `horizon_1.json`, `horizon_2.json`,
`horizon_4.json`, `horizon_8.json`. No worker writes `results.json`; that is written
once by `finalize_drev.py` on CPU after both workers finish.

Physical GPU indices are detected before launch and recorded in
`artifacts/DREV_P0/environment.json`.

## 19. Source feature extraction

Layer-4 `b_i` and `r_i` are identical for all horizons and are computed **once**,
saved to `artifacts/DREV_P0/source_fit.npz` and
`artifacts/DREV_P0/source_test.npz`. Rejected experts are not re-executed per
horizon; both workers reuse the cached arrays.

## 20. V100 settings

`torch.inference_mode()`, `model.eval()`, all parameters frozen, FP16 inference,
router softmax keeping the installed fp32 semantics. Initial microbatch 8, reduced
8 → 4 → 2 → 1 on OOM. Only microbatch may change: sample count, context length,
precision, and model are fixed, and no quantization is used. One model instance per
GPU.

## 21. Tests before the formal run

A. model revision/config exactly correct; B. native forced route reproduces stock
OLMoE output; C. rejected ranks 9–12 computed correctly; D. the counterfactual
target-layer intervention changes only the target experimental token's route;
E. all intermediate layers remain native; F. Layer-4 source features are identical
regardless of which horizon worker consumes them; G. REDV-V1 / REDV-V2 / DREV
manifests have zero overlap; H. PCA is fit on the fit set only; I. real and shuffled
features are both exactly 128 dimensions; J. the shuffle has zero fixed points;
K. the shuffled rejected matrix is exactly a row permutation of the real one;
L. no DREV verdict statistic is produced during smoke tests.

## 22. Output artifacts

Only `artifacts/DREV_P0/`: `environment.json`, `model_provenance.json`,
`data_manifest.json`, `source_fit.npz`, `source_test.npz`, `horizon_1.json`,
`horizon_2.json`, `horizon_4.json`, `horizon_8.json`, `results.json`, `RESULTS.md`,
`run.log`.

`results.json` carries, for each Δ in {1, 2, 4, 8}: `target_layer`, `R2_real`,
`R2_shuffle`, `D`, `mean_G`, `median_G`, `proportion_G_positive`; plus
`mean_D_delayed` and `final_verdict`.

## 23. Governance

Do not ask "How can we make delayed memory work?" Ask only: "Does Layer-4 rejected
evidence become more informative about routing regret at later depths than at the
immediate next layer?" If the frozen pilot says no, stop.
