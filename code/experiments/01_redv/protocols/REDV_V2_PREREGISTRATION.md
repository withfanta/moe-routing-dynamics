# REDV-V2 Preregistration

**Dimension-Matched Rejected-Evidence Test.**

The single, final authorized follow-up to REDV-V1. Written before any REDV-V2
rejected vector, regret value, probe prediction, or R² was computed. There will be
no REDV-V3.

## 0. Status of REDV-V1 — permanent

REDV-V1 remains permanently recorded exactly as executed. Its preregistered
formal verdict remains **SUPPORTED**. That verdict is not changed here.

REDV-V1 is nonetheless scientifically **non-interpretable** for the delayed-value
claim, because its primary comparison was confounded: baseline 2112 dimensions vs
augmented 4160 dimensions, with only n = 512 fitting samples. A post-hoc
diagnostic showed that adding 2048 dimensions of scale-matched random noise
produced a *larger* Delta_R2 than adding the real rejected evidence, at all three
transitions.

Therefore REDV-V1 does **not** establish that rejected evidence has delayed
predictive value, and does **not** establish that such value is absent. REDV-V2
exists only to repair this single measurement problem.

## 1. User-locked question

Does the rejected near-miss expert evidence belonging to the CURRENT sample
contain sample-specific information about routing regret at the NEXT MoE layer?

Precisely: is `[baseline_i ; real_rejected_i]` a better held-out predictor of
next-layer routing regret than `[baseline_i ; rejected_from_another_sample_i]`,
when both models have identical dimensionality, identical Ridge model, identical
alpha, identical preprocessing, and rejected vectors from the same empirical
distribution?

This tests **sample-specific information**. It does not test a new architecture.

## 2. Compute device

Formal extraction runs on ONE NVIDIA Tesla V100 GPU, FP16 weights and inference,
`torch.inference_mode()`, `model.eval()`, all parameters frozen.

Recorded device facts (verified before extraction):

| item | value |
|---|---|
| `torch.cuda.is_available()` | `True` |
| `torch.cuda.device_count()` | 1 |
| `torch.cuda.get_device_name(0)` | `Tesla V100-SXM2-32GB` |
| compute capability | 7.0 |
| total memory | 32494 MB |
| multiprocessors | 80 |
| torch | 2.4.1+cu121 |
| CUDA | 12.1 |
| transformers | 4.45.1 |

**Device selection note.** The protocol specifies `CUDA_VISIBLE_DEVICES=0`.
On this shared DGX-1 host, *physical* GPU 0 was occupied by another user's
8-day-old job with only 2061 MiB of 32768 MiB free; loading OLMoE in FP16 needs
roughly 14 GB, so physical GPU 0 would OOM immediately and forcing it risked
destabilizing another user's work. All eight GPUs on this host are identical
Tesla V100-SXM2-32GB devices, so the V100-only requirement is met regardless of
index. With user authorization, `CUDA_VISIBLE_DEVICES=2` is used, which torch
then sees as its only device, `cuda:0`: `torch.cuda.get_device_name(0)` returns
`Tesla V100-SXM2-32GB`, satisfying the device-0 semantics of the protocol. The
physical index is recorded in `artifacts/REDV_V2/environment.json`. No CPU
inference, no quantization, no 8-bit, no 4-bit, no precision or model
substitution.

If GPU memory proves insufficient: microbatch 8 → 4 → 2 → 1, then sequential.
Sample counts, context length, and precision semantics are never reduced. If the
exact experiment still cannot run, stop with `TECHNICAL_BLOCKER`.

## 3. Model — identical to REDV-V1

| item | value |
|---|---|
| model id | `allenai/OLMoE-1B-7B-0125` |
| revision | `9b0c1aa87e34a20052389dce1f0cf01da783f654` |
| instruct checkpoint | no |
| architecture | `OlmoeForCausalLM` |
| hidden_size | 2048 |
| num_hidden_layers | 16 |
| num_experts | 64 |
| num_experts_per_tok | 8 |
| norm_topk_prob | `false` |

Router semantics unchanged from REDV-V1: router logits, fp32 softmax across all
64 experts, Top-8 selection, no Top-k renormalization, original 64-way
probabilities scale expert outputs. OLMoE is not modified.

## 4. Dataset and forbidden blocks

`Salesforce/wikitext`, `wikitext-103-raw-v1`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`.

The 512 validation and 512 test blocks sampled by REDV-V1 have already been
observed and are **forbidden** in REDV-V2. REDV-V1 fingerprints and manifests
remain read-only.

Untouched-pool audit (from the frozen V1 manifest):

| split | blocks available | used by V1 | untouched |
|---|---|---|---|
| validation | 1957 | 512 | 1445 |
| test | 2238 | 512 | 1726 |
| **pool** | 4195 | 1024 | **3171** |

3171 untouched blocks ≥ 2048 required, margin 1123. No WikiText train split is
used, and no sample-size reduction is needed.

## 5. Fresh V2 sample construction

Same token-stream construction as REDV-V1: exact OLMoE tokenizer, EOS inserted
between original WikiText records, non-overlapping 129-token blocks, first 128
tokens the context, token 129 the next-token target, experimental token at context
position 127.

From blocks NOT present in REDV-V1, select exactly:

* **1024 fitting contexts**
* **1024 final test contexts**

Both drawn from previously unused validation and test blocks. No context appears
in both sets, and no context appeared anywhere in REDV-V1.

V2 sampling seed: **20260915**.

Sampling procedure: for each source split, enumerate all blocks, remove every
block id used by REDV-V1 in that split, and keep the remainder in ascending
order. Concatenate the validation remainder followed by the test remainder into
one candidate pool, each candidate tagged with its source split and original block
index. Draw `2048` distinct candidates from that pool without replacement using
`np.random.default_rng(20260915)`, in drawn order; the first 1024 become the
fitting set and the last 1024 become the final test set.

Frozen into `artifacts/REDV_V2/data_manifest.json`, recording per context: source
split, original block index, token-stream offset, target token, role (`fit` or
`test`), plus per-role fingerprints and the V1-exclusion audit.

## 6. Layer transitions — identical to V1

| human-readable | 0-based code |
|---|---|
| Layer 4 → Layer 5 | 3 → 4 |
| Layer 8 → Layer 9 | 7 → 8 |
| Layer 12 → Layer 13 | 11 → 12 |

No other layers.

## 7. Rejected experts and rejected evidence — identical to V1

At source layer `l`, native selected experts are ranks 1–8 and near-miss rejected
experts are exactly ranks **9, 10, 11, 12**. For analysis only, those four are
executed on the SAME expert input `x_{l,t}`.

    r_{l,t} = sum_{j in ranks 9..12}  p_{l,t,j} * E_{l,j}(x_{l,t})

* `l` — source MoE layer.
* `t` — final token in the 128-token context.
* `x_{l,t}` — exact expert-block input.
* `j` — rejected expert identity.
* `p_{l,t,j}` — original 64-way router softmax probability for rejected expert j.
* `E_{l,j}` — full frozen FFN of expert j.
* `r_{l,t}` — 2048-dimensional rejected-evidence vector.

Probabilities are **not** renormalized over ranks 9–12. This representation is
unchanged.

## 8. Baseline feature — identical to V1

    b_{l,t} = [h_{l,t} ; g_{l,t}]        2048 + 64 = 2112

`h_{l,t}` is the native hidden state after decoder layer `l` for token `t`;
`g_{l,t}` is the complete 64-dimensional router-logit vector at source layer `l`.
The baseline is not weakened or changed.

## 9. Next-layer routing regret — identical to V1

At layer `l+1`, native route is ranks 1–8, and exactly four alternatives are
built, changing only the eighth expert identity:

    [1..7, 9]   [1..7, 10]   [1..7, 11]   [1..7, 12]

Each forced expert keeps its original 64-way router probability; no
renormalization. All later computation remains native. With `L_native` the
native-route next-token NLL and `L_best_alt` the minimum NLL among the four
alternatives:

    G = max(0, L_native - L_best_alt)

Definition unchanged.

## 10. The only scientific change in REDV-V2

REDV-V1 compared baseline against baseline + rejected evidence, which changed
dimensionality. REDV-V2 compares two **equal-dimension** models:

    REAL:     X_real_i    = [b_i ; r_i]
    CONTROL:  X_shuffle_i = [b_i ; r_perm(i)]

Both have 2112 + 2048 = **4160** dimensions. `perm(i)` assigns another sample by a
fixed permutation, so `r_perm(i)` is rejected evidence from a different sample.

The shuffled control preserves exact dimensionality, empirical feature
distribution, feature scale, within-vector covariance structure, and Ridge
geometry. It destroys only the sample-specific correspondence between rejected
evidence and routing regret.

## 11. Shuffle — exactly one fixed control

One deterministic permutation, seed **314159**, via
`np.random.default_rng(314159)`. Permutations are generated independently for the
fitting set, the final test set, and each of the three layer transitions (six
permutations total), drawn in a fixed order from that single seeded generator:
for each transition in order 3→4, 7→8, 11→12, first the fit-set permutation then
the test-set permutation.

`perm(i) != i` is guaranteed for every sample.

**Derangement algorithm (documented, deterministic).** Draw
`p = rng.permutation(n)`. Then scan `i` from `0` to `n-1`; whenever `p[i] == i`,
swap `p[i]` with `p[(i + 1) % n]`. A single forward pass suffices: a swap at `i`
sends the fixed point to position `i+1`, and position `i` receives a value that
was not `i`; the final wrap-around swap at `i = n-1` cannot reintroduce a fixed
point at any earlier index, because every index below `n-1` has already been
fixed and the value moved into position `n-1` came from position `0`. The result
is verified to be a derangement by assertion. The seed is not changed to obtain
it.

No multiple shuffle seeds, no averaging over permutations, no selection of a
favorable permutation.

## 12. Probes — exactly three per transition

| probe | input | dim |
|---|---|---|
| A — baseline | `b` | 2112 |
| B — real-augmented | `[b ; r]` | 4160 |
| C — shuffled-augmented | `[b ; r_shuffled]` | 4160 |

All three are `sklearn.linear_model.Ridge(alpha=1.0)`. No alpha search, no MLP,
no nonlinear predictor, no feature selection, no PCA, no random projection.

## 13. Standardization

Every probe standardizes feature columns using **fit-set statistics only**:
`(value - fit_mean) / fit_std`, with zero-variance columns set to standardized
value 0. The REAL and SHUFFLED augmented probes use identical preprocessing
logic. The target `G` is not standardized. Test statistics are never used.

## 14. Formal metrics

On the untouched final test set, per transition: `R2_baseline`, `R2_real`,
`R2_shuffle`, and

    D = R2_real - R2_shuffle

**D is the primary matched-control statistic.** `D > 0` means correctly matched
rejected evidence predicts regret better than an equal-dimensional,
equal-distribution rejected vector belonging to another sample.

Also reported: `gain_over_baseline = R2_real - R2_baseline` (descriptive, not
primary), and descriptively `mean G`, `median G`, `proportion G > 0`. No
additional verdict metrics.

## 15. Pre-registered final judgement rule

REDV-V2 is **SUPPORTED** only if ALL three conditions hold:

1. `R2_real > 0` for all three transitions;
2. `D > 0` for all three transitions;
3. `mean(D) >= +0.02`.

If ANY condition fails: **NOT_SUPPORTED**.

There is no INCONCLUSIVE category for ordinary numerical outcomes. INCONCLUSIVE is
allowed only for a genuine technical failure that prevents the experiment from
producing its frozen statistics. This rule is not modified after any REDV-V2
result exists.

## 16. Why `R2_real > 0` is required

A model that predicts held-out regret worse than the test-set mean is not
sufficient evidence for building a new routing mechanism, even if it is less bad
than a shuffled control. Therefore `R2_real > 0` is required independently of `D`.
This prevents REDV-V2 from declaring success based only on a comparison between
two unusable regressors — the exact failure mode that made REDV-V1
non-interpretable.

## 17. Formal run order

1. `CUDA_VISIBLE_DEVICES` set to the verified V100; 2. verify V100; 3. load pinned
checkpoint in FP16; 4. build and freeze the V2 manifest; 5. extract all
fitting-set `b`, `r`, `G`; 6. extract all final-test `b`, `r`, `G`; 7. construct
the one fixed shuffled control; 8. fit all three probes for all three
transitions; 9. only after all nine probe fits are complete, compute final-test
R² values; 10. apply the rule mechanically; 11. write results.

Probes are not fitted before all extraction completes. No early transition is
inspected and acted upon before later transitions finish.

## 18. Tests before formal run

All REDV-V1 tests A–F are retained. V2 adds only:

* **G. No sample overlap** — V2 fit ∩ V2 test = ∅, and V2 all ∩ REDV-V1 all = ∅.
* **H. Derangement** — `perm(i) != i` for all samples, for every V2 control
  permutation.
* **I. Distribution identity** — the shuffled rejected matrix is exactly a row
  permutation of the real one (identical sorted row hashes).
* **J. Equal dimension** — `X_real.shape == X_shuffle.shape`, both exactly 4160
  columns.
* **K. Probe identity** — REAL and SHUFFLED use `Ridge(alpha=1.0)`, identical
  standardization implementation, identical target, identical sample count.

Tests must not inspect whether `D` is positive.

## 19. Result files

Only `artifacts/REDV_V2/`: `environment.json`, `model_provenance.json`,
`data_manifest.json`, `fit_features.npz`, `test_features.npz`, `results.json`,
`RESULTS.md`, `run.log`. `artifacts/REDV_V1/` is never overwritten.

`results.json` carries per transition: `n_fit`, `n_test`, `R2_baseline`,
`R2_real`, `R2_shuffle`, `D`, `gain_over_baseline`, `mean_G`, `median_G`,
`proportion_G_positive`; and globally `mean_D` and `final_verdict`.

## 20. Final stopping rule

**SUPPORTED** → stop, and record only: "The prerequisite phenomenon is supported
under the frozen OLMoE/WikiText setting using a dimension-matched shuffled
control." Do not implement attention, memory, reconsideration, or Routing
Surprise; those require explicit user authorization.

**NOT_SUPPORTED** → stop the entire rejected-expert delayed-value direction. No
REDV-V3. Do not increase sample size, add a dataset or model, change Ridge alpha,
try PCA or MLP, change rejected ranks, add layers, add shuffle seeds, reinterpret
subgroups, or rescue the hypothesis. Preserve the negative result.

## 21. Governance

REDV-V2 is not an attempt to make rejected-expert memory work. It asks exactly one
final question: does the rejected evidence belonging to the correct sample contain
more held-out information about future routing regret than an equal-dimensional
rejected vector drawn from another sample? If not, the direction ends.
