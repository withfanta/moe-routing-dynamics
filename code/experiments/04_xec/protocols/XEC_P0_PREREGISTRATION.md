# XEC-P0 Preregistration

**Cross-Layer Expert Cache Actionability Pilot.**

Written after the environment/provenance audit and **before** any formal oracle label,
policy training, or TEST NLL result existed. Everything below is frozen from this commit.

## 0. Motivation — frozen

EIPC-P0 established, under frozen OLMoE/WikiText, that historical selected-expert
provenance carries more held-out information about future router states than the same
history after early fusion (+0.10923 R² at Layer 8, +0.17506 R² at Layer 12) and than an
identity-destroyed control. It did **not** show that a cache improves language-model
performance.

XEC-P0 tests actionability: can those states be converted into an actual routing decision
that improves next-token likelihood? It does not attempt to build a final architecture.

## 1. Question

At target Layer 12, with the native router selecting Top-8, exactly five actions exist:

| action | route |
|---|---|
| 0 | keep native Top-8 |
| 1 | ranks 1–7 + native rank 9 |
| 2 | ranks 1–7 + native rank 10 |
| 3 | ranks 1–7 + native rank 11 |
| 4 | ranks 1–7 + native rank 12 |

Does a policy that dynamically reads historical selected-expert states from Layers 1–11
choose better Layer-12 actions than (1) native OLMoE, (2) a learned policy using only
current Layer-12 information, and (3) the same cache mechanism using early-fused
historical MoE states?

Primary outcome: **true next-token NLL** after applying the predicted action.

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
| dtype | float16, not quantized |
| mode | `eval()`, all parameters frozen |

Routing semantics under `transformers 4.45.1`: router logits → fp32 softmax over all 64
experts → Top-8 → **no** Top-k renormalization → original probabilities weight expert
outputs. Forced routes apply the override before the renormalization check, so the only
intended difference from native execution is the identity of the eighth expert.

**No OLMoE parameter receives gradients.** Only the policy modules train, and they consume
detached arrays extracted under `torch.inference_mode()`, so the guarantee is structural.

## 3. Packages — recorded

python 3.10 (`envs/sensorllm`), torch 2.4.1+cu121, CUDA 12.1, transformers 4.45.1,
datasets 4.5.0, numpy 1.24.4, scipy 1.10.1, scikit-learn 1.3.2. No package installed for
this experiment.

## 4. Compute — two V100 GPUs

`nvidia-smi` was inspected first; GPU indices are not assumed. All eight devices are
`Tesla V100-SXM2-32GB`; two idle ones are selected at launch and their physical index,
UUID, free VRAM, and torch-visible device are recorded in
`artifacts/XEC_P0/environment.json`.

No DDP, no model parallelism; each worker loads one independent frozen OLMoE.
`torch.inference_mode()` for extraction and oracle generation. Initial microbatch 8,
reduced 8 → 4 → 2 → 1 on OOM; sample counts never change. After extraction, no two 7B
model copies remain loaded — policy training is tiny and runs separately.

| worker | job |
|---|---|
| GPU A | TRAIN extraction + oracle generation |
| GPU B | VALIDATION + TEST extraction |

## 5. Data — fresh WikiText train blocks

`Salesforce/wikitext`, `wikitext-103-raw-v1`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, **train** split only.

Every train block used by EIPC-P0 is excluded. REDV-V1, REDV-V2 and DREV-P0 each record
`train_split_used = False`, and that was verified rather than assumed. The XEC stream is
rebuilt from the same 40000-record window EIPC-P0 used and reproduces its exact 20628-block
indexing, so exclusion is exact rather than nominal.

| quantity | value |
|---|---|
| blocks in the shared window | 20628 |
| excluded (EIPC-P0 fit + test) | 2048 |
| untouched pool | 18580 |
| required | 3584 |

Non-overlapping 129-token blocks: tokens 0..127 the context, token 128 the next-token
target; experimental token at context position **127**.

| split | n |
|---|---|
| TRAIN | 2048 |
| VALIDATION | 512 |
| TEST | 1024 |

All mutually disjoint. Sampling seed **20260918**. Frozen in
`artifacts/XEC_P0/data_manifest.json`, sha256
`8d77d76970b9635c10cabb8de564186ef7e932d25b10ddc6e81af7b12f92bc55`; fingerprints TRAIN
`151e51fcc47bd11c…`, VALIDATION `fc544a68ff5fb921…`, TEST `ed456eedcb42f295…`.

## 6. Target layer

Only human-readable **Layer 12** (code index 11), because EIPC-P0 showed the larger
expert-provenance gain there. That choice was fixed by a committed prior result, before
XEC-P0 produced any statistic. Layer 8 and every other layer are not tested.

Historical cache: Layers 1–11 (code 0–10).

## 7. Historical selected-expert states

For sample `i`, historical layer `l`, selected expert `e`, with `o_{i,l,e}` the full expert
FFN output at the experimental token and `p_{i,l,e}` its **original** 64-way router softmax
probability:

    c_{i,l,e} = p_{i,l,e} * o_{i,l,e}

Only native Top-8 selected experts are used; **no rejected expert is executed** for the
cache. Per layer, the sum over selected experts of `c` must reproduce the native fused MoE
contribution within FP16 tolerance, which is asserted before formal extraction.

## 8. Fixed cache projection

One fixed, non-trainable `P : R^2048 -> R^64`, from
`numpy.random.default_rng(20260918)` with entries `Normal(0, 1/sqrt(64))`, sha256
`11a1228c725b2a952b6cb966ac0316bd102b8a48bacdbf76391f0d44476556a6`, recorded in
`artifacts/XEC_P0/projection.json`.

    s_{i,l,e} = P(c_{i,l,e})        dim 64

No projection-dimension sweep, and no learned projection at this stage.

## 9. Current Layer-12 context

Captured before the Layer-12 routing decision, for the experimental token: `x_i`, the exact
Layer-12 expert/router input hidden state (2048), and `g_i`, the native Layer-12 router
logits (64).

    u_i = [LayerNorm(x_i) ; centered(g_i)]        dim 2112

LayerNorm is fixed with `elementwise_affine=False`; `centered(g_i) = g_i - mean(g_i)`. No
test statistics are used anywhere in feature construction.

## 10. Five routing actions

As tabulated in §1. Forced alternatives use the original 64-way router probability of every
executed expert, with **no renormalization**. All other tokens and all other layers remain
native; after the Layer-12 local intervention, Layers 13–16 execute normally, and the true
next-token NLL is computed.

## 11. Oracle training label

For TRAIN and VALIDATION only, all five NLLs `L_i(0..4)` are computed and

    a_i* = argmin_a L_i(a)

with ties broken to the **lowest action index**, so native action 0 wins exact ties.

The oracle is a supervised training/validation signal only. The TEST oracle may be computed
only **after** all policy training is frozen, and only for descriptive headroom reporting;
it must not affect model selection. This ordering is enforced by a marker file that the
training script writes on completion, so the TEST oracle loader refuses to run early.

## 12. Three learned policy variants

`CURRENT_ONLY`, `FUSED_CACHE`, `EXPERT_CACHE`. Native OLMoE is an additional non-learned
baseline. All three learned variants share the same query architecture, key/value
architecture, policy head, trainable parameter shapes, optimizer, training schedule, and
initialization seed.

## 13. Cache item format

Metadata: layer one-hot 11, expert one-hot 64.

EXPERT_CACHE item:

    z_{i,l,e} = [LayerNorm(s_{i,l,e}) ; layer_one_hot(l) ; expert_one_hot(e)]

Total item dimension 64 + 11 + 64 = **139**, with 11 × 8 = **88** items per sample. Only
actually selected experts create items.

FUSED_CACHE item, per historical layer, with `f_{i,l} = sum_e s_{i,l,e}`:

    z_fused_{i,l} = [LayerNorm(f_{i,l}) ; layer_one_hot(l) ; zeros(64)]

Also 139-dimensional, with exactly **11** items. FUSED_CACHE and EXPERT_CACHE use the SAME
trainable attention parameters; only memory contents differ.

CURRENT_ONLY uses no historical information: its memory vector is forcibly a 64-d zero
vector. It gets no other architecture, which keeps the policy head comparable.

## 14. Policy architecture — frozen before the run

Attention dimension `d = 64`. Trainable modules: `W_q : R^2112 -> R^64`,
`W_k : R^139 -> R^64`, `W_v : R^139 -> R^64`, `W_policy : R^128 -> R^5`.

    k_j = W_k(z_j),  v_j = W_v(z_j),  q = W_q(u)
    score_j = q^T k_j / sqrt(64)
    alpha = softmax(scores)
    m = sum_j alpha_j v_j
    logits_action = W_policy([q ; m])

No hidden MLP, no multi-head attention, no residual stack, no LayerNorm with trainable
affine parameters. CURRENT_ONLY sets `m = 0` and otherwise uses the same `q` and
`W_policy`.

## 15. Parameter matching

The same module class is instantiated for all three variants, each containing `W_q`, `W_k`,
`W_v`, `W_policy` even where CURRENT_ONLY leaves some unused, so trainable parameter counts
are exactly identical. For a given seed, all three variants are initialized from the **same
initial state_dict** and then trained independently.

## 16. Training

Objective: 5-way cross entropy against frozen oracle labels, no class weighting. AdamW,
learning rate 1e-3, weight decay 0.01, batch size 64, maximum 20 epochs. Validation cross
entropy is computed after every epoch; the checkpoint with the lowest validation cross
entropy is selected. No learning-rate scheduler, no gradient accumulation unless
technically required, no hyperparameter search.

## 17. Training seeds

Exactly three: **42, 123, 2026**. For each seed all three variants are trained, giving
3 × 3 = **9** checkpoints. The expensive OLMoE features and oracle labels are extracted
**once** and reused; OLMoE is not rerun per seed.

## 18. Test execution

For each frozen checkpoint: read TEST current context and cache, predict one of the five
actions, apply that exact action at Layer 12, run Layers 13–16 natively, and compute the
true next-token NLL. The policy never receives TEST oracle labels.

Saved per TEST sample: `native_NLL`, and `current_only_NLL`, `fused_cache_NLL`,
`expert_cache_NLL` for each seed. Only after policy evaluation is complete is the TEST
five-action oracle NLL computed, for descriptive upper-bound reporting.

## 19. Primary metric

Per TEST sample, with EXPERT_CACHE averaged over the three seeds:

    d_native_i  = mean_seeds(EXPERT_i) - native_i
    d_current_i = mean_seeds(EXPERT_i) - mean_seeds(CURRENT_ONLY_i)
    d_fused_i   = mean_seeds(EXPERT_i) - mean_seeds(FUSED_CACHE_i)

Negative means EXPERT_CACHE is better. These three paired differences are the primary
quantities.

## 20. Fixed paired bootstrap

For each difference, a paired bootstrap 95% CI for the mean: **10000** resamples, seed
**314159**, resampling TEST sample indices with replacement. Seeds are not resampled
independently. This procedure is not changed after results exist.

## 21. Frozen decision rule

XEC-P0 is **PROMISING** only if ALL hold:

1. mean `d_native` ≤ −0.005 nats/token
2. 95% CI upper bound for `d_native` < 0
3. mean `d_current` ≤ −0.005 nats/token
4. 95% CI upper bound for `d_current` < 0
5. mean `d_fused` ≤ −0.005 nats/token
6. 95% CI upper bound for `d_fused` < 0
7. EXPERT_CACHE beats FUSED_CACHE in mean TEST NLL in **all three** training seeds

If ANY condition fails: **NOT_PROMISING**. There is no ordinary INCONCLUSIVE category;
`TECHNICAL_BLOCKER` is allowed only if the frozen experiment cannot be executed.
Thresholds are not modified after observing results.

## 22. Descriptive metrics — reported, not used for the verdict

Five-action classification accuracy against the TEST oracle; percentage predicted native;
percentage predicted swap; percentage of TEST samples improved and worsened vs native;
native mean NLL; five-action oracle mean NLL; oracle headroom; action-frequency
distribution. No subgroup analyses.

## 23. What is cached

Projected historical selected-expert states; historical layer IDs; historical expert IDs;
Layer-12 `x`; Layer-12 native router logits `g`; native Top-12 expert IDs; native Top-12
original router probabilities; target token ID; oracle action for TRAIN/VALIDATION; five
action NLLs for TRAIN/VALIDATION. Full 2048-d historical expert outputs are projected
immediately and not retained. TEST oracle files are generated only after policy training is
frozen.

## 24. Tests before the formal run

A. exact model ID/revision/config; B. frozen/eval/FP16 model with no OLMoE gradients;
C. native Top-8 router semantics reproduced; D. sum of separately executed native selected
experts reproduces the stock Layer-12 MoE output within FP16 tolerance; E. action 0 exactly
reproduces the native NLL within numerical tolerance; F. actions 1–4 differ only at
Layer-12 experimental-token routing; G. no Top-k renormalization occurs; H. the historical
cache contains selected experts only; I. cached expert IDs match the true native selected
IDs; J. FUSED_CACHE is derived from exactly the same projected expert items as
EXPERT_CACHE; K. the projection hash is deterministic; L. TRAIN/VALIDATION/TEST manifests
are mutually disjoint; M. EIPC-P0 train blocks are excluded; N. all three variants have
identical trainable parameter counts; O. for a given seed all three start from the identical
initial state_dict; P. the TEST oracle is unavailable to training and model-selection code;
Q. smoke tests do not compute formal XEC verdict statistics.

## 25. Output artifacts

`artifacts/XEC_P0/`: `environment.json`, `model_provenance.json`, `data_manifest.json`,
`projection.json`, `train_features.npz`, `validation_features.npz`, `test_features.npz`,
`train_oracle.npz`, `validation_oracle.npz`, `test_policy_results.npz`,
`test_oracle_descriptive.npz`, `results.json`, `RESULTS.md`, `run.log`.

Large regenerable `.npz` caches are git-ignored; their sha256 values and the exact
regeneration command are documented in `RESULTS.md`.

## 26. Interpretation

If PROMISING, the allowed conclusion is exactly: *under frozen OLMoE/WikiText and a
restricted five-action Layer-12 intervention, dynamically retrieving historical
selected-expert states provides actionable routing information that improves true
next-token likelihood beyond native routing, a current-only learned policy, and a
fused-history cache.*

It must not be claimed that the final architecture is solved, that MoE improves generally,
that latency or memory improves, that all layers benefit, or that all datasets or models
benefit. Stop after recording the result; a broader architecture experiment requires
explicit user authorization.

If NOT_PROMISING, the expert-cache method direction stops: no more actions, no target-layer
change, no cache-dimension tuning, no larger sample count, no attention heads, no optimizer
change, no Layer 8, no other model or dataset, no rescue.

## 27. Governance

XEC-P0 is not asking "Can we invent an attention cache that looks useful?" It asks: "Can
expert-specific historical states, whose future-routing information was established by
EIPC-P0, be converted into an actual routing decision that improves true next-token
likelihood beyond current-state and early-fused-history controls?" If no, stop.
