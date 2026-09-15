# REDV-V1 Preregistration

**Rejected-Expert Delayed Value V1.**

Written after the environment/config audit and **before** any formal REDV
evidence, regret value, or probe statistic was produced. Everything below is
frozen from this commit onward.

## 1. Question

Does information contained in near-miss experts that are rejected by Top-k
routing at MoE layer `l` provide incremental information for predicting
counterfactual routing regret at the NEXT MoE layer `l+1`?

REDV-V1 is only a phenomenon test. It proposes no architecture.

## 2. Model — frozen

| item | value |
|---|---|
| model id | `allenai/OLMoE-1B-7B-0125` |
| revision (pinned, resolved) | `9b0c1aa87e34a20052389dce1f0cf01da783f654` |
| instruct checkpoint | no |
| runtime dtype | float16 |
| mode | `eval()`, all parameters `requires_grad = False` |

Verified config, all matching the expected scientific configuration:

| field | value |
|---|---|
| architecture | `OlmoeForCausalLM` |
| hidden_size | 2048 |
| num_hidden_layers | 16 |
| num_experts | 64 |
| num_experts_per_tok | 8 |
| norm_topk_prob | `false` |
| intermediate_size | 1024 |
| vocab_size | 50304 |
| eos_token_id | 50279 |

No substitute checkpoint is permitted. If this exact model cannot be loaded, the
run stops and reports a technical blocker.

## 3. Dataset — frozen

| item | value |
|---|---|
| dataset | `Salesforce/wikitext` |
| configuration | `wikitext-103-raw-v1` |
| revision (resolved) | `b08601e04326c79dfdd32d625aee71d232d685c3` |
| validation split | probe fitting |
| test split | final held-out evaluation |
| train split | **never used** |

## 4. Packages — recorded

| package | version |
|---|---|
| python | 3.10 (`envs/sensorllm`) |
| torch | 2.4.1+cu121 |
| transformers | 4.45.1 |
| datasets | 4.5.0 |
| numpy | 1.24.4 |
| scipy | 1.10.1 |
| scikit-learn | 1.3.2 |

No packages were installed for this experiment; the existing environment already
satisfied every direct requirement. Hardware: Tesla V100-SXM2-32GB (V100-class,
FP16).

## 5. Context construction — frozen

Tokenizer: the exact tokenizer shipped with `allenai/OLMoE-1B-7B-0125`
(`GPTNeoXTokenizerFast`).

One token stream is built independently for validation and for test. Between
original WikiText text records the model EOS token (50279) is inserted. Each
stream is cut into **non-overlapping** blocks of 129 tokens. No sliding windows.

Per block:

* tokens 1–128 → model context
* token 129 → next-token target

| parameter | frozen value |
|---|---|
| context length | 128 |
| block length | 129 |
| validation n | 512 |
| test n | 512 |
| sampling seed | 20260914 |

Sampling is without replacement from all available blocks in the split. The
frozen samples are written to `artifacts/REDV_V1/data_manifest.json`, including
block ids, stream start offsets, targets, and a content fingerprint sufficient to
reconstruct the exact sampled blocks.

These values do not change after this commit.

## 6. Layers — frozen

Exactly three adjacent transitions:

| human-readable (1-based) | code (0-based) |
|---|---|
| Layer 4 → Layer 5 | 3 → 4 |
| Layer 8 → Layer 9 | 7 → 8 |
| Layer 12 → Layer 13 | 11 → 12 |

No other layers are tested, and layers are not chosen based on observed results.

For every context, the experimental token `t` is the **last** context token,
position 127 in the 0-based 128-token context. Its output position predicts token
128, the held-out next-token target.

## 7. Definitions — frozen

* `l` — current MoE layer index.
* `t` — the experimental token, always the final token of the 128-token context.
* `x_{l,t}` — the exact input vector received by the MoE expert block at layer `l`
  for token `t`.
* `g_{l,t}` — the 64-dimensional router-logit vector at layer `l` for token `t`.
* `p_{l,t}` — `softmax(g_{l,t})`, router probabilities before Top-8 masking.

Ranks by descending `p_{l,t}`:

* ranks 1–8 — native selected experts;
* **ranks 9, 10, 11, 12 — the near-miss rejected experts.**

No other rejected ranks are tested.

## 8. Rejected-evidence vector — frozen

Normally OLMoE executes only ranks 1–8. For REDV-V1 analysis only, the four
near-miss rejected experts at ranks 9–12 are explicitly executed on the SAME
expert input `x_{l,t}`. This is an oracle analysis and is intentionally
expensive, because REDV-V1 asks whether rejected information exists at all.

With `E_{l,j}(x_{l,t})` the full FFN output of rejected expert `j`:

    r_{l,t} = sum_{j in ranks 9..12}  p_{l,t,j} * E_{l,j}(x_{l,t})

`p_{l,t,j}` is the **original unmodified** router probability of expert `j`.
Probabilities are **not** renormalized over ranks 9–12, so weakly scored rejected
experts remain weak. `dim(r) = 2048`.

No learned attention weights, no learned aggregator, no max/mean/concatenation/MLP
substitute.

## 9. Baseline — frozen

`h_{l,t}` is the normal model hidden state **after** completing decoder layer `l`
for token `t` under native Top-8 routing.

    b_{l,t} = [h_{l,t} ; g_{l,t}]        dim 2048 + 64 = 2112

This deliberately gives the baseline the normal propagated hidden state and the
complete native router score pattern. The rejected-evidence test must show value
beyond these normal signals. The baseline is not weakened.

## 10. Counterfactual target — frozen

At layer `l+1` for token `t`, take the native ranking of all 64 experts and build
exactly four equal-compute alternatives, each keeping native ranks 1–7 and
replacing native rank 8:

| route | experts (ranks) |
|---|---|
| native | 1,2,3,4,5,6,7,8 |
| alt A | 1,2,3,4,5,6,7,**9** |
| alt B | 1,2,3,4,5,6,7,**10** |
| alt C | 1,2,3,4,5,6,7,**11** |
| alt D | 1,2,3,4,5,6,7,**12** |

Every route executes exactly 8 experts; no alternative changes compute count.
Only this one MoE routing decision, for the experimental token at layer `l+1`, is
overridden. All other tokens and all later routing decisions remain native. After
the intervention the rest of the frozen model continues normally.

With `L_native(l+1,t)` the final next-token NLL under the native Top-8 route and
`L_alt_a(l+1,t)` that of alternative `a`:

    L_best_alt(l+1,t) = min over the four alternatives
    G_{l+1,t}         = max(0, L_native(l+1,t) - L_best_alt(l+1,t))

`G = 0`: no tested equal-compute alternative improves the native route.
`G > 0`: at least one gives a better next-token prediction.

## 11. Routing-weight semantics — frozen

Read from the installed implementation,
`transformers/models/olmoe/modeling_olmoe.py::OlmoeSparseMoeBlock.forward`
(transformers 4.45.1). The native path:

1. computes router logits via a bias-free linear gate;
2. softmax over all 64 experts in float32;
3. takes the Top-8 probabilities and identities;
4. since `norm_topk_prob` is **false**, does **not** renormalize them;
5. casts weights back to the input dtype;
6. scales each expert's output by its own router probability and sums.

For a forced alternative route: the SAME original 64-way router probabilities are
used, the forced eight identities are selected, and their original probabilities
are applied exactly according to these native semantics. The alternative set is
**not** renormalized, because the installed native implementation would not
renormalize it. The only intended difference between native and alternative
execution is the identity of the eighth expert.

## 12. Implementation validation before formal run

Unit tests, all of which must pass before formal extraction:

* **A. Native-route equivalence** — forcing exactly the eight experts the stock
  router selected reproduces the stock MoE output within FP16 tolerance, on
  several random tokens.
* **B. Expert isolation** — changing the forced eighth expert does not alter
  experts 1–7, unrelated tokens, or unrelated layers.
* **C. Rejected expert correctness** — the function computing `E_{l,j}(x)`
  numerically matches the corresponding expert computation inside the model.
* **D. No gradient / frozen model** — all parameters `requires_grad = False`;
  model in `eval()`.
* **E. Dataset manifest determinism** — rebuilding with seed 20260914 reproduces
  the same validation and test block ids / token ranges.
* **F. Native loss equivalence** — the non-intervened path in the counterfactual
  runner reproduces the ordinary model next-token loss within FP16 tolerance.

Smoke tests verify implementation correctness only. They must not compute or
inspect ΔR², research correlations, success/failure, or subgroup performance.

## 13. Probes — exactly two

Both are `sklearn.linear_model.Ridge(alpha=1.0)`. No MLP, no nonlinear model, no
hyperparameter search. Two probes per transition, three transitions.

Every input feature is standardized using the **validation-set** mean and
standard deviation only. Zero-variance dimensions are left at standardized value
zero. The target is not standardized.

| probe | input | dim | target |
|---|---|---|---|
| A — baseline | `b_{l,t} = [h_{l,t}; g_{l,t}]` | 2112 | `G_{l+1,t}` |
| B — augmented | `[b_{l,t}; r_{l,t}]` | 4160 | `G_{l+1,t}` |

The augmented probe differs ONLY by access to rejected evidence `r`.

Both probes are fit on the 512 validation contexts and evaluated once on the 512
test contexts. Test results are not inspected until all three transitions and
both probe fits are complete.

## 14. Primary metric

Per transition, held-out on the 512 test contexts:

    Delta_R2 = R²_augmented - R²_baseline

This is the primary quantity. `Delta_R2 > 0` means rejected evidence provides
held-out predictive information beyond the normal hidden state and native router
scores.

Also reported, **descriptively only**: mean `G`, median `G`, and the proportion of
samples with `G > 0`. These do not change the verdict. No additional primary
metric is added.

## 15. Pre-registered judgement rule

Three primary `Delta_R2` values: 4→5, 8→9, 12→13. Take their arithmetic mean.

**SUPPORTED** if:

    mean Delta_R2 >= +0.02   AND   all three Delta_R2 > 0

**NOT_SUPPORTED** if either:

    mean Delta_R2 < +0.01    OR    at least two of the three Delta_R2 <= 0

**INCONCLUSIVE** otherwise.

These thresholds are committed before formal results exist and are not modified
after seeing results. There is exactly one statistical rule.

## 16. Stopping rule

* **NOT_SUPPORTED** → stop this scientific direction. Do not implement
  rejected-expert attention, rejected-expert memory, cross-layer accumulation,
  reconsideration, or Routing Surprise gating. Do not create REDV-V2 as a rescue.
* **INCONCLUSIVE** → stop. Record INCONCLUSIVE. Do not increase sample count, add
  seeds, add layers, change alpha, or add datasets.
* **SUPPORTED** → stop REDV-V1. Do not implement the method. Record only: "The
  prerequisite phenomenon is supported." A later experiment requires explicit
  user authorization.

A negative result is an acceptable and final outcome, and is not reinterpreted.

## 17. Compute rules

Model frozen, FP16 on V100-class GPUs, no training of OLMoE. Probe fitting may
run on CPU. Numerically equivalent engineering optimizations are allowed
(caching hidden states and router logits, batching the five native/alternative
branches, caching prefix computation, smaller microbatches, sequential
processing). They may not change sample count, context length, model, precision
semantics in a materially output-changing way, layer set, alternatives, expert
ranks, or scientific definitions.

If GPU memory is insufficient: first reduce microbatch size, then process
samples sequentially. Do not reduce n=512, do not shorten context length, do not
quantize as a fallback. If the exact experiment remains technically impossible,
stop and report `TECHNICAL_BLOCKER`; do not substitute another model.
