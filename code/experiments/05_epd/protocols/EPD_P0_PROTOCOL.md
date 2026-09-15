# EPD-P0 Protocol

**Expert Provenance Decomposition Pilot.** Exploratory, small scale, no architecture
authorized. Frozen before any EPD statistic was produced.

## 0. Question

EIPC-P0 established that preserving historical selected-expert provenance predicts future
routing better than early fusion, but did not identify **what** carries that information.
Candidates: (A) expert-selection identity/path itself, (B) the activation contents produced
by selected experts, (C) the interaction — which expert produced which activation.

Single target: native **Layer-12** router logits. History: Layers 1–11.

## 1. Four representations — fixed meanings

**FUSED** — per historical layer, sum the weighted outputs of the native Top-8. Preserves
what the layer produced overall; destroys individual decomposition and expert identity.
Raw dim 11 × 32 = **352**.

**ID_PATH** — per layer, a 64-d binary vector with 1 for each native Top-8 expert, 0
otherwise. Contains **no** expert activation output, **no** full router logits, and **no**
unselected expert scores. Intentionally a pure selection-path representation. Raw dim
11 × 64 = **704**.

**CONTENT_RANK** — per layer, the eight projected weighted contributions kept separately and
ordered by native router rank 1..8, with **no** expert IDs. The probe learns what the k-th
ranked selected expert computed without learning whether that was Expert 3, 17 or 42. Raw
dim 11 × 8 × 32 = **2816**.

**FULL_PROVENANCE** — per layer, 64 expert slots; slot e holds the projected weighted
contribution if expert e was selected, else zero. Retains layer identity, expert identity,
and individual activation content. Raw dim 11 × 64 × 32 = **22528**.

## 2. Interpretation logic — exploratory, not hypothesis testing

- `FULL ≈ ID_PATH`, both substantially above `CONTENT_RANK` → primarily expert-path /
  identity.
- `FULL ≈ CONTENT_RANK`, both substantially above `ID_PATH` → primarily activation content.
- `FULL` substantially above **both** → identity and content interact.
- `FUSED ≈ FULL` → the provenance advantage does not replicate cleanly here.

No binary verdict is produced. No pattern is forced when results are ambiguous.

## 3. Out of scope

No other model, dataset, target layer, or history range; no rejected experts,
counterfactual routing, routing regret, or oracle actions; no cache, attention, GRU, MLP, or
nonlinear probe; no additional projection dimensions, alpha search, multiple projection
seeds, or extra downstream tasks. This is for understanding the EIPC signal, not for
obtaining a positive method result.

## 4. Model

`allenai/OLMoE-1B-7B-0125` @ `9b0c1aa87e34a20052389dce1f0cf01da783f654`;
`OlmoeForCausalLM`, hidden 2048, 16 layers, 64 experts, Top-8, `norm_topk_prob = false`;
transformers 4.45.1 native routing semantics (fp32 softmax over 64, Top-8, no
renormalization, original probabilities weight outputs). Frozen, `eval()`, FP16, not
quantized. **No model parameter receives gradients.**

## 5. Reused EIPC projection

The **exact** EIPC-P0 fixed projection `R^2048 -> R^32` is reused, keeping EPD directly
comparable to EIPC-P0:

| item | value |
|---|---|
| source artifact | `expert_identity_preservation/artifacts/EIPC_P0/projection.json` |
| seed | 20260917 |
| distribution | `Normal(0, 1/sqrt(32))` |
| expected sha256 | `fb9e6e9b7568f6d4a5ce74a7bee1ba2eab5a48a02fb2a93f80520e4ec0480282` |

EIPC-P0 recorded the projection's metadata and hash rather than the matrix, so the matrix is
regenerated from the recorded seed and distribution and then hash-checked against both the
artifact and the expected constant. Any mismatch aborts extraction; a different projection is
never silently substituted.

## 6. Dataset

`Salesforce/wikitext`, `wikitext-103-raw-v1`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, **train** split only.

Every train block used by EIPC-P0 (2048) and XEC-P0 (3584) is excluded; those two sets are
disjoint, so 5632 blocks are excluded in total. REDV-V1, REDV-V2 and DREV-P0 are verified to
record `train_split_used = False`. All prior projects built their stream from the same 40000
record window, so block ids align and exclusion is exact.

Token stream: EOS between original records; non-overlapping 129-token blocks; tokens 0..127
context, token 128 next token; experimental token at context position **127**.

## 7. Small exploratory sample

FIT = **512**, TEST = **256**, total **768**, disjoint, sampling seed **20260919**. All
blocks fresh relative to EIPC/XEC. The sample is intentionally small because this is
directional exploration. Sample size is not increased after observing results.

## 8. Four-V100 execution

Four idle `Tesla V100-SXM2-32GB` devices, chosen after inspecting `nvidia-smi` (device IDs
not assumed). Physical index, UUID, free VRAM, and the torch-visible device are recorded.
No DDP, no model parallelism; each worker loads one independent frozen OLMoE.

The 768 blocks are split into four deterministic shards of **192** each. Shard assignment is
round-robin over the drawn order *within* each role, so every shard carries 128 FIT + 64
TEST and FIT/TEST identity is preserved. Each worker extracts only its own shard; shards are
merged once on CPU afterwards.

`torch.inference_mode()`, `model.eval()`. Initial microbatch 8, reduced 8 → 4 → 2 → 1 on
OOM. Sample count never changes.

## 9. Historical expert contributions

For every sample, historical layer 1..11, and selected expert: capture the native expert ID,
the native router rank 1..8, the original 64-way probability `p`, and the full expert output
`o`. Then `c = p * o`, projected to `s = P(c)` with dim 32. Unselected experts are never
executed.

## 10. Reconstruction checks

On smoke-test examples, verify that the sum of the Top-8 `c` reproduces the native fused MoE
contribution within expected FP16 tolerance, and that `P(sum_e c_e) ≈ sum_e P(c_e)`.
Tolerances are judged on a relative basis; no unrealistically strict bitwise threshold is
imposed.

## 11. Dimension matching

Each representation is compressed independently to exactly **32** dimensions, fit on FIT data
only: `StandardScaler` → `PCA(n_components=32, svd_solver="randomized",
random_state=20260919)` → `StandardScaler`. The fitted pipeline is applied unchanged to TEST.
No target information enters PCA. The comparison therefore holds the dimensional budget
constant across all four representations.

## 12. Target

Native Layer 12 (code 11) complete 64-d router-logit vector at the experimental token,
`g_i`. Centred per sample, `q_i = g_i - mean(g_i)`, then each of the 64 target dimensions is
standardized using FIT statistics only, applied unchanged to TEST. No target-layer
information appears in historical features.

## 13. Probes

Exactly four: FUSED, ID_PATH, CONTENT_RANK, FULL_PROVENANCE, each → Layer-12 routing. All
`sklearn.linear_model.Ridge(alpha=1.0)`, multi-output. No alpha search, no nonlinear model,
no neural network.

## 14. Primary output

On TEST, `r2_score(y_true, y_pred, multioutput="uniform_average")` giving `R2_fused`,
`R2_id`, `R2_content`, `R2_full`, plus descriptive gaps:

    G_full_vs_fused          = R2_full - R2_fused
    G_identity_given_content = R2_full - R2_content
    G_content_given_identity = R2_full - R2_id

These names are descriptive; they are **not** causal effects.

## 15. Small layerwise analysis

For each historical layer l = 1..11, two separate probes: ID_PATH at layer l only, and
CONTENT_RANK at layer l only, each → Layer-12 router logits. Each single-layer
representation is compressed to exactly **16** dimensions with FIT-only StandardScaler +
PCA(16), then `Ridge(alpha=1.0)`. Records `R2_id_layer_l` and `R2_content_layer_l`, giving
two 11-point curves.

Purpose: whether the relation is dominated by recent layers, distributed across depth, or
stronger for identity than content. Exploratory only; not used to create post-hoc subgroups.

## 16. Descriptive transition summary — no new model

For each source layer 1..11, a 64 × 64 co-selection count matrix between source-layer
selected expert IDs and Layer-12 selected expert IDs. Each source row is normalized to a
probability distribution and the Layer-12 marginal selection frequency is subtracted. Only
the **10** strongest enriched source→target pairs are reported. No statistical testing, and
these pairs never contribute to a verdict.

## 17. Exploratory interpretation guide

Report a DECOMPOSITION PATTERN rather than a verdict:

- **PATTERN A — PATH-DOMINANT**: `R2_id` close to `R2_full` and substantially above
  `R2_content`.
- **PATTERN B — CONTENT-DOMINANT**: `R2_content` close to `R2_full` and substantially above
  `R2_id`.
- **PATTERN C — INTERACTION**: `R2_full` clearly exceeds both.
- **PATTERN D — NO CLEAN DECOMPOSITION**: none of the above is clear.

For discussion, a gap around **0.02 R²** may be described as non-trivial, but it is **not** a
preregistered significance threshold, and no pattern is forced.

## 18. Tests

A. exact model/revision/config; B. frozen/eval/FP16; C. native Top-8 IDs correctly captured;
D. no rejected experts executed; E. weighted selected-expert reconstruction works; F. EIPC
projection hash matches exactly; G. projection linearity; H. ID_PATH contains only 0/1
selected identities; I. CONTENT_RANK contains no absolute expert IDs; J. CONTENT_RANK rank
ordering correct; K. FULL_PROVENANCE slot index equals true expert ID; L. FUSED is exactly
the sum of the same projected contributions used in CONTENT/FULL; M. FIT/TEST disjoint;
N. EIPC and XEC train blocks excluded; O. Layer-12 logits never used as input features;
P. all final probe inputs exactly 32 dims; Q. layerwise probe inputs exactly 16 dims.

## 19. Run order

Create project; freeze protocol; build fresh manifest; commit protocol + manifest; implement
tests; run smoke test; launch four V100 extraction workers; merge shards; build four
representations; fit preprocessing on FIT only; fit all probes; evaluate TEST once; run
layerwise analysis; generate final report.

Partial R² values are not inspected to change the experiment midway.

## 20. Output

`artifacts/EPD_P0/`: `environment.json`, `model_provenance.json`, `data_manifest.json`,
`shard_0.npz` … `shard_3.npz`, `merged_raw.npz`, `results.json`, `RESULTS.md`, `run.log`.

`results.json` carries the four main R² values, the three descriptive gaps, the layerwise
`R2_id_layer` / `R2_content_layer` for Layers 1..11, the top 10 enriched transition pairs,
and `decomposition_pattern`. No method recommendation is automatically executed.

## 21. Governance

Not "How can we make expert cache work?" but "Why did preserving expert provenance improve
future-routing predictability in EIPC-P0?" — decomposed into expert identity/path, expert
activation content, and their interaction. Do not build a method from the answer
automatically.
