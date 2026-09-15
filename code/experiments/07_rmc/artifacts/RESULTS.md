# RMC-P0 — Routing Memory Cross-Model Pilot: result

Verdict: **REPLICATED** (all six preregistered conditions hold).

Model `jetmoe/jetmoe-8b` BASE at immutable commit `d8fd02cc`, official in-tree
`transformers.models.jetmoe` (4.45.1), FP16, `eval()`, frozen, `torch.inference_mode()`.
MLP expert router only. 512 FIT / 256 TEST wikitext-103-raw-v1 blocks, one experimental
token each. Thresholds and the verdict rule were frozen at commit `f4924dc`, before any
router prediction existed; nothing was modified after results were seen.

## Held-out R² (Ridge alpha=1.0, 16-d feature, `uniform_average`)

| target | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| Layer 12 | +0.20924 | +0.34982 | +0.35199 | +0.38275 |
| Layer 20 | +0.13898 | +0.24936 | +0.34425 | +0.35463 |

## Primary comparison (k=1 vs k=4), paired bootstrap 10000, seed 314159

| target | Delta4 | 95% CI | Delta2 | Delta8 |
|---|---|---|---|---|
| Layer 12 | **+0.14275** | [+0.10998, +0.17916] | +0.14057 | +0.17351 |
| Layer 20 | **+0.20528** | [+0.16891, +0.24396] | +0.11038 | +0.21565 |

In both targets every one of the 10000 resamples gave a positive difference.

## Preregistered rule (section 16)

| condition | Layer 12 | Layer 20 |
|---|---|---|
| R²(k=1) > 0 | PASS | PASS |
| Delta4 ≥ +0.02 | PASS | PASS |
| bootstrap CI lower bound > 0 | PASS | PASS |

Both Delta4 values clear the +0.02 threshold by roughly seven to ten times, so unlike the
prior OLMoE exploratory analysis this verdict does not sit near its boundary.

## Descriptive shape (not part of the rule)

Stepwise gains — Layer 12: Step2 +0.14057, Step4 +0.00217, Step8 +0.03076. Layer 20:
Step2 +0.11038, Step4 +0.09489, Step8 +0.01038. Both curves rise monotonically in k with
no material drop, but the depth at which the gain arrives differs between the two targets,
so the shape is reported and not interpreted.

## Caveats

The older-history block is compressed to a fixed 8 dimensions at every k, so retained
variance falls as the window widens (Layer 12: 1.00 at k=2, 0.574 at k=4, 0.366 at k=8;
Layer 20: 1.00, 0.654, 0.526). At k=2 the compression is lossless — 8 raw dimensions into 8
components — so only the k=4 and k=8 numbers carry a capacity penalty. This works against
deeper history rather than for it, so it cannot manufacture the observed gains, but it does
mean the k curve understates what deeper windows contain and is not a memory-decay curve.

The verdict rests on two target layers, one dataset, one token position per block, and a
linear probe. R² is predictive, not causal.

## Router identification

JetMoE's attention mixture (`JetMoeMoA`) and its MLP mixture (`JetMoeMoE`) use the same
`JetMoeTopKGating` class with the identical `Linear(2048 -> 8)` shape, so shape can never
distinguish them and the router was identified by object identity, never by name. On real
FP16 weights across all 24 blocks the captured tensor equals `W_mlp_router @ x` recomputed
by hand from the hidden state entering `.mlp` exactly (max abs diff 0.00e+00), while
differing from the MoA router logits by at least 1.0039 on every block. Top-2 identities
were recomputed from the same captured tensor and agreed with it in every shard
(agreement 1.000000). Evidence: `router_identification.json`.

The library's `output_router_logits=True` path was unused by design because it interleaves
attention and MLP logits in one flat tuple; it is additionally broken in transformers
4.45.1 (`AttributeError: JetMoeForCausalLM has no attribute num_experts`).

## Execution

Four independent single-GPU workers on Tesla V100-SXM2-32GB physical indices 2, 4, 5, 6.
FP16 load used 15.87 GiB on one GPU, so no memory blocker applied and no model parallelism
or quantisation was used. All four shards completed at the initial microbatch of 8; the OOM
ladder never triggered. Analysis ran once, on CPU, after 45 implementation checks passed.

Allowed conclusion (protocol section 22): in two architecturally different pretrained
sparse MoEs, routing history beyond the immediately preceding layer contains incremental
held-out information about future MLP expert routing. Combined with the prior OLMoE
exploratory result, this supports cross-architecture short-range higher-order routing
dependence. Not claimed: formal Markov order, causal memory, long-term memory, that a
recurrent router improves performance, better NLL, or universal behavior across all MoEs.
