# XEC-P0 Research Charter

Experiment ID: **XEC-P0** — Cross-Layer Expert Cache Actionability Pilot.

This charter is the scope boundary. It is authored by the user and is not open to
reinterpretation by the agent.

## Motivation — frozen

EIPC-P0 established, under frozen OLMoE/WikiText, that historical selected-expert
provenance contains more held-out information about future router states than (1) the
same history after early expert fusion, and (2) an identity-destroyed control. Its
conservative improvement over FUSED was **+0.10923 R²** at target Layer 8 and
**+0.17506 R²** at target Layer 12.

EIPC-P0 does **not** show that a cache improves language-model performance. That gap is
exactly what XEC-P0 tests.

## User-locked question

At target Layer 12, with the native router selecting Top-8, exactly five routing
actions are permitted:

| action | route |
|---|---|
| 0 | keep native Top-8 |
| 1 | replace native rank 8 with native rank 9 |
| 2 | replace native rank 8 with native rank 10 |
| 3 | replace native rank 8 with native rank 11 |
| 4 | replace native rank 8 with native rank 12 |

> Does a policy that dynamically reads historical selected-expert states from Layers
> 1–11 choose better Layer-12 routing actions than (1) native OLMoE, (2) a learned
> policy using only current Layer-12 information, and (3) the same cache mechanism using
> early-fused historical MoE states?

The primary outcome is **true next-token NLL** after applying the predicted routing
action.

## What XEC-P0 is

An actionability test: does established information content convert into a decision that
improves likelihood?

## What XEC-P0 is not

It does **not** attempt to build a final architecture. No final cache architecture is
authorized by this charter.

## Interpretation limit

If PROMISING, the only permitted conclusion is: *under frozen OLMoE/WikiText and a
restricted five-action Layer-12 intervention, dynamically retrieving historical
selected-expert states provides actionable routing information that improves true
next-token likelihood beyond native routing, a current-only learned policy, and a
fused-history cache.*

It must NOT be claimed that the final architecture is solved, that MoE improves
generally, that latency or memory improves, that all layers benefit, or that other
datasets or models benefit.

If NOT_PROMISING, the expert-cache method direction stops: no more actions, no target
layer change, no cache-dimension tuning, no larger sample count, no attention heads, no
optimizer change, no Layer 8, no other model or dataset, no rescue.

## Anti-drift rule

Only fix issues that make XEC-P0 incapable of answering the frozen question. Do not add
another model, dataset, target layer, or history range; no rejected-expert history, no
extra candidate actions, no rank 13+, no dynamic-k, no full 64-way route search; no
other cache dimension or attention architecture, no GRU, no recurrent memory, no
multi-head attention, no MLP policy towers; no end-to-end OLMoE, expert, router, or
LM-head finetuning; no other loss, optimizer search, or learning-rate search; no
subgroup analyses or additional benchmarks.

## Frozen-model boundary

OLMoE is `eval()`, FP16, not quantized, and **no OLMoE parameter receives gradients**.
Only the policy modules train. Frozen-model features and oracle labels are extracted
once under `torch.inference_mode()` and consumed as detached arrays.

## Relationship to the closed prior projects

`rejected_expert_delayed_value` (REDV-V1, REDV-V2), `delayed_rejected_evidence`
(DREV-P0), and `expert_identity_preservation` (EIPC-P0) are closed and read-only. Their
results are not modified or reinterpreted. XEC-P0 inherits from EIPC-P0 only its
established phenomenon and the resulting target-layer choice, both fixed before XEC-P0
produced any statistic, and it executes no rejected experts.

## Agent authority

The agent may implement, test, organize, and record. The agent may not redefine the
contribution or expand scope.

## Governing sentence

XEC-P0 is not asking "Can we invent an attention cache that looks useful?"

It asks: "Can expert-specific historical states, whose future-routing information was
established by EIPC-P0, be converted into an actual routing decision that improves true
next-token likelihood beyond current-state and early-fused-history controls?"

If no, stop.
