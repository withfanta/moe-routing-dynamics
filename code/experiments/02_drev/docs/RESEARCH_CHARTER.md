# DREV-P0 Research Charter

Experiment ID: **DREV-P0** — Delayed Rejected-Evidence Value Pilot.

This charter is the scope boundary. It is authored by the user and is not open to
reinterpretation by the agent.

## User-locked question

> Does rejected near-miss expert evidence produced at an early MoE layer contain
> sample-specific predictive information about routing regret several layers
> later, even when immediate next-layer value is weak?

The hypothesis is specifically about **delayed** value: immediate next-layer value
may be absent while delayed cross-layer value may still exist.

## Fixed design

One source layer, human-readable **Layer 4** (code index 3), chosen because it is
early enough to leave multiple future depths available — not chosen based on DREV
results.

Four target layers:

| target layer | code index | Δ | role |
|---|---|---|---|
| 5 | 4 | 1 | immediate |
| 6 | 5 | 2 | delayed |
| 8 | 7 | 4 | delayed |
| 12 | 11 | 8 | delayed |

Δ is the number of MoE layers between the source routing decision and the target
routing decision.

## Primary quantity

Per target layer `m`, on the held-out test split:

    D_m = R2_real(m) - R2_shuffle(m)

`D_m` is the sample-specific predictive value of the *correct* rejected evidence at
target layer `m`, measured against an equal-dimension derangement control.

## What DREV-P0 is

A small screening experiment, and a prerequisite phenomenon test only.

## What DREV-P0 does not implement

Memory, attention, recurrent routing, Routing Surprise, dynamic-k, expert
reconsideration, or a new MoE architecture.

## Interpretation limit

If PROMISING, the only permitted conclusion is: *Layer-4 rejected evidence contains
sample-specific information about some later routing-regret horizons under this
frozen pilot setting.* This does **not** mean cross-layer memory works. No memory
and no attention is built automatically.

If NOT_PROMISING, this delayed-rejected-evidence hypothesis stops. No source layers,
horizons, PCA dimension, alpha, sample size, model, or dataset are added. No
DREV-P1 is automatically authorized.

A screening result is not a universal theorem in either direction.

## Anti-drift rule

Only fix problems that make DREV-P0 incapable of answering the question above. Do
not add another model, dataset, source layer, or horizon; do not change rejected
ranks or routing regret; do not add attention, memory, GRU, or MLP probes; do not
add multiple alpha values, PCA dimensions, or shuffle seeds; do not add subgroup
analyses; do not try GSM8K or MMLU; do not increase sample size after seeing
results; do not search for a positive result.

## Relationship to the closed REDV project

`/home/h-li/work/rejected_expert_delayed_value` is closed and **read-only**. Its
negative result is not modified, reinterpreted, or rescued. DREV-P0 reuses
validated REDV *implementation* only, never REDV *conclusions*. DREV-P0 is not
REDV-V3: it asks a different question, about delayed rather than immediate value.

## Agent authority

The agent may implement, test, organize, and record. The agent may not redefine
the contribution or expand scope.

## Governing sentence

Do not ask "How can we make delayed memory work?"

Ask only: "Does Layer-4 rejected evidence become more informative about routing
regret at later depths than at the immediate next layer?"

If the frozen pilot says no, stop.
