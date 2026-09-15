# EIPC-P0 Research Charter

Experiment ID: **EIPC-P0** — Expert-Identity Preservation Cache Pilot.

This charter is the scope boundary. It is authored by the user and is not open to
reinterpretation by the agent.

## User-locked question

> Does preserving individual selected-expert provenance across depth contain
> incremental information about future routing states beyond early-fused history
> and identity-destroyed controls?

Standard sparse MoE aggregates immediately, `y_l = sum_e p_{l,e} * o_{l,e}`, after
which future layers retain the fused representation but lose explicit provenance —
*this part came from Layer l, Expert e*.

## The three representations

    FUSED
    EXPERT_IDENTITY
    SHUFFLED_IDENTITY

`FUSED` is the projected ordinary fused expert contribution per historical layer.
`EXPERT_IDENTITY` places each selected expert's projected weighted contribution in
the slot indexed by its true expert identity, zero elsewhere. `SHUFFLED_IDENTITY`
holds exactly the same vectors in the same layers, cyclically shifted so no expert
occupies its own slot.

All three are constructed from exactly the same native information: the fused
contribution is by definition the sum of the individual selected contributions. The
only conceptual difference is whether expert identity survives compression.

## The target

    future-layer router state

Specifically the native 64-dimensional router-logit vector at the target layer,
mean-centred per sample (softmax routing is invariant to a constant logit shift).

## What EIPC-P0 is

An information-content experiment, and a prerequisite phenomenon test only.

## What EIPC-P0 is not

It does not test whether an attention cache improves LM accuracy. **No new MoE
architecture is authorized.**

The possible future

    expert-state cache -> retrieval -> future-layer use

is explicitly **out of scope**. It is not designed, prototyped, or scaffolded here.

## Meaning of a negative result

A negative result terminates this premise. It is not reinterpreted or rescued, and
no follow-up experiment is created automatically.

## Anti-drift rule

Only fix an issue if it prevents EIPC-P0 from answering the locked question. Do not
add rejected experts, another model, dataset, target layer, history definition,
projection dimension, PCA dimension, probe family, or Ridge alpha; no MLP,
attention, memory, GRU, Routing Surprise, counterfactual routing, routing regret, or
dynamic-k; no multiple shuffle seeds, subgroup analysis, extra benchmarks, or
post-hoc thresholds. Do not attempt to make expert caching work.

## Relationship to the closed prior projects

The rejected-expert projects at `/home/h-li/work/rejected_expert_delayed_value`
(REDV-V1, REDV-V2) and `/home/h-li/work/delayed_rejected_evidence` (DREV-P0) are
closed and read-only. EIPC-P0 is not REDV-V3 and not a rescue of any prior
hypothesis; it concerns *selected* expert provenance rather than rejected experts,
executes no rejected experts, and imports no prior conclusion.

## Agent authority

The agent may implement, test, organize, and record. The agent may not redefine the
contribution or expand scope.

## Governing sentence

Do not ask "Can we make an expert cache architecture work?"

Ask only: "Before selected expert outputs are fused and their provenance
disappears, does preserving that expert-specific provenance retain additional
information about future routing states?"

If the answer is no, the cache premise ends.
