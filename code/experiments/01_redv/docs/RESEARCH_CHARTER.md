# REDV-V1 Research Charter

Experiment ID: **REDV-V1** — Rejected-Expert Delayed Value V1.

This charter is the scope boundary for the project. It is authored by the user
and is not open to reinterpretation by the agent.

## User-locked question

> Does rejected near-miss expert information at layer `l` add held-out
> predictive information about counterfactual routing regret at layer `l+1`?

That is the only scientific question in REDV-V1.

## Primary object

One context × one layer transition.

A context is a 128-token WikiText-103-raw block; the experimental token is
always its final token. A layer transition is an adjacent MoE layer pair
`l -> l+1`.

## Primary comparison

    baseline probe
    versus
    baseline + rejected-evidence probe

The two probes are identical in family, regularization, and target. They differ
only in whether the rejected-evidence vector is present in the input.

## Primary outcome

Held-out `ΔR²`:

    Delta_R2 = R²_augmented - R²_baseline

## What REDV-V1 is

A phenomenon test. It measures whether a signal exists, on held-out data,
beyond what the model's own propagated hidden state and native router scores
already carry.

## What REDV-V1 is not

REDV-V1 does not:

- propose a new MoE architecture;
- implement cross-layer memory;
- implement attention over experts;
- implement Routing Surprise;
- implement a new router;
- claim rejected experts are generally useful;
- test multiple models;
- test multiple datasets;
- search for a positive result.

No architecture is proposed in REDV-V1.

The possible future idea

    rejected evidence -> cross-layer memory -> reconsideration

is explicitly **out of scope**. It is not designed, prototyped, sketched, or
scaffolded here, and no dormant node for it exists in this repository.

## Meaning of a negative result

A negative result is an acceptable and final outcome.

If the premise is not supported, **no Rejected-Expert Memory method is built
from this premise.** The direction stops. A negative result is not reinterpreted,
rescued, or converted into a follow-up experiment.

## Agent authority

The agent **may**: implement, test, organize, and record.

The agent **may NOT**: redefine the contribution or expand scope.

Concretely, the agent may not add another dataset, model, corpus, routing
metric, rejected-expert definition, rejected-expert count, context length, layer
set, probe family, or seed; may not add attention, memory, GRU, transformer
memory, Routing Surprise, dynamic-k, expert-expert attention, or Shapley
analysis; may not run hyperparameter sweeps or post-hoc subgroup analysis; and
may not introduce a success threshold after results exist.

## Anti-drift rule

Only fix an issue if the issue prevents REDV-V1 from answering its
pre-registered scientific question. Do not improve, broaden, rescue, strengthen,
or generalize the experiment.

## Governing sentence

The question is not "How can I make rejected expert memory work?"

The question is only: "Is there enough delayed information in rejected
near-miss expert outputs to justify trying such a method at all?"

If the answer is no, the negative result is preserved and the direction
terminates.
