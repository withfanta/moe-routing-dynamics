# EPD-P0 Research Question

Experiment ID: **EPD-P0** — Expert Provenance Decomposition Pilot.

Exploratory. Smaller scale than previous experiments. **No architecture is authorized.**

## The question

EIPC-P0 established that preserving historical selected-expert provenance predicts future
routing better than early fusion. It did not identify **what** carries that information.

Candidate sources:

- **A. expert-selection identity/path itself** — which experts were chosen;
- **B. the activation contents** produced by selected experts — what they computed;
- **C. the interaction** — *which expert produced which activation*.

EPD-P0 decomposes these factors.

Single target: the native **Layer-12** router logits. Historical information: Layers 1–11.

## The four representations

`FUSED` sums the weighted Top-8 outputs per layer: it keeps what the layer produced overall
and destroys individual decomposition and expert identity.

`ID_PATH` is a 64-d binary vector per layer, 1 for each native Top-8 expert. It contains
**no** activation output, no router logits, and no unselected expert scores — a pure
selection-path representation.

`CONTENT_RANK` keeps the eight projected contributions separately, ordered by native router
rank 1..8, with **no** expert IDs. The probe learns what the k-th ranked expert computed
without learning whether that was Expert 3, 17 or 42.

`FULL_PROVENANCE` places each selected expert's projected contribution in the slot indexed
by its true expert ID, zero elsewhere: layer identity, expert identity, and content
together.

## Interpretation logic

These are **exploratory interpretation rules, not hypothesis-testing thresholds**:

- `FULL ≈ ID_PATH`, both substantially above `CONTENT_RANK` → primarily an expert-path /
  identity phenomenon.
- `FULL ≈ CONTENT_RANK`, both substantially above `ID_PATH` → primarily carried by expert
  activation content.
- `FULL` substantially above **both** → identity and content interact: knowing what was
  computed *and* who computed it beats either alone.
- `FUSED ≈ FULL` → the provenance advantage does not replicate cleanly under this
  decomposition.

No binary scientific verdict is produced, and no pattern is forced when results are
ambiguous. A gap around 0.02 R² may be *described* as non-trivial, but it is not a
preregistered significance threshold.

## Out of scope

No other model, dataset, target layer, or history range. No rejected experts,
counterfactual routing, routing regret, or oracle actions. No cache, attention, GRU, MLP,
or nonlinear probe. No additional projection dimensions, alpha search, multiple projection
seeds, or extra downstream tasks.

This experiment is for understanding the positive EIPC signal, not for obtaining a positive
method result.

## Governance

Not "How can we make expert cache work?" but: "Why did preserving expert provenance improve
future-routing predictability in EIPC-P0?" — decomposed into expert identity/path, expert
activation content, and their interaction.

Do not build a method from the answer automatically.
