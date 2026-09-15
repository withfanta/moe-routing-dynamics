# EIPC-P0 Decision Log

One line per decision. The result decision is appended once, after the formal run,
and is not reinterpreted afterwards.

## Scaffold

- **Project created.** `expert_identity_preservation` initialized as a fresh
  repository with a minimal structure. No research tree, no future architecture
  roadmap, no dormant speculative nodes.
- **Fresh data confirmed available.** All three prior manifests (REDV-V1, REDV-V2,
  DREV-P0) record `train_split_used = False`, so the WikiText-103-raw train split is
  genuinely untouched. EIPC-P0 uses train exclusively and needs no prior manifest as
  data.
- **Prior projects treated as closed and read-only.** No file in either prior
  repository is modified, and no prior conclusion is imported. EIPC-P0 is not
  REDV-V3: it tests selected-expert provenance, executes no rejected experts, and
  uses no counterfactual routing or routing regret.
- **Charter accepted as locked.** The user-locked question, the three
  representations, the future-routing target, the interpretation limit, and the
  out-of-scope list in `docs/RESEARCH_CHARTER.md` are frozen.

## Result (final)

- **EIPC-P0 is PROMISING under the frozen five-condition rule.** R2_identity > 0 at
  both targets (+0.29939 at Layer 8, +0.46211 at Layer 12); A > 0 at both (+0.10923,
  +0.17506); B > 0 at both (+0.30088, +0.45815); mean(A) = +0.14215 >= +0.02; and
  mean(B) = +0.37952 >= +0.02. Permitted conclusion only: individual selected-expert
  provenance contains incremental held-out information about future routing states
  beyond early fusion and an identity-destroyed control, under this frozen
  OLMoE/WikiText setting. EIPC-P0 stops here; no cache attention was implemented and a
  later method experiment requires explicit user authorization.
- **Positive result audited before reporting.** Because this is the first positive
  outcome across these pilots, three post-hoc diagnostics were run after the verdict was
  recorded: history strictly precedes each target (no leakage); FUSED inputs are full
  rank and receive the same 64-component budget (not a PCA-capacity artifact); and the
  shuffle preserves per-layer sums to 4.8e-07, the vector multiset, and occupied-slot
  counts. None modified the rule, thresholds, or verdict.
- **Probe-class limit recorded.** R2_shuffled is near zero although the fused sum remains
  arithmetically recoverable from shuffled slots, because the per-sample per-layer cyclic
  shift is not linearly invertible by a 64-component linear probe. B thus measures the
  value of expert identity to this probe class under this compression rather than an
  information-theoretic bound, and A is the more conservative of the two quantities. This
  is a limit on interpretation, not a change to the result.
