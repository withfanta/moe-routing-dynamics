# EIPC-P0 Current State

EIPC-P0 is closed.

Verdict under the frozen five-condition rule: **PROMISING**.

**No cache architecture is authorized.** Cache attention was not implemented. A later
method experiment requires explicit user authorization.

## Result

| target | history | R² fused | R² identity | R² shuffled | A | B |
|---|---|---|---|---|---|---|
| Layer 8 | L1–7 | +0.19016 | +0.29939 | -0.00149 | +0.10923 | +0.30088 |
| Layer 12 | L1–11 | +0.28705 | +0.46211 | +0.00396 | +0.17506 | +0.45815 |

mean A = +0.14215 (identity vs early fusion); mean B = +0.37952 (identity vs
identity-destroyed control). All five conditions passed.

Permitted conclusion, and nothing more: individual selected-expert provenance contains
incremental held-out information about future routing states beyond early fusion and an
expert-identity-destroyed control, under this frozen OLMoE/WikiText setting.

## Caveats recorded with the result

Post-hoc diagnostics (run after the verdict, changing nothing) confirmed no target
leakage, that FUSED is not starved of PCA components relative to EXPERT_IDENTITY, and
that the shuffle preserves the per-layer sum to 4.8e-07 while destroying only slot
identity.

The important limit: `R2_shuffled` is near zero even though the fused sum stays
arithmetically recoverable from shuffled slots, because the per-sample per-layer cyclic
shift is not linearly invertible by a 64-component probe. B therefore measures the value
of expert identity to this probe class under this compression, not an
information-theoretic bound. A is the more conservative quantity.

This says nothing about whether an expert-state cache would improve accuracy, latency,
or memory cost.

## Scope note

The prior rejected-expert projects remain closed and read-only, their results
unmodified. EIPC-P0 is not REDV-V3: it tested selected-expert provenance, executed no
rejected experts, and used no counterfactual routing or routing regret.
