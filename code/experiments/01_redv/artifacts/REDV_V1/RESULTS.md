# REDV-V1 Results

Rejected-Expert Delayed Value V1. Frozen setting, single run, no rescue
modification. Verdict follows the rule committed in
`protocols/REDV_V1_PREREGISTRATION.md` before these numbers existed.

## Provenance

- model: `allenai/OLMoE-1B-7B-0125` revision `9b0c1aa87e34a20052389dce1f0cf01da783f654`
- data: WikiText-103-raw, validation (probe fitting) / test (held-out); train unused
- data manifest sha256: `e8581e6b3a301a9ddce5ae46b5b436c9ed5ba769bc9771047ea9d2b680cf1714`
- probe: `sklearn.linear_model.Ridge(alpha=1.0)`

## Primary result

| transition | R² baseline | R² augmented | Delta_R2 |
|---|---|---|---|
| 4 -> 5 | -1.63915 | -0.81271 | +0.82645 |
| 8 -> 9 | -0.72434 | -0.27779 | +0.44654 |
| 12 -> 13 | -0.16347 | -0.01428 | +0.14919 |

**mean Delta_R2 = +0.47406**

Baseline feature dim 2112 = [h 2048 ; g 64]. Augmented dim 4160 adds r (2048).
Both probes fit on 512 validation contexts, evaluated once on 512 test contexts.

### Interpretation caveat (reported, not a verdict change)

Every R² above is **negative**: both probes predict held-out regret worse than
predicting the test-set mean (R² = 0). The positive Delta_R2 is therefore the gap
between two badly-overfit fits, not a gap between a working predictor and a
better one. With 2112 (and 4160) features on 512 samples, Ridge(alpha=1.0) is in
the p >> n regime, and adding 2048 columns shrinks the fitted weights and pulls
predictions toward the mean, which mechanically moves a very negative R² toward
zero.

A post-hoc control run after the verdict was recorded confirms this reading:
substituting 2048 columns of scale-matched **random noise** for `r` yields a
*larger* Delta_R2 than the real rejected evidence at all three transitions.

| transition | Delta_R2 (real r) | Delta_R2 (random noise control) |
|---|---|---|
| 4 -> 5 | +0.82645 | +1.33435 |
| 8 -> 9 | +0.44654 | +0.48339 |
| 12 -> 13 | +0.14919 | +0.17967 |

The rejected-evidence vector performs strictly worse than noise of the same
dimension. Under this protocol Delta_R2 does not isolate incremental information,
so the SUPPORTED verdict below should not be read as evidence that rejected
near-miss experts carry delayed predictive value. This control is diagnostic
only; it does not modify the pre-registered rule, the thresholds, or the recorded
verdict.

## Routing regret (descriptive only)

These values do not change the verdict.

| transition | n val | n test | mean G | median G | proportion G > 0 |
|---|---|---|---|---|---|
| 4 -> 5 | 512 | 512 | 0.05327 | 0.01130 | 0.7871 |
| 8 -> 9 | 512 | 512 | 0.05694 | 0.01720 | 0.7910 |
| 12 -> 13 | 512 | 512 | 0.04644 | 0.01005 | 0.7539 |

## Verdict

**SUPPORTED**

REDV-V1 supports delayed predictive value of rejected near-miss expert evidence under the frozen OLMoE/WikiText setting.

Rule applied:

- SUPPORTED: mean Delta_R2 >= +0.02 AND all three Delta_R2 > 0
- NOT_SUPPORTED: mean Delta_R2 < +0.01 OR at least two Delta_R2 <= 0
- INCONCLUSIVE: otherwise

The prerequisite phenomenon is supported. REDV-V1 stops here. No method is implemented. A later experiment requires explicit user authorization.

Rescue modifications: none.
