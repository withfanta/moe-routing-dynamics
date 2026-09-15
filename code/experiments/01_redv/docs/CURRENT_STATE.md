# Current State

REDV-V1 is closed.
REDV-V2 is closed.

**The rejected-expert delayed-value direction is stopped.**

No architecture is authorized. No method was implemented. There is no REDV-V3.

## REDV-V1 (closed, permanent)

Verdict under its pre-registered rule: **SUPPORTED** (mean Delta_R2 = +0.47406).
Permanently recorded, unchanged, artifacts read-only.

Scientifically non-interpretable for the delayed-value claim: baseline 2112 dims
vs augmented 4160 dims at n = 512, and scale-matched random noise produced a
larger Delta_R2 than the real rejected evidence. It establishes neither presence
nor absence of delayed predictive value.

## REDV-V2 (closed, final)

Verdict under its pre-registered rule: **NOT_SUPPORTED**.

Equal-dimension comparison (4160 vs 4160) against one fixed derangement control
(seed 314159), on 1024 fit + 1024 test previously untouched contexts. All three
pre-registered conditions failed:

1. R2_real > 0 for all three transitions — FAIL, all three negative
   (-0.94185, -0.49803, -0.27184).
2. D > 0 for all three transitions — FAIL, negative at 4->5 (-0.19634) and
   8->9 (-0.13224); +0.02939 at 12->13.
3. mean(D) >= +0.02 — FAIL, mean D = -0.09973.

The shuffled control matched or beat correctly matched rejected evidence, and the
real-augmented probes predict worse than the test-set mean. The apparent
`real - baseline` gain (+2.44, +1.02, +0.79) is the REDV-V1 dimensionality
artifact reproduced, and it vanishes against the matched control.

Direction stopped. Negative result preserved.
