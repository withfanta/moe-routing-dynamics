# DREV-P0 Current State

DREV-P0 is closed.

**The delayed-rejected-evidence hypothesis is stopped.**

Verdict under the frozen four-condition pilot rule: **NOT_PROMISING**.

No architecture authorized. No memory and no attention was built. No automatic
DREV-P1.

## Result

| Δ | target layer | R² real | R² shuffled | D |
|---|---|---|---|---|
| 1 | 5 | -0.08378 | -0.04037 | -0.04340 |
| 2 | 6 | -0.05771 | -0.03739 | -0.02032 |
| 4 | 8 | -0.07582 | -0.02717 | -0.04865 |
| 8 | 12 | -0.13461 | -0.02565 | -0.10895 |

mean D over delayed horizons (Δ = 2, 4, 8) = -0.05931; immediate D (Δ = 1) = -0.04340.

All four conditions failed: no delayed horizon has R2_real > 0 (0 of 3), no delayed
horizon has D > 0 (0 of 3), mean_D_delayed is below +0.02, and the delayed mean is
more negative than the immediate value rather than larger.

At every horizon the derangement control predicted regret at least as well as a
sample's own rejected evidence, and the most negative D sits at the largest distance.
There is no sign of information emerging with depth.

The PCA compression worked as designed: at 128 dimensions against n = 512 the probes
are out of the p >> n regime, so the matched comparison is readable. It reads
negative. Routing regret is present and non-trivial at every horizon (mean G 0.047 to
0.054, positive for 74-78% of samples), so the null is not an artifact of a
degenerate target.

## Scope note

This is a screening result under one frozen pilot setting, not a universal theorem.

The prior REDV project remains closed and read-only; its results were not modified or
reinterpreted, and none of its conclusions were imported here.
