# XEC-P0 Current State

XEC-P0 is closed.
EIPC-P0 remains closed and unchanged (verdict PROMISING).

**The expert-cache method direction is stopped.**

Verdict under the frozen seven-condition rule: **NOT_PROMISING**. All seven conditions
failed. No final cache architecture was built, and no EIPC-P1 or XEC-P1 is authorized.

## Result

True next-token NLL on TEST (lower is better):

| method | seed 42 | seed 123 | seed 2026 | mean |
|---|---|---|---|---|
| Native OLMoE | 2.692297 (deterministic) | — | — | 2.692297 |
| Current Only | 2.688796 | 2.692549 | 2.689986 | 2.690444 |
| Fused Cache | 2.689150 | 2.690900 | 2.689154 | 2.689735 |
| Expert Cache | 2.688811 | 2.692697 | 2.690864 | 2.690791 |
| Five-action Oracle | 2.647312 (deterministic) | — | — | 2.647312 |

| comparison | mean paired ΔNLL | 95% bootstrap CI |
|---|---|---|
| Expert − Native | -0.001506 | [-0.005743, +0.002760] |
| Expert − Current | +0.000347 | [-0.000761, +0.001670] |
| Expert − Fused | +0.001056 | [-0.000023, +0.002156] |

Expert Cache did not reach the -0.005 nats/token threshold against native, no CI excluded
zero, it was worse than both learned controls on average, and it lost to Fused Cache in
two of three seeds.

## Why the null is readable

All nine checkpoints selected epoch 0 (initialization): training cross entropy collapsed
to 0.03-0.08 while validation cross entropy never improved on ~1.62 (chance is 1.609). The
training loop works; the target is near-noise. The oracle's median margin between best and
second-best action is 0.0087 nats, a third of samples have under 0.001 nats of oracle
gain, and total oracle headroom is only ~0.049 nats.

So the restricted five-action Layer-12 decision carries very little learnable signal. The
three variants were parameter-matched (153600 each), shared a per-seed initialization, and
saw identical data, so the failure is not a handicap to the cache mechanism. But because
the checkpoints are effectively untrained, the comparisons among the learned variants
reflect near-random action choices and are uninformative about the cache mechanism itself.

## Scope note

EIPC-P0's information-content result stands unchanged: expert provenance does carry
future-routing information. XEC-P0 shows only that, under this frozen setting and this
restricted action space, that information did not convert into a better routing decision.
It does not establish that no cache architecture could ever work.

Prior projects (REDV-V1/V2, DREV-P0, EIPC-P0) remain closed and read-only, results
unmodified.
