# XEC-P0 Results

Cross-Layer Expert Cache Actionability Pilot. The verdict follows the
seven-condition rule committed in `protocols/XEC_P0_PREREGISTRATION.md` before
these numbers existed.

## Provenance

- model: `allenai/OLMoE-1B-7B-0125` revision `9b0c1aa87e34a20052389dce1f0cf01da783f654`, frozen, FP16, not quantized
- **no OLMoE parameter received gradients**: `False`
- target: Layer 12 (code 11), history Layers 1-11
- data: WikiText-103-raw train, EIPC-P0 blocks excluded (2048)
- manifest sha256: `8d77d76970b9635c10cabb8de564186ef7e932d25b10ddc6e81af7b12f92bc55`
  - train fingerprint `151e51fcc47bd11c53ca19b36e932406…`
  - validation fingerprint `fc544a68ff5fb92189e0841038fe92d2…`
  - test fingerprint `ed456eedcb42f295b10b9b8438902ce6…`
- n_train 2048, n_validation 512, n_test 1024
- projection sha256: `11a1228c725b2a952b6cb966ac0316bd102b8a48bacdbf76391f0d44476556a6`
- policy seeds: [42, 123, 2026]

## Policy NLL table (true next-token NLL on TEST, lower is better)

| method | seed 42 | seed 123 | seed 2026 | mean |
|---|---|---|---|---|
| Native OLMoE | 2.692297 (det.) | — | — | **2.692297** |
| Current Only | 2.688796 | 2.692549 | 2.689986 | **2.690444** |
| Fused Cache | 2.689150 | 2.690900 | 2.689154 | **2.689735** |
| Expert Cache | 2.688811 | 2.692697 | 2.690864 | **2.690791** |
| Five-action Oracle | 2.647312 (det.) | — | — | **2.647312** |

Native and the oracle are deterministic, so their seed columns are marked (det.).

Oracle headroom vs native: -0.044984 nats/token.

## Primary paired comparisons

Per TEST sample, EXPERT_CACHE averaged over the three seeds. Negative is better.

| comparison | mean paired ΔNLL | bootstrap 95% CI |
|---|---|---|
| Expert − Native | **-0.001506** | [-0.005743, +0.002760] |
| Expert − Current | **+0.000347** | [-0.000761, +0.001670] |
| Expert − Fused | **+0.001056** | [-0.000023, +0.002156] |

Paired bootstrap: 10000 resamples over TEST sample indices, seed 314159; seeds not resampled independently.

## Verdict

**NOT_PROMISING**

| condition | requirement | observed | outcome |
|---|---|---|---|
| 1 | mean d_native <= -0.005 | mean = -0.001506 | **FAIL** |
| 2 | 95% CI upper bound for d_native < 0 | CI [-0.005743, +0.002760] | **FAIL** |
| 3 | mean d_current <= -0.005 | mean = +0.000347 | **FAIL** |
| 4 | 95% CI upper bound for d_current < 0 | CI [-0.000761, +0.001670] | **FAIL** |
| 5 | mean d_fused <= -0.005 | mean = +0.001056 | **FAIL** |
| 6 | 95% CI upper bound for d_fused < 0 | CI [-0.000023, +0.002156] | **FAIL** |
| 7 | EXPERT_CACHE beats FUSED_CACHE in mean TEST NLL in all three seeds | seed 42: 2.68881 vs 2.68915, seed 123: 2.69270 vs 2.69090, seed 2026: 2.69086 vs 2.68915 | **FAIL** |

XEC-P0 does not show that historical selected-expert states can be converted into a Layer-12 routing decision that improves true next-token likelihood beyond native routing, a current-only learned policy, and a fused-history cache, under this frozen setting. The expert-cache method direction is stopped.

## Descriptive metrics (not used for the verdict)

- native mean NLL: 2.692297
- five-action oracle mean NLL: 2.647312
- oracle headroom vs native: -0.044984
- TEST oracle action distribution: [271, 225, 188, 170, 170] (native 0.2646, swap 0.7354)
- five-action accuracy vs TEST oracle (seed-averaged):
  - CURRENT_ONLY: 0.2100
  - FUSED_CACHE: 0.2119
  - EXPERT_CACHE: 0.2106
- predicted action distribution (pooled over seeds):
  - CURRENT_ONLY: [865, 489, 640, 554, 524] (native 0.2816, swap 0.7184)
  - FUSED_CACHE: [803, 553, 625, 574, 517] (native 0.2614, swap 0.7386)
  - EXPERT_CACHE: [843, 513, 633, 557, 526] (native 0.2744, swap 0.7256)
- EXPERT_CACHE vs native: improved 0.4736, worsened 0.5264, equal 0.0000

## Why the policies failed to learn

All nine checkpoints selected **epoch 0** — the untrained initialization — because
validation cross entropy never improved on its starting value (~1.62, marginally above
chance ln(5) = 1.609) while training cross entropy collapsed to 0.03-0.08. The training
loop is functional: gradients flow and the 153600-parameter policy has ample capacity. It
simply memorized 2048 training samples without learning anything that generalized.

A diagnostic on the frozen oracle labels explains why, and it is a property of the target
rather than of the policy:

| quantity | TRAIN | VALIDATION |
|---|---|---|
| median margin, best vs second-best action | 0.0087 nats | 0.0087 nats |
| fraction with oracle gain < 0.01 nats | 0.4653 | 0.4980 |
| fraction with oracle gain < 0.001 nats | 0.3301 | 0.3164 |
| mean oracle gain over native | 0.0492 nats | 0.0461 nats |

For roughly a third of samples the five actions are separated by under 0.001 nats, so
which action "wins" is essentially arbitrary at that scale. The argmin label is close to
noise, and the total headroom available even to a perfect oracle is only ~0.049 nats. The
overfitting is the expected consequence of fitting a near-noise target.

This makes the null result readable rather than ambiguous: the failure is not that the
cache mechanism was starved of capacity or data relative to its competitors — all three
variants were parameter-matched, shared an initialization per seed, and saw the same
number of samples — but that the restricted five-action decision at Layer 12 carries very
little learnable signal in the first place.

Because the checkpoints are effectively untrained, the measured differences between the
three learned variants reflect near-random action choices rather than learned policies.
That is precisely why conditions 3-7, which compare EXPERT_CACHE against the learned
controls, are uninformative about the cache mechanism itself and are reported as failing
on their own terms.

## Numerical note recorded with the result

The selected-expert reconstruction invariant (weighted sum of the eight separately
executed experts vs the stock fused MoE output) showed a maximum ABSOLUTE error of
2.24e-02 on validation and 1.90e-02 on test across ~11.5M values per split. A
diagnostic on real contexts showed the RELATIVE error is ~1e-3 at every layer, at
fp16 eps (9.77e-04), with the largest absolute deviations occurring exactly where
|y| is large (5 to 12.7). This is fp16 accumulation noise, not a defect.

The decisive correctness gate is independent of that tolerance: **action 0 reproduced
the plain native NLL at 0.0e+00 error** on every split, so the intervention machinery
is exact where it matters for the metric.

## Interpretation limit

The expert-cache method direction is stopped. No more actions, no target-layer change, no cache-dimension tuning, no larger sample count, no attention heads, no optimizer change, no Layer 8, no other model or dataset, no rescue.

## Regenerating ignored caches

`.npz` caches are git-ignored. sha256:

- `test_policy_results.npz`: `3099c08c144f9fc6359a06424109540831c765a08a04dfe5d356124209a066b6`
- `test_oracle_descriptive.npz`: `cfc24c3ed24f82e6732c52331221432a90ea96df29148213f186feb68626d08b`
- `train_features.npz`: `df79814e9d91a2066bded4c61fa9866c46f8262a9020fdb9b5418727fc5d19bb`
- `validation_features.npz`: `45a09ec67e936a4e18fe314a590c038df19dd5e473248c4f64a4d7eea7faa35a`
- `test_features.npz`: `4bc771f893ae48bb8712599100570e5920c41249faacf2db1434a205f997632c`
- `train_oracle.npz`: `111c95efdc159873860734f55d4a0634d6483ff35170afddab9454d228bc54df`
- `validation_oracle.npz`: `91ea518fa662f00344e74bae61535a395f5fa3a94064bf545da6136824a1b507`

Regenerate with the command block in `README.md`.

## Relationship to prior projects

Prior projects are closed, read-only, results unmodified. The Layer-12 target was inherited
from the committed EIPC-P0 result, and no rejected expert was executed here.

Rescue modifications: none.
