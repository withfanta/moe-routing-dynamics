# DREV-P0 Results

Delayed Rejected-Evidence Value Pilot. A small screening experiment. The verdict
follows the four-condition rule committed in
`protocols/DREV_P0_PREREGISTRATION.md` before these numbers existed.

## Provenance

- model: `allenai/OLMoE-1B-7B-0125` revision `9b0c1aa87e34a20052389dce1f0cf01da783f654`, frozen, FP16, not quantized
- source layer: Layer 4 (code 3)
- data: WikiText-103-raw, only blocks never used by REDV-V1 or REDV-V2
- untouched pool: 1123 blocks; prior excluded: {'redv_v1': 1024, 'redv_v2': 2048, 'union': 3072}
- manifest sha256: `66cbc0807aeb144e69f270c5b5b26cd7cdb1a23ba346ff06b85e9102a82be15c`
- fit fingerprint: `feac5447b92787043b10d4804327695b…`
- test fingerprint: `1b0285636d5b8a1da9cd894a00924231…`
- compression: StandardScaler + PCA(n_components=64, svd_solver='randomized') per block, fit split only; probe dim 128
- probe: `sklearn.linear_model.Ridge(alpha=1.0)`, shuffle seed 271828
- GPUs: Tesla V100-SXM2-32GB (two independent single-GPU workers, no DDP)

## Primary result

| Δ | target layer | R² real | R² shuffled | D |
|---|---|---|---|---|
| 1 | 5 | -0.08378 | -0.04037 | **-0.04340** |
| 2 | 6 | -0.05771 | -0.03739 | **-0.02032** |
| 4 | 8 | -0.07582 | -0.02717 | **-0.04865** |
| 8 | 12 | -0.13461 | -0.02565 | **-0.10895** |

- **mean D over delayed horizons (Δ = 2, 4, 8) = -0.05931**
- immediate D (Δ = 1) = -0.04340

n_fit = 512, n_test = 512 per horizon; probe dim 128.

## Routing regret (descriptive only)

| Δ | target layer | mean G | median G | proportion G > 0 |
|---|---|---|---|---|
| 1 | 5 | 0.05226 | 0.01060 | 0.7520 |
| 2 | 6 | 0.05386 | 0.01319 | 0.7383 |
| 4 | 8 | 0.05406 | 0.01412 | 0.7793 |
| 8 | 12 | 0.04732 | 0.01167 | 0.7500 |

## Verdict

**NOT_PROMISING**

Mechanical application of the frozen rule:

| condition | requirement | observed | outcome |
|---|---|---|---|
| 1 | at least 2 of 3 delayed horizons have R2_real > 0 | 0 of 3 positive | **FAIL** |
| 2 | at least 2 of 3 delayed horizons have D > 0 | 0 of 3 positive | **FAIL** |
| 3 | mean_D_delayed >= +0.02 | mean_D_delayed = -0.05931 | **FAIL** |
| 4 | mean_D_delayed > D at the immediate horizon | -0.05931 vs D_5 = -0.04340 | **FAIL** |

All four conditions fail. Any single failure yields NOT_PROMISING.

### Reading the result

The pilot asked whether Layer-4 rejected evidence becomes *more* informative about
routing regret at later depths than at the immediate next layer. It does not.

Every `R2_real` is negative, so at no horizon does the real-augmented probe predict
held-out regret better than the test-set mean. Every `D` is negative, so at every
horizon the derangement control — rejected evidence belonging to a *different*
sample — predicts regret at least as well as a sample's own. And the delayed mean
is more negative than the immediate value, so there is no sign of information
emerging with depth; if anything the gap widens against the hypothesis, with the
most negative D at the largest distance (Δ = 8, D = -0.10895).

The compression worked as intended: at 128 dimensions against n = 512 the probes are
no longer in the p >> n regime that made the earlier REDV measurement
uninterpretable, and the R² values are now close to zero rather than large and
negative. The matched comparison is therefore readable, and it reads negative.

Routing regret itself is present and non-trivial at every horizon (mean G ≈ 0.047 to
0.054, positive for roughly 74-78% of samples), so the null result is not caused by
an absent or degenerate target.

## Interpretation limit

This delayed-rejected-evidence hypothesis is stopped. No added source layers or horizons, no PCA-dimension change, no alpha change, no larger n, no other model or dataset. No automatic DREV-P1.

This is a screening result under one frozen pilot setting, not a universal theorem.
It does not prove that no delayed cross-layer signal exists anywhere; it says this
pilot found none, and the pre-registered rule therefore stops the hypothesis.

## Relationship to the closed REDV project

REDV is closed, read-only, result unmodified. DREV-P0 reused validated REDV implementation only
(commits listed in `README.md`) and imported none of its conclusions.

Rescue modifications: none.
