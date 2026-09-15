# REDV-V2 Results

Dimension-Matched Rejected-Evidence Test. The single, final authorized
follow-up to REDV-V1. Verdict follows the rule committed in
`protocols/REDV_V2_PREREGISTRATION.md` before these numbers existed.

## Provenance

- model: `allenai/OLMoE-1B-7B-0125` revision `9b0c1aa87e34a20052389dce1f0cf01da783f654`
- GPU: Tesla V100-SXM2-32GB (FP16, `torch.inference_mode()`, frozen, not quantized)
- data: WikiText-103-raw, untouched blocks only; every REDV-V1 block excluded; train unused
- data manifest sha256: `52cd97f5daea74abcd588cb580e6324c3c0f552ac759e6bf98be96b5bbe53c38`
- fit fingerprint: `bf4bcf5fd59a583517bdf72df713680b…`
- test fingerprint: `6c3159fbfdb817d5d5f51cee2065648b…`
- sampling seed 20260915, shuffle seed 314159
- probe: `sklearn.linear_model.Ridge(alpha=1.0)`

## The matched comparison

REDV-V1's confound was a dimensionality gap (2112 vs 4160 at n=512). REDV-V2
compares two **equal-dimension 4160-d** models under identical Ridge, identical
alpha, and identical preprocessing:

    REAL:     [b_i ; r_i]
    CONTROL:  [b_i ; r_perm(i)]      one fixed derangement, seed 314159

The control preserves dimensionality, empirical distribution, scale, and
within-vector covariance. It destroys only the sample-specific correspondence
between rejected evidence and routing regret.

## Primary result

| transition | R² baseline | R² real | R² shuffled | D | real − baseline |
|---|---|---|---|---|---|
| 4 -> 5 | -3.38419 | -0.94185 | -0.74550 | **-0.19634** | +2.44235 |
| 8 -> 9 | -1.51443 | -0.49803 | -0.36580 | **-0.13224** | +1.01640 |
| 12 -> 13 | -1.06260 | -0.27184 | -0.30124 | **+0.02939** | +0.79075 |

**mean D = -0.09973**

n_fit = 1024, n_test = 1024 per transition.

## Routing regret (descriptive only)

| transition | mean G | median G | proportion G > 0 |
|---|---|---|---|
| 4 -> 5 | 0.05372 | 0.01148 | 0.7832 |
| 8 -> 9 | 0.05263 | 0.01292 | 0.7500 |
| 12 -> 13 | 0.04575 | 0.01035 | 0.7373 |

## Verdict

**NOT_SUPPORTED**

Mechanical application of the pre-registered rule:

| condition | requirement | outcome |
|---|---|---|
| 1 | R²_real > 0 for all three | **FAIL** — all three negative |
| 2 | D > 0 for all three | **FAIL** — negative at 4→5 and 8→9 |
| 3 | mean(D) ≥ +0.02 | **FAIL** — mean D = -0.09973 |

All three conditions fail. Any single failure yields NOT_SUPPORTED.

### Reading the result

Two independent reasons the premise is unsupported:

1. **The regressors are unusable.** Every R²_real is negative, so the real-augmented
   probe predicts held-out regret worse than the test-set mean. Condition 1 exists
   precisely to block a success claim resting on two unusable regressors.
2. **Matched control beats real evidence.** At 4→5 and 8→9 the shuffled control
   scores *higher* than correctly matched evidence (D = −0.196, −0.132). Rejected
   evidence from another sample predicts regret no worse — in fact better — than a
   sample's own. The small positive D at 12→13 (+0.029) does not survive the rule,
   and is not interpreted separately.

The large positive `real − baseline` column (+2.44, +1.02, +0.79) is exactly the
REDV-V1 artifact reproduced: adding 2048 columns to a p≫n Ridge shrinks predictions
toward the mean and lifts a very negative R² toward zero. Against the
dimension-matched control that apparent gain disappears, which is what REDV-V2 was
built to detect.

The entire rejected-expert delayed-value direction is stopped. No REDV-V3. No sample-size increase, no new dataset or model, no alpha change, no PCA, no MLP, no rank change, no added layers, no additional shuffle seeds, no subgroup reinterpretation. The negative result is preserved.

## Relationship to REDV-V1

REDV-V1's verdict remains permanently recorded as **SUPPORTED**
and is not changed. Its artifacts were read-only throughout REDV-V2.

Rescue modifications: none.
