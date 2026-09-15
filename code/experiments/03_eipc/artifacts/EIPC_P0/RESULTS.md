# EIPC-P0 Results

Expert-Identity Preservation Cache Pilot. An information-content experiment.
The verdict follows the five-condition rule committed in
`protocols/EIPC_P0_PREREGISTRATION.md` before these numbers existed.

## Provenance

- model: `allenai/OLMoE-1B-7B-0125` revision `9b0c1aa87e34a20052389dce1f0cf01da783f654`, frozen, FP16, not quantized
- data: WikiText-103-raw **train** split (fresh; prior projects never used it)
- manifest sha256: `23d75e6ee9f2a76249001e3e9411254ad07efcc20b667c6b317e8d523761fc4c`
- FIT fingerprint: `7a7910a0f6653d4912ed9d458e9b78fc…`
- TEST fingerprint: `ac0dce7ccfcba1cba05527e2d0ab020c…`
- fixed projection: R^2048 -> R^32, seed 20260917, sha256 `fb9e6e9b7568f6d4a5ce74a7bee1ba2eab5a48a02fb2a93f80520e4ec0480282`
- compression: StandardScaler -> PCA(n_components=64, svd_solver='randomized') -> StandardScaler, fit on FIT only
- probe: `sklearn.linear_model.Ridge(alpha=1.0), multi-output`; shuffle seed 314159
- n_fit = 1024, n_test = 1024, probe input 64-d, target 64-d

## Primary result

| target | history | R² fused | R² identity | R² shuffled | A | B |
|---|---|---|---|---|---|---|
| Layer 8 | L1–7 | +0.19016 | +0.29939 | -0.00149 | **+0.10923** | **+0.30088** |
| Layer 12 | L1–11 | +0.28705 | +0.46211 | +0.00396 | **+0.17506** | **+0.45815** |

- **mean A = +0.14215**  (identity vs early fusion)
- **mean B = +0.37952**  (identity vs identity-destroyed control)

Raw dimensions before the shared 64-d compression:

| target | FUSED | EXPERT_IDENTITY / SHUFFLED_IDENTITY |
|---|---|---|
| Layer 8 | 224 | 14336 |
| Layer 12 | 352 | 22528 |

## Verdict

**PROMISING**

| condition | requirement | observed | outcome |
|---|---|---|---|
| 1 | R2_identity > 0 at both targets | L8 +0.29939, L12 +0.46211 | **PASS** |
| 2 | A > 0 at both targets | L8 +0.10923, L12 +0.17506 | **PASS** |
| 3 | B > 0 at both targets | L8 +0.30088, L12 +0.45815 | **PASS** |
| 4 | mean(A) >= +0.02 | mean(A) = +0.14215 | **PASS** |
| 5 | mean(B) >= +0.02 | mean(B) = +0.37952 | **PASS** |

All five conditions pass.

Individual selected-expert provenance contains incremental held-out information about future routing states beyond early fusion and an expert-identity-destroyed control, under this frozen OLMoE/WikiText setting.

## Reading the result — and its limits

Both A and B are positive at both depths, and the effect is larger at the later
target (Layer 12) than the middle one (Layer 8). Three checks were run after the
verdict was recorded, to test whether the positive result is an artifact rather
than the phenomenon. All are diagnostic only and changed nothing.

**No target leakage.** History for each target is strictly earlier than the target
layer (Layers 1–7 for target 8, Layers 1–11 for target 12), verified against the
frozen layer indices.

**Not a PCA-capacity artifact.** FUSED inputs are full rank (224 and 352) and every
representation receives the identical 64-component budget, so FUSED is not starved
of components relative to EXPERT_IDENTITY.

**The control behaves as specified.** Shuffling preserves the per-layer vector sum
to 4.8e-07, the same vector multiset, and the same occupied-slot count; only which
expert slot holds which vector changes.

**The most important caveat.** `R2_shuffled` is approximately zero at both targets
even though the fused sum remains arithmetically recoverable from the shuffled
slots. The reason is that the cyclic shift varies per sample and per layer, so a
*linear* probe on 64 PCA components cannot undo it. B therefore measures the value
of expert identity **to this probe class under this compression**, not an
information-theoretic bound. B being much larger than A should be read in that
light: A is the more conservative of the two quantities, since FUSED and
EXPERT_IDENTITY are both linearly accessible to the probe.

## Interpretation limit

This does NOT prove an expert cache architecture improves performance. EIPC-P0 stops here; cache attention is not implemented. A later method experiment requires explicit user authorization.

This is a screening result about information content under one frozen setting. It
does not show that an expert-state cache would improve language-model accuracy,
latency, or memory cost, and it is not evidence about any specific architecture.

## Relationship to the closed prior projects

The prior rejected-expert projects are closed, read-only, results unmodified. EIPC-P0 asks a
different question about *selected*-expert provenance, executed no rejected experts,
used no counterfactual routing or routing regret, and imported no prior conclusion.

Rescue modifications: none.
