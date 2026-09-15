# EPD-P0 Results

Expert Provenance Decomposition Pilot. **Exploratory**: this reports a decomposition
pattern, not a SUPPORTED/NOT_SUPPORTED verdict, and builds no architecture.

## Provenance

- model: `allenai/OLMoE-1B-7B-0125` revision `9b0c1aa87e34a20052389dce1f0cf01da783f654`, frozen, FP16, no gradients
- target: native Layer 12 router logits; history Layers 1-11
- data: WikiText-103-raw train, fresh relative to EIPC-P0 and XEC-P0
  - excluded: EIPC-P0 2048 + XEC-P0 3584 blocks (overlap 0), union 5632
- manifest sha256: `bbbe06fcb451b217de3b03cf75d37921a4e19f555d95d92302b8898d8a3466ba`
  - FIT fingerprint `e88b138a2c496524dfd53a9c35030e57…`, TEST `190ee4bb4e562d24c1396319cb42c7e4…`
- n_fit 512, n_test 256 (intentionally small: directional exploration)
- **reused EIPC-P0 projection**: seed 20260917, R^2048 -> R^32, sha256 `fb9e6e9b7568f6d4a5ce74a7bee1ba2eab5a48a02fb2a93f80520e4ec0480282` (verified, never regenerated)
- compression: StandardScaler -> PCA(32, randomized) -> StandardScaler, fit on FIT only
- probe: `sklearn.linear_model.Ridge(alpha=1.0), multi-output`

## Main result

| representation | raw dim | PCA evr @32 | R² future Layer-12 router |
|---|---|---|---|
| FUSED | 352 | 0.5375 | **+0.20114** |
| ID_PATH | 704 | 0.5217 | **+0.66969** |
| CONTENT_RANK | 2816 | 0.1969 | **+0.06732** |
| FULL_PROVENANCE | 22528 | 0.1518 | **+0.34007** |

Descriptive gaps (not causal effects):

| gap | value |
|---|---|
| FULL − FUSED (`G_full_vs_fused`) | +0.13893 |
| FULL − CONTENT (`G_identity_given_content`) | +0.27275 |
| FULL − ID (`G_content_given_identity`) | -0.32962 |

## Layerwise exploratory curves

Single-layer probes, each compressed to 16 dims.

| historical layer | ID_PATH R² | CONTENT_RANK R² |
|---|---|---|
| 1 | +0.14253 | +0.00914 |
| 2 | +0.25597 | -0.00721 |
| 3 | +0.35859 | -0.00605 |
| 4 | +0.41664 | +0.00556 |
| 5 | +0.40384 | -0.01048 |
| 6 | +0.44991 | -0.00814 |
| 7 | +0.44797 | -0.00657 |
| 8 | +0.49381 | +0.02139 |
| 9 | +0.53850 | +0.00420 |
| 10 | +0.57216 | +0.00700 |
| 11 | +0.59927 | +0.01017 |

Best ID_PATH layer 11, best CONTENT_RANK layer 8;
mean ID_PATH +0.42538 vs mean CONTENT_RANK +0.00173.

ID_PATH rises monotonically with depth (0.14 at Layer 1 to 0.60 at Layer 11), while
CONTENT_RANK stays near zero at every depth. The relation is distributed across depth
and strengthens toward the target, and it is carried by identity, not content.

## Interpretation

**PATTERN: PATH_DOMINANT**

Historical expert-selection path carries most of the signal: which experts were chosen predicts future routing, largely without their activation content.

### Reading the ordering honestly

`ID_PATH` (+0.670) exceeds `FULL_PROVENANCE` (+0.340) even though FULL contains strictly
more information — binarizing FULL's slot occupancy reproduces ID_PATH exactly, which was
verified. The ordering is a **compression artifact of the shared 32-dim budget**, not an
information paradox: PCA retains 52% of ID_PATH's variance but only 15% of FULL's, because
FULL is 22528 dense dimensions whose variance is dominated by activation magnitudes rather
than the sparse identity pattern. A post-hoc control confirmed FULL stays below ID_PATH
even at 64 and 128 components (0.376 and 0.444 vs 0.695 and 0.706).

Two observations make the path reading robust rather than an artifact of that budget:

1. `CONTENT_RANK` is near zero (+0.067) and near zero at *every* individual layer, so
   rank-ordered activation content on its own carries almost no future-routing information
   at this scale.
2. Concatenating content onto the path changes nothing: ID_PATH alone +0.66969 versus
   ID_PATH+CONTENT +0.66728, a delta of −0.0024.

So the EIPC-P0 provenance advantage appears to be carried by **which experts were selected**
across depth, not by what those experts computed, and not by an identity-content interaction.

### What this does not establish

This is a single small pilot (FIT 512 / TEST 256) at one target layer in one model on one
dataset, with linear probes under one fixed projection. It does not show that a cache or any
architecture would benefit — XEC-P0 already found the related five-action Layer-12 decision
carries little actionable signal. A plausible and untested reading is that the selection path
is predictable partly because routing is autocorrelated across depth, which would make it
informative without being useful. Nothing here was built into a method.

## Descriptive expert transitions

Source-layer expert -> Layer-12 expert enrichment above the Layer-12 marginal
(mean marginal 0.12500). No statistical testing; these do not
contribute to the pattern.

| source layer | source expert | target expert | P(target\|source) | marginal | enrichment | source support |
|---|---|---|---|---|---|---|
| 7 | 42 | 5 | 0.1250 | 0.0560 | +0.0690 | 1 |
| 11 | 56 | 47 | 0.1250 | 0.0612 | +0.0638 | 7 |
| 10 | 46 | 16 | 0.1250 | 0.0716 | +0.0534 | 1 |
| 8 | 2 | 38 | 0.0938 | 0.0482 | +0.0456 | 12 |
| 11 | 56 | 27 | 0.0536 | 0.0182 | +0.0353 | 7 |
| 6 | 42 | 26 | 0.1250 | 0.0951 | +0.0299 | 1 |
| 11 | 56 | 23 | 0.0893 | 0.0599 | +0.0294 | 7 |
| 7 | 42 | 48 | 0.1250 | 0.0964 | +0.0286 | 1 |
| 8 | 2 | 51 | 0.1146 | 0.0872 | +0.0273 | 12 |
| 9 | 29 | 33 | 0.1133 | 0.0859 | +0.0273 | 64 |

**Caveat:** several top pairs rest on very low support (1-12 source selections out of 768
samples), so they are weak illustrations rather than findings. Only the Layer-9 expert-29 pair
(support 64) and Layer-11 expert-56 pairs have appreciable backing.

## Numerical note

Selected-expert reconstruction across shards: max absolute 1.300e-02, max relative
2.515e-03. Judged on a relative basis, since fp16 accumulation error scales with
the magnitude of the fused output. Projection linearity held at ~1e-8 on real captures.

## Scope

Architecture built: none. Rescue modifications: none.
Prior projects: closed, read-only, results unmodified.
