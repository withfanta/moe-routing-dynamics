# Experiment index

The complete scientific lineage of the sparse-MoE routing-dynamics line, in order. Both
positive and negative experiments are recorded. Nothing was cherry-picked, relabelled, or
removed. The frozen numbers in each experiment's own `results.json` are authoritative; the
values quoted here are transcribed from them.

Each entry states the experiment's **formal status** (what its own preregistration or
protocol concluded under its own frozen decision rule) separately from its **scientific
interpretation** (what may legitimately be read from it). Where these differ, both are
kept.

| # | Experiment | Model | Kind | Status | Directory |
|---|---|---|---|---|---|
| 1 | REDV-V1 | OLMoE-1B-7B | preregistered confirmation | SUPPORTED (formal) / UNINTERPRETABLE (scientific) | [experiments/01_redv/](experiments/01_redv/) |
| 2 | REDV-V2 | OLMoE-1B-7B | preregistered confirmation | NOT_SUPPORTED | [experiments/01_redv/](experiments/01_redv/) |
| 3 | DREV-P0 | OLMoE-1B-7B | pilot | NOT_PROMISING | [experiments/02_drev/](experiments/02_drev/) |
| 4 | EIPC-P0 | OLMoE-1B-7B | preregistered confirmation | PROMISING | [experiments/03_eipc/](experiments/03_eipc/) |
| 5 | XEC-P0 | OLMoE-1B-7B | preregistered method test | NOT_PROMISING | [experiments/04_xec/](experiments/04_xec/) |
| 6 | EPD-P0 | OLMoE-1B-7B | exploratory | PATH-DOMINANT (exploratory label) | [experiments/05_epd/](experiments/05_epd/) |
| 7 | RMO-P0 | OLMoE-1B-7B | exploratory | SHORT-HISTORY (exploratory label) | [experiments/06_rmo/](experiments/06_rmo/) |
| 8 | RMC-P0 | JetMoE-8B | preregistered replication | REPLICATED | [experiments/07_rmc/](experiments/07_rmc/) |
| 9 | NHD-P0 | JetMoE-8B (reused artifacts) | mechanism disambiguation | RESIDUAL-HISTORY-VALUE | [experiments/08_nhd/](experiments/08_nhd/) |

---

## REDV-V1

**Formal preregistered result: SUPPORTED**

**Scientific interpretation: UNINTERPRETABLE**

Reason: p ≫ n / unequal dimensionality confound. Scale-matched random noise improved even
more.

The preregistered decision rule was met — mean ΔR² = +0.47406, and the frozen decision
sentence reads "REDV-V1 supports delayed predictive value of rejected near-miss expert
evidence under the frozen OLMoE/WikiText setting." But the comparison gave the rejected-
evidence condition more input dimensions than its control, in a regime where predictors
outnumber samples. A scale-matched random-noise condition produced an even larger apparent
improvement, which means the measured gain cannot be attributed to rejected-expert content.

**REDV-V1 is never relabelled as NOT_SUPPORTED.** Its formal verdict stands as recorded.
What changed is the interpretation, not the verdict: the design cannot distinguish signal
from dimensionality. This distinction is the reason REDV-V2 exists.

Source: `01_redv/artifacts/REDV_V1/results.json`, preregistration
`01_redv/protocols/REDV_V1_PREREGISTRATION.md`.

---

## REDV-V2

**Result: NOT_SUPPORTED**

Equal-dimensional real-vs-shuffled rejected evidence did not show stable incremental
routing-regret information.

With the dimensionality confound removed by construction, mean D = −0.09973 — the real
rejected evidence did *worse* than its dimension-matched shuffled control. The frozen
decision sentence: "REDV-V2 does not support sample-specific delayed predictive value of
rejected near-miss expert evidence under the frozen OLMoE/WikiText setting with a
dimension-matched shuffled control."

**The rejected-expert immediate-value line is closed.** No REDV-V3. The frozen stopping note
explicitly forbids sample-size increases, new datasets or models, alpha changes, PCA, MLPs,
rank changes, added layers, additional shuffle seeds, and subgroup reinterpretation. This is
a negative result and is preserved as one.

Source: `01_redv/artifacts/REDV_V2/results.json`, preregistration
`01_redv/protocols/REDV_V2_PREREGISTRATION.md`.

---

## DREV-P0

**Result: NOT_PROMISING**

Rejected near-miss expert outputs did not develop delayed predictive value at longer layer
gaps.

Layer-4 rejected evidence was tested against routing regret at horizons 1, 2, 4, and 8
(target layers 5, 6, 8, 12). Mean D over the delayed horizons = −0.05931, against D at the
immediate next layer = −0.04340: the delayed conditions were not better than the immediate
one, and neither was positive. The frozen decision sentence: "DREV-P0 does not show that
Layer-4 rejected evidence becomes more informative about routing regret at later depths than
at the immediate next layer, under this frozen pilot setting."

**The rejected-expert delayed-memory direction is closed.** No DREV-P1.

Source: `02_drev/artifacts/DREV_P0/results.json` plus per-horizon files
`horizon_{1,2,4,8}.json`.

---

## EIPC-P0

**Result: PROMISING**

Preserving selected-expert provenance predicted future native routing better than early
fused historical outputs.

Main conservative evidence: **identity − fused > 0**, i.e. mean A = +0.14215 (Layer 8
+0.10923, Layer 12 +0.17506), positive at both targets. Held-out R² for the identity-
preserving representation was +0.29939 at Layer 8 and +0.46211 at Layer 12.

The second recorded quantity, mean B = +0.37952, compares against an expert-identity-
destroyed (shuffled) control. **Do not interpret the shuffled-identity comparison as a clean
information-theoretic identity effect.** The conservative claim is the identity-vs-fused
contrast; the shuffled contrast is larger but is not a clean isolation of identity
information.

The frozen stopping note is explicit that this "does NOT prove an expert cache architecture
improves performance." Cache attention was not implemented here. That question is what
XEC-P0 went on to test.

Source: `03_eipc/artifacts/EIPC_P0/results.json`, preregistration
`03_eipc/protocols/EIPC_P0_PREREGISTRATION.md`.

---

## XEC-P0

**Result: NOT_PROMISING**

Historical expert provenance was not shown actionable for choosing the NLL-optimal
five-action routing intervention.

Three learned policies (CURRENT_ONLY, EXPERT_CACHE, FUSED_CACHE) × three seeds (42, 123,
2026) were trained to pick among five Layer-12 routing actions and scored by true next-token
NLL against native routing. All four preregistered conditions failed: mean d_native =
−0.001506 with 95% CI [−0.005743, +0.002760], against a required threshold of ≤ −0.005;
mean d_current = +0.000347 with its CI likewise not below zero.

The important distinction this experiment establishes:

```
predictive routing information
    !=
routing-regret information
    !=
actionable rerouting information
```

Being able to predict future routing from history does not imply that history can be used to
choose a better route. **No XEC-P1.** The expert-cache method direction is closed.

Source: `04_xec/artifacts/XEC_P0/results.json`, preregistration
`04_xec/protocols/XEC_P0_PREREGISTRATION.md`, nine trained policy checkpoints under
`04_xec/artifacts/XEC_P0/checkpoints/`.

---

## EPD-P0

**Exploratory interpretation: PATH-DOMINANT**

Main result: future-routing predictability is dominated by historical expert selection
identity/path rather than selected-expert activation content.

Key values (Layer 12 target, history layers 1–11, held-out R²):

```
FUSED            R² = +0.20114
ID_PATH          R² = +0.66969
CONTENT_RANK     R² = +0.06732
FULL_PROVENANCE  R² = +0.34007
```

Which experts were chosen predicts future routing largely without their activation content;
adding content on top of identity did not help (G_content_given_identity = −0.32962, while
G_identity_given_content = +0.27275).

**The PATH-DOMINANT label is not preregistered.** EPD-P0 is an exploratory decomposition
under a frozen protocol (`protocols/EPD_P0_PROTOCOL.md`, not a preregistration), and its own
`results.json` records `exploratory: true` and `no_formal_verdict: true`. The 0.02 gap figure
used in the pattern description is a discussion aid, not a significance threshold.

Source: `05_epd/artifacts/EPD_P0/results.json`.

---

## RMO-P0

**Exploratory result: SHORT-HISTORY**

This is the OLMoE follow-up. JetMoE is not involved here.

Layer-12 history curve (held-out R², cumulative windows):

```
L11 only   +0.59879
L10-11     +0.62320
L8-11      +0.64705
L4-11      +0.66265
L1-11      +0.66544
```

Stepwise gains: +0.02441, +0.02385, +0.01561, +0.00279. The curve rises then flattens, hence
the SHORT-HISTORY label.

Allowed interpretation: earlier expert-selection history contains incremental predictive
structure beyond the immediately preceding layer under this frozen linear probe.

**Do not claim formal Markov order.** RMO-P0's own recorded scientific limit: it "does not
prove causal memory, a formal Markov order, that a recurrent router will help, that storing
old paths improves NLL, or that routing should use long-term memory." It is exploratory and
post-hoc on already-observed data — it ran no inference of its own, reusing EPD-P0's
extracted arrays read-only under verified hashes.

Source: `06_rmo/artifacts/results.json`.

---

## RMC-P0

**Model: JetMoE-8B**

**Verdict: REPLICATED**

An independent cross-model test of the RMO-P0 phenomenon in an architecturally different
sparse MoE, under a preregistered six-condition rule.

Layer 12 (held-out R² by history window size k):

```
k=1  +0.20924
k=2  +0.34982
k=4  +0.35199
k=8  +0.38275

Delta4 = +0.14275
bootstrap CI [+0.10998, +0.17916]
```

Layer 20:

```
k=1  +0.13898
k=2  +0.24936
k=4  +0.34425
k=8  +0.35463

Delta4 = +0.20528
bootstrap CI [+0.16891, +0.24396]
```

Paired bootstrap, 10000 resamples, seed 314159; fraction above zero 1.0 at both targets.

Allowed conclusion: in a second, architecturally different sparse MoE, routing history beyond
the immediately preceding layer retains substantial held-out predictive information.

### JetMoE router-identification audit (preserved)

JetMoE has two mixtures per block, and this audit is why the target router is known to be the
right one:

- MoA (attention) and MLP routers use **the same router class and the same 2048→8 shape**, so
  shape alone can never distinguish them.
- Identification was performed by **object identity** (`model.layers[i].mlp.router`), not by
  shape or position.
- Captured MLP logits **exactly matched** a manual `W_router @ x` recomputation from the
  hidden state entering `.mlp` (max absolute difference 0.00e+00; minimum difference against
  MoA logits 1.0039; agreement 1.000000).
- The transformers 4.45.1 convenience path `output_router_logits=True` was **not used**. It
  interleaves attention and MLP logits in one flat tuple, and in that version it is also
  outright broken (`AttributeError: 'JetMoeForCausalLM' object has no attribute
  'num_experts'`). The breakage is recorded in `artifacts/router_identification.json`.

Note on the k curve: RMC-P0 applied PCA to the older-history block to hold every probe at
exactly 16 dimensions, and explained variance falls with k (Layer 12: k=2 1.00, k=4 0.5735,
k=8 0.366). This works *against* larger k, so it cannot manufacture the gains, but it means
the k curve is **not** a memory-decay curve.

Source: `07_rmc/artifacts/results.json`, `07_rmc/artifacts/router_identification.json`,
preregistration `07_rmc/protocol.md`.

---

## NHD-P0

**Classification: RESIDUAL-HISTORY-VALUE**

A minimal mechanism-disambiguation experiment. RMC-P0 showed recent + older history beats
recent only, but that alone does not establish that older history carries information absent
from the recent routing state: the future-routing information might already be encoded in the
recent state R and merely related to future routing Y nonlinearly, where a linear probe would
fail to decode it. NHD-P0 asks only whether the history gain survives a small nonlinear
recent-state predictor.

Layer 12:

```
LINEAR recent     +0.20972
LINEAR history    +0.37188
gain              +0.16216

MLP_RECENT mean           +0.21111
MLP_RECENT_MATCHED mean   +0.21809
MLP_HISTORY mean          +0.38249

A          +0.00139
B          +0.17137
B_matched  +0.16440

residual TEST R²    +0.20549
permuted history    -0.33325
```

Layer 20:

```
LINEAR recent     +0.13948
LINEAR history    +0.35382
gain              +0.21434

MLP_RECENT mean           +0.14884
MLP_RECENT_MATCHED mean   +0.14834
MLP_HISTORY mean          +0.36745

A          +0.00936
B          +0.21861
B_matched  +0.21912

residual TEST R²    +0.23556
permuted history    -0.32149
```

A is the nonlinear gain over Ridge given the recent state alone; B is the history gain over
the nonlinear recent-state model; B_matched is the same against a parameter-matched control
(hidden width 77, 1317 parameters against MLP_HISTORY's 1320, fixed mechanically before any
TEST number existed). Both targets received the RESIDUAL-HISTORY-VALUE label independently.

Allowed conclusion: the observed history gain is not explained by simple linear-decoding
insufficiency. Historical routing retains substantial held-out predictive value after the
tested nonlinear recent-state decoding and cross-fitted residualization.

**Do not claim that H is mathematically proven to contain information absent from R.** Even
RESIDUAL-HISTORY-VALUE shows only that the tested nonlinear recent-state predictor does not
explain the full held-out history gain. B is not conditional information, and no
information-theoretic conditional-independence claim is made.

NHD-P0 ran no inference: it reuses the frozen RMC-P0 artifacts read-only under four pinned
sha256 hashes.

Source: `08_nhd/artifacts/results.json`, protocol `08_nhd/protocol.md`.
