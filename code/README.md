# code — sparse-MoE routing-dynamics experimental record

Archival package of eight experiments on routing dynamics in sparse mixture-of-experts
language models. Each experiment is preserved as an independently understandable unit with its
own protocol, source code, tests, and frozen results, copied from its source repository's
frozen commit. Nothing was refactored, recomputed, or unified into a common framework.

## Current strongest supported statement

> Across OLMoE and JetMoE, routing history beyond the immediately preceding expert-selection
> state provides substantial held-out predictive value for future routing.
>
> In JetMoE, this gain persists under small nonlinear recent-state decoders and cross-fitted
> residualization.

That is the ceiling of what these experiments support. In particular the record does **not**
establish that history contains information mathematically absent from the recent state, any
causal routing memory, any formal Markov order, that recurrent routing improves language-model
performance, or that any of this is a universal property of sparse MoEs. XEC-P0 specifically
found that predictive value did **not** translate into a better routing decision.

## The eight experiments

| # | Directory | Experiment | Model | Kind | Status |
|---|---|---|---|---|---|
| 1 | [experiments/01_redv/](experiments/01_redv/) | REDV-V1 | OLMoE-1B-7B | preregistered | SUPPORTED (formal) but **UNINTERPRETABLE** (p ≫ n / dimensionality confound) |
| 1 | [experiments/01_redv/](experiments/01_redv/) | REDV-V2 | OLMoE-1B-7B | preregistered | **NOT_SUPPORTED** — rejected-expert immediate-value line closed |
| 2 | [experiments/02_drev/](experiments/02_drev/) | DREV-P0 | OLMoE-1B-7B | pilot | **NOT_PROMISING** — rejected-expert delayed-memory direction closed |
| 3 | [experiments/03_eipc/](experiments/03_eipc/) | EIPC-P0 | OLMoE-1B-7B | preregistered | **PROMISING** — selected-expert provenance predicts future routing (identity − fused > 0) |
| 4 | [experiments/04_xec/](experiments/04_xec/) | XEC-P0 | OLMoE-1B-7B | preregistered method test | **NOT_PROMISING** — prediction did not imply actionable rerouting |
| 5 | [experiments/05_epd/](experiments/05_epd/) | EPD-P0 | OLMoE-1B-7B | exploratory | **PATH-DOMINANT** (exploratory label, not preregistered) |
| 6 | [experiments/06_rmo/](experiments/06_rmo/) | RMO-P0 | OLMoE-1B-7B | exploratory | **SHORT-HISTORY** (exploratory label) |
| 7 | [experiments/07_rmc/](experiments/07_rmc/) | RMC-P0 | JetMoE-8B | preregistered replication | **REPLICATED** |
| 8 | [experiments/08_nhd/](experiments/08_nhd/) | NHD-P0 | JetMoE-8B (reused artifacts) | mechanism disambiguation | **RESIDUAL-HISTORY-VALUE** |

Four of the nine results are negative or uninterpretable. They are kept as such: see
[EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md) for each experiment's frozen verdict, key numbers,
and the specific interpretation limits that apply to it.

## Headline numbers

Two experiments carry the positive claim.

**RMC-P0** (JetMoE-8B, held-out R² by history window size k, Δ4 = k=4 − k=1 with paired
bootstrap CI):

| Target | k=1 | k=2 | k=4 | k=8 | Δ4 | 95% CI |
|---|---|---|---|---|---|---|
| Layer 12 | +0.20924 | +0.34982 | +0.35199 | +0.38275 | +0.14275 | [+0.10998, +0.17916] |
| Layer 20 | +0.13898 | +0.24936 | +0.34425 | +0.35463 | +0.20528 | [+0.16891, +0.24396] |

**NHD-P0** (same model, testing whether a nonlinear recent-state decoder explains the gain):

| Target | linear gain | A | B | B_matched | residual R² | permuted |
|---|---|---|---|---|---|---|
| Layer 12 | +0.16216 | +0.00139 | +0.17137 | +0.16440 | +0.20549 | −0.33325 |
| Layer 20 | +0.21434 | +0.00936 | +0.21861 | +0.21912 | +0.23556 | −0.32149 |

A is the nonlinear gain given the recent state alone; B is the history gain over the nonlinear
recent-state model; B_matched uses a parameter-matched control (1317 vs 1320 parameters). B is
not conditional information.

## Documents

- [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md) — full scientific lineage, per-experiment verdicts,
  key values, and interpretation limits.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) — dependency chain, kinds of evidence, environment,
  model and dataset revisions, and what can and cannot be reproduced here.
- [SOURCE_AUDIT.md](SOURCE_AUDIT.md) / [source_audit.json](source_audit.json) — read-only audit
  of every source repository at packaging time, with frozen commit SHAs.
- [ARTIFACTS_NOT_VERSIONED.md](ARTIFACTS_NOT_VERSIONED.md) — the 16 large artifacts omitted,
  with sizes, sha256 hashes, and regeneration commands.
- Per experiment: `PACKAGING_PROVENANCE.md` (source commit, files included and omitted, exact
  scientific status, reproduction entry point) and `REPRODUCTION.md` (environment, model
  revision, dataset, hardware, exact commands, expected outputs).

## Environment

All experiments ran under one pinned environment: python 3.10.19, torch 2.4.1+cu121,
transformers 4.45.1, numpy 1.24.4, scikit-learn 1.3.2, pytest 8.3.5. The transformers pin is
load-bearing — see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

Models: `allenai/OLMoE-1B-7B-0125` at revision
`9b0c1aa87e34a20052389dce1f0cf01da783f654` (experiments 1–6) and `jetmoe/jetmoe-8b` at
revision `d8fd02ccf7911aa8148a63c7984ffd2e465b0352` (experiments 7–8), both BASE, float16, no
quantization. Data: `Salesforce/wikitext` / `wikitext-103-raw-v1` at revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, train split.

## Layout

```
code/
├── README.md                     this file
├── EXPERIMENT_INDEX.md           scientific lineage, all nine results
├── SOURCE_AUDIT.md               read-only source audit
├── source_audit.json             machine-readable audit
├── REPRODUCIBILITY.md            dependency chain and evidence kinds
├── ARTIFACTS_NOT_VERSIONED.md    omitted large artifacts, hashes, regeneration
├── _make_audit.py                regenerates source_audit.json (read-only on sources)
└── experiments/
    ├── 01_redv/   REDV-V1, REDV-V2
    ├── 02_drev/   DREV-P0
    ├── 03_eipc/   EIPC-P0
    ├── 04_xec/    XEC-P0
    ├── 05_epd/    EPD-P0
    ├── 06_rmo/    RMO-P0
    ├── 07_rmc/    RMC-P0
    └── 08_nhd/    NHD-P0
```

Each experiment keeps its original internal structure (`src/`, `scripts/`, `tests/`,
`protocols/`, `artifacts/` for the earlier ones; a flat layout for RMO/RMC/NHD) so relative
paths and imports still work.
