# Manuscript review and source audit

## Deliverable

- Title: **Beyond the Previous Layer: Residual Predictive Structure in Sparse MoE Routing**
- Authors, in order: **Hao Li; Yasuyuki Tahara; Yuichi Sei**.
- Corresponding author: **Yuichi Sei**.
- Affiliation: Department of Informatics, Graduate School of Informatics and Engineering,
  The University of Electro-Communications, Tokyo, Japan.
- All three supplied email addresses and ORCIDs appear in the first-page footnote.
- Repository: `https://github.com/withfanta/moe-routing-dynamics`.
- Base commit: `febc169a08827239e0e2057a4f81915cfbcc4aba`.
- Repository root used: `/workspace/scratch/b62cf76649a4/moe-routing-dynamics`.
- Existing preprint directory: `preprint/`.
- Original template: `preprint/The Yale arXiv Paper Template/`.
- Experimental code: `code/`.
- New directory: `preprint/Beyond-the-Previous-Layer-Residual-Predictive-Structure-in-Sparse-MoE-Routing/`.
- Printed code URL: `https://github.com/withfanta/moe-routing-dynamics/tree/main/code`.
- Total length: **5 PDF pages, including references**.

## Frozen source map

Paths below are relative to the repository root. Each JSON is accompanied by the
`RESULTS.md` in the same directory. These reports and their protocols are included
in the SHA-256 manifest. No experiment was rerun to write the paper.

| Experiment | Frozen source-repository commit | Numerical artifact | Paper location |
|---|---|---|---|
| EIPC-P0 | `517f74edb05d50b9d512248eeff2b5864afc59af` | `code/experiments/03_eipc/artifacts/EIPC_P0/results.json` | Table 1, Section 3.1 |
| EPD-P0 | `995d23ebb2cc37772b38abefc8d16d1af35586e3` | `code/experiments/05_epd/artifacts/EPD_P0/results.json` | Table 1, Section 3.1 |
| RMO-P0 | `a8e9d83cc2a6dd4800db72e610bd010ff80a83bb` | `code/experiments/06_rmo/artifacts/results.json` | Table 2, Section 3.2 |
| RMC-P0 | `cc60edd9313c15c9a0f433d9f897a38fb44ef1a3` | `code/experiments/07_rmc/artifacts/results.json` | Table 2, Section 3.2 |
| NHD-P0 | `f1a7d4ab9313798a20297921dd8a8125dee71763` | `code/experiments/08_nhd/artifacts/results.json` | Table 3, Section 3.3 |
| XEC-P0 | `a664cb5de30900119453d25eba2c4034bfef7b05` | `code/experiments/04_xec/artifacts/XEC_P0/results.json` | Section 4 |

Additional sources used for specific statements:

- EPD path + content: `code/experiments/05_epd/artifacts/EPD_P0/RESULTS.md`,
  section “Reading the ordering honestly,” item 2. The value **0.66728** is a
  post-hoc report value, not a field in the main JSON. The generator extracts it
  from that hash-verified report.
- JetMoE MLP router audit:
  `code/experiments/07_rmc/artifacts/router_identification.json`.
- NHD cross-fitting and early stopping: `code/experiments/08_nhd/analyze.py`,
  `protocol.md`, and `artifacts/RESULTS.md`.
- XEC epoch-zero checkpoint selection:
  `code/experiments/04_xec/artifacts/XEC_P0/RESULTS.md`, section
  “Why the policies failed to learn.”
- Independent source commit identities: each experiment's
  `PACKAGING_PROVENANCE.md`.

## Numerical reconciliation

Tables use five decimal places, formatted directly from stored values. XEC NLL
values and intervals use six decimal places. Differences are taken from the
stored unrounded differences, rather than recalculated from rounded table cells.

The supplied planning text quoted EIPC mean Identity minus Fused as approximately
**0.1422**. The frozen JSON and report give **0.14215** at five decimal places.
The manuscript consistently uses the source value; no scientific result changed.
Rounding intermediate values can account for the planning-text discrepancy.

RMC and NHD ridge scores intentionally differ: RMC compresses older history into
eight PCA components; NHD uses 24 raw history coordinates. Neither baseline was
silently substituted for the other. The residual scores predict a residual target,
so Table 3 distinguishes them from the joint router-logit scores.

All table decimals were checked against the six frozen JSON files or the explicitly
identified EPD report value. The abstract's decimals and all 14 XEC means,
differences, and interval endpoints were checked separately. The 27 manifest
entries matched their SHA-256 values before table generation and after writing.

## Interpretation and coverage

The central claim is additional held-out predictive value of older selected-expert
states when the immediately preceding selection is already supplied to the probe.
The paper retains the provenance/path decomposition, exploratory OLMoE history
curve, preregistered JetMoE replication, nonlinear controls, parameter matching,
cross-fitted residual prediction, and closed XEC actionability result.

RMO is explicitly exploratory on EPD samples. RMC is the preregistered
cross-architecture replication. NHD reuses RMC's captures and split. Residualizers
use seed 42; the three-seed table concerns the nonlinear joint predictors.

The paper reports the EPD curve as generally strengthening, rather than strictly
monotone: the frozen layerwise values contain small reversals. The evidence concerns
probe-accessible structure under the stated representations and decoder families.
Formal Markov order, causal memory, information-theoretic absence from the recent
state, universal MoE behavior, and improved language-model quality are not claimed.

XEC's three primary paired intervals cross zero. Validation selected untrained
initializations for all nine policy checkpoints. The paper preserves the negative
finding for that decision and training setup without extending it to all possible
uses of history. REDV/DREV are separate branches and are excluded as requested.

The short paper includes the main numerical evidence and conclusions. Complete
source reports also retain auxiliary reconstruction checks, transition examples,
policy action distributions, and other implementation diagnostics.

## Bibliography and writing

The nine BibTeX records were fetched from arXiv's export endpoint and checked
against the corresponding paper metadata. The exact titles and author lists were
retained. Years follow the exported arXiv records, which may reflect a revised
version rather than the first submission year.

| Key | Primary source | Role |
|---|---|---|
| `readme` | https://arxiv.org/abs/2410.19123 | Router decoupling and pre-gating |
| `rmoe` | https://arxiv.org/abs/2408.06793 | Recurrent cross-layer router |
| `pathmoe` | https://arxiv.org/abs/2603.18297 | Expert paths and shared router parameters |
| `geometry` | https://arxiv.org/abs/2609.02404 | Shared routing geometry and dynamics |
| `esrl` | https://arxiv.org/abs/2609.13058 | Recording and replaying expert paths |
| `hero` | https://arxiv.org/abs/2609.08189 | History-aware dynamic layer skipping |
| `olmoe` | https://arxiv.org/abs/2409.02060 | OLMoE backbone |
| `jetmoe` | https://arxiv.org/abs/2404.07413 | JetMoE backbone |
| `wikitext` | https://arxiv.org/abs/1609.07843 | WikiText corpus |

Writing follows a single claim with a sequence of discriminating comparisons.
Results lead the abstract and experimental subsections. Necessary scope conditions
are concentrated in the evaluation description and Discussion. No borrowed prose,
priority claim, invented reference, or speculative mechanism was added.

## Build and delivery checks

- Compiled in a fresh directory from `.tex`, `.bib`, and the unchanged `.cls`.
- `latexmk` completed successfully, including BibTeX and cross-reference passes.
- No undefined references, undefined citations, missing assets, or horizontal overflow.
- Standard font packages were supplied through the build environment; the class
  was not modified to work around missing fonts.
- Underfull spacing diagnostics and a 1.95 pt vertical-box diagnostic from balancing
  the reference columns remain. Rendered pages have no overlapping or clipped text.
- PDF title and author metadata match the manuscript.
- Exactly three author commands; the removed author and affiliation are absent.
- Code directory and URL were verified against the pinned repository using
  authenticated access. This does not assert anonymous public access.
- All changes are confined to the new paper directory. No push, release, or arXiv
  submission is included in this task. Local author and committer identity:
  `withfanta <1909037265@qq.com>`.
- The copied class retains its original trailing whitespace to preserve byte identity;
  whitespace checks pass for the other manuscript files.
