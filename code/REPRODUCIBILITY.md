# Reproducibility

How the eight experiments depend on one another, what kind of evidence each one is, and what
can and cannot be reproduced from this repository alone.

## Dependency chain

```
REDV-V1 / REDV-V2  ──► rejected-expert hypothesis closed
DREV-P0            ──┘

EIPC-P0            ──► selected-expert provenance predicts future routing

XEC-P0             ──► prediction did not imply actionable rerouting

EPD-P0             ──► signal decomposed as expert-selection-path dominant

RMO-P0             ──► OLMoE recent history beyond previous layer adds prediction

RMC-P0             ──► phenomenon independently replicated in JetMoE

NHD-P0             ──► simple nonlinear-decoding alternative does not explain the gain
```

Read as a narrative: the line began with a hypothesis about *rejected* experts, which failed
twice and was closed. Attention moved to *selected* expert provenance, which did predict
future routing (EIPC-P0) — but when that prediction was asked to do work, choosing a better
route, it did not (XEC-P0). EPD-P0 then decomposed where the predictive signal lives, finding
it in the selection path rather than activation content. RMO-P0 asked how far back the useful
history extends, RMC-P0 checked the phenomenon survives in a different architecture, and
NHD-P0 tested the most obvious alternative explanation for it.

Two artifact-level dependencies matter for reproduction:

- **RMO-P0 reuses EPD-P0's extracted arrays** read-only, verified by sha256
  (`merged_raw.npz` `39c9b60…`, `data_manifest.json` `bbbe06f…`). RMO-P0 ran no model
  inference of its own.
- **NHD-P0 reuses RMC-P0's artifacts** read-only under four pinned sha256 hashes recorded in
  `08_nhd/artifacts/source_verification.json`. NHD-P0 ran no model inference and needed no
  GPU.

Consequently RMO-P0 cannot be reproduced without first reproducing EPD-P0's extraction, and
NHD-P0 cannot be reproduced without RMC-P0's extraction. NHD-P0 halts with
`REQUIRED_ARTIFACT_NOT_AVAILABLE` rather than silently re-running inference if those inputs
are missing.

## Kinds of evidence — these are not interchangeable

### Preregistered confirmation experiments

A hypothesis, a decision rule, and a threshold were all fixed in writing before the data were
seen; the recorded verdict is whatever that rule produced.

- **REDV-V1** (`01_redv/protocols/REDV_V1_PREREGISTRATION.md`) → SUPPORTED
- **REDV-V2** (`01_redv/protocols/REDV_V2_PREREGISTRATION.md`) → NOT_SUPPORTED
- **EIPC-P0** (`03_eipc/protocols/EIPC_P0_PREREGISTRATION.md`) → PROMISING
- **XEC-P0** (`04_xec/protocols/XEC_P0_PREREGISTRATION.md`) → NOT_PROMISING
- **RMC-P0** (`07_rmc/protocol.md`, six-condition replication rule) → REPLICATED

### Exploratory / descriptive experiments

Frozen protocols and frozen analysis code, but the *label* is a description chosen from a
pre-committed set rather than a hypothesis test. No p-value, no formal verdict.

- **EPD-P0** → PATH-DOMINANT. Its own `results.json` records `exploratory: true` and
  `no_formal_verdict: true`. The label is **not** preregistered.
- **RMO-P0** → SHORT-HISTORY. Exploratory and post-hoc on already-observed data.
- **NHD-P0** → RESIDUAL-HISTORY-VALUE. A mechanism-disambiguation experiment: the label
  criteria were fixed operationally before results (protocol.md section 10), but the protocol
  explicitly refuses to force a binary scientific verdict.

### Negative stopping results

Experiments that closed a research direction. They are recorded as negatives, not as
successful steps toward something else.

- **REDV-V2** — NOT_SUPPORTED. Rejected-expert immediate-value line closed. No REDV-V3.
- **DREV-P0** — NOT_PROMISING. Rejected-expert delayed-memory direction closed. No DREV-P1.
- **XEC-P0** — NOT_PROMISING. Expert-cache method direction closed. No XEC-P1.

These three are not stepping stones that happened to work out. They are results that ended
lines of inquiry, and each carries a frozen stopping note forbidding the usual rescue moves
(more samples, new model or dataset, threshold changes, added capacity, extra seeds, subgroup
reinterpretation). **REDV-V1 is a fourth special case:** its formal verdict is SUPPORTED and
stays SUPPORTED, but it is scientifically UNINTERPRETABLE because of a p ≫ n / unequal
dimensionality confound — scale-matched random noise improved even more. It is not counted as
positive evidence for anything.

## Shared environment

Every experiment ran in the same pinned conda environment:

```
python        3.10.19        (/home/h-li/miniconda3/envs/sensorllm/bin/python)
torch         2.4.1+cu121
transformers  4.45.1
numpy         1.24.4
scikit-learn  1.3.2
pytest        8.3.5
```

The `transformers` pin is load-bearing, not incidental. JetMoE support and the exact router
module layout differ across versions, and the default interpreter on the original host is
python 3.13 / transformers 4.57, which is **wrong** for these projects. Several experiments
record the implementation file's sha256 in their provenance for this reason: a pinned file
hash is worthless if a different file would actually execute.

On the 80-core host, BLAS threads were capped for the CPU analysis stages:

```
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
```

## Models and data

| Experiment | Model | Immutable revision | Dataset | Split |
|---|---|---|---|---|
| REDV-V1/V2, DREV-P0, EIPC-P0, XEC-P0, EPD-P0 | `allenai/OLMoE-1B-7B-0125` | `9b0c1aa87e34a20052389dce1f0cf01da783f654` | `Salesforce/wikitext` / `wikitext-103-raw-v1` rev `b08601e04326c79dfdd32d625aee71d232d685c3` | train |
| RMO-P0 | (no inference; reuses EPD-P0 capture from the same OLMoE revision) | — | — | — |
| RMC-P0 | `jetmoe/jetmoe-8b` | `d8fd02ccf7911aa8148a63c7984ffd2e465b0352` | same WikiText revision | train |
| NHD-P0 | (no inference; reuses RMC-P0 JetMoE artifacts) | — | — | — |

All checkpoints are BASE, not instruct; `float16` runtime dtype; no quantization; no
gradients on the language model. Tokenizer for OLMoE is `GPTNeoXTokenizerFast`. JetMoE ships
a Llama tokenizer class — a property of that model, not a defect.

## What is and is not reproducible from this repository

**Reproducible from what is here, no GPU needed:** every recorded number can be *checked*
against the frozen `results.json` and `RESULTS.md` files, and RMC-P0's and NHD-P0's full
analyses can be re-run, because their input arrays are small enough to be included
(`07_rmc/artifacts/merged.npz`, 311 KB).

**Requires re-extraction on GPU:** REDV, DREV, EIPC, XEC, and EPD analyses depend on feature
caches of 6–54 MB that are not versioned here. Re-running them means re-running extraction
against the pinned model and dataset revisions. Sizes, sha256 hashes, and the exact
regeneration command for each are in [ARTIFACTS_NOT_VERSIONED.md](ARTIFACTS_NOT_VERSIONED.md).

**Bit-exact reproduction is not guaranteed.** GPU extraction under float16 is sensitive to
device, driver, and kernel-selection differences. The original runs used Tesla V100-SXM2-32GB
GPUs. Sample selection, splits, and seeds are frozen in each experiment's `data_manifest.json`
and provenance files, so the *sample* is reproducible even where the last digits of a float
may not be. CPU-side analyses (RMO-P0, NHD-P0, and the analysis halves of the others, given
their caches) are deterministic under the pinned environment.

**Nothing in this repository recomputes or overrides a frozen number.** This package is
archival: the frozen values in each experiment's artifacts are authoritative. Where a
document here quotes a number, it is transcribed from those files.
