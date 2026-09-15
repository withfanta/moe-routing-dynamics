# Packaging provenance — EPD-P0

**Source repository:** `/home/h-li/work/expert_provenance_decomposition`

**Source commit SHA:** `995d23ebb2cc37772b38abefc8d16d1af35586e3` (branch `master`, 4 commits,
working tree clean at packaging time)

**Packaging method:** `git archive 995d23e | tar -x`. The frozen commit is the canonical
source; no working-tree file was packaged.

## Exact scientific status

**Exploratory interpretation: PATH-DOMINANT. No formal verdict.**

Future-routing predictability is dominated by historical expert selection identity/path rather
than by selected-expert activation content. Layer-12 target, history layers 1–11, held-out R²
on 256 TEST samples (512 FIT):

```
FUSED            R² = +0.20114
ID_PATH          R² = +0.66969
CONTENT_RANK     R² = +0.06732
FULL_PROVENANCE  R² = +0.34007
```

Which experts were chosen predicts future routing largely without their activation content, and
adding content on top of identity did not help: G_content_given_identity = −0.32962 while
G_identity_given_content = +0.27275.

**The PATH-DOMINANT label is not preregistered.** EPD-P0 ran under a frozen protocol
(`protocols/EPD_P0_PROTOCOL.md`), not a preregistration, and its own `results.json` records
`exploratory: true` and `no_formal_verdict: true`. The 0.02 gap figure used in the pattern
description is a discussion aid, not a significance threshold. `architecture_built: none`.

## Files included

23 files, 166,102 B total.

```
README.md
docs/RESEARCH_QUESTION.md
docs/DECISION_LOG.md
protocols/EPD_P0_PROTOCOL.md            frozen protocol (not a preregistration)
src/epd/__init__.py
src/epd/analysis.py
src/epd/data.py
src/epd/extraction.py
src/epd/probes.py
src/epd/representations.py
scripts/build_manifest.py
scripts/extract_shard.py
scripts/merge_shards.py
scripts/analyze_epd.py
scripts/smoke_epd.py
scripts/write_results_md.py
tests/test_epd.py
artifacts/EPD_P0/results.json           frozen result
artifacts/EPD_P0/RESULTS.md
artifacts/EPD_P0/data_manifest.json     frozen sample
artifacts/EPD_P0/model_provenance.json
artifacts/EPD_P0/environment.json
.gitignore
```

## Files intentionally omitted

Five NPZ files, 17.1 MB total: four per-GPU extraction shards and their merge. All regenerable
from the frozen manifest by `scripts/extract_shard.py` + `scripts/merge_shards.py`.

| Path | Size | sha256 |
|---|---|---|
| `artifacts/EPD_P0/merged_raw.npz` | 8,551,137 B | `39c9b609d89d7fe568e88221b35bdcba6f686e4cec7af1ba1cdd76527d030442` |
| `artifacts/EPD_P0/shard_0.npz` | 2,140,283 B | `690e9ffe2e624830df18356fbd2150ed6a47ba5c4f7f259b9ce7f67b6a55675c` |
| `artifacts/EPD_P0/shard_1.npz` | 2,141,366 B | `42d05fe99010e4c6903de4b209f122ae5f58bf5d178ee0ca36d6606cd32ce31b` |
| `artifacts/EPD_P0/shard_2.npz` | 2,140,990 B | `f9820d78864c029167f32199ae0be4023242876b8f187b080962a6254748cd30` |
| `artifacts/EPD_P0/shard_3.npz` | 2,141,125 B | `590d974a2cd8a2644b59ea0f02e977493bd84c8f39b195e782285ec0bf800c0c` |

`merged_raw.npz` is the one omitted artifact with a downstream consumer: RMO-P0 read it
read-only. RMO-P0 therefore cannot be re-run from this repository until it is regenerated. See
[../06_rmo/PACKAGING_PROVENANCE.md](../06_rmo/PACKAGING_PROVENANCE.md).

Also omitted by policy: `__pycache__`, run logs, and the source `.git` directory.

## Artifact hashes (included files)

EPD-P0 did not create its own random projection. It reused EIPC-P0's read-only, recorded in
`artifacts/EPD_P0/model_provenance.json` with the identical hash:

```
projection seed    20260917
shape              (2048, 32),  Normal(0, 1/sqrt(32)), float64, not trainable
sha256             fb9e6e9b7568f6d4a5ce74a7bee1ba2eab5a48a02fb2a93f80520e4ec0480282
```

Frozen sample identity is pinned by the manifest hash inside `results.json`; that recorded
value is authoritative.

## Reproduction entry point

See [REPRODUCTION.md](REPRODUCTION.md). Summary:

```bash
PY=/home/h-li/miniconda3/envs/sensorllm/bin/python
$PY scripts/build_manifest.py
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=<gpu_i> $PY scripts/extract_shard.py $i & done; wait
$PY scripts/merge_shards.py
$PY scripts/analyze_epd.py
```
