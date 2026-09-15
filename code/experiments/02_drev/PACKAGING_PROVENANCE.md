# Packaging provenance — DREV-P0

**Source repository:** `/home/h-li/work/delayed_rejected_evidence`

**Source commit SHA:** `4cce387df1cc5f62989417f7882f33f905e09d47` (branch `master`, 4 commits,
working tree clean at packaging time)

**Packaging method:** `git archive 4cce387 | tar -x`. The frozen commit is the canonical
source; no working-tree file was packaged.

## Exact scientific status

**Result: NOT_PROMISING.**

Rejected near-miss expert outputs did not develop delayed predictive value at longer layer
gaps. Layer-4 rejected evidence was tested against routing regret at horizons 1, 2, 4 and 8
(target layers 5, 6, 8, 12). Mean D over the delayed horizons = −0.05931 against D at the
immediate next layer = −0.04340: the delayed conditions were not better than the immediate
one, and none was positive. Held-out R²_real was negative at every horizon
(−0.08378, −0.05771, −0.07582, −0.13461).

The frozen decision sentence recorded in `artifacts/DREV_P0/results.json`: "DREV-P0 does not
show that Layer-4 rejected evidence becomes more informative about routing regret at later
depths than at the immediate next layer, under this frozen pilot setting."

This is a negative result and is preserved as one. The rejected-expert delayed-memory
direction is closed; there is no DREV-P1. It is not rewritten as a stepping stone toward any
later experiment.

## Files included

32 files, 176,286 B total.

```
README.md
docs/RESEARCH_CHARTER.md
docs/DECISION_LOG.md
docs/CURRENT_STATE.md
protocols/DREV_P0_PREREGISTRATION.md      preregistration, fixed before data
src/drev/__init__.py
src/drev/analysis.py
src/drev/counterfactual.py
src/drev/data.py
src/drev/probes.py
src/drev/projection.py
src/drev/source_features.py
scripts/build_manifest.py
scripts/extract_source.py
scripts/finalize_drev.py
scripts/_horizon_worker.py
scripts/inspect_environment.py
scripts/run_horizons_gpu_a.py
scripts/run_horizons_gpu_b.py
scripts/smoke_drev.py
scripts/write_results_md.py
tests/test_drev.py
artifacts/DREV_P0/results.json            frozen result
artifacts/DREV_P0/RESULTS.md
artifacts/DREV_P0/horizon_1.json          per-horizon frozen result
artifacts/DREV_P0/horizon_2.json
artifacts/DREV_P0/horizon_4.json
artifacts/DREV_P0/horizon_8.json
artifacts/DREV_P0/data_manifest.json      frozen sample
artifacts/DREV_P0/model_provenance.json
artifacts/DREV_P0/environment.json
.gitignore
```

The count above is the archive listing; `src/` and `scripts/` are shown by directory.

## Files intentionally omitted

Two NPZ source-feature caches, 12.8 MB total. Both are regenerable from the frozen manifest
by `scripts/extract_source.py`.

| Path | Size | sha256 |
|---|---|---|
| `artifacts/DREV_P0/source_fit.npz` | 6,419,957 B | `8cbef264c1663b456e719c5627490e21f6d46617a0ae4592fdb80ddf01c7f195` |
| `artifacts/DREV_P0/source_test.npz` | 6,424,881 B | `f2fbad596e2162698dfe98ee6b4d8249a19ce1a2e0676f69be1251a5552624cc` |

Also omitted by policy: `__pycache__`, run logs, and the source `.git` directory.

## Artifact hashes (included files)

Frozen sample identity is pinned inside the artifacts themselves; `artifacts/DREV_P0/results.json`
records the manifest hash. That recorded value is authoritative — verify a regenerated
manifest against it rather than against any hash written here.

## Reproduction entry point

See [REPRODUCTION.md](REPRODUCTION.md). Summary:

```bash
PY=/home/h-li/miniconda3/envs/sensorllm/bin/python
$PY scripts/extract_source.py
CUDA_VISIBLE_DEVICES=<A> $PY scripts/run_horizons_gpu_a.py   # targets 5, 6
CUDA_VISIBLE_DEVICES=<B> $PY scripts/run_horizons_gpu_b.py   # targets 8, 12
$PY scripts/finalize_drev.py
```
