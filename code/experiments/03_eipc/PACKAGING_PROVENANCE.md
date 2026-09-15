# Packaging provenance — EIPC-P0

**Source repository:** `/home/h-li/work/expert_identity_preservation`

**Source commit SHA:** `517f74edb05d50b9d512248eeff2b5864afc59af` (branch `master`, 4 commits,
working tree clean at packaging time)

**Packaging method:** `git archive 517f74e | tar -x`. The frozen commit is the canonical
source; no working-tree file was packaged.

## Exact scientific status

**Result: PROMISING.**

Preserving selected-expert provenance predicted future native routing better than early fused
historical outputs. The conservative evidence is the identity-vs-fused contrast: mean A =
+0.14215, positive at both targets (Layer 8 +0.10923, Layer 12 +0.17506). Held-out R² for the
identity-preserving representation was +0.29939 at Layer 8 and +0.46211 at Layer 12, against
+0.19016 and +0.28705 for fused.

The second recorded quantity, mean B = +0.37952, compares against an expert-identity-destroyed
(shuffled) control. **Do not interpret the shuffled-identity comparison as a clean
information-theoretic identity effect.** The conservative claim is identity-vs-fused; the
shuffled contrast is larger but does not isolate identity information.

The frozen stopping note is explicit that this result **does not prove an expert cache
architecture improves performance**. No cache attention was implemented here. That question
was tested separately by XEC-P0, which returned NOT_PROMISING.

## Files included

28 files, 202,375 B total.

```
README.md
docs/RESEARCH_CHARTER.md
docs/DECISION_LOG.md
docs/CURRENT_STATE.md
protocols/EIPC_P0_PREREGISTRATION.md     preregistration, fixed before data
src/eipc/__init__.py
src/eipc/analysis.py
src/eipc/data.py
src/eipc/expert_states.py
src/eipc/olmoe.py
src/eipc/probes.py
src/eipc/representations.py
scripts/build_manifest.py
scripts/extract_fit.py
scripts/extract_test.py
scripts/_extract_worker.py
scripts/finalize_eipc.py
scripts/inspect_environment.py
scripts/smoke_eipc.py
scripts/write_results_md.py
tests/test_eipc.py
artifacts/EIPC_P0/results.json           frozen result
artifacts/EIPC_P0/RESULTS.md
artifacts/EIPC_P0/data_manifest.json     frozen sample
artifacts/EIPC_P0/projection.json        frozen random projection record
artifacts/EIPC_P0/model_provenance.json
artifacts/EIPC_P0/environment.json
.gitignore
```

## Files intentionally omitted

Two raw NPZ extraction caches, 23.0 MB total. Both are regenerable from the frozen manifest by
`scripts/extract_fit.py` and `scripts/extract_test.py`.

| Path | Size | sha256 |
|---|---|---|
| `artifacts/EIPC_P0/fit_raw.npz` | 11,474,244 B | `379fe7d3d0c14723f3c5d1f898f63e75450c551bdb10430d742e3302c4cf9a43` |
| `artifacts/EIPC_P0/test_raw.npz` | 11,476,809 B | `088e0a06982ea5bdb637cedceeafcce51921e111cb7bd6c00c3edb4ce42d36b9` |

Also omitted by policy: `__pycache__`, run logs, and the source `.git` directory.

## Artifact hashes (included files)

The fixed random projection used to compress expert states is pinned by hash inside
`artifacts/EIPC_P0/model_provenance.json` and `projection.json`:

```
projection seed    20260917
shape              (2048, 32),  Normal(0, 1/sqrt(32)), float64, not trainable
sha256             fb9e6e9b7568f6d4a5ce74a7bee1ba2eab5a48a02fb2a93f80520e4ec0480282
```

The same projection was later reused read-only by EPD-P0, where the identical hash is
recorded. Frozen sample identity is pinned by the manifest hash inside `results.json`; those
recorded values are authoritative.

## Reproduction entry point

See [REPRODUCTION.md](REPRODUCTION.md). Summary:

```bash
PY=/home/h-li/miniconda3/envs/sensorllm/bin/python
CUDA_VISIBLE_DEVICES=<A> $PY scripts/extract_fit.py
CUDA_VISIBLE_DEVICES=<B> $PY scripts/extract_test.py
$PY scripts/finalize_eipc.py
```
