# Artifacts not versioned

Large artifacts deliberately excluded from this repository, with everything needed to
regenerate or verify them. No file ≥ 25 MB is committed, and Git LFS is **not** used.

All 16 omitted files were already untracked in their source repositories at the frozen
commit — they were excluded there too, as regenerable caches. Total omitted: **263.6 MB**.

Every entry records original path, size, sha256, experiment, reason, and the command needed
to regenerate it. Hashes were computed from the files as they exist on the original host, so
a regenerated file can be compared against them.

Environment for every regeneration command below:

```bash
PY=/home/h-li/miniconda3/envs/sensorllm/bin/python   # python 3.10.19, transformers 4.45.1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
```

Regeneration requires the pinned model and dataset revisions listed in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) and, for all of these, a GPU (originals used Tesla
V100-SXM2-32GB).

---

## REDV — `rejected_expert_delayed_value` @ `910482e`

| Original path | Size | sha256 |
|---|---|---|
| `artifacts/REDV_V1/validation_features.npz` | 19,255,764 B (18.4 MB) | `90c1313f6791e010bdf162bcc7d56d501809d3b7da93a74d6bd5f5e0f35cdcca` |
| `artifacts/REDV_V1/test_features.npz` | 19,256,745 B (18.4 MB) | `b28ea83f3817595fedbc1091eca272e473f293c0b653e63ad44c670f14e36d56` |
| `artifacts/REDV_V2/fit_features.npz` | 38,488,901 B (36.7 MB) | `546f6dd4495fc1ff12b11db7ef79e6064dd4d53515b1168d5b4fcb26e9683f85` |
| `artifacts/REDV_V2/test_features.npz` | 38,484,343 B (36.7 MB) | `c654d1a8253fa55e8afc700dbba45d63e5b6df681a941036fc2e1a7de2a57883` |

**Reason omitted:** two files exceed the 25 MB ceiling outright; all four are regenerable
extraction caches of hidden states and expert evidence, and were untracked in the source
(commit `910482e` is titled "untrack regenerable V2 feature caches").

**Regenerate:**

```bash
cd 01_redv
$PY scripts/run_redv_v1.py     # writes REDV_V1 validation_features.npz + test_features.npz
$PY scripts/run_redv_v2.py     # writes REDV_V2 fit_features.npz + test_features.npz
```

Both scripts extract and analyse in one pass; the caches are a by-product. The frozen sample
is fixed by `artifacts/REDV_V{1,2}/data_manifest.json`.

---

## DREV-P0 — `delayed_rejected_evidence` @ `4cce387`

| Original path | Size | sha256 |
|---|---|---|
| `artifacts/DREV_P0/source_fit.npz` | 6,419,957 B (6.1 MB) | `8cbef264c1663b456e719c5627490e21f6d46617a0ae4592fdb80ddf01c7f195` |
| `artifacts/DREV_P0/source_test.npz` | 6,424,881 B (6.1 MB) | `f2fbad596e2162698dfe98ee6b4d8249a19ce1a2e0676f69be1251a5552624cc` |

**Reason omitted:** regenerable Layer-4 source-feature caches (b and r), untracked in source.
Below 25 MB but excluded to match the source repository's own policy and because they are
mechanically reproducible from the frozen manifest.

**Regenerate:**

```bash
cd 02_drev
$PY scripts/build_manifest.py      # only if data_manifest.json is absent; it is included here
$PY scripts/extract_source.py      # writes source_fit.npz and source_test.npz
```

---

## EIPC-P0 — `expert_identity_preservation` @ `517f74e`

| Original path | Size | sha256 |
|---|---|---|
| `artifacts/EIPC_P0/fit_raw.npz` | 11,474,244 B (10.9 MB) | `379fe7d3d0c14723f3c5d1f898f63e75450c551bdb10430d742e3302c4cf9a43` |
| `artifacts/EIPC_P0/test_raw.npz` | 11,476,809 B (10.9 MB) | `088e0a06982ea5bdb637cedceeafcce51921e111cb7bd6c00c3edb4ce42d36b9` |

**Reason omitted:** regenerable raw expert-state extractions, untracked in source.

**Regenerate** (two GPUs, or run sequentially on one):

```bash
cd 03_eipc
CUDA_VISIBLE_DEVICES=<A> $PY scripts/extract_fit.py     # 1024 FIT  -> fit_raw.npz
CUDA_VISIBLE_DEVICES=<B> $PY scripts/extract_test.py    # 1024 TEST -> test_raw.npz
```

The frozen random projection (seed 20260917, 2048→32, sha256 `fb9e6e9b7568f6d4a5ce74a7bee1ba2eab5a48a02fb2a93f80520e4ec0480282`)
is recorded in `artifacts/EIPC_P0/projection.json` and is regenerated deterministically from
its seed, so it is not an omitted artifact.

---

## XEC-P0 — `cross_layer_expert_cache` @ `a664cb5`

| Original path | Size | sha256 |
|---|---|---|
| `artifacts/XEC_P0/train_features.npz` | 54,405,353 B (51.9 MB) | `df79814e9d91a2066bded4c61fa9866c46f8262a9020fdb9b5418727fc5d19bb` |
| `artifacts/XEC_P0/test_features.npz` | 27,201,555 B (25.9 MB) | `4bc771f893ae48bb8712599100570e5920c41249faacf2db1434a205f997632c` |
| `artifacts/XEC_P0/validation_features.npz` | 13,607,589 B (13.0 MB) | `45a09ec67e936a4e18fe314a590c038df19dd5e473248c4f64a4d7eea7faa35a` |

**Reason omitted:** the first two exceed 25 MB; all three are regenerable feature and oracle-
label caches, untracked in source.

**Included instead:** the nine trained policy checkpoints
(`artifacts/XEC_P0/checkpoints/{CURRENT_ONLY,EXPERT_CACHE,FUSED_CACHE}_seed{42,123,2026}.pt`,
617 KB each). These are the trained policies whose TEST NLL produced the verdict, they are
small, and they were tracked in the source commit. Keeping them means the verdict's inputs
survive even though the feature caches do not.

**Regenerate:**

```bash
cd 04_xec
$PY scripts/build_manifest.py
CUDA_VISIBLE_DEVICES=<A> $PY scripts/extract_worker_a.py   # TRAIN + oracle labels
CUDA_VISIBLE_DEVICES=<B> $PY scripts/extract_worker_b.py   # VALIDATION + TEST
```

The frozen projection (seed 20260918, 2048→64, sha256 `11a1228c725b2a952b6cb966ac0316bd102b8a48bacdbf76391f0d44476556a6`)
is recorded in `artifacts/XEC_P0/projection.json`.

---

## EPD-P0 — `expert_provenance_decomposition` @ `995d23e`

| Original path | Size | sha256 |
|---|---|---|
| `artifacts/EPD_P0/merged_raw.npz` | 8,551,137 B (8.2 MB) | `39c9b609d89d7fe568e88221b35bdcba6f686e4cec7af1ba1cdd76527d030442` |
| `artifacts/EPD_P0/shard_0.npz` | 2,140,283 B (2.0 MB) | `690e9ffe2e624830df18356fbd2150ed6a47ba5c4f7f259b9ce7f67b6a55675c` |
| `artifacts/EPD_P0/shard_1.npz` | 2,141,366 B (2.0 MB) | `42d05fe99010e4c6903de4b209f122ae5f58bf5d178ee0ca36d6606cd32ce31b` |
| `artifacts/EPD_P0/shard_2.npz` | 2,140,990 B (2.0 MB) | `f9820d78864c029167f32199ae0be4023242876b8f187b080962a6254748cd30` |
| `artifacts/EPD_P0/shard_3.npz` | 2,141,125 B (2.0 MB) | `590d974a2cd8a2644b59ea0f02e977493bd84c8f39b195e782285ec0bf800c0c` |

**Reason omitted:** regenerable extraction caches, untracked in source.

**This is the one omission that blocks a second experiment.** RMO-P0 consumes
`merged_raw.npz` read-only and verifies it by sha256 — the value `39c9b60…` above is the exact
hash RMO-P0 checks against (`06_rmo/artifacts/results.json` → `source.file_sha256`). Without
regenerating it, RMO-P0's analysis cannot be re-run.

**Regenerate:**

```bash
cd 05_epd
$PY scripts/build_manifest.py
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=<gpu_i> $PY scripts/extract_shard.py $i & done; wait
$PY scripts/merge_shards.py        # shards -> merged_raw.npz
```

---

## RMO-P0, RMC-P0, NHD-P0 — nothing omitted

- **RMO-P0** produced no large artifacts of its own; it is pure CPU analysis over EPD-P0's
  arrays. Its dependency is EPD-P0's `merged_raw.npz` above.
- **RMC-P0**'s JetMoE extraction is small enough to be fully versioned here:
  `artifacts/merged.npz` (311 KB) and `artifacts/shard_{0,1,2,3}.npz` (~80 KB each). This is
  why RMC-P0's analysis is re-runnable from this repository with no GPU.
- **NHD-P0** produced no large artifacts and needs none beyond RMC-P0's, which are included.
  Its analysis is therefore also re-runnable here.

---

## Never included, by policy

Not omitted-with-a-hash but categorically excluded: Hugging Face model weights, raw or
downloaded WikiText data, `~/.cache`, conda and virtual environments, source `.git`
directories, SSH keys, tokens, `.env` credentials, API keys, `__pycache__` and compiled
caches, large temporary logs, tmux logs, and core dumps. A secret scan for `hf_`,
`github_pat_`, `ghp_`, `sk-`, `PRIVATE KEY`, `BEGIN OPENSSH PRIVATE KEY`, `HF_TOKEN`, and
`GITHUB_TOKEN` returned zero matches across the packaged tree before staging.

The only file matching any of those patterns is this one, because the sentence above names the
patterns themselves. No credential value appears anywhere in the repository. A rerun of the
scan should expect exactly this single self-referential match and no other.
