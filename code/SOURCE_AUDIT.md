# Source audit

Read-only audit of every source research repository at packaging time. No source repository
was modified, checked out, reset, cleaned, or committed to. Every packaged file was taken
from the frozen commit named below using `git archive <sha>`, so no uncommitted working-tree
content could be packaged as canonical.

Audit performed: 2026-09-15. Host: dgx1.

## Summary table

| # | Slug | Source repository | Experiments | HEAD (frozen commit) | Branch | Commits | Working tree | Repo size |
|---|---|---|---|---|---|---|---|---|
| 1 | `01_redv` | `/home/h-li/work/rejected_expert_delayed_value` | REDV-V1, REDV-V2 | `910482e4e990a7ad7e792fb9df8b648a2ba37b77` | master | 8 | clean (0 entries) | 185 MB |
| 2 | `02_drev` | `/home/h-li/work/delayed_rejected_evidence` | DREV-P0 | `4cce387df1cc5f62989417f7882f33f905e09d47` | master | 4 | clean (0 entries) | 14 MB |
| 3 | `03_eipc` | `/home/h-li/work/expert_identity_preservation` | EIPC-P0 | `517f74edb05d50b9d512248eeff2b5864afc59af` | master | 4 | clean (0 entries) | 23 MB |
| 4 | `04_xec` | `/home/h-li/work/cross_layer_expert_cache` | XEC-P0 | `a664cb5de30900119453d25eba2c4034bfef7b05` | master | 4 | clean (0 entries) | 103 MB |
| 5 | `05_epd` | `/home/h-li/work/expert_provenance_decomposition` | EPD-P0 | `995d23ebb2cc37772b38abefc8d16d1af35586e3` | master | 4 | clean (0 entries) | 18 MB |
| 6 | `06_rmo` | `/home/h-li/work/routing_markov_order` | RMO-P0 | `a8e9d83cc2a6dd4800db72e610bd010ff80a83bb` | master | 3 | clean (0 entries) | 460 KB |
| 7 | `07_rmc` | `/home/h-li/work/routing_memory_crossmodel` | RMC-P0 | `cc60edd9313c15c9a0f433d9f897a38fb44ef1a3` | master | 3 | clean (0 entries) | 2.2 MB |
| 8 | `08_nhd` | `/home/h-li/work/nonlinear_history_decoding` | NHD-P0 | `f1a7d4ab9313798a20297921dd8a8125dee71763` | master | 3 | clean (0 entries) | 584 KB |

Expected commits stated in the packaging instruction were confirmed exactly: EPD-P0
`995d23e`, RMO-P0 `a8e9d83`, RMC-P0 `cc60edd`. NHD-P0's closed result commit is `f1a7d4a`
(recorded here as its canonical SHA; it was not specified in advance).

**No repository had uncommitted modifications.** All eight working trees reported zero
`git status --porcelain` entries both before and after packaging.

## Per repository detail

### 1. `01_redv` — rejected_expert_delayed_value @ `910482e`

Commit history (newest first):

```
910482e untrack regenerable V2 feature caches
7988430 REDV-V2 formal result and closure
f3dc29b REDV-V2 implementation and tests
1892e97 REDV-V2 preregistration: dimension-matched rejected-evidence test
7964f31 formal result and closure
6d1c57f implementation and tests
0d0ce3e REDV-V1 preregistration: rejected-expert delayed-value test
6d733c5 project scaffold and research charter
```

34 tracked files. Result and protocol files found: `protocols/REDV_V1_PREREGISTRATION.md`,
`protocols/REDV_V2_PREREGISTRATION.md`, `artifacts/REDV_V1/{results.json,RESULTS.md,
data_manifest.json,model_provenance.json,environment.json}`, `artifacts/REDV_V2/{...}`,
`docs/{RESEARCH_CHARTER.md,DECISION_LOG.md,CURRENT_STATE.md}`, `tests/test_redv.py`,
`tests/test_redv_v2.py`.

Untracked large artifacts (not packaged): four NPZ feature caches, 19–38 MB each, 115 MB
total. Already untracked in the source by commit `910482e` itself.

### 2. `02_drev` — delayed_rejected_evidence @ `4cce387`

```
4cce387 formal result and closure
28f166d implementation and tests
a88835a DREV-P0 preregistration: delayed rejected-evidence pilot
3313b80 scaffold and research charter
```

32 tracked files, including four per-horizon result files
(`artifacts/DREV_P0/horizon_{1,2,4,8}.json`). Untracked large artifacts: `source_fit.npz`,
`source_test.npz`, 6.4 MB each.

### 3. `03_eipc` — expert_identity_preservation @ `517f74e`

```
517f74e formal result and closure
351c4d0 implementation and tests
04c48a3 EIPC-P0 preregistration and frozen data manifest
f27ff22 scaffold and research charter
```

28 tracked files. Includes `artifacts/EIPC_P0/projection.json` recording the frozen random
projection (seed 20260917, 2048→32, sha256 `fb9e6e9b…`). Untracked large artifacts:
`fit_raw.npz`, `test_raw.npz`, 11.5 MB each.

### 4. `04_xec` — cross_layer_expert_cache @ `a664cb5`

```
a664cb5 formal result and closure
65b4603 implementation and tests
b39e594 XEC-P0 preregistration and frozen manifest
491da9f scaffold and research charter
```

42 tracked files, the largest packaged set. Includes **nine trained policy checkpoints**
(`artifacts/XEC_P0/checkpoints/{CURRENT_ONLY,EXPERT_CACHE,FUSED_CACHE}_seed{42,123,2026}.pt`,
617 KB each, 5.6 MB total). These were tracked in the source commit and are packaged; they
are the trained policies whose TEST NLL produced the verdict, and they are small.

Note: the repository's own `.gitignore` lists `artifacts/*/*.pt`, but git continues tracking
files added before an ignore rule. To preserve the frozen record exactly, these nine files
were force-added during packaging. This is a packaging mechanism, not a content change.

Untracked large artifacts: `train_features.npz` (54 MB), `test_features.npz` (27 MB),
`validation_features.npz` (13.6 MB).

### 5. `05_epd` — expert_provenance_decomposition @ `995d23e`

```
995d23e exploratory result
949d737 implementation and tests
a1a6983 frozen fresh-data manifest
6556d29 scaffold and protocol
```

23 tracked files. `protocols/EPD_P0_PROTOCOL.md` is a protocol, not a preregistration, which
matches EPD-P0's exploratory status. Untracked large artifacts: `merged_raw.npz` (8.6 MB)
and four shards (2.1 MB each).

### 6. `06_rmo` — routing_markov_order @ `a8e9d83`

```
a8e9d83 RMO-P0 result and closure
07f0854 RMO-P0 implementation and tests
3b70f14 RMO-P0 frozen exploratory protocol
```

7 tracked files, flat layout. RMO-P0 ran no inference of its own: it reuses EPD-P0's
extracted arrays read-only, verified by hash (`merged_raw.npz` sha256 `39c9b60…`,
`data_manifest.json` sha256 `bbbe06f…`). No large artifacts of its own.

### 7. `07_rmc` — routing_memory_crossmodel @ `cc60edd`

```
cc60edd RMC-P0 result and closure
85013f5 RMC-P0 implementation and tests
f4924dc RMC-P0 preregistration and frozen provenance
```

21 tracked files, flat layout. The JetMoE extraction artifacts are small enough to be
tracked in the source and are fully packaged: `artifacts/merged.npz` (311 KB) and
`artifacts/shard_{0,1,2,3}.npz` (~80 KB each). Also includes
`artifacts/router_identification.json`, the MoA-vs-MLP router audit. No untracked large
artifacts.

### 8. `08_nhd` — nonlinear_history_decoding @ `f1a7d4a`

```
f1a7d4a NHD-P0 result and closure
973bbf8 NHD-P0 implementation and tests
cc45fa4 NHD-P0 frozen protocol and source-artifact verification
```

9 tracked files, flat layout. NHD-P0 ran no inference: it reuses RMC-P0 artifacts read-only
under four pinned sha256 hashes recorded in `artifacts/source_verification.json`. No large
artifacts of its own.

## Artifact size overview

| Slug | Packaged files | Packaged size | Largest packaged file | Large artifacts omitted |
|---|---|---|---|---|
| `01_redv` | 34 | 408 KB | `artifacts/REDV_V2/data_manifest.json` 108 KB | 4 files, 115 MB |
| `02_drev` | 32 | 276 KB | `tests/test_drev.py` 19 KB | 2 files, 12.8 MB |
| `03_eipc` | 28 | 296 KB | `artifacts/EIPC_P0/data_manifest.json` 79 KB | 2 files, 23 MB |
| `04_xec` | 42 | 5.8 MB | 9 checkpoints, 617 KB each | 3 files, 95 MB |
| `05_epd` | 23 | 232 KB | `artifacts/EPD_P0/data_manifest.json` 38 KB | 5 files, 17 MB |
| `06_rmo` | 7 | 80 KB | `test_analysis.py` 19 KB | none |
| `07_rmc` | 21 | 796 KB | `test_rmc.py` 39 KB | none |
| `08_nhd` | 9 | 124 KB | `test_nhd.py` 31 KB | none |
| **total** | **196** | **8.1 MB** | 0.59 MB | 16 files, ~263 MB |

The 196 figure counts files copied out of the frozen source commits. It excludes the
documentation written for this packaging task itself — `SOURCE_AUDIT.md`, `EXPERIMENT_INDEX.md`,
`REPRODUCIBILITY.md`, `ARTIFACTS_NOT_VERSIONED.md`, `code/README.md`, and the per-experiment
`PACKAGING_PROVENANCE.md` / `REPRODUCTION.md` pairs (16 files). Those are packaging artifacts,
not packaged research code, and they exist in no source repository.

No packaged file reaches 25 MB; the largest is 0.59 MB. Omitted artifacts are itemised with
sizes, sha256 hashes, and regeneration commands in
[ARTIFACTS_NOT_VERSIONED.md](ARTIFACTS_NOT_VERSIONED.md).

## Verification after packaging

All eight source repositories re-checked after copying: HEAD unchanged, branch unchanged,
`git status --porcelain` still zero entries. See [source_audit.json](source_audit.json) for
the machine-readable record.
