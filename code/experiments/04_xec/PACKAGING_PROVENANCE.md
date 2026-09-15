# Packaging provenance — XEC-P0

**Source repository:** `/home/h-li/work/cross_layer_expert_cache`

**Source commit SHA:** `a664cb5de30900119453d25eba2c4034bfef7b05` (branch `master`, 4 commits,
working tree clean at packaging time)

**Packaging method:** `git archive a664cb5 | tar -x`. The frozen commit is the canonical
source; no working-tree file was packaged.

## Exact scientific status

**Result: NOT_PROMISING.**

Historical expert provenance was not shown actionable for choosing the NLL-optimal five-action
Layer-12 routing intervention. Three policies (CURRENT_ONLY, EXPERT_CACHE, FUSED_CACHE) × three
seeds (42, 123, 2026) were trained to pick among five routing actions and scored by true
next-token NLL on 1024 held-out samples.

Frozen mean test NLL:

```
native                 2.69230
CURRENT_ONLY           2.69044
FUSED_CACHE            2.68974
EXPERT_CACHE           2.69079
five-action oracle     2.64731
```

The preregistered rule required each of d_native, d_current, d_fused ≤ −0.005 with every 95%
paired-bootstrap CI upper bound below zero, and EXPERT_CACHE below FUSED_CACHE in all three
seeds. None held: d_native = −0.00151 (CI −0.00574 … +0.00276), d_current = +0.00035, d_fused =
+0.00106, and `seed_check.all_seeds_expert_lower = false`.

The frozen decision sentence: "XEC-P0 does not show that historical selected-expert states can
be converted into a Layer-12 routing decision that improves true next-token NLL."

The distinction this establishes: predictive routing information ≠ routing-regret information ≠
actionable rerouting information. Being able to predict future routing from history does not
imply history can be used to choose a better route. This is a negative result and is preserved
as one. The expert-cache method direction is closed; there is no XEC-P1.

## Files included

42 files, 5,855,552 B total — the largest packaged experiment, because the nine trained policy
checkpoints are included.

```
README.md
docs/RESEARCH_CHARTER.md
docs/DECISION_LOG.md
docs/CURRENT_STATE.md
protocols/XEC_P0_PREREGISTRATION.md          preregistration, fixed before data
src/xec/__init__.py
src/xec/analysis.py
src/xec/cache_features.py
src/xec/counterfactual.py
src/xec/data.py
src/xec/olmoe.py
src/xec/oracle.py
src/xec/policy.py
src/xec/training.py
scripts/build_manifest.py
scripts/_extract_common.py
scripts/extract_worker_a.py
scripts/extract_worker_b.py
scripts/finalize_xec.py
scripts/inspect_environment.py
scripts/smoke_xec.py
scripts/train_policies.py
scripts/write_results_md.py
tests/test_xec.py
artifacts/XEC_P0/results.json                frozen result
artifacts/XEC_P0/RESULTS.md
artifacts/XEC_P0/data_manifest.json          frozen sample
artifacts/XEC_P0/projection.json             frozen random projection record
artifacts/XEC_P0/model_provenance.json
artifacts/XEC_P0/environment.json
artifacts/XEC_P0/training_summary.json
artifacts/XEC_P0/training_complete.json
artifacts/XEC_P0/checkpoints/*.pt            9 policy checkpoints, 5.55 MB
.gitignore
```

### Packaging note on the nine checkpoints

The checkpoints are tracked in source commit `a664cb5`, but the repository's own `.gitignore`
carries the pattern `artifacts/*/*.pt`. Git keeps tracking files added before an ignore rule
exists, so they are versioned in the source while a fresh `git add` in a new repository would
silently skip them. They were therefore staged here with `git add -f`.

This is a packaging mechanism only. No checkpoint content was altered, regenerated or
reinterpreted; the nine files are byte-identical to the frozen commit. They are kept because
they are small (617 KB each) and are the trained policies the NOT_PROMISING verdict was
computed from.

| Checkpoint | Size | sha256 |
|---|---|---|
| `CURRENT_ONLY_seed42.pt` | 616,752 B | `b34b9fd61b839bb53152f3221839abae664591d59ed02eaaff4150c0e69fc767` |
| `CURRENT_ONLY_seed123.pt` | 616,760 B | `a3d053f05126f72061f0fbad68899431acefd14c870ed87fdceadc24da493913` |
| `CURRENT_ONLY_seed2026.pt` | 616,768 B | `bfe9f68d1e47b7793eeecb3e6c087b297c0b841b1b134a3b26f5e601033cb078` |
| `EXPERT_CACHE_seed42.pt` | 616,752 B | `654a3dfefd2704edce9e0087e45ee391bd961002a5d7d81edebeda0dce4f5322` |
| `EXPERT_CACHE_seed123.pt` | 616,760 B | `73f76a2aa3904129c6b7560ad57d1a7666f2ef4a8991f04f04d430de67c1abb4` |
| `EXPERT_CACHE_seed2026.pt` | 616,768 B | `434aca4a7ac38100ab1da8414070a4c139abcfbf247cd8bc3f4df9a4a715ca17` |
| `FUSED_CACHE_seed42.pt` | 616,744 B | `8922cc9adc4b4be9f84e7e4e19af3445a80ee22bdf5dcf1d18dcacfef42279b5` |
| `FUSED_CACHE_seed123.pt` | 616,752 B | `6a44fff3cfe84236bc3c593f923b994d4ef2b83eb279409d20c3486589d67337` |
| `FUSED_CACHE_seed2026.pt` | 616,760 B | `d0c37db33a3c5b841efb1a6768a2a861f79d1838bb27b274245ffdf9454c15a6` |

## Files intentionally omitted

Three NPZ feature caches, 95.2 MB total. Two exceed the 25 MB ceiling. All are regenerable
from the frozen manifest by the two extraction workers.

| Path | Size | sha256 |
|---|---|---|
| `artifacts/XEC_P0/train_features.npz` | 54,405,353 B | `df79814e9d91a2066bded4c61fa9866c46f8262a9020fdb9b5418727fc5d19bb` |
| `artifacts/XEC_P0/test_features.npz` | 27,201,555 B | `4bc771f893ae48bb8712599100570e5920c41249faacf2db1434a205f997632c` |
| `artifacts/XEC_P0/validation_features.npz` | 13,607,589 B | `45a09ec67e936a4e18fe314a590c038df19dd5e473248c4f64a4d7eea7faa35a` |

The oracle and policy-result NPZ files (`train_oracle.npz`, `validation_oracle.npz`,
`test_oracle_descriptive.npz`, `test_policy_results.npz`) are likewise not packaged; their
sha256 values are all recorded inside `artifacts/XEC_P0/results.json` under `artifact_sha256`.

Also omitted by policy: `__pycache__`, run logs, and the source `.git` directory.

## Artifact hashes (included files)

`artifacts/XEC_P0/results.json` carries an `artifact_sha256` block pinning every NPZ the
verdict depended on, including the omitted ones. Those recorded values are authoritative. The
fixed projection is pinned as:

```
projection seed    20260918
shape              (2048, 64),  Normal(0, 1/sqrt(64)), float64, not trainable, not learned
sha256             11a1228c725b2a952b6cb966ac0316bd102b8a48bacdbf76391f0d44476556a6
```

Bootstrap seed 314159, 10,000 resamples; policy seeds 42 / 123 / 2026.

## Reproduction entry point

See [REPRODUCTION.md](REPRODUCTION.md). Summary:

```bash
PY=/home/h-li/miniconda3/envs/sensorllm/bin/python
$PY scripts/build_manifest.py
CUDA_VISIBLE_DEVICES=<A> $PY scripts/extract_worker_a.py   # TRAIN + oracle
CUDA_VISIBLE_DEVICES=<B> $PY scripts/extract_worker_b.py   # VALIDATION + TEST
$PY scripts/train_policies.py                              # 9 checkpoints
CUDA_VISIBLE_DEVICES=<A> $PY scripts/finalize_xec.py       # TEST NLL, bootstrap, verdict
```
