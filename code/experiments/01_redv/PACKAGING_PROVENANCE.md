# Packaging provenance — REDV-V1 and REDV-V2

**Source repository:** `/home/h-li/work/rejected_expert_delayed_value`

**Source commit SHA:** `910482e4e990a7ad7e792fb9df8b648a2ba37b77` (branch `master`, 8 commits,
working tree clean at packaging time)

**Packaging method:** `git archive 910482e | tar -x`. The frozen commit is the canonical
source; no working-tree file was packaged.

## Exact scientific status

Two experiments live in this repository.

**REDV-V1 — formal preregistered result: SUPPORTED. Scientific interpretation:
UNINTERPRETABLE.**

Mean ΔR² = +0.47406. The preregistered rule was met and the verdict recorded in
`artifacts/REDV_V1/results.json` is `SUPPORTED`. It is not relabelled. But the design gave the
rejected-evidence condition more input dimensions than its control in a p ≫ n regime, and a
scale-matched random-noise condition improved even more, so the gain cannot be attributed to
rejected-expert content. It is not counted as positive evidence for anything.

**REDV-V2 — result: NOT_SUPPORTED.**

Mean D = −0.09973 with an equal-dimensional real-vs-shuffled control: real rejected evidence
did worse than its dimension-matched control. Equal-dimensional rejected evidence did not show
stable incremental routing-regret information. The rejected-expert immediate-value line is
closed; no REDV-V3.

## Files included

34 files, 408 KB total.

```
README.md
docs/RESEARCH_CHARTER.md
docs/DECISION_LOG.md
docs/CURRENT_STATE.md
protocols/REDV_V1_PREREGISTRATION.md          preregistration, fixed before data
protocols/REDV_V2_PREREGISTRATION.md          preregistration, fixed before data
src/redv/__init__.py
src/redv/analysis.py
src/redv/counterfactual.py
src/redv/data.py
src/redv/olmoe_hooks.py
src/redv/probes.py
src/redv/rejected.py
src/redv/v2_probes.py
src/redv/v2_sampling.py
scripts/inspect_environment.py
scripts/run_redv_v1.py
scripts/run_redv_v2.py
scripts/smoke_redv.py
scripts/write_results_md.py
scripts/write_results_md_v2.py
tests/test_redv.py
tests/test_redv_v2.py
artifacts/REDV_V1/results.json                frozen result
artifacts/REDV_V1/RESULTS.md
artifacts/REDV_V1/data_manifest.json          frozen sample
artifacts/REDV_V1/model_provenance.json
artifacts/REDV_V1/environment.json
artifacts/REDV_V2/results.json                frozen result
artifacts/REDV_V2/RESULTS.md
artifacts/REDV_V2/data_manifest.json          frozen sample
artifacts/REDV_V2/model_provenance.json
artifacts/REDV_V2/environment.json
.gitignore
```

## Files intentionally omitted

Four NPZ feature caches, 115.5 MB total, already untracked in the source at commit `910482e`
(whose message is "untrack regenerable V2 feature caches"). Two exceed the 25 MB ceiling.

| Path | Size | sha256 |
|---|---|---|
| `artifacts/REDV_V1/validation_features.npz` | 19,255,764 B | `90c1313f6791e010bdf162bcc7d56d501809d3b7da93a74d6bd5f5e0f35cdcca` |
| `artifacts/REDV_V1/test_features.npz` | 19,256,745 B | `b28ea83f3817595fedbc1091eca272e473f293c0b653e63ad44c670f14e36d56` |
| `artifacts/REDV_V2/fit_features.npz` | 38,488,901 B | `546f6dd4495fc1ff12b11db7ef79e6064dd4d53515b1168d5b4fcb26e9683f85` |
| `artifacts/REDV_V2/test_features.npz` | 38,484,343 B | `c654d1a8253fa55e8afc700dbba45d63e5b6df681a941036fc2e1a7de2a57883` |

Also omitted by policy: `__pycache__`, run logs, and the source `.git` directory.

## Artifact hashes (included files)

Frozen sample identity is pinned inside the artifacts themselves:
`artifacts/REDV_V1/results.json` and `artifacts/REDV_V2/results.json` each record
`data_manifest_sha256`, and REDV-V2 additionally records `fit_fingerprint` and
`test_fingerprint`. Those recorded values are authoritative; verify a regenerated manifest
against them rather than against any hash written here.

## Reproduction entry point

See [REPRODUCTION.md](REPRODUCTION.md). Summary:

```bash
PY=/home/h-li/miniconda3/envs/sensorllm/bin/python
$PY scripts/run_redv_v1.py     # REDV-V1, GPU
$PY scripts/run_redv_v2.py     # REDV-V2, GPU
```
