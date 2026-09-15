# rejected_expert_delayed_value

REDV-V1 — Rejected-Expert Delayed Value V1.

A single phenomenon test. It asks one question and stops:

> Does information contained in near-miss experts that are rejected by Top-k
> routing at MoE layer `l` provide incremental information for predicting
> counterfactual routing regret at the NEXT MoE layer `l+1`?

REDV-V1 proposes no architecture. It is not a method, and it is not a
precursor to one unless the user separately authorizes that.

- Frozen scientific question and scope: [docs/RESEARCH_CHARTER.md](docs/RESEARCH_CHARTER.md)
- Frozen protocol (written before any result existed): [protocols/REDV_V1_PREREGISTRATION.md](protocols/REDV_V1_PREREGISTRATION.md)
- Where the project currently stands: [docs/CURRENT_STATE.md](docs/CURRENT_STATE.md)
- Recorded decisions: [docs/DECISION_LOG.md](docs/DECISION_LOG.md)

## Setting

- Model: `allenai/OLMoE-1B-7B-0125` (frozen, eval, FP16, single revision pin)
- Data: WikiText-103 raw — `validation` for probe fitting, `test` for held-out evaluation
- Probes: two Ridge regressions per layer transition, baseline vs. baseline+rejected-evidence
- Primary quantity: held-out `Delta_R2 = R²_augmented - R²_baseline`

## Layout

```
src/redv/        library code (data, hooks, rejected evidence, counterfactuals, probes, analysis)
scripts/         environment audit, smoke tests, formal run
tests/           implementation-correctness unit tests (no research statistics)
artifacts/REDV_V1/  provenance, manifest, features, results
```

## Reproducing

```bash
PY=/home/h-li/miniconda3/envs/sensorllm/bin/python
$PY scripts/inspect_environment.py     # provenance audit, no research metric
$PY -m pytest tests/ -v                # implementation correctness
$PY scripts/smoke_redv.py              # end-to-end plumbing on a few contexts
$PY scripts/run_redv_v1.py             # formal run
```
