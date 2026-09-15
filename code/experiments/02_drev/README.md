# delayed_rejected_evidence

DREV-P0 — Delayed Rejected-Evidence Value Pilot.

A small screening experiment with one question:

> Does rejected near-miss expert evidence produced at an early MoE layer contain
> sample-specific predictive information about routing regret several layers
> later, even when immediate next-layer value is weak?

One fixed source layer (Layer 4) and four fixed horizons:

| target layer | Δ (layers ahead) |
|---|---|
| 5 | 1 (immediate) |
| 6 | 2 (delayed) |
| 8 | 4 (delayed) |
| 12 | 8 (delayed) |

DREV-P0 is a prerequisite phenomenon test. It implements no memory, no attention,
no recurrent routing, no Routing Surprise, no dynamic-k, no expert
reconsideration, and no new MoE architecture.

- Frozen scope: [docs/RESEARCH_CHARTER.md](docs/RESEARCH_CHARTER.md)
- Frozen protocol, written before any formal statistic: [protocols/DREV_P0_PREREGISTRATION.md](protocols/DREV_P0_PREREGISTRATION.md)
- Current status: [docs/CURRENT_STATE.md](docs/CURRENT_STATE.md)
- Decisions: [docs/DECISION_LOG.md](docs/DECISION_LOG.md)

## Setting

- Model: `allenai/OLMoE-1B-7B-0125` @ `9b0c1aa87e34a20052389dce1f0cf01da783f654`, frozen, FP16
- Data: WikiText-103-raw, only blocks never used by REDV-V1 or REDV-V2
- Compression: StandardScaler + PCA to 64 dims, fit on the fit split only, so p = 128 against n = 512
- Probes: two Ridge(alpha=1.0) per horizon — real vs. one fixed derangement control
- Primary quantity per horizon: `D_m = R2_real(m) - R2_shuffle(m)`

## Implementation reused from REDV

The previous project at `/home/h-li/work/rejected_expert_delayed_value` is
**read-only**. It was not edited. These validated implementation pieces were
copied from it, at these commits:

| piece | copied from | REDV commit |
|---|---|---|
| OLMoE expert interception, forced-route execution, native capture hooks | `src/redv/olmoe_hooks.py` | `6d1c57f3b4c9472145aff2ea5aa89e89c0143794` |
| Router probability semantics (fp32 softmax over 64, top-8, no renormalization) | `src/redv/olmoe_hooks.py`, `src/redv/rejected.py` | `6d1c57f3b4c9472145aff2ea5aa89e89c0143794` |
| Rejected-expert execution and evidence vector | `src/redv/rejected.py` | `6d1c57f3b4c9472145aff2ea5aa89e89c0143794` |
| Next-token NLL and routing-regret construction | `src/redv/counterfactual.py`, `src/redv/olmoe_hooks.py` | `6d1c57f3b4c9472145aff2ea5aa89e89c0143794` |
| WikiText block construction and token stream | `src/redv/data.py` | `6d1c57f3b4c9472145aff2ea5aa89e89c0143794` |
| Deterministic derangement | `src/redv/v2_sampling.py` | `f3dc29b70a1ff889b29ec600a3f6cc9428ef3e2b` |

No REDV scientific conclusion is imported into DREV. REDV's verdicts concern
REDV's own questions and settings, and say nothing about this pilot's outcome.

## Reproducing

```bash
PY=/home/h-li/miniconda3/envs/sensorllm/bin/python
$PY scripts/inspect_environment.py      # provenance audit, no research metric
$PY scripts/build_manifest.py           # freeze untouched-block sample
$PY -m pytest tests/ -q                 # tests A-L
$PY scripts/smoke_drev.py               # plumbing only
$PY scripts/extract_source.py           # Layer-4 b and r, computed once
CUDA_VISIBLE_DEVICES=<A> $PY scripts/run_horizons_gpu_a.py   # targets 5, 6
CUDA_VISIBLE_DEVICES=<B> $PY scripts/run_horizons_gpu_b.py   # targets 8, 12
$PY scripts/finalize_drev.py            # CPU, once, after both workers
```
