# expert_identity_preservation

EIPC-P0 — Expert-Identity Preservation Cache Pilot.

Standard sparse MoE immediately aggregates selected expert outputs:

    y_l = sum_e p_{l,e} * o_{l,e}

Future layers keep the fused representation, but explicit provenance — *this part
came from Layer l, Expert e* — is gone. EIPC-P0 asks one question:

> Does preserving the identities and individual outputs of previously SELECTED
> experts provide additional held-out information about future routing states
> beyond (1) the same historical information after normal expert fusion, and
> (2) the same individual expert outputs with expert identities destroyed?

This is an information-content experiment. It does **not** test whether an
attention cache improves LM accuracy, and it does not implement a cache
architecture.

Three representations, one target:

| representation | what it keeps |
|---|---|
| `FUSED` | historical MoE information, no individual provenance |
| `EXPERT_IDENTITY` | layer identity + expert identity + selected contributions |
| `SHUFFLED_IDENTITY` | same vectors and layers, expert identity destroyed |

Target: the future-layer native router state (64-d router logits), at target
Layer 8 (history Layers 1–7) and target Layer 12 (history Layers 1–11).

- Frozen scope: [docs/RESEARCH_CHARTER.md](docs/RESEARCH_CHARTER.md)
- Frozen protocol, written before any formal statistic: [protocols/EIPC_P0_PREREGISTRATION.md](protocols/EIPC_P0_PREREGISTRATION.md)
- Current status: [docs/CURRENT_STATE.md](docs/CURRENT_STATE.md)
- Decisions: [docs/DECISION_LOG.md](docs/DECISION_LOG.md)

## Setting

- Model: `allenai/OLMoE-1B-7B-0125` @ `9b0c1aa87e34a20052389dce1f0cf01da783f654`, frozen, FP16
- Data: WikiText-103-raw **train** split only — fresh data, never touched by prior projects
- Fixed non-trainable random projection `P: R^2048 -> R^32`, seed 20260917
- Compression: StandardScaler → PCA(64) → StandardScaler, fit on FIT only, so every representation is exactly 64-d
- Probes: `Ridge(alpha=1.0)`, multi-output, three per target
- Primary quantities: `A_m = R2_identity - R2_fused` and `B_m = R2_identity - R2_shuffled`

## Relationship to prior projects

The earlier rejected-expert projects are **closed and read-only**:

- `/home/h-li/work/rejected_expert_delayed_value` (REDV-V1, REDV-V2)
- `/home/h-li/work/delayed_rejected_evidence` (DREV-P0)

EIPC-P0 is not REDV-V3 and not a rescue of any prior hypothesis. It asks a
different question — about *selected* expert provenance across depth, not about
rejected experts — and it executes **no** rejected experts, no counterfactual
routing, and no routing regret. No prior conclusion is imported.

Those projects deliberately never used the WikiText train split, which is why
EIPC-P0 uses train exclusively and needs no prior manifest as data.

## Reproducing

```bash
PY=/home/h-li/miniconda3/envs/sensorllm/bin/python
$PY scripts/inspect_environment.py     # provenance audit, no research metric
$PY scripts/build_manifest.py          # freeze train-split FIT/TEST sample
$PY -m pytest tests/ -q                # invariants A-P
$PY scripts/smoke_eipc.py              # plumbing only
CUDA_VISIBLE_DEVICES=<A> $PY scripts/extract_fit.py    # 1024 FIT
CUDA_VISIBLE_DEVICES=<B> $PY scripts/extract_test.py   # 1024 TEST
$PY scripts/finalize_eipc.py           # CPU, once, six probes, frozen rule
```
