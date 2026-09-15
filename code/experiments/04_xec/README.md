# cross_layer_expert_cache

XEC-P0 — Cross-Layer Expert Cache Actionability Pilot.

EIPC-P0 established, under frozen OLMoE/WikiText, that historical selected-expert
provenance carries more held-out information about future router states than the same
history after early fusion (+0.10923 R² at Layer 8, +0.17506 R² at Layer 12) and than
an identity-destroyed control. It did **not** show that a cache improves
language-model performance.

XEC-P0 asks the next question — the first **actionability** test:

> Can historical selected-expert states be used to make an ACTUAL routing decision
> that improves next-token likelihood?

At target Layer 12 the policy chooses one of exactly five actions:

| action | route |
|---|---|
| 0 | keep native Top-8 |
| 1 | ranks 1–7 + native rank 9 |
| 2 | ranks 1–7 + native rank 10 |
| 3 | ranks 1–7 + native rank 11 |
| 4 | ranks 1–7 + native rank 12 |

Four methods are compared, three of them learned and parameter-matched:

| method | memory contents |
|---|---|
| Native OLMoE | none (not learned) |
| CURRENT_ONLY | forced zero memory vector |
| FUSED_CACHE | 11 early-fused historical items |
| EXPERT_CACHE | 88 individual selected-expert items with layer + expert identity |

Primary outcome: **true next-token NLL** after applying the predicted action, as three
paired differences against Native, CURRENT_ONLY, and FUSED_CACHE, each with a fixed
paired bootstrap 95% CI.

- Frozen scope: [docs/RESEARCH_CHARTER.md](docs/RESEARCH_CHARTER.md)
- Frozen protocol, written before any formal statistic: [protocols/XEC_P0_PREREGISTRATION.md](protocols/XEC_P0_PREREGISTRATION.md)
- Current status: [docs/CURRENT_STATE.md](docs/CURRENT_STATE.md)
- Decisions: [docs/DECISION_LOG.md](docs/DECISION_LOG.md)

## Setting

- Model: `allenai/OLMoE-1B-7B-0125` @ `9b0c1aa87e34a20052389dce1f0cf01da783f654`, frozen, eval, FP16, not quantized. **No OLMoE parameter receives gradients.**
- Data: WikiText-103-raw **train** split, excluding every block used by EIPC-P0
- Fixed non-trainable projection `P: R^2048 -> R^64`, seed 20260918
- Policy: single-head attention over cache items, `d = 64`, no hidden MLP, 5-way action head
- Training: AdamW, lr 1e-3, wd 0.01, batch 64, ≤20 epochs, best-validation checkpoint, seeds 42 / 123 / 2026

Only the policy modules train. OLMoE features and oracle labels are extracted once
under `torch.inference_mode()` and consumed as detached arrays, so the frozen-model
guarantee is structural rather than a flag.

## Read-only prior projects

These are closed and were not modified:

- `/home/h-li/work/rejected_expert_delayed_value` (REDV-V1, REDV-V2)
- `/home/h-li/work/delayed_rejected_evidence` (DREV-P0)
- `/home/h-li/work/expert_identity_preservation` (EIPC-P0, commit `517f74e`)

XEC-P0 reuses validated implementation patterns from EIPC-P0 — native capture,
selected-expert re-execution, router semantics, block construction — and takes its
target-layer choice (Layer 12) from the EIPC-P0 result, which was recorded before XEC
existed. It executes no rejected experts.

## Regenerating the ignored caches

`.npz` feature/oracle caches are ignored by git. Their sha256 values are recorded in
`artifacts/XEC_P0/RESULTS.md`; regenerate with:

```bash
PY=/home/h-li/miniconda3/envs/sensorllm/bin/python
$PY scripts/build_manifest.py
CUDA_VISIBLE_DEVICES=<A> $PY scripts/extract_worker_a.py   # TRAIN + oracle
CUDA_VISIBLE_DEVICES=<B> $PY scripts/extract_worker_b.py   # VALIDATION + TEST
$PY scripts/train_policies.py                              # 9 checkpoints
CUDA_VISIBLE_DEVICES=<A> $PY scripts/finalize_xec.py       # TEST NLL, bootstrap, verdict
```
