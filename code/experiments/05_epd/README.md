# expert_provenance_decomposition

EPD-P0 — Expert Provenance Decomposition Pilot.

A small **exploratory** experiment. It builds no cache, no attention, and no method, and
it does not attempt to rescue XEC-P0.

EIPC-P0 found that preserving historical selected-expert provenance predicts future
routing better than early fusion. It did not identify *what* carries that information.
EPD-P0 decomposes the candidate sources:

| representation | keeps | destroys |
|---|---|---|
| `FUSED` | what the layer produced overall | individual decomposition, expert identity |
| `ID_PATH` | which experts were selected (binary) | all activation content |
| `CONTENT_RANK` | what each rank-ordered expert computed | absolute expert identity |
| `FULL_PROVENANCE` | layer identity, expert identity, and content | — |

Single target: the native **Layer-12** router logits. History: Layers 1–11.

- Question and scope: [docs/RESEARCH_QUESTION.md](docs/RESEARCH_QUESTION.md)
- Frozen protocol: [protocols/EPD_P0_PROTOCOL.md](protocols/EPD_P0_PROTOCOL.md)
- Decisions: [docs/DECISION_LOG.md](docs/DECISION_LOG.md)

## Setting

- Model: `allenai/OLMoE-1B-7B-0125` @ `9b0c1aa87e34a20052389dce1f0cf01da783f654`, frozen, eval, FP16, no gradients
- Data: WikiText-103-raw **train**, excluding every block used by EIPC-P0 and XEC-P0
- Projection: the **exact** EIPC-P0 fixed projection `R^2048 -> R^32`, hash verified as `fb9e6e9b7568f6d4a5ce74a7bee1ba2eab5a48a02fb2a93f80520e4ec0480282` — loaded, never regenerated, so EPD is directly comparable to EIPC-P0
- Sample: FIT 512 / TEST 256, disjoint, seed 20260919 (deliberately small: this is directional exploration)
- Compression: StandardScaler → PCA(32) → StandardScaler, FIT only, so all four representations share one dimensional budget
- Probe: `Ridge(alpha=1.0)`, multi-output

This experiment reports a **decomposition pattern**, not a SUPPORTED/NOT_SUPPORTED verdict.

## Read-only prior projects

- `/home/h-li/work/rejected_expert_delayed_value` — REDV-V1 (formal SUPPORTED, scientifically uninterpretable), REDV-V2 (NOT_SUPPORTED)
- `/home/h-li/work/delayed_rejected_evidence` — DREV-P0 (NOT_PROMISING)
- `/home/h-li/work/expert_identity_preservation` — EIPC-P0 (PROMISING, information-content)
- `/home/h-li/work/cross_layer_expert_cache` — XEC-P0 (NOT_PROMISING, actionability)

None were modified. EPD-P0 executes no rejected experts, no counterfactual routing, no
routing regret, and no oracle actions.

## Running

```bash
PY=/home/h-li/miniconda3/envs/sensorllm/bin/python
$PY scripts/build_manifest.py                                  # freeze fresh sample
$PY -m pytest tests/ -q                                        # invariants A-Q
$PY scripts/smoke_epd.py                                       # plumbing only
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=<gpu_i> $PY scripts/extract_shard.py $i & done; wait
$PY scripts/merge_shards.py                                    # CPU
$PY scripts/analyze_epd.py                                     # CPU, probes + layerwise + transitions
```
