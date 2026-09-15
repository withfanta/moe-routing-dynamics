# Reproduction — XEC-P0

Commands are taken from this experiment's own `README.md` ("Regenerating the ignored caches").
Nothing is invented.

## 1. Required environment

```
python        3.10.19    (/home/h-li/miniconda3/envs/sensorllm/bin/python)
torch         2.4.1+cu121   (CUDA 12.1)
transformers  4.45.1
numpy         1.24.4
scikit-learn  1.3.2
```

```bash
PY=/home/h-li/miniconda3/envs/sensorllm/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
```

## 2. Model and immutable revision

```
allenai/OLMoE-1B-7B-0125
revision 9b0c1aa87e34a20052389dce1f0cf01da783f654   (requested == resolved)
```

BASE checkpoint, `float16`, not quantized, `config_mismatches: []`. The OLMoE weights are frozen
throughout: `olmoe_receives_gradients: false` — only the small policies are trained.

## 3. Dataset and split

```
Salesforce/wikitext
config   wikitext-103-raw-v1
revision b08601e04326c79dfdd32d625aee71d232d685c3
split    train
```

TRAIN / VALIDATION / TEST are frozen in `artifacts/XEC_P0/data_manifest.json` (included).
`n_test = 1024`.

## 4. Required hardware

GPU required for extraction and for the final NLL pass; originals ran on **Tesla
V100-SXM2-32GB**. Worker A and worker B run on two GPUs concurrently. `train_policies.py` trains
the nine small policies; `finalize_xec.py` needs a GPU because it scores true next-token NLL
through OLMoE.

## 5. Exact entry commands

```bash
$PY scripts/build_manifest.py
CUDA_VISIBLE_DEVICES=<A> $PY scripts/extract_worker_a.py   # TRAIN + oracle
CUDA_VISIBLE_DEVICES=<B> $PY scripts/extract_worker_b.py   # VALIDATION + TEST
$PY scripts/train_policies.py                              # 9 checkpoints
CUDA_VISIBLE_DEVICES=<A> $PY scripts/finalize_xec.py       # TEST NLL, bootstrap, verdict
```

Also present: `scripts/inspect_environment.py` (provenance audit), `scripts/smoke_xec.py`
(plumbing only), `tests/test_xec.py` via `$PY -m pytest tests/ -q`, and
`scripts/write_results_md.py`. The README's regeneration block does not order the test step
explicitly; run it before extraction as in the sibling experiments.

## 6. Expected primary output

```
artifacts/XEC_P0/results.json
artifacts/XEC_P0/RESULTS.md
artifacts/XEC_P0/training_summary.json
artifacts/XEC_P0/training_complete.json
artifacts/XEC_P0/checkpoints/{CURRENT_ONLY,EXPERT_CACHE,FUSED_CACHE}_seed{42,123,2026}.pt
```

The nine checkpoints are packaged here with per-file sha256 in
[PACKAGING_PROVENANCE.md](PACKAGING_PROVENANCE.md). By-products not versioned: the three feature
NPZ caches and the oracle/policy-result NPZ files.

## 7. Expected result values

```
final_verdict   NOT_PROMISING
```

Mean TEST NLL (1024 samples), policy seeds 42 / 123 / 2026:

```
native                 2.6922965893329205
CURRENT_ONLY           2.690443709407912
FUSED_CACHE            2.6897350254719554
EXPERT_CACHE           2.6907906432290534
five-action oracle     2.6473121643066406
```

Paired differences, 10,000-resample bootstrap, seed 314159:

| Comparison | mean | 95% CI |
|---|---|---|
| d_native | −0.00150595 | [−0.00574278, +0.00275956] |
| d_current | +0.00034693 | [−0.00076092, +0.00166994] |
| d_fused | +0.00105562 | [−0.00002334, +0.00215620] |

`seed_check.all_seeds_expert_lower = false`. Action accuracy against the TEST oracle was ≈0.21
for all three policies, against a native-action fraction of 0.2646.

The fixed projection must hash to
`11a1228c725b2a952b6cb966ac0316bd102b8a48bacdbf76391f0d44476556a6` (seed 20260918, shape
(2048, 64), Normal(0, 1/√64), float64, not trainable, not learned). Every NPZ the verdict
depended on has its sha256 recorded in `results.json` under `artifact_sha256`; verify against
those. Bit-exact reproduction of float16 GPU extraction is not guaranteed across devices.
