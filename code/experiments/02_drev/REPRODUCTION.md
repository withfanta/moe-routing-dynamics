# Reproduction — DREV-P0

Commands are taken from this experiment's own `README.md`. Nothing is invented.

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

BASE checkpoint, `float16`, not quantized. 16 layers, 64 experts, Top-8, `norm_topk_prob=false`,
hidden 2048. Tokenizer `GPTNeoXTokenizerFast`, `eos_token_id` 50279. `config_mismatches: []`.

## 3. Dataset and split

```
Salesforce/wikitext
config   wikitext-103-raw-v1
revision b08601e04326c79dfdd32d625aee71d232d685c3
```

The sampled blocks are frozen in `artifacts/DREV_P0/data_manifest.json` (included). 512 FIT /
512 TEST at every horizon.

## 4. Required hardware

GPU required for extraction and the horizon workers; originals ran on **Tesla V100-SXM2-32GB**
(shared DGX-1, 8 visible). The pipeline is written for two GPUs in parallel — worker A takes
target layers 5 and 6, worker B takes 8 and 12. `finalize_drev.py` is CPU.

## 5. Exact entry commands

```bash
$PY scripts/inspect_environment.py      # provenance audit, no research metric
$PY scripts/build_manifest.py           # freeze untouched-block sample
$PY -m pytest tests/ -q                 # tests A-L
$PY scripts/smoke_drev.py               # plumbing only
$PY scripts/extract_source.py           # Layer-4 b and r, computed once
CUDA_VISIBLE_DEVICES=<A> $PY scripts/run_horizons_gpu_a.py   # targets 5, 6
CUDA_VISIBLE_DEVICES=<B> $PY scripts/run_horizons_gpu_b.py   # targets 8, 12
$PY scripts/finalize_drev.py            # CPU, once, after both workers
```

`scripts/write_results_md.py` regenerates `RESULTS.md` from an existing `results.json`.

## 6. Expected primary output

```
artifacts/DREV_P0/results.json          aggregate verdict
artifacts/DREV_P0/horizon_1.json        per-horizon results, written by the workers
artifacts/DREV_P0/horizon_2.json
artifacts/DREV_P0/horizon_4.json
artifacts/DREV_P0/horizon_8.json
artifacts/DREV_P0/RESULTS.md
```

By-products not versioned here: `source_fit.npz`, `source_test.npz` (see
[../../ARTIFACTS_NOT_VERSIONED.md](../../ARTIFACTS_NOT_VERSIONED.md)).

## 7. Expected result values

```
final_verdict        NOT_PROMISING
mean_D_delayed       -0.05930933632977128
```

Per horizon (held-out R², 512 FIT / 512 TEST):

| Horizon | Target layer | R²_real | R²_shuffle | mean G | prop. G > 0 |
|---|---|---|---|---|---|
| 1 | 5 | −0.08378 | −0.04037 | +0.05226 | 0.75195 |
| 2 | 6 | −0.05771 | −0.03739 | +0.05386 | 0.73828 |
| 4 | 8 | −0.07582 | −0.02717 | +0.05406 | 0.77930 |
| 8 | 12 | −0.13461 | −0.02565 | +0.04732 | 0.75000 |

All four preregistered conditions were required; the recorded rule is in `results.json` under
`judgement_rule`, with `frozen_before_results: true`. Bit-exact reproduction of float16 GPU
extraction is not guaranteed across devices and drivers; the frozen sample and seeds are.
