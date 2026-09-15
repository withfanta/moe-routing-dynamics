# Reproduction — EIPC-P0

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

BASE checkpoint, `float16`, not quantized, `config_mismatches: []`.

## 3. Dataset and split

```
Salesforce/wikitext
config   wikitext-103-raw-v1
revision b08601e04326c79dfdd32d625aee71d232d685c3
split    train
```

1024 FIT / 1024 TEST, frozen in `artifacts/EIPC_P0/data_manifest.json` (included).

## 4. Required hardware

GPU required for the two extraction passes; originals ran on **Tesla V100-SXM2-32GB**. FIT and
TEST extraction are independent and were run on two GPUs concurrently. `finalize_eipc.py` is
CPU.

## 5. Exact entry commands

```bash
$PY scripts/inspect_environment.py     # provenance audit, no research metric
$PY scripts/build_manifest.py          # freeze train-split FIT/TEST sample
$PY -m pytest tests/ -q                # invariants A-P
$PY scripts/smoke_eipc.py              # plumbing only
CUDA_VISIBLE_DEVICES=<A> $PY scripts/extract_fit.py    # 1024 FIT
CUDA_VISIBLE_DEVICES=<B> $PY scripts/extract_test.py   # 1024 TEST
$PY scripts/finalize_eipc.py           # CPU, once, six probes, frozen rule
```

`scripts/write_results_md.py` regenerates `RESULTS.md` from an existing `results.json`.

## 6. Expected primary output

```
artifacts/EIPC_P0/results.json
artifacts/EIPC_P0/RESULTS.md
artifacts/EIPC_P0/projection.json    written before extraction, pins the fixed projection
```

By-products not versioned here: `fit_raw.npz`, `test_raw.npz`.

## 7. Expected result values

```
final_verdict   PROMISING
mean_A          +0.1421454697718327     (identity - fused; the conservative quantity)
mean_B          +0.37951644215333513    (vs identity-destroyed control)
```

Per target (held-out R², 1024 FIT / 1024 TEST):

| Target layer | R²_fused | R²_identity | R²_shuffled |
|---|---|---|---|
| 8 | +0.19016 | +0.29939 | −0.00149 |
| 12 | +0.28705 | +0.46211 | +0.00396 |

The fixed projection must hash to
`fb9e6e9b7568f6d4a5ce74a7bee1ba2eab5a48a02fb2a93f80520e4ec0480282` (seed 20260917, shape
(2048, 32), Normal(0, 1/√32), float64, not trainable). If it does not, the run is not comparable.

Reminder: mean B is the shuffled-identity contrast and must not be read as a clean
information-theoretic identity effect. See [PACKAGING_PROVENANCE.md](PACKAGING_PROVENANCE.md).
