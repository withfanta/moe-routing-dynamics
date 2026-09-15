# Reproduction — NHD-P0

Commands are taken from this experiment's own `README.md` ("Run order"). Nothing is invented.

## 1. Required environment

```
python        3.10.19    (/home/h-li/miniconda3/envs/sensorllm/bin/python)
torch         2.4.1+cu121   (used on CPU only, 4 threads)
numpy         1.24.4
scikit-learn  1.3.2
pytest        8.3.5
```

```bash
PY=/home/h-li/miniconda3/envs/sensorllm/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
```

## 2. Model and immutable revision

No model is loaded. `new_model_inference: false`, `gpu_used: false`. The reused RMC-P0 data came
from `jetmoe/jetmoe-8b` revision `d8fd02ccf7911aa8148a63c7984ffd2e465b0352`, BASE, float16,
unquantized, MLP router only.

## 3. Dataset and split

No dataset access. Input is the frozen RMC-P0 artifact set, pinned at commit `cc60edd` and
hash-verified before anything is computed:

```
artifacts/merged.npz              7dafe8048772937c239bf9ab61577d9cce648847af4a6227e4ca5d86a9758726
artifacts/data_manifest.json      371601f7ef6c491d571b1528fb15ef5dcd8cd361844786aee9fc8c4aef8afe02
artifacts/model_provenance.json   e26d8ec79271f1312baad4c7993b6bd153d69f5426abb104221cae9e25585335
artifacts/results.json            595ee0b98ed70bd5a9758e8ede0042ac1ab7ccaef3192f8c44077584795c0851
```

Split: RMC-P0's own 512 FIT / 256 TEST via the stored `is_test` mask, not resampled. Within FIT,
384 train / 128 val (split seed recorded in `results.json`).

**Availability in this packaged repository:** all four pinned files are packaged under
[../07_rmc/artifacts/](../07_rmc/artifacts/), and `merged.npz` there hashes to `7dafe804…`,
matching the pin. Repoint `nhd.py`'s source directory at `../07_rmc` and the verification passes
without regenerating anything. This is the only experiment in the series that is fully
re-runnable from this repository as packaged.

## 4. Required hardware

CPU only, 4 threads. No GPU. Minutes.

## 5. Exact entry commands

```bash
$PY -c "import nhd; nhd.verify_source()"   # checks A, B
$PY analyze.py                             # gated on test S; writes results
```

`analyze.py` runs `test_nhd.py` as a subprocess and refuses to compute any TEST statistic unless
every check passes. To run the checks alone: `$PY -m pytest test_nhd.py -q`.

## 6. Expected primary output

```
artifacts/results.json
artifacts/RESULTS.md
artifacts/source_verification.json
```

## 7. Expected result values

```
classification   RESIDUAL-HISTORY-VALUE
checks           35 passed, 1 skipped
```

Per target, agreeing at both:

| Quantity | Layer 12 | Layer 20 |
|---|---|---|
| A (nonlinear gain on R) | +0.0013939501891806483 | +0.009361533892784463 |
| B | +0.17137406662770485 | +0.2186123962775228 |
| B_matched | +0.164401761783238 | +0.21911607474031944 |
| residual R² | +0.20548814488885064 | +0.23556024835469053 |
| nonlinear_improves | False | False |
| residual_positive | True | True |

Variables and capacity control:

```
R      S_{m-1} raw binary Top-2 MLP selection, 8 dims
H      [S_{m-4}, S_{m-3}, S_{m-2}] raw, 24 dims
full   [H;R], 32 dims                     no PCA, no compression
L12    recent layer 11, history [8, 9, 10]
L20    recent layer 19, history [16, 17, 18]

MLP_RECENT           Linear(8,32)  -> GELU -> Linear(32,8)
MLP_HISTORY          Linear(32,32) -> GELU -> Linear(32,8)    1320 params
MLP_RECENT_MATCHED   Linear(8,77)  -> GELU -> Linear(77,8)    1317 params
```

Width 77 was chosen mechanically by parameter count before TEST was seen. Training: AdamW,
lr 1e-3, weight decay 1e-4, max 300 epochs, patience 30, full batch, best state restored.
Init seeds 42 / 123 / 2026. Probe: Ridge(alpha=1.0). `near_zero_threshold = 0.02`.
Residualization is cross-fitted: no sample is residualized by a model trained on it.

Reading limits, restated because they are part of the result: B is not conditional information;
nothing here is a claim of information-theoretic conditional independence; and **even
RESIDUAL-HISTORY-VALUE does not prove that historical information is mathematically absent from
R — it only shows that the tested nonlinear recent-state predictor does not explain the full
held-out history gain.**
