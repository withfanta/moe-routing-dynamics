# Reproduction — RMO-P0

Commands are taken from this experiment's own `README.md` ("Run"). Nothing is invented.

## 1. Required environment

```
python        3.10.19    (/home/h-li/miniconda3/envs/sensorllm/bin/python)
numpy         1.24.4
scikit-learn  1.3.2
```

The `sensorllm` env is required specifically because it holds the sklearn 1.3.2 / numpy 1.24.4
versions EPD-P0's own analysis ran under. The system default python (3.13 / transformers 4.57)
is the wrong environment for this repository.

```bash
PY=/home/h-li/miniconda3/envs/sensorllm/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
```

## 2. Model and immutable revision

No model is loaded. `new_model_inference_run: false`. The underlying capture that produced the
data used `allenai/OLMoE-1B-7B-0125` revision `9b0c1aa87e34a20052389dce1f0cf01da783f654`, frozen,
FP16, unquantized, no gradients.

## 3. Dataset and split

No dataset access. Every quantity comes from the closed, read-only EPD-P0 artifact

```
/home/h-li/work/expert_provenance_decomposition/artifacts/EPD_P0/merged_raw.npz
```

with `data_manifest.json` asserted to hash
`bbbe06fcb451b217de3b03cf75d37921a4e19f555d95d92302b8898d8a3466ba` before anything is computed.
The FIT 512 / TEST 256 split is EPD-P0's own stored `is_test` mask; nothing is resampled.

**Availability in this packaged repository:** `merged_raw.npz` is an omitted artifact (8.55 MB).
It must be regenerated first — see [../05_epd/REPRODUCTION.md](../05_epd/REPRODUCTION.md) — and
the source path in `analyze.py` repointed at it. Until then the frozen result is readable but the
analysis is not re-runnable here.

## 4. Required hardware

CPU only. `gpu_used: false`. Minutes, not hours.

## 5. Exact entry commands

```bash
$PY -m pytest test_analysis.py -q
$PY analyze.py
```

`analyze.py` refuses to write results unless the tests passed in the same invocation chain
(protocol test J).

## 6. Expected primary output

```
artifacts/results.json
artifacts/RESULTS.md
```

## 7. Expected result values

No formal verdict — `exploratory: true`, `no_formal_verdict: true`,
`post_hoc_on_already_observed_data: true`, `independent_confirmation: false`,
`architecture_built: none`. Exploratory label: SHORT-HISTORY.

Layer-12 history curve, held-out R² over cumulative windows:

| Window | Layers | R² |
|---|---|---|
| k=1 | L11 only | +0.5987857371764951 |
| k=2 | L10–11 | +0.623200212340848 |
| k=4 | L8–11 | +0.647045882367207 |
| k=8 | L4–11 | +0.6626511098179462 |
| k=11 | L1–11 | +0.6654387153209018 |

Stepwise gains +0.02441, +0.02385, +0.01561, +0.00279; max cumulative gain +0.06665;
`curve_irregular: false`, `max_drop_along_curve: 0.0`.

Compression: `StandardScaler → PCA(randomized) → StandardScaler`, fit on FIT only, 16
components, `random_state=20260920`. Probe: Ridge(alpha=1.0), multi-output.

Do not read the SHORT-HISTORY label as a formal Markov order. See
[PACKAGING_PROVENANCE.md](PACKAGING_PROVENANCE.md).
