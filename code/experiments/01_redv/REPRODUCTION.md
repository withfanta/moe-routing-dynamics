# Reproduction — REDV-V1 and REDV-V2

Commands below are taken from this experiment's own `README.md` and scripts. Nothing is
invented.

## 1. Required environment

```
python        3.10.19    (/home/h-li/miniconda3/envs/sensorllm/bin/python)
torch         2.4.1+cu121
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
revision 9b0c1aa87e34a20052389dce1f0cf01da783f654
```

BASE checkpoint (`instruct_checkpoint: false`), runtime dtype `float16`, not quantized.
Tokenizer `GPTNeoXTokenizerFast`, vocab 50280, `eos_token_id` 50279. `intermediate_size` 1024.

## 3. Dataset and split

```
Salesforce/wikitext
config   wikitext-103-raw-v1
revision b08601e04326c79dfdd32d625aee71d232d685c3
```

The exact sampled blocks are frozen in `artifacts/REDV_V1/data_manifest.json` and
`artifacts/REDV_V2/data_manifest.json`, both included here. Their sha256 values are recorded
inside the corresponding `results.json` as `data_manifest_sha256`.

## 4. Required hardware

GPU required. Originals ran on **Tesla V100-SXM2-32GB** (capability 7.0, 32 GB).

`run_redv_v2.py` enforces this: it raises `TECHNICAL_BLOCKER: CUDA unavailable; OLMoE must not
run on CPU` if no GPU is visible, and `TECHNICAL_BLOCKER: expected a Tesla V100-class device`
if the device is not V100-class. Reproducing on a different GPU class requires relaxing that
check, which changes the frozen conditions — do not do so silently.

REDV-V2 ran with `CUDA_VISIBLE_DEVICES=2` on the original shared host, visible as `cuda:0`.
Analysis stages are CPU.

## 5. Exact entry commands

```bash
$PY scripts/inspect_environment.py     # provenance audit, no research metric
$PY -m pytest tests/ -v                # implementation correctness
$PY scripts/smoke_redv.py              # end-to-end plumbing on a few contexts
$PY scripts/run_redv_v1.py             # REDV-V1 formal run
```

REDV-V2 (the dimension-matched follow-up):

```bash
$PY -m pytest tests/test_redv_v2.py -v
CUDA_VISIBLE_DEVICES=<gpu> $PY scripts/run_redv_v2.py
```

Results-markdown writers, if the JSON exists but the Markdown does not:

```bash
$PY scripts/write_results_md.py        # REDV-V1
$PY scripts/write_results_md_v2.py     # REDV-V2
```

Each formal run extracts features and analyses in one pass, writing the NPZ caches as a
by-product.

## 6. Expected primary output

| Run | Writes |
|---|---|
| REDV-V1 | `artifacts/REDV_V1/results.json`, `RESULTS.md`, `validation_features.npz`, `test_features.npz` |
| REDV-V2 | `artifacts/REDV_V2/results.json`, `RESULTS.md`, `fit_features.npz`, `test_features.npz` |

## 7. Expected result values

**REDV-V1**

```
verdict            SUPPORTED
mean_delta_r2      +0.47405942463408407
```

**REDV-V2**

```
final_verdict      NOT_SUPPORTED
mean_D             -0.09973059672145974
```

Feature-cache sha256 values for byte-level comparison are in
[../../ARTIFACTS_NOT_VERSIONED.md](../../ARTIFACTS_NOT_VERSIONED.md). Bit-exact reproduction
of float16 GPU extraction is not guaranteed across devices and drivers; the frozen sample and
seeds are reproducible.

Reminder on interpretation: REDV-V1's `SUPPORTED` verdict is formal only and the result is
scientifically UNINTERPRETABLE (p ≫ n / dimensionality confound; scale-matched random noise
improved even more). REDV-V2 is the negative result that closed the line. See
[PACKAGING_PROVENANCE.md](PACKAGING_PROVENANCE.md).
