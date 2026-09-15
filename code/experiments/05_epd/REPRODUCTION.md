# Reproduction — EPD-P0

Commands are taken from this experiment's own `README.md` ("Running"). Nothing is invented.

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
revision 9b0c1aa87e34a20052389dce1f0cf01da783f654
```

BASE checkpoint, `float16`, not quantized, `olmoe_receives_gradients: false`,
`config_mismatches: []`.

## 3. Dataset and split

```
Salesforce/wikitext
config   wikitext-103-raw-v1
revision b08601e04326c79dfdd32d625aee71d232d685c3
split    train
```

A fresh sample, frozen in `artifacts/EPD_P0/data_manifest.json` (included). 512 FIT / 256 TEST,
with the `is_test` mask stored in the extraction output — RMO-P0 later reused that exact mask.

## 4. Required hardware

GPU required for extraction; originals ran on **Tesla V100-SXM2-32GB**. Extraction is sharded
four ways and was run across four GPUs concurrently. `merge_shards.py` and `analyze_epd.py` are
CPU.

## 5. Exact entry commands

```bash
$PY scripts/build_manifest.py                                  # freeze fresh sample
$PY -m pytest tests/ -q                                        # invariants A-Q
$PY scripts/smoke_epd.py                                       # plumbing only
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=<gpu_i> $PY scripts/extract_shard.py $i & done; wait
$PY scripts/merge_shards.py                                    # CPU
$PY scripts/analyze_epd.py                                     # CPU, probes + layerwise + transitions
```

`scripts/write_results_md.py` regenerates `RESULTS.md` from an existing `results.json`.

## 6. Expected primary output

```
artifacts/EPD_P0/results.json
artifacts/EPD_P0/RESULTS.md
```

By-products not versioned here: `shard_0..3.npz` and `merged_raw.npz`. **`merged_raw.npz` is
also RMO-P0's only input**, so regenerating it is a prerequisite for re-running RMO-P0 from this
repository.

## 7. Expected result values

No formal verdict — `exploratory: true`, `no_formal_verdict: true`, `architecture_built: none`.
Exploratory label: PATH-DOMINANT.

Main probes, Layer-12 target, history layers 1–11, held-out R² (512 FIT / 256 TEST):

```
R2_fused            +0.20114033648445187
R2_id               +0.6696914181719282
R2_content          +0.06732391703802035
R2_full             +0.34007424419871485
```

Conditional gains: `G_identity_given_content` = +0.27275, `G_content_given_identity` = −0.32962.

Layerwise identity R² rises monotonically from +0.14253 (layer 1) to +0.59927 (layer 11);
best content layer is 8 at +0.02139. The 0.02 figure used in the pattern description is a
discussion aid, not a significance threshold.

The reused EIPC-P0 projection must hash to
`fb9e6e9b7568f6d4a5ce74a7bee1ba2eab5a48a02fb2a93f80520e4ec0480282` (seed 20260917, shape
(2048, 32)). Bit-exact reproduction of float16 GPU extraction is not guaranteed across devices.
