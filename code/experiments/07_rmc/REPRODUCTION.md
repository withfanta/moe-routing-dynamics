# Reproduction — RMC-P0

Commands are taken from this experiment's own `README.md` ("Run"). Nothing is invented.

## 1. Required environment

```
python        3.10.19    (/home/h-li/miniconda3/envs/sensorllm/bin/python)
torch         2.4.1+cu121   (CUDA 12.1)
transformers  4.45.1        (JetMoE support must be the official in-tree implementation)
numpy         1.24.4
scikit-learn  1.3.2
pytest        8.3.5
```

```bash
PY=/home/h-li/miniconda3/envs/sensorllm/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
```

The modelling file itself is pinned:
`transformers/models/jetmoe/modeling_jetmoe.py` sha256
`2f02f04ef4ce3da366bd7ee58da4d922968fba949e9aeb82e9a9201898531a20`,
`implementation_rewritten: false`, `third_party_jetmoe_installed: false`.

Known limitation, recorded not worked around: `output_router_logits=True` raises
`AttributeError` in transformers 4.45.1 (`JetMoeForCausalLM` has no attribute `num_experts`).
That path is unused by design; router logits are read from the modules directly.

## 2. Model and immutable revision

```
jetmoe/jetmoe-8b
revision d8fd02ccf7911aa8148a63c7984ffd2e465b0352   (immutable sha)
```

BASE checkpoint (not SFT/chat), `float16`, not quantized, no gradients. 24 layers, 8 local
experts, Top-2, hidden 2048, intermediate 5632.

**Only the MLP MoE router is studied** (`model.layers[i].mlp.router`); `moa_router_used: false`.
MoA and MLP routers share the identical class and 2048→8 shape, so `record_provenance.py`
identifies the right one by object identity plus numerical recomputation of `W_router @ x`. The
expected evidence, in `artifacts/router_identification.json`:

```
blocks checked                              24
max |diff| vs recomputed MLP router logits  0.0
min |diff| vs MoA router logits             0.7412109375
top-k identity/logit agreement              1.0
```

## 3. Dataset and split

```
Salesforce/wikitext
config   wikitext-103-raw-v1
revision b08601e04326c79dfdd32d625aee71d232d685c3
split    train
```

512 FIT / 256 TEST, frozen in `artifacts/data_manifest.json` (included), 4 shards of 128 FIT +
64 TEST each. `reused_prior_block_indices: false` — tokenization is model-specific, so no OLMoE
manifest was read.

## 4. Required hardware

GPU required for extraction; originals ran on a single **Tesla V100-SXM2-32GB** at 15.93 GiB
allocated (`single_gpu_fp16_load: true`). The four shards can run sequentially on one GPU.
`merge.py` and `analyze.py` are CPU.

## 5. Exact entry commands

```bash
$PY build_manifest.py                 # before anything else
$PY record_provenance.py
$PY -m pytest test_rmc.py -q          # A-P, includes a real 2-layer smoke load
CUDA_VISIBLE_DEVICES=<gpu> $PY extract.py --shard <0..3>
$PY merge.py
$PY analyze.py
```

## 6. Expected primary output

```
artifacts/results.json
artifacts/RESULTS.md
artifacts/router_identification.json
artifacts/shard_0..3.npz     packaged here
artifacts/merged.npz         packaged here
```

## 7. Expected result values

```
verdict   REPLICATED
checks    45 passed, 2 skipped, 2 warnings
```

Held-out R² (512 FIT / 256 TEST), input = 8-d binary Top-2 selection indicators over the
previous k layers, feature dim 16 after `StandardScaler → PCA(8, randomized,
random_state=20260920) → StandardScaler` fit on FIT only, probe Ridge(alpha=1.0):

| Target | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| L12 | +0.20924 | +0.34982 | +0.35199 | +0.38275 |
| L20 | +0.13898 | +0.24936 | +0.34425 | +0.35463 |

Primary comparison k=1 vs k=4, paired bootstrap 10,000 resamples, seed 314159:

```
L12   Delta4 mean +0.14368   95% CI [+0.10998, +0.17916]   frac above zero 1.0
L20   Delta4 mean +0.20587   95% CI [+0.16891, +0.24396]   frac above zero 1.0
```

Threshold `delta4_threshold = 0.02`. Expected artifact hashes:

```
data_manifest.json   371601f7ef6c491d571b1528fb15ef5dcd8cd361844786aee9fc8c4aef8afe02
merged.npz           7dafe8048772937c239bf9ab61577d9cce648847af4a6227e4ca5d86a9758726
fit fingerprint      7951ab9004769a2c923ed1a0f21fd64784743eed5f42a4789178601e72337a6e
test fingerprint     c4441a9beaf2583ed7327e9c289e11ab17631a06e3675c370d178507a3c1abd7
```

The `merged.npz` hash above is the value NHD-P0 pinned, and the packaged file matches it.
Do not read REPLICATED as a formal Markov order claim. See
[PACKAGING_PROVENANCE.md](PACKAGING_PROVENANCE.md).
