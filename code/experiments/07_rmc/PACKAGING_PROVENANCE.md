# Packaging provenance — RMC-P0

**Source repository:** `/home/h-li/work/routing_memory_crossmodel`

**Source commit SHA:** `cc60edd9313c15c9a0f433d9f897a38fb44ef1a3` (branch `master`, 3 commits,
working tree clean at packaging time)

**Packaging method:** `git archive cc60edd | tar -x`. The frozen commit is the canonical
source; no working-tree file was packaged.

## Exact scientific status

**Preregistered result: REPLICATED.**

The layer-history routing signal first seen in OLMoE reappears in a second, architecturally
different sparse MoE — JetMoE-8B, 24 layers, 8 MLP experts, Top-2 — under a rule frozen before
any statistic existed. Held-out R² predicting native MLP router logits from binary Top-2
selection identities over the previous k layers (FIT 512 / TEST 256):

```
target Layer 12    k=1 +0.20924   k=2 +0.34982   k=4 +0.35199   k=8 +0.38275
target Layer 20    k=1 +0.13898   k=2 +0.24936   k=4 +0.34425   k=8 +0.35463
```

Primary comparison k=1 vs k=4, paired bootstrap (10,000 resamples, seed 314159):

```
Layer 12   Δ4 = +0.14368   95% CI [+0.10998, +0.17916]   fraction above zero 1.0
Layer 20   Δ4 = +0.20587   95% CI [+0.16891, +0.24396]   fraction above zero 1.0
```

All four preregistered conditions held (k=1 R² positive and bootstrap CI lower bound above zero
at both targets), so the verdict is REPLICATED.

Constraints on how this may be read:

- **Do not claim a formal Markov order.** The measured quantity is incremental held-out R² from
  a wider history window under one compression and one probe.
- **Do not claim that history is mathematically proven to contain information absent from recent
  state.** That is a stronger claim than any R² comparison here supports; NHD-P0 was the
  follow-up that examined the recent-state alternative, and it also does not establish it.
- `architecture_built: none`, `hyperparameter_search: none`, `additional_model: none`,
  `additional_dataset: none`, `rescue_modifications: none`.

### Router identification

JetMoE has two mixtures. The MoA (attention) and MLP routers use the identical class with the
identical 2048→8 shape, so shape alone can never distinguish them. Identification was done by
object identity plus numerical recomputation, recorded in `artifacts/router_identification.json`:

```
blocks checked                                   24
max |diff| vs recomputed MLP router logits       0.0
min |diff| vs MoA router logits                  0.7412109375
top-k identity/logit agreement                   1.0
```

Only `model.layers[i].mlp.router` was studied; `moa_router_used: false`.

`transformers` 4.45.1's built-in `output_router_logits=True` raises `AttributeError`
(`JetMoeForCausalLM` has no attribute `num_experts`) for this model. That path is unused by
design and the failure is recorded rather than worked around.

## Files included

21 files, 750,578 B total. Flat layout.

```
README.md
protocol.md                          preregistration, fixed before data
rmc.py                               library
build_manifest.py
record_provenance.py
extract.py                           GPU, sharded
merge.py
analyze.py                           CPU, gated on tests
test_rmc.py                          checks A-P, includes a real 2-layer smoke load
artifacts/results.json               frozen result
artifacts/RESULTS.md
artifacts/data_manifest.json         frozen sample
artifacts/model_provenance.json
artifacts/environment.json
artifacts/router_identification.json numerical proof of which router was read
artifacts/merged.npz                 packaged: 311 KB
artifacts/shard_0.npz                packaged
artifacts/shard_1.npz
artifacts/shard_2.npz
artifacts/shard_3.npz
.gitignore
```

## Files intentionally omitted

None. `omitted_bytes_total: 0`. Because RMC-P0 records only 8-d selection indicators and router
logits rather than hidden states, its extraction output is small enough to version in full: the
four shards plus `merged.npz` total well under 1 MB. This is the only experiment in the series
whose extracted data is packaged, which is also why NHD-P0 could reuse it on CPU with no
regeneration step.

Omitted by policy only: `__pycache__`, run logs, and the source `.git` directory.

## Artifact hashes

Recorded inside `artifacts/results.json`:

```
data_manifest.json  sha256   371601f7ef6c491d571b1528fb15ef5dcd8cd361844786aee9fc8c4aef8afe02
fit fingerprint             7951ab9004769a2c923ed1a0f21fd64784743eed5f42a4789178601e72337a6e
test fingerprint            c4441a9beaf2583ed7327e9c289e11ab17631a06e3675c370d178507a3c1abd7
shard_0                     dd568f2cd45b672412b17737587ded68529d94f5b40b34d6263d22eb52d2deb3
shard_1                     e8d3bd8db2a6ee8319f9d5b4bdf9e20397963662456c40b543f853b348dff711
```

The full `shard_sha256` block covering all four shards is in `results.json`; those recorded
values are authoritative. The `transformers` JetMoE modelling file itself is pinned:
`2f02f04ef4ce3da366bd7ee58da4d922968fba949e9aeb82e9a9201898531a20`, `implementation_rewritten:
false`.

The sample is model-specific: `reused_prior_block_indices: false` — OLMoE block indices were not
reused and no prior manifest was read, because tokenization differs between the two models.

## Reproduction entry point

See [REPRODUCTION.md](REPRODUCTION.md). Summary:

```bash
PY=/home/h-li/miniconda3/envs/sensorllm/bin/python
$PY build_manifest.py
$PY record_provenance.py
$PY -m pytest test_rmc.py -q
CUDA_VISIBLE_DEVICES=<gpu> $PY extract.py --shard <0..3>
$PY merge.py
$PY analyze.py
```
