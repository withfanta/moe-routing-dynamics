# Packaging provenance — RMO-P0

**Source repository:** `/home/h-li/work/routing_markov_order`

**Source commit SHA:** `a8e9d83cc2a6dd4800db72e610bd010ff80a83bb` (branch `master`, 3 commits,
working tree clean at packaging time)

**Packaging method:** `git archive a8e9d83 | tar -x`. The frozen commit is the canonical
source; no working-tree file was packaged.

## Exact scientific status

**Exploratory result: SHORT-HISTORY. No formal verdict.**

This is the OLMoE follow-up to EPD-P0. JetMoE is not involved. It asks whether the strong EPD-P0
`ID_PATH` signal is mostly trivial local routing persistence, or whether routing history from
before the immediately previous layer still adds incremental predictive information.

Layer-12 history curve, held-out R² over cumulative windows (FIT 512 / TEST 256):

```
L11 only   +0.59879
L10-11     +0.62320
L8-11      +0.64705
L4-11      +0.66265
L1-11      +0.66544
```

Stepwise gains +0.02441, +0.02385, +0.01561, +0.00279; max cumulative gain over L11-only
+0.06665. The curve rises then flattens, hence SHORT-HISTORY.

Constraints on how this may be read, from the experiment's own record:

- `exploratory: true`, `no_formal_verdict: true`, `post_hoc_on_already_observed_data: true`,
  `independent_confirmation: false`, `new_model_inference_run: false`, `gpu_used: false`.
- **This is a post-hoc analysis of an already-observed dataset. It is not independent
  confirmation of anything.**
- **Do not claim a formal Markov order.** The label describes the shape of a held-out R² curve
  under one compression and one probe. It is not a statement about conditional independence or
  about any formal Markov property of the routing process.
- `architecture_built: none`, `hyperparameter_search: none`, `rescue_modifications: none`.

## Files included

7 files, 61,151 B total. Flat layout, no `src/` package.

```
README.md
protocol.md              frozen exploratory protocol
analyze.py               the whole analysis, CPU only
test_analysis.py         invariant checks, gate for analyze.py
artifacts/results.json   frozen result
artifacts/RESULTS.md
.gitignore
```

## Files intentionally omitted

None. Nothing was omitted from this repository — every tracked file at `a8e9d83` is packaged,
and RMO-P0 produced no large artifacts of its own (`omitted_bytes_total: 0`).

## Artifact hashes

RMO-P0 stores no data of its own; it asserts hashes of the read-only EPD-P0 artifacts it
consumes before computing anything. Recorded in `artifacts/results.json` under `source`:

```
EPD-P0 commit                  995d23e
data_manifest.json  sha256     bbbe06fcb451b217de3b03cf75d37921a4e19f555d95d92302b8898d8a3466ba
merged_raw.npz      sha256     39c9b609d89d7fe568e88221b35bdcba6f686e4cec7af1ba1cdd76527d030442
shard_0.npz         sha256     690e9ffe2e624830df18356fbd2150ed6a47ba5c4f7f259b9ce7f67b6a55675c
shard_1.npz         sha256     42d05fe99010e4c6903de4b209f122ae5f58bf5d178ee0ca36d6606cd32ce31b
shard_2.npz         sha256     f9820d78864c029167f32199ae0be4023242876b8f187b080962a6254748cd30
shard_3.npz         sha256     590d974a2cd8a2644b59ea0f02e977493bd84c8f39b195e782285ec0bf800c0c
```

The FIT 512 / TEST 256 split is EPD-P0's own stored `is_test` mask; nothing was resampled. PCA
compression at 16 components, `random_state=20260920`.

**Dependency consequence for this packaged repository:** `merged_raw.npz` is an omitted
artifact (8.55 MB, see [../05_epd/PACKAGING_PROVENANCE.md](../05_epd/PACKAGING_PROVENANCE.md)).
RMO-P0's analysis therefore cannot be re-run from this repository as packaged until EPD-P0's
extraction is regenerated on a GPU host. Its frozen result remains fully readable.

RMO-P0 re-derives, rather than imports, the EPD-P0 representations it needs, so it has no
dependency on EPD-P0 source code and cannot modify it.

## Reproduction entry point

See [REPRODUCTION.md](REPRODUCTION.md). Summary:

```bash
PY=/home/h-li/miniconda3/envs/sensorllm/bin/python
$PY -m pytest test_analysis.py -q
$PY analyze.py
```

`analyze.py` refuses to write results unless the tests passed in the same invocation chain
(protocol test J).
