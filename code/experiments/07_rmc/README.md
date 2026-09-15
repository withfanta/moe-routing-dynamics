# RMC-P0 — Routing Memory Cross-Model Pilot

One cross-model replication. Asks whether predictive routing dependence beyond the
immediately preceding layer replicates in a sparse MoE architecturally distant from
OLMoE.

Scope, thresholds, and the verdict rule are fixed in [protocol.md](protocol.md), committed
before any router prediction or R² value existed.

## What this is not

Not a method experiment. No recurrent router, no memory, no architecture. No OLMoE rerun,
no second model, no second dataset, no alpha or PCA search. Previous projects are CLOSED
and READ-ONLY; none is read or modified here.

## Question

After the immediately preceding MLP-MoE routing state is already known, does earlier MLP
expert-selection history still provide incremental held-out information about the current
MLP router state?

The primary comparison is k=1 versus k=4, at two fixed targets: Layer 12 and Layer 20.

## Model

`jetmoe/jetmoe-8b`, BASE checkpoint, pinned at immutable commit
`d8fd02ccf7911aa8148a63c7984ffd2e465b0352`. 24 blocks, 8 MLP experts, Top-2, ~8B total /
~2.2B active parameters. Frozen, `eval()`, FP16, `torch.inference_mode()`, not quantized,
no gradients.

Implementation is the official in-tree `transformers.models.jetmoe` at transformers
4.45.1, used verbatim and pinned by file content hash in
`artifacts/model_provenance.json`.

## Only the MLP router

Every JetMoE block holds two mixtures — a Mixture of Attention heads and a Mixture of MLP
Experts — and **both use the same gating class**, so the router is identified structurally
rather than by name. Hooks attach to `model.layers[i].mlp.router` only.

The library's own `output_router_logits=True` is deliberately unused: it returns attention
and MLP router logits interleaved in one flat tuple, which is exactly the confusion to
avoid. Test D pins the captured module to `layers[i].mlp.router` by identity and rejects
the MoA router.

## Design

Input per layer is `S_l ∈ R^8`, a binary indicator of the native Top-2 MLP experts at the
experimental token — identities only, no probabilities, no rank. Target is the native MLP
router logits at the target block, per-sample centred then standardized on FIT statistics
only.

Every probe sees exactly 16 dimensions: a fixed 8-d `z_recent` (the scaler for `S_{m-1}`,
fitted once per target and reused for every k) concatenated with an 8-d `z_old(k)`
(`StandardScaler -> PCA(8, randomized, random_state=20260920) -> StandardScaler`, fit on
FIT only, over the history strictly before `S_{m-1}`). For k=1, `z_old = zeros(8)`.

Eight probes total: 2 targets × 4 history depths, each `Ridge(alpha=1.0)`, multi-output.

## Verdict rule (frozen)

REPLICATED requires all six: for each of Layer 12 and Layer 20, `R2_1 > 0`,
`Delta4 >= +0.02`, and a 95% paired-bootstrap CI lower bound above 0 (10000 resamples,
seed 314159, resampling TEST sample indices jointly for both models). Otherwise
NOT_REPLICATED. No INCONCLUSIVE category for ordinary numerical results.

## Data

`Salesforce/wikitext` / `wikitext-103-raw-v1` TRAIN, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, tokenized with the JetMoE tokenizer, EOS
between original records, non-overlapping 129-token blocks. Experimental token is context
position 127. FIT = 512 / TEST = 256, disjoint, seed 20260920.

Tokenization is model-specific, so OLMoE block indices are **not** reused and no prior
manifest is read.

## Layout

    README.md  protocol.md
    rmc.py                 shared frozen constants and manifest types
    build_manifest.py      freezes artifacts/data_manifest.json
    record_provenance.py   freezes provenance and environment
    extract.py             one single-GPU shard worker
    merge.py               merges four shards
    analyze.py             eight probes, bootstrap, mechanical verdict
    test_rmc.py            protocol tests A-P
    artifacts/             manifest, provenance, shards, results

## Run

    python build_manifest.py                 # before anything else
    python record_provenance.py
    python -m pytest test_rmc.py -q          # A-P, includes a real 2-layer smoke load
    CUDA_VISIBLE_DEVICES=<gpu> python extract.py --shard <0..3>
    python merge.py
    python analyze.py

Run with `/home/h-li/miniconda3/envs/sensorllm/bin/python`.

## Result

See [artifacts/RESULTS.md](artifacts/RESULTS.md).

## Status

Closed on completion. No method is derived from the outcome; any further experiment
requires explicit authorization.
