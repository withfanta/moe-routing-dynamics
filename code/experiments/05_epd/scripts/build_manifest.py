"""Freeze the EPD-P0 sample from fresh WikiText train blocks, and record provenance.

Excludes every block used by EIPC-P0 and XEC-P0, verifies REDV/DREV never used train, and
verifies the reused EIPC projection hash. Runs on CPU; produces no research statistic.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy
import sklearn
import torch
import transformers
from transformers import AutoTokenizer

from epd import (
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    MODEL_ID,
    MODEL_REVISION,
    N_FIT,
    N_SHARDS,
    N_TEST,
    N_TOTAL,
    SAMPLE_SEED,
)
from epd.data import (
    build_token_stream,
    draw_sample,
    fresh_blocks,
    load_prior_used_blocks,
    load_train_texts,
    make_blocks,
    manifest_payload,
    verify_prior_projects_avoided_train,
    write_manifest,
)
from epd.extraction import load_eipc_projection

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "EPD_P0")
DATASET_DIR = os.environ.get("EPD_DATASET_DIR")
MODEL_DIR = os.environ.get("EPD_MODEL_DIR")

EXPECTED_CONFIG = {
    "architectures": ["OlmoeForCausalLM"],
    "hidden_size": 2048,
    "num_hidden_layers": 16,
    "num_experts": 64,
    "num_experts_per_tok": 8,
    "norm_topk_prob": False,
}


def gpu_table():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader"], text=True)
        rows = []
        for line in out.strip().splitlines():
            idx, name, uuid, total, free, util = [p.strip() for p in line.split(",")]
            rows.append({"index": int(idx), "name": name, "uuid": uuid,
                         "memory_total": total, "memory_free": free, "utilization": util})
        return rows
    except Exception as e:  # pragma: no cover
        return [{"error": str(e)}]


def main():
    os.makedirs(ART, exist_ok=True)

    # Environment record.
    gpus = gpu_table()
    idle = [g for g in gpus if "V100" in g.get("name", "")
            and int(g.get("memory_free", "0 MiB").split()[0]) > 25000]
    env = {
        "experiment": "EPD-P0",
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "packages": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "transformers": transformers.__version__,
            "numpy": numpy.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "gpus_detected": gpus,
        "idle_v100_candidates": [g["index"] for g in idle],
        "n_shards_required": N_SHARDS,
        "gradients_on_olmoe": False,
    }
    with open(os.path.join(ART, "environment.json"), "w") as fh:
        json.dump(env, fh, indent=2)
    print("environment.json written")
    for g in gpus:
        print(f"  GPU {g.get('index')}: free {g.get('memory_free')} util {g.get('utilization')}")
    print("idle V100 candidates:", env["idle_v100_candidates"])
    if len(idle) < N_SHARDS:
        print(f"WARNING: only {len(idle)} idle V100s; {N_SHARDS} are needed")

    # Reused EIPC projection, hash-verified.
    P, proj_meta = load_eipc_projection()
    print(f"\nEIPC projection verified: sha256 {proj_meta['sha256']} shape {P.shape}")

    # Prior-use exclusion.
    checked = verify_prior_projects_avoided_train()
    print("REDV/DREV verified never to have used train:", checked)
    excluded, prior_info = load_prior_used_blocks()
    print(f"prior exclusion: {prior_info}")

    src = MODEL_DIR or MODEL_ID
    kwargs = {} if MODEL_DIR else {"revision": MODEL_REVISION}
    tok = AutoTokenizer.from_pretrained(src, **kwargs)
    eos = 50279

    texts = load_train_texts(dataset_dir=DATASET_DIR)
    stream = build_token_stream(texts, tok, eos)
    blocks = make_blocks(stream)
    print(f"\ntrain: {len(texts)} records -> {len(stream)} tokens -> {len(blocks)} blocks")

    if len(blocks) != prior_info["eipc_blocks_available"]:
        raise AssertionError(
            f"stream mismatch: {len(blocks)} blocks vs prior {prior_info['eipc_blocks_available']}")
    print("stream matches prior projects' block indexing, so exclusion is exact")

    pool = fresh_blocks(blocks, excluded)
    print(f"fresh pool: {len(pool)} blocks (need {N_TOTAL})")
    if len(pool) < N_TOTAL:
        print(f"TECHNICAL_BLOCKER: only {len(pool)} fresh blocks")
        return 1

    contexts = draw_sample(pool, seed=SAMPLE_SEED)

    # Model provenance.
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    verified, mismatches = {}, []
    for k, exp in EXPECTED_CONFIG.items():
        act = cfg.architectures if k == "architectures" else getattr(cfg, k, None)
        verified[k] = {"expected": exp, "actual": act, "ok": act == exp}
        if act != exp:
            mismatches.append(k)
    with open(os.path.join(ART, "model_provenance.json"), "w") as fh:
        json.dump({
            "experiment": "EPD-P0",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "instruct_checkpoint": False,
            "dataset_id": DATASET_ID,
            "dataset_config": DATASET_CONFIG,
            "dataset_revision": DATASET_REVISION,
            "dataset_split_used": DATASET_SPLIT,
            "config_verification": verified,
            "config_mismatches": mismatches,
            "tokenizer_class": tok.__class__.__name__,
            "runtime_dtype": "float16",
            "quantized": False,
            "olmoe_receives_gradients": False,
            "reused_eipc_projection": proj_meta,
        }, fh, indent=2)
    print("model_provenance.json written; config mismatches:", mismatches)
    if mismatches:
        return 1

    payload = manifest_payload(
        contexts, len(pool), len(texts), len(stream), len(blocks), prior_info, checked,
        proj_meta, extra={"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                          "tokenizer_class": tok.__class__.__name__})
    h = write_manifest(os.path.join(ART, "data_manifest.json"), payload)

    print(f"\ndata_manifest.json written, sha256 {h}")
    print(f"  fit  n={payload['fit']['n']} fingerprint {payload['fit']['fingerprint'][:16]}")
    print(f"  test n={payload['test']['n']} fingerprint {payload['test']['fingerprint'][:16]}")
    print(f"  combined fingerprint {payload['fingerprint_all'][:16]}")
    fit_ids = set(payload["fit"]["block_ids"])
    test_ids = set(payload["test"]["block_ids"])
    print(f"  fit ∩ test = {len(fit_ids & test_ids)}")
    print(f"  EPD ∩ prior = {len((fit_ids | test_ids) & excluded)}")
    print(f"  shard sizes {payload['shard_sizes']}")
    print(f"  shard role counts {payload['shard_role_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
