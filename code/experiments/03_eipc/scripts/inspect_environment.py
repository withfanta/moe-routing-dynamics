"""EIPC-P0 environment and provenance audit. Computes NO research metric."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy
import scipy
import sklearn
import torch
import transformers
from huggingface_hub import HfApi

from eipc import (
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    MODEL_ID,
    MODEL_REVISION,
)
from eipc.expert_states import build_projection, projection_metadata

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "EIPC_P0")

EXPECTED = {
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
            ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free",
             "--format=csv,noheader"], text=True)
        rows = []
        for line in out.strip().splitlines():
            idx, name, uuid, total, free = [p.strip() for p in line.split(",")]
            rows.append({"index": int(idx), "name": name, "uuid": uuid,
                         "memory_total": total, "memory_free": free})
        return rows
    except Exception as e:  # pragma: no cover
        return [{"error": str(e)}]


def main():
    os.makedirs(ART, exist_ok=True)

    gpus = gpu_table()
    free_v100 = [g for g in gpus if "V100" in g.get("name", "")
                 and int(g.get("memory_free", "0 MiB").split()[0]) > 20000]

    env = {
        "experiment": "EIPC-P0",
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
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "gpus_detected": gpus,
        "free_v100_candidates": [g["index"] for g in free_v100],
        "packages_installed_by_this_run": [],
    }
    try:
        import datasets

        env["packages"]["datasets"] = datasets.__version__
    except Exception:
        env["packages"]["datasets"] = "unavailable"

    with open(os.path.join(ART, "environment.json"), "w") as fh:
        json.dump(env, fh, indent=2)
    print("environment.json written")
    print(json.dumps(env["packages"], indent=2))
    print("\nGPUs detected:")
    for g in gpus:
        print(f"  {g.get('index')}: {g.get('name')} free {g.get('memory_free')} uuid {g.get('uuid')}")
    print("free V100 candidates (>20 GiB):", env["free_v100_candidates"])

    api = HfApi()
    minfo = api.model_info(MODEL_ID, revision=MODEL_REVISION)
    dinfo = api.dataset_info(DATASET_ID, revision=DATASET_REVISION)

    from transformers import AutoConfig, AutoTokenizer

    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    verified, mismatches = {}, []
    for k, exp in EXPECTED.items():
        act = cfg.architectures if k == "architectures" else getattr(cfg, k, None)
        verified[k] = {"expected": exp, "actual": act, "ok": act == exp}
        if act != exp:
            mismatches.append(k)

    P = build_projection()
    proj_meta = projection_metadata(P)

    prov = {
        "experiment": "EIPC-P0",
        "model_id": MODEL_ID,
        "model_revision_requested": MODEL_REVISION,
        "model_revision_resolved": minfo.sha,
        "instruct_checkpoint": False,
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision_requested": DATASET_REVISION,
        "dataset_revision_resolved": dinfo.sha,
        "dataset_split_used": DATASET_SPLIT,
        "config_verification": verified,
        "config_mismatches": mismatches,
        "eos_token_id": cfg.eos_token_id,
        "tokenizer_class": tok.__class__.__name__,
        "runtime_dtype": "float16",
        "quantized": False,
        "projection": proj_meta,
    }
    with open(os.path.join(ART, "model_provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)

    print("\nmodel revision resolved :", minfo.sha)
    print("dataset revision resolved:", dinfo.sha)
    print("dataset split            :", DATASET_SPLIT)
    for k, v in verified.items():
        print(f"  {'OK ' if v['ok'] else 'BAD'} {k}: {v['actual']}")
    print("projection sha256:", proj_meta["sha256"])

    if mismatches or minfo.sha != MODEL_REVISION:
        print("\nSTOP: provenance mismatch", mismatches)
        return 1
    if len(free_v100) < 2:
        print(f"\nWARNING: only {len(free_v100)} free V100 candidates; two are needed")
    print("\nAudit passed. No research metric computed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
