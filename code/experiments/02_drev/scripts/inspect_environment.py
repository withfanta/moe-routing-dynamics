"""DREV-P0 environment and provenance audit. Computes NO research metric."""

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

from drev import DATASET_CONFIG, DATASET_ID, DATASET_REVISION, MODEL_ID, MODEL_REVISION

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "DREV_P0")

EXPECTED = {
    "architectures": ["OlmoeForCausalLM"],
    "hidden_size": 2048,
    "num_hidden_layers": 16,
    "num_experts": 64,
    "num_experts_per_tok": 8,
    "norm_topk_prob": False,
}


def gpus():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader"], text=True)
        return [l.strip() for l in out.strip().splitlines()]
    except Exception as e:  # pragma: no cover
        return [f"unavailable: {e}"]


def main():
    os.makedirs(ART, exist_ok=True)

    env = {
        "experiment": "DREV-P0",
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
        "gpus_visible": gpus(),
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
    print("\nGPUs:")
    for g in env["gpus_visible"]:
        print("  ", g)

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

    prov = {
        "experiment": "DREV-P0",
        "model_id": MODEL_ID,
        "model_revision_requested": MODEL_REVISION,
        "model_revision_resolved": minfo.sha,
        "instruct_checkpoint": False,
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision_requested": DATASET_REVISION,
        "dataset_revision_resolved": dinfo.sha,
        "config_verification": verified,
        "config_mismatches": mismatches,
        "eos_token_id": cfg.eos_token_id,
        "tokenizer_class": tok.__class__.__name__,
        "runtime_dtype": "float16",
        "quantized": False,
    }
    with open(os.path.join(ART, "model_provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)

    print("\nmodel revision resolved :", minfo.sha)
    print("dataset revision resolved:", dinfo.sha)
    for k, v in verified.items():
        print(f"  {'OK ' if v['ok'] else 'BAD'} {k}: {v['actual']}")

    if mismatches or minfo.sha != MODEL_REVISION:
        print("\nSTOP: provenance mismatch", mismatches)
        return 1
    print("\nAudit passed. No research metric computed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
