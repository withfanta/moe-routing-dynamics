"""Record RMC-P0 provenance and environment before any formal statistic exists.

Writes artifacts/model_provenance.json and artifacts/environment.json. Pins the checkpoint
SHA, the exact JetMoE implementation file by content hash, package versions, and the GPUs
actually available. Loads no model weights.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys

from rmc import (
    ART, DATASET_CONFIG, DATASET_ID, DATASET_REVISION, DATASET_SPLIT, IMPLEMENTATION,
    MICROBATCH, MODEL_ID, MODEL_REVISION, N_SHARDS, NUM_EXPERTS, N_LAYERS, TOP_K,
    sha256_file,
)


def nvidia_smi():
    q = ("index,name,memory.total,memory.free,utilization.gpu,uuid")
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    gpus = []
    for line in out:
        f = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(f[0]), "name": f[1], "memory_total": f[2],
                     "memory_free": f[3], "utilization": f[4], "uuid": f[5]})
    return gpus


def main() -> int:
    os.makedirs(ART, exist_ok=True)
    import torch
    import transformers
    from transformers.models.jetmoe import modeling_jetmoe
    from transformers import AutoConfig

    impl_path = modeling_jetmoe.__file__
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    # Architecture facts are asserted, never assumed.
    facts = {
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "num_local_experts": int(cfg.num_local_experts),
        "num_experts_per_tok": int(cfg.num_experts_per_tok),
        "hidden_size": int(cfg.hidden_size),
        "intermediate_size": int(cfg.intermediate_size),
        "model_type": cfg.model_type,
        "architectures": list(getattr(cfg, "architectures", []) or []),
    }
    assert facts["num_hidden_layers"] == N_LAYERS, facts
    assert facts["num_local_experts"] == NUM_EXPERTS, facts
    assert facts["num_experts_per_tok"] == TOP_K, facts

    prov = {
        "experiment": "RMC-P0",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "revision_is_immutable_sha": True,
        "checkpoint_kind": "BASE (not SFT/chat)",
        "implementation": IMPLEMENTATION,
        "implementation_version": transformers.__version__,
        "implementation_file": impl_path,
        "implementation_file_sha256": sha256_file(impl_path),
        "implementation_rewritten": False,
        "third_party_jetmoe_installed": False,
        "config_facts": facts,
        "router_studied": "MLP MoE router only (model.layers[i].mlp.router)",
        "moa_router_used": False,
        "builtin_output_router_logits_used": False,
        "builtin_note": ("transformers returns attention and MLP router logits interleaved "
                         "in one flat tuple, so direct module hooks are used instead"),
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "dataset_split": DATASET_SPLIT,
        "gradients_on_model": False,
        "quantized": False,
        "dtype": "float16",
    }
    with open(os.path.join(ART, "model_provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)

    gpus = nvidia_smi()
    free = [g for g in gpus
            if int(g["memory_free"].split()[0]) > 30000 and "V100" in g["name"]]
    env = {
        "experiment": "RMC-P0",
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "packages": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "transformers": transformers.__version__,
        },
        "gpus_detected": gpus,
        "free_v100_candidates": [g["index"] for g in free],
        "n_shards_required": N_SHARDS,
        "initial_microbatch": MICROBATCH,
        "ddp": False,
        "model_parallelism": False,
    }
    import numpy, sklearn
    env["packages"]["numpy"] = numpy.__version__
    env["packages"]["scikit-learn"] = sklearn.__version__
    with open(os.path.join(ART, "environment.json"), "w") as fh:
        json.dump(env, fh, indent=2)

    print(f"[provenance] checkpoint {MODEL_REVISION}")
    print(f"[provenance] impl {impl_path}")
    print(f"[provenance] impl sha256 {prov['implementation_file_sha256']}")
    print(f"[provenance] config: {facts['num_hidden_layers']} layers, "
          f"{facts['num_local_experts']} experts, top-{facts['num_experts_per_tok']}")
    print(f"[provenance] free V100 candidates {env['free_v100_candidates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
