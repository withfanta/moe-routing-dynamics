"""EPD-P0 shard extraction worker. Usage: extract_shard.py <shard_index>

Pin with CUDA_VISIBLE_DEVICES to one idle Tesla V100 before launching. Each worker loads its
own frozen OLMoE and extracts only its assigned 192 samples.

Captured per sample: for Layers 1-11, the native Top-8 expert IDs, their original router
probabilities, and their projected weighted contributions in native rank order; plus the
Layer-12 router logits and Top-8 IDs (target side only) and the next token.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from epd import (
    HISTORY_CODE,
    MODEL_REVISION,
    N_SHARDS,
    PROJ_DIM,
    TARGET_LAYER,
    TOP_K,
)
from epd.data import rebuild_from_manifest, shard_contexts
from epd.extraction import (
    assert_frozen,
    check_context_len,
    fused_from_contributions,
    load_eipc_projection,
    load_model,
    native_forward,
    native_topk,
    project_torch,
    selected_expert_contributions,
    verify_config,
)

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "EPD_P0")
DATASET_DIR = os.environ.get("EPD_DATASET_DIR")
MODEL_DIR = os.environ.get("EPD_MODEL_DIR")
MICROBATCH = int(os.environ.get("EPD_MICROBATCH", "8"))


def make_log(shard):
    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [shard{shard}] {msg}"
        print(line, flush=True)
        with open(os.path.join(ART, "run.log"), "a") as fh:
            fh.write(line + "\n")
    return log


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if len(sys.argv) != 2:
        print("usage: extract_shard.py <shard_index>")
        return 2
    shard = int(sys.argv[1])
    if not 0 <= shard < N_SHARDS:
        print(f"shard index must be 0..{N_SHARDS - 1}")
        return 2

    log = make_log(shard)
    log(f"=== EPD-P0 shard {shard} extraction ===")

    if not torch.cuda.is_available():
        raise RuntimeError("TECHNICAL_BLOCKER: CUDA unavailable")
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    log(f"GPU: {name} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    log(f"  capability {props.major}.{props.minor}, total {props.total_memory/2**20:.0f} MiB")
    if "V100" not in name:
        raise RuntimeError(f"TECHNICAL_BLOCKER: expected Tesla V100, got {name}")

    P, proj_meta = load_eipc_projection()
    log(f"reused EIPC projection verified, sha256 {proj_meta['sha256']}")

    model, tok = load_model(model_dir=MODEL_DIR)
    model.eval()
    assert_frozen(model)
    verify_config(model)
    log(f"model frozen FP16, revision {MODEL_REVISION}, no gradients")

    P_t = torch.from_numpy(P).float().cuda()

    all_contexts = rebuild_from_manifest(
        os.path.join(ART, "data_manifest.json"), tok, model.config.eos_token_id,
        dataset_dir=DATASET_DIR)
    mine = shard_contexts(all_contexts, shard)
    log(f"manifest rebuilt and fingerprints verified; shard {shard} has {len(mine)} samples "
        f"({sum(c.role == 'fit' for c in mine)} fit, {sum(c.role == 'test' for c in mine)} test)")

    capture_layers = sorted(set(HISTORY_CODE) | {TARGET_LAYER})
    store = {"s": [], "hist_ids": [], "hist_probs": [], "g12": [], "top8_12": [],
             "next_token": []}
    recon_max_abs, recon_max_rel = 0.0, 0.0

    n, mb, done, t0 = len(mine), MICROBATCH, 0, time.time()
    while done < n:
        chunk = mine[done : done + mb]
        ids = torch.tensor([c.context for c in chunk]).cuda()
        check_context_len(ids)
        try:
            with torch.inference_mode():
                cap = native_forward(model, ids, capture_layers)

                s_layers, id_layers, p_layers = [], [], []
                for li in HISTORY_CODE:
                    block = model.model.layers[li].mlp
                    e_ids, e_probs, contrib = selected_expert_contributions(
                        block, cap.x[li], cap.g[li])
                    y = cap.y[li].float()
                    diff = (fused_from_contributions(contrib) - y)
                    recon_max_abs = max(recon_max_abs, diff.abs().max().item())
                    recon_max_rel = max(recon_max_rel,
                                        (diff.norm() / y.norm().clamp_min(1e-9)).item())
                    s_layers.append(project_torch(P_t, contrib).cpu().numpy())
                    id_layers.append(e_ids.to(torch.int16).cpu().numpy())
                    p_layers.append(e_probs.float().cpu().numpy())

                store["s"].append(np.stack(s_layers, axis=1))
                store["hist_ids"].append(np.stack(id_layers, axis=1))
                store["hist_probs"].append(np.stack(p_layers, axis=1))

                g12 = cap.g[TARGET_LAYER]
                t8, _ = native_topk(g12, TOP_K)
                store["g12"].append(g12.float().cpu().numpy())
                store["top8_12"].append(t8.to(torch.int16).cpu().numpy())
                store["next_token"].append(
                    np.array([c.next_token for c in chunk], dtype=np.int64))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if mb == 1:
                raise RuntimeError("TECHNICAL_BLOCKER: OOM at microbatch 1")
            mb = max(1, mb // 2)
            log(f"  OOM -> microbatch {mb} (sample count unchanged)")
            continue

        done += len(chunk)
        if done % 64 == 0 or done == n:
            log(f"  {done}/{n}, {time.time()-t0:.0f}s elapsed, mb={mb}")

    arrays = {k: np.concatenate(v, axis=0) for k, v in store.items()}
    log(f"  reconstruction invariant: max abs {recon_max_abs:.3e}, max relative {recon_max_rel:.3e}")

    # Roles and block ids travel with the shard so the merge can restore order.
    roles = np.array([1 if c.role == "test" else 0 for c in mine], dtype=np.int64)
    block_ids = np.array([c.block_id for c in mine], dtype=np.int64)

    path = os.path.join(ART, f"shard_{shard}.npz")
    np.savez_compressed(
        path,
        s=arrays["s"].astype(np.float32),
        hist_ids=arrays["hist_ids"].astype(np.int16),
        hist_probs=arrays["hist_probs"].astype(np.float32),
        g12=arrays["g12"].astype(np.float32),
        top8_12=arrays["top8_12"].astype(np.int16),
        next_token=arrays["next_token"].astype(np.int64),
        is_test=roles, block_id=block_ids,
        recon_max_abs=np.array([recon_max_abs]), recon_max_rel=np.array([recon_max_rel]),
    )
    log(f"shard_{shard}.npz written, sha256 {sha256_file(path)}")
    log(f"  s{arrays['s'].shape} (expect (n, 11, {TOP_K}, {PROJ_DIM})), g12{arrays['g12'].shape}")
    log(f"shard {shard} complete; no representation built, no probe fitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
