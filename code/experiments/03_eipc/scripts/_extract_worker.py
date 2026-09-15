"""Shared extraction logic for EIPC-P0.

One worker handles one split. It records, for Layers 1..11 at the experimental token:
native Top-8 IDs, native Top-8 probabilities, and the selected-expert weighted
contributions after the fixed 32-d projection. It also records native router logits
for target Layers 8 and 12.

Full 2048-d expert outputs are projected immediately and never retained. No worker
computes any R2, A, or B.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eipc import ALL_HISTORY_CODE, MODEL_REVISION, PROJ_DIM, TARGETS, TOP_K
from eipc.data import rebuild_from_manifest
from eipc.expert_states import (
    build_projection,
    extract_layer_states,
    projection_hash,
)
from eipc.olmoe import (
    assert_frozen,
    check_context_len,
    fused_from_contributions,
    load_model,
    native_forward,
    selected_expert_contributions,
    verify_config,
)

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "EIPC_P0")
DATASET_DIR = os.environ.get("EIPC_DATASET_DIR")
MODEL_DIR = os.environ.get("EIPC_MODEL_DIR")
MICROBATCH = int(os.environ.get("EIPC_MICROBATCH", "8"))

TARGET_CODES = sorted({v["code"] for v in TARGETS.values()})


def make_log(role):
    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{role}] {msg}"
        print(line, flush=True)
        with open(os.path.join(ART, "run.log"), "a") as fh:
            fh.write(line + "\n")
    return log


def run_extraction(role: str):
    """Extract one split: role is 'fit' or 'test'."""
    if role not in ("fit", "test"):
        raise ValueError(f"role must be fit or test, got {role!r}")
    log = make_log(role)
    log(f"=== EIPC-P0 {role.upper()} extraction ===")

    if not torch.cuda.is_available():
        raise RuntimeError("TECHNICAL_BLOCKER: CUDA unavailable")
    gpu = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    log(f"GPU: {gpu} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    log(f"  capability {props.major}.{props.minor}, total {props.total_memory/2**20:.0f} MiB")
    if "V100" not in gpu:
        raise RuntimeError(f"TECHNICAL_BLOCKER: expected Tesla V100, got {gpu}")

    model, tok = load_model(model_dir=MODEL_DIR)
    model.eval()
    assert_frozen(model)
    verify_config(model)
    log(f"model frozen FP16, revision {MODEL_REVISION}")

    P = build_projection()
    log(f"projection sha256 {projection_hash(P)}")
    P_t = torch.from_numpy(P).float().cuda()

    fit_ctx, test_ctx = rebuild_from_manifest(
        os.path.join(ART, "data_manifest.json"), tok, model.config.eos_token_id,
        dataset_dir=DATASET_DIR)
    contexts = fit_ctx if role == "fit" else test_ctx
    log(f"manifest rebuilt and fingerprints verified: {len(contexts)} {role} contexts")

    capture_layers = sorted(set(ALL_HISTORY_CODE) | set(TARGET_CODES))
    log(f"capturing layers (code) {capture_layers}")

    n = len(contexts)
    store_s = {li: [] for li in ALL_HISTORY_CODE}
    store_ids = {li: [] for li in ALL_HISTORY_CODE}
    store_probs = {li: [] for li in ALL_HISTORY_CODE}
    store_logits = {li: [] for li in TARGET_CODES}
    recon_err = []

    mb, done, t0 = MICROBATCH, 0, time.time()
    while done < n:
        chunk = contexts[done : done + mb]
        ids = torch.tensor([c.context for c in chunk]).cuda()
        check_context_len(ids)
        try:
            with torch.inference_mode():
                cap = native_forward(model, ids, capture_layers)

                for li in ALL_HISTORY_CODE:
                    block = model.model.layers[li].mlp
                    st = extract_layer_states(block, cap.x[li], cap.g[li], P_t)
                    store_s[li].append(st["s"])
                    store_ids[li].append(st["topk_ids"])
                    store_probs[li].append(st["topk_probs"])

                # Invariant check on the first microbatch of the first layer:
                # weighted sum of separately executed selected experts must
                # reproduce the stock fused MoE output.
                if done == 0:
                    li = ALL_HISTORY_CODE[0]
                    block = model.model.layers[li].mlp
                    _, _, contrib = selected_expert_contributions(block, cap.x[li], cap.g[li])
                    recon = fused_from_contributions(contrib)
                    err = (recon - cap.y[li].float()).abs().max().item()
                    recon_err.append(err)
                    log(f"  reconstruction invariant: max|sum(c_e) - stock y| = {err:.3e}")

                for li in TARGET_CODES:
                    store_logits[li].append(cap.g[li].float().cpu().numpy())
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if mb == 1:
                raise RuntimeError("TECHNICAL_BLOCKER: OOM at microbatch 1")
            mb = max(1, mb // 2)
            log(f"  OOM -> microbatch {mb} (sample count unchanged)")
            continue

        done += len(chunk)
        if done % 128 == 0 or done == n:
            el = time.time() - t0
            log(f"  {done}/{n}, {el:.0f}s elapsed, {el/done*(n-done):.0f}s remaining, mb={mb}")

    arrays = {}
    for li in ALL_HISTORY_CODE:
        arrays[f"L{li+1}_s"] = np.concatenate(store_s[li], axis=0).astype(np.float32)
        arrays[f"L{li+1}_ids"] = np.concatenate(store_ids[li], axis=0).astype(np.int16)
        arrays[f"L{li+1}_probs"] = np.concatenate(store_probs[li], axis=0).astype(np.float32)
    for li in TARGET_CODES:
        arrays[f"L{li+1}_logits"] = np.concatenate(store_logits[li], axis=0).astype(np.float32)

    path = os.path.join(ART, f"{role}_raw.npz")
    np.savez_compressed(path, **arrays)

    example = arrays[f"L{ALL_HISTORY_CODE[0]+1}_s"]
    log(f"{role}_raw.npz written: s shape {example.shape} (expect (n, {TOP_K}, {PROJ_DIM}))")
    log(f"  history layers stored: {[li+1 for li in ALL_HISTORY_CODE]}")
    log(f"  target logits stored: {[li+1 for li in TARGET_CODES]}")
    log(f"{role} extraction complete; no R2, A, or B computed")
    return 0
