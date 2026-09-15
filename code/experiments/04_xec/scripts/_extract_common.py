"""Shared frozen-model extraction for XEC-P0.

Per sample at the experimental token: projected historical selected-expert states for
Layers 1-11 with their layer and expert IDs, the Layer-12 hidden state and native router
logits, native Top-12 identities and original probabilities, and the target token. For
TRAIN and VALIDATION it additionally computes all five action NLLs and the oracle label.

All frozen-model work runs under torch.inference_mode(); nothing here trains.
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

from xec import (
    HISTORY_CODE,
    MAX_RANK,
    MODEL_REVISION,
    N_ACTIONS,
    PROJ_DIM,
    TARGET_LAYER,
    TOP_K,
)
from xec.cache_features import build_projection, project_contributions, projection_hash
from xec.counterfactual import all_action_nlls, native_nll_check
from xec.data import rebuild_from_manifest
from xec.olmoe import (
    assert_frozen,
    check_context_len,
    fused_from_contributions,
    load_model,
    native_forward,
    native_topk,
    selected_expert_contributions,
    verify_config,
)
from xec.oracle import action_distribution, oracle_action

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "XEC_P0")
DATASET_DIR = os.environ.get("XEC_DATASET_DIR")
MODEL_DIR = os.environ.get("XEC_MODEL_DIR")
MICROBATCH = int(os.environ.get("XEC_MICROBATCH", "8"))

# Roles needing oracle labels for supervision. TEST is excluded by protocol.
ORACLE_ROLES = ("train", "validation")


def make_log(worker):
    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{worker}] {msg}"
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


def verify_device(log):
    if not torch.cuda.is_available():
        raise RuntimeError("TECHNICAL_BLOCKER: CUDA unavailable")
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    log(f"GPU: {name} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    log(f"  capability {props.major}.{props.minor}, total {props.total_memory/2**20:.0f} MiB")
    if "V100" not in name:
        raise RuntimeError(f"TECHNICAL_BLOCKER: expected Tesla V100, got {name}")
    return name


def extract_role(model, contexts, P_t, role, log, want_oracle):
    """Extract features (and oracle labels when requested) for one role."""
    capture_layers = sorted(set(HISTORY_CODE) | {TARGET_LAYER})
    n = len(contexts)
    store = {"s": [], "hist_ids": [], "hist_probs": [], "x12": [], "g12": [],
             "top12_ids": [], "top12_probs": [], "target": []}
    action_rows = []
    native_rows = []
    recon_err = 0.0

    mb, done, t0 = MICROBATCH, 0, time.time()
    while done < n:
        chunk = contexts[done : done + mb]
        ids = torch.tensor([c.context for c in chunk]).cuda()
        tgt = torch.tensor([c.target for c in chunk]).cuda()
        check_context_len(ids)
        try:
            with torch.inference_mode():
                cap = native_forward(model, ids, tgt, capture_layers)

                s_layers, id_layers, p_layers = [], [], []
                for li in HISTORY_CODE:
                    block = model.model.layers[li].mlp
                    e_ids, e_probs, contrib = selected_expert_contributions(
                        block, cap.x[li], cap.g[li])
                    # Invariant: the eight weighted contributions rebuild the fused output.
                    err = (fused_from_contributions(contrib) - cap.y[li].float()).abs().max().item()
                    recon_err = max(recon_err, err)
                    s_layers.append(project_contributions(P_t, contrib).cpu().numpy())
                    id_layers.append(e_ids.to(torch.int16).cpu().numpy())
                    p_layers.append(e_probs.float().cpu().numpy())

                store["s"].append(np.stack(s_layers, axis=1))            # (n, 11, 8, 64)
                store["hist_ids"].append(np.stack(id_layers, axis=1))     # (n, 11, 8)
                store["hist_probs"].append(np.stack(p_layers, axis=1))    # (n, 11, 8)

                t12_ids, t12_probs = native_topk(cap.g[TARGET_LAYER], MAX_RANK)
                store["x12"].append(cap.x[TARGET_LAYER].float().cpu().numpy())
                store["g12"].append(cap.g[TARGET_LAYER].float().cpu().numpy())
                store["top12_ids"].append(t12_ids.to(torch.int16).cpu().numpy())
                store["top12_probs"].append(t12_probs.float().cpu().numpy())
                store["target"].append(tgt.cpu().numpy())
                native_rows.append(cap.nll.float().cpu().numpy())

                if want_oracle:
                    action_rows.append(all_action_nlls(model, ids, tgt, t12_ids))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if mb == 1:
                raise RuntimeError("TECHNICAL_BLOCKER: OOM at microbatch 1")
            mb = max(1, mb // 2)
            log(f"  OOM -> microbatch {mb} (sample count unchanged)")
            continue

        done += len(chunk)
        if done % 256 == 0 or done == n:
            el = time.time() - t0
            log(f"  {role}: {done}/{n}, {el:.0f}s elapsed, {el/done*(n-done):.0f}s remaining, mb={mb}")

    arrays = {k: np.concatenate(v, axis=0) for k, v in store.items()}
    arrays["native_nll"] = np.concatenate(native_rows, axis=0)
    log(f"  {role}: reconstruction invariant max err {recon_err:.3e}")

    feat_path = os.path.join(ART, f"{role}_features.npz")
    np.savez_compressed(
        feat_path,
        s=arrays["s"].astype(np.float32),
        hist_ids=arrays["hist_ids"].astype(np.int16),
        hist_probs=arrays["hist_probs"].astype(np.float32),
        x12=arrays["x12"].astype(np.float32),
        g12=arrays["g12"].astype(np.float32),
        top12_ids=arrays["top12_ids"].astype(np.int16),
        top12_probs=arrays["top12_probs"].astype(np.float32),
        target=arrays["target"].astype(np.int64),
        native_nll=arrays["native_nll"].astype(np.float64),
    )
    log(f"  {role}_features.npz written, sha256 {sha256_file(feat_path)}")
    log(f"    s{arrays['s'].shape} x12{arrays['x12'].shape} top12_ids{arrays['top12_ids'].shape}")

    if want_oracle:
        A = np.concatenate(action_rows, axis=0)
        assert A.shape == (n, N_ACTIONS), A.shape
        err = native_nll_check(A, arrays["native_nll"])
        log(f"  {role}: action 0 vs native NLL max err {err:.3e}")
        a_star = oracle_action(A)
        orc_path = os.path.join(ART, f"{role}_oracle.npz")
        np.savez_compressed(orc_path, action_nlls=A.astype(np.float64),
                            oracle_action=a_star.astype(np.int64))
        dist = action_distribution(a_star)
        log(f"  {role}_oracle.npz written, sha256 {sha256_file(orc_path)}")
        log(f"    oracle action distribution {dist['counts']} "
            f"(native {dist['fraction_native_action']:.4f}, swap {dist['fraction_swap_actions']:.4f})")

    return arrays


def run_worker(worker_name, roles):
    log = make_log(worker_name)
    log(f"=== XEC-P0 extraction worker {worker_name}: roles {list(roles)} ===")
    verify_device(log)

    model, tok = load_model(model_dir=MODEL_DIR)
    model.eval()
    assert_frozen(model)
    verify_config(model)
    log(f"model frozen FP16, revision {MODEL_REVISION}, no OLMoE gradients")

    P = build_projection()
    log(f"projection sha256 {projection_hash(P)}")
    P_t = torch.from_numpy(P).float().cuda()

    contexts = rebuild_from_manifest(
        os.path.join(ART, "data_manifest.json"), tok, model.config.eos_token_id,
        dataset_dir=DATASET_DIR)
    log("manifest rebuilt and all fingerprints verified: "
        + ", ".join(f"{r} {len(contexts[r])}" for r in ("train", "validation", "test")))

    for role in roles:
        want_oracle = role in ORACLE_ROLES
        log(f"extracting {role} (oracle labels: {want_oracle})")
        extract_role(model, contexts[role], P_t, role, log, want_oracle)

    log(f"worker {worker_name} complete; no policy trained, no verdict statistic computed")
    return 0
