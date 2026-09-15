"""XEC-P0 smoke test: plumbing on a handful of real contexts.

Checks capture, selected-expert reconstruction, the five actions, action-0 equivalence,
projection, cache item shapes, and a forward pass through an untrained policy. Per the
preregistration this must NOT compute formal XEC verdict statistics: no oracle label is
saved, no policy is trained, and no paired difference or bootstrap is computed.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from xec import (
    HISTORY_CODE,
    ITEM_DIM,
    MAX_RANK,
    N_ACTIONS,
    N_EXPERT_ITEMS,
    N_FUSED_ITEMS,
    PROJ_DIM,
    TARGET_LAYER,
    TARGET_LAYER_HUMAN,
    TOP_K,
    VARIANTS,
)
from xec.cache_features import (
    build_projection,
    current_context,
    expert_cache_items,
    fused_cache_items,
    project_contributions,
    projection_hash,
)
from xec.data import build_token_stream, load_train_texts, make_blocks
from xec.olmoe import (
    action_identities,
    action_nll,
    assert_frozen,
    check_context_len,
    fused_from_contributions,
    load_model,
    native_forward,
    native_topk,
    selected_expert_contributions,
    verify_config,
)
from xec.policy import RoutingPolicy, count_trainable

N_SMOKE = 4
DATASET_DIR = os.environ.get("XEC_DATASET_DIR")
MODEL_DIR = os.environ.get("XEC_MODEL_DIR")


def main():
    print("loading frozen model (fp16)")
    model, tok = load_model(model_dir=MODEL_DIR)
    model.eval()
    assert_frozen(model)
    verify_config(model)
    print("  frozen OK, eval OK, config OK, no OLMoE grads, dtype",
          next(model.parameters()).dtype)
    print("  GPU", torch.cuda.get_device_name(0))

    texts = load_train_texts(dataset_dir=DATASET_DIR, max_records=2000)
    stream = build_token_stream(texts, tok, model.config.eos_token_id)
    blocks = make_blocks(stream)[:N_SMOKE]
    ids = torch.tensor([b.tokens[:128] for b in blocks]).cuda()
    tgt = torch.tensor([b.tokens[128] for b in blocks]).cuda()
    check_context_len(ids)
    print(f"  {len(blocks)} smoke contexts from the train split")

    P = build_projection()
    print(f"  projection sha256 {projection_hash(P)} shape {P.shape}")
    P_t = torch.from_numpy(P).float().cuda()

    with torch.inference_mode():
        cap = native_forward(model, ids, tgt, sorted(set(HISTORY_CODE) | {TARGET_LAYER}))
    print(f"\nnative next-token NLL: {[round(float(v),4) for v in cap.nll]}")

    print("\nhistorical selected-expert states:")
    s_layers, id_layers = [], []
    for li in HISTORY_CODE:
        block = model.model.layers[li].mlp
        with torch.inference_mode():
            e_ids, e_probs, contrib = selected_expert_contributions(block, cap.x[li], cap.g[li])
            err = (fused_from_contributions(contrib) - cap.y[li].float()).abs().max().item()
            s = project_contributions(P_t, contrib)
        s_layers.append(s.cpu().numpy())
        id_layers.append(e_ids.cpu().numpy())
        if li < 2:
            print(f"  Layer {li+1}: s{tuple(s.shape)} reconstruction err {err:.3e} "
                  f"ids {e_ids[0].tolist()} prob sum {float(e_probs[0].sum()):.5f}")
    print(f"  all {len(HISTORY_CODE)} history layers captured")

    s_arr = np.stack(s_layers, axis=1)
    ids_arr = np.stack(id_layers, axis=1)
    expert_items = expert_cache_items(s_arr, ids_arr)
    fused_items = fused_cache_items(s_arr)
    print(f"\ncache items: expert {expert_items.shape} (expect (n,{N_EXPERT_ITEMS},{ITEM_DIM})), "
          f"fused {fused_items.shape} (expect (n,{N_FUSED_ITEMS},{ITEM_DIM}))")

    u = current_context(cap.x[TARGET_LAYER].float().cpu().numpy(),
                        cap.g[TARGET_LAYER].float().cpu().numpy())
    print(f"  current context u {u.shape}")

    print(f"\nfive actions at Layer {TARGET_LAYER_HUMAN}:")
    t12_ids, t12_probs = native_topk(cap.g[TARGET_LAYER], MAX_RANK)
    for a in range(N_ACTIONS):
        acts = action_identities(t12_ids, a)
        with torch.inference_mode():
            nll = action_nll(model, ids, tgt, t12_ids, a)
        tag = "native" if a == 0 else f"swap rank {8 + a}"
        print(f"  action {a} ({tag}): experts[0] {acts[0].tolist()} "
              f"NLL {[round(float(v),4) for v in nll]}")
        if a == 0:
            err = (nll - cap.nll).abs().max().item()
            print(f"    action 0 vs plain native NLL max err {err:.3e}")

    print("\nuntrained policy forward pass (parameter matching only):")
    u_t = torch.from_numpy(u.astype(np.float32))
    for variant in VARIANTS:
        m = RoutingPolicy(variant)
        it = None
        if variant == "EXPERT_CACHE":
            it = torch.from_numpy(expert_items)
        elif variant == "FUSED_CACHE":
            it = torch.from_numpy(fused_items)
        with torch.no_grad():
            out = m(u_t, it)
        print(f"  {variant:14s} logits {tuple(out.shape)} trainable params {count_trainable(m)}")

    print("\nSmoke OK. No oracle saved, no policy trained, no paired difference or bootstrap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
