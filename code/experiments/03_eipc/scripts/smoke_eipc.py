"""EIPC-P0 smoke test: plumbing on a handful of real contexts.

Verifies capture, selected-expert re-execution, the reconstruction invariant, the
fixed projection, and its linearity. Per the preregistration this must NOT compute
formal R2, A, B, or any verdict statistic: no compression is fit and no probe is
fitted.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eipc import ALL_HISTORY_CODE, NUM_EXPERTS, PROJ_DIM, TARGETS, TOP_K
from eipc.data import build_token_stream, load_train_texts, make_blocks
from eipc.expert_states import (
    build_projection,
    fused_projected,
    identity_slots,
    project_torch,
    projection_hash,
)
from eipc.olmoe import (
    assert_frozen,
    check_context_len,
    fused_from_contributions,
    load_model,
    native_forward,
    native_topk,
    selected_expert_contributions,
    verify_config,
)

N_SMOKE = 4
DATASET_DIR = os.environ.get("EIPC_DATASET_DIR")
MODEL_DIR = os.environ.get("EIPC_MODEL_DIR")


def main():
    print("loading frozen model (fp16)")
    model, tok = load_model(model_dir=MODEL_DIR)
    model.eval()
    assert_frozen(model)
    verify_config(model)
    print("  frozen OK, eval OK, config OK, dtype", next(model.parameters()).dtype)
    print("  GPU", torch.cuda.get_device_name(0))

    texts = load_train_texts(dataset_dir=DATASET_DIR, max_records=2000)
    stream = build_token_stream(texts, tok, model.config.eos_token_id)
    blocks = make_blocks(stream)[:N_SMOKE]
    ids = torch.tensor([b.tokens[:128] for b in blocks]).cuda()
    check_context_len(ids)
    print(f"  {len(blocks)} smoke contexts from the train split")

    P = build_projection()
    print(f"  projection sha256 {projection_hash(P)} shape {P.shape}")
    P_t = torch.from_numpy(P).float().cuda()

    target_codes = sorted({v["code"] for v in TARGETS.values()})
    with torch.inference_mode():
        cap = native_forward(model, ids, sorted(set(ALL_HISTORY_CODE) | set(target_codes)))

    print("\nselected-expert states at history layers:")
    for li in ALL_HISTORY_CODE[:3]:
        block = model.model.layers[li].mlp
        with torch.inference_mode():
            identities, probs, contrib = selected_expert_contributions(block, cap.x[li], cap.g[li])
            recon = fused_from_contributions(contrib)
            err = (recon - cap.y[li].float()).abs().max().item()
            s = project_torch(P_t, contrib).cpu().numpy()

        assert identities.shape == (len(blocks), TOP_K)
        assert s.shape == (len(blocks), TOP_K, PROJ_DIM)
        print(f"  Layer {li+1}: ids{tuple(identities.shape)} probs{tuple(probs.shape)} "
              f"s{s.shape}, reconstruction max err {err:.3e}")
        print(f"    example ids {identities[0].tolist()}")
        print(f"    example probs {[round(v,5) for v in probs[0].tolist()]} sum {float(probs[0].sum()):.5f}")

        # Projection linearity, on real captured contributions.
        lhs = project_torch(P_t, recon).cpu().numpy()
        rhs = fused_projected(s)
        print(f"    linearity max|P(sum c) - sum P(c)| = {np.abs(lhs - rhs).max():.3e}")

        slots = identity_slots(s, identities.cpu().numpy())
        assert slots.shape == (len(blocks), NUM_EXPERTS, PROJ_DIM)
        nonzero = (np.abs(slots[0]).sum(axis=1) > 0).sum()
        print(f"    identity slots {slots.shape}, non-zero slots for sample 0: {nonzero} "
              f"(expect <= {TOP_K})")

    print("\nfuture routing targets (raw capture only):")
    for m in sorted(TARGETS):
        g = cap.g[TARGETS[m]["code"]]
        assert g.shape == (len(blocks), NUM_EXPERTS)
        top_ids, _ = native_topk(g)
        print(f"  Layer {m}: logits{tuple(g.shape)} example top8 {sorted(top_ids[0].tolist())}")

    print("\nSmoke OK. No compression fit, no probe fitted, no R2/A/B computed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
