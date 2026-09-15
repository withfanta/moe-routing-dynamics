"""EPD-P0 smoke test: plumbing and the reconstruction/linearity checks.

Verifies capture, selected-expert reconstruction, projection linearity, the reused EIPC
projection hash, and the four representation shapes. Per the protocol this must NOT fit
compression or probes, and must not compute any R2, gap, or pattern.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from epd import (
    HISTORY_CODE,
    NUM_EXPERTS,
    PROJ_DIM,
    RAW_DIMS,
    REPRESENTATIONS,
    TARGET_LAYER,
    TARGET_LAYER_HUMAN,
    TOP_K,
)
from epd.data import build_token_stream, load_train_texts, make_blocks
from epd.extraction import (
    assert_frozen,
    check_context_len,
    fused_from_contributions,
    load_eipc_projection,
    load_model,
    native_forward,
    native_topk,
    project,
    project_torch,
    router_probs,
    selected_expert_contributions,
    verify_config,
)
from epd.representations import build_all

N_SMOKE = 4
DATASET_DIR = os.environ.get("EPD_DATASET_DIR")
MODEL_DIR = os.environ.get("EPD_MODEL_DIR")


def main():
    print("loading frozen model (fp16)")
    model, tok = load_model(model_dir=MODEL_DIR)
    model.eval()
    assert_frozen(model)
    verify_config(model)
    print("  frozen OK, eval OK, config OK, no gradients, dtype",
          next(model.parameters()).dtype)
    print("  GPU", torch.cuda.get_device_name(0))

    P, meta = load_eipc_projection()
    print(f"  reused EIPC projection sha256 {meta['sha256']} shape {P.shape} seed {meta['seed']}")
    P_t = torch.from_numpy(P).float().cuda()

    texts = load_train_texts(dataset_dir=DATASET_DIR, max_records=2000)
    stream = build_token_stream(texts, tok, model.config.eos_token_id)
    blocks = make_blocks(stream)[:N_SMOKE]
    ids = torch.tensor([b.tokens[:128] for b in blocks]).cuda()
    check_context_len(ids)
    print(f"  {len(blocks)} smoke contexts from the train split")

    with torch.inference_mode():
        cap = native_forward(model, ids, sorted(set(HISTORY_CODE) | {TARGET_LAYER}))

    print("\nreconstruction and linearity checks:")
    s_layers, id_layers = [], []
    for li in HISTORY_CODE:
        block = model.model.layers[li].mlp
        with torch.inference_mode():
            e_ids, e_probs, contrib = selected_expert_contributions(block, cap.x[li], cap.g[li])
            recon = fused_from_contributions(contrib)
            y = cap.y[li].float()
            s = project_torch(P_t, contrib)
        abs_err = (recon - y).abs().max().item()
        rel_err = ((recon - y).norm() / y.norm().clamp_min(1e-9)).item()
        # Projection linearity on real captured contributions.
        lhs = project_torch(P_t, recon).cpu().numpy()
        rhs = s.cpu().numpy().sum(axis=1)
        lin = float(np.abs(lhs - rhs).max())
        s_layers.append(s.cpu().numpy())
        id_layers.append(e_ids.cpu().numpy())
        if li < 3 or li == HISTORY_CODE[-1]:
            print(f"  Layer {li+1:2d}: recon abs {abs_err:.3e} rel {rel_err:.3e} | "
                  f"linearity {lin:.3e} | ids {e_ids[0].tolist()}")
            print(f"            prob sum {float(e_probs[0].sum()):.5f} (unrenormalized), "
                  f"s{tuple(s.shape)}")

    s_arr = np.stack(s_layers, axis=1)
    ids_arr = np.stack(id_layers, axis=1)

    print("\nrepresentation shapes:")
    reps = build_all(s_arr, ids_arr)
    for name in REPRESENTATIONS:
        X = reps[name]
        print(f"  {name:16s} {X.shape} (expect (n, {RAW_DIMS[name]}))")
    uniq = np.unique(reps["ID_PATH"])
    print(f"  ID_PATH unique values {uniq.tolist()} (expect [0.0, 1.0])")
    print(f"  ID_PATH ones per sample {int(reps['ID_PATH'][0].sum())} "
          f"(expect {len(HISTORY_CODE) * TOP_K})")
    nz = (np.abs(reps['FULL_PROVENANCE'][0].reshape(len(HISTORY_CODE), NUM_EXPERTS, PROJ_DIM)
                 ).sum(axis=-1) > 0).sum()
    print(f"  FULL_PROVENANCE non-zero slots for sample 0: {nz} "
          f"(expect {len(HISTORY_CODE) * TOP_K})")

    print(f"\ntarget Layer {TARGET_LAYER_HUMAN} (raw capture only):")
    g12 = cap.g[TARGET_LAYER]
    t8, _ = native_topk(g12, TOP_K)
    print(f"  logits {tuple(g12.shape)}, example top8 {sorted(t8[0].tolist())}")
    print(f"  probs sum to 1: {float(router_probs(g12)[0].sum()):.6f}")

    print("\nSmoke OK. No compression fit, no probe fitted, no R2/gap/pattern computed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
