"""DREV-P0 smoke test: plumbing on a handful of real contexts.

Verifies the data path, Layer-4 capture, rejected evidence, and target-layer
counterfactual all execute with well-formed shapes. Per the preregistration this
must NOT produce any verdict statistic: no PCA is fit, no probe is fitted, and no
R2 or D is computed.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from drev import HORIZONS, REJECTED_RANKS, SOURCE_LAYER, SOURCE_LAYER_HUMAN
from drev.counterfactual import routing_regret
from drev.data import build_token_stream, load_split_texts, make_blocks
from drev.source_features import (
    assert_frozen,
    check_context_len,
    load_model,
    native_forward,
    probs_from_logits,
    rejected_evidence,
    rejected_identities,
    verify_config,
)

N_SMOKE = 4
DATASET_DIR = os.environ.get("DREV_DATASET_DIR")
MODEL_DIR = os.environ.get("DREV_MODEL_DIR")


def main():
    print("loading frozen model (fp16)")
    model, tok = load_model(model_dir=MODEL_DIR)
    model.eval()
    assert_frozen(model)
    verify_config(model)
    print("  frozen OK, eval OK, config OK, dtype", next(model.parameters()).dtype)
    print("  GPU", torch.cuda.get_device_name(0))

    texts = load_split_texts("validation", dataset_dir=DATASET_DIR)
    stream = build_token_stream(texts, tok, model.config.eos_token_id)
    blocks = make_blocks(stream)[:N_SMOKE]
    ids = torch.tensor([b.tokens[:128] for b in blocks]).cuda()
    targets = torch.tensor([b.tokens[128] for b in blocks]).cuda()
    check_context_len(ids)
    print(f"  {len(blocks)} smoke contexts built")

    all_layers = [SOURCE_LAYER] + [c for _, c in HORIZONS.values()]
    with torch.inference_mode():
        cap = native_forward(model, ids, targets, sorted(set(all_layers)))

    print(f"\nsource Layer {SOURCE_LAYER_HUMAN} (code {SOURCE_LAYER}):")
    print(f"  x{tuple(cap.x[SOURCE_LAYER].shape)} g{tuple(cap.g[SOURCE_LAYER].shape)} "
          f"h{tuple(cap.h[SOURCE_LAYER].shape)}")
    block = model.model.layers[SOURCE_LAYER].mlp
    with torch.inference_mode():
        r = rejected_evidence(block, cap.x[SOURCE_LAYER])
    probs_src = probs_from_logits(cap.g[SOURCE_LAYER])
    rej_ids, rej_p = rejected_identities(probs_src)
    print(f"  r shape {tuple(r.shape)}, mean norm {float(r.norm(dim=1).mean()):.4f}")
    print(f"  rejected ranks {list(REJECTED_RANKS)} renormalized=False")
    print(f"  example identities {rej_ids[0].tolist()}")
    print(f"  example probs {[round(v,5) for v in rej_p[0].tolist()]} sum {float(rej_p[0].sum()):.5f}")

    print("\nnative next-token NLL (plumbing only):",
          [round(float(v), 4) for v in cap.token_nll])

    for delta in sorted(HORIZONS):
        target_human, target_code = HORIZONS[delta]
        probs_t = probs_from_logits(cap.g[target_code])
        with torch.inference_mode():
            res = routing_regret(model, ids, targets, target_code, cap.token_nll, probs_t)
        assert res.regret.shape == (len(blocks),)
        assert np.all(res.regret >= 0.0)
        print(f"  delta={delta} target Layer {target_human}: regret well-formed "
              f"shape {res.regret.shape}, all >= 0 OK, alt matrix {res.alt_nll.shape}")

    print("\nSmoke OK. No PCA fit, no probe fitted, no R2 or D computed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
