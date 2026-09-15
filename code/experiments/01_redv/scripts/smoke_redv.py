"""REDV-V1 smoke test: end-to-end plumbing on a handful of real contexts.

Verifies that the real data path, capture, rejected evidence, and counterfactual
runner all execute and produce well-formed shapes. Per the preregistration, this
must NOT compute or inspect Delta_R2, research correlations, success/failure, or
subgroup performance: no probe is fitted here.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from redv import CONTEXT_LEN, TRANSITIONS
from redv.counterfactual import routing_regret
from redv.data import build_token_stream, load_split_texts, make_blocks
from redv.olmoe_hooks import assert_frozen, check_context_len, load_model, native_forward
from redv.rejected import probs_from_logits, rejected_evidence, rejected_report

N_SMOKE = 4
DATASET_DIR = os.environ.get("REDV_DATASET_DIR")
MODEL_DIR = os.environ.get("REDV_MODEL_DIR")


def main():
    print("loading frozen model (fp16)")
    model, tok = load_model(model_dir=MODEL_DIR)
    assert_frozen(model)
    print("  frozen OK, eval OK, dtype", next(model.parameters()).dtype)

    print("building a few real validation contexts")
    texts = load_split_texts("validation", dataset_dir=DATASET_DIR)
    stream = build_token_stream(texts, tok, model.config.eos_token_id)
    blocks = make_blocks(stream)[:N_SMOKE]
    print(f"  {len(texts)} records -> {len(stream)} tokens -> using {len(blocks)} blocks")

    ids = torch.tensor([b.context for b in blocks]).cuda()
    targets = torch.tensor([b.target for b in blocks]).cuda()
    check_context_len(ids)

    capture_layers = sorted({l for pair in TRANSITIONS for l in pair})
    cap = native_forward(model, ids, targets, capture_layers)
    print("  captured layers", capture_layers)
    for src, tgt in TRANSITIONS:
        print(f"  transition {src+1}->{tgt+1} (0-based {src}->{tgt}): "
              f"x{tuple(cap.x[src].shape)} g{tuple(cap.g[src].shape)} h{tuple(cap.h[src].shape)}")

    print("native next-token NLL (plumbing only):", [round(float(v), 4) for v in cap.token_nll])

    for src, tgt in TRANSITIONS:
        block = model.model.layers[src].mlp
        r = rejected_evidence(block, cap.x[src])
        assert r.shape == (len(blocks), 2048)
        probs_src = probs_from_logits(cap.g[src])
        rep = rejected_report(probs_src)
        print(f"\n  layer {src+1}: r shape {tuple(r.shape)}, norm "
              f"{float(r.norm(dim=1).mean()):.4f}")
        print(f"    rejected ranks {rep['ranks']} renormalized={rep['renormalized']}")
        print(f"    example identities {rep['example_identities']}")
        print(f"    example probs {[round(p,5) for p in rep['example_probs']]}")

        probs_tgt = probs_from_logits(cap.g[tgt])
        res = routing_regret(model, ids, targets, tgt, cap.token_nll, probs_tgt)
        assert res.regret.shape == (len(blocks),)
        assert np.all(res.regret >= 0.0)
        print(f"    regret at layer {tgt+1} well-formed: shape {res.regret.shape}, "
              f"all >= 0 OK, alt matrix {res.alt_nll.shape}")

    print("\nSmoke OK. No probe fitted, no Delta_R2 computed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
