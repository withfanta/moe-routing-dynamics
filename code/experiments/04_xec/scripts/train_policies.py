"""Train the nine XEC-P0 policy checkpoints (3 variants x 3 seeds).

Runs on CPU (or one GPU if free) after frozen feature extraction. No OLMoE is loaded here,
so no OLMoE parameter can receive a gradient. TRAIN and VALIDATION oracle labels supervise;
the TEST oracle is never read. On completion this writes a marker file that gates TEST
oracle access.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from xec import POLICY_SEEDS, VARIANTS
from xec.cache_features import current_context, expert_cache_items, fused_cache_items
from xec.policy import build_matched_variants, count_trainable
from xec.training import predict_actions, train_policy

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "XEC_P0")
CKPT_DIR = os.path.join(ART, "checkpoints")
MARKER = os.path.join(ART, "training_complete.json")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [train] {msg}"
    print(line, flush=True)
    with open(os.path.join(ART, "run.log"), "a") as fh:
        fh.write(line + "\n")


def load_role(role):
    f = np.load(os.path.join(ART, f"{role}_features.npz"))
    u = current_context(f["x12"], f["g12"])
    expert_items = expert_cache_items(f["s"], f["hist_ids"])
    fused_items = fused_cache_items(f["s"])
    return {
        "u": torch.from_numpy(u.astype(np.float32)),
        "EXPERT_CACHE": torch.from_numpy(expert_items),
        "FUSED_CACHE": torch.from_numpy(fused_items),
        "CURRENT_ONLY": None,
        "n": u.shape[0],
    }


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    log("=== XEC-P0 policy training (no OLMoE loaded) ===")

    tr = load_role("train")
    va = load_role("validation")
    log(f"loaded TRAIN n={tr['n']}, VALIDATION n={va['n']}")
    log(f"  u {tuple(tr['u'].shape)}, expert items {tuple(tr['EXPERT_CACHE'].shape)}, "
        f"fused items {tuple(tr['FUSED_CACHE'].shape)}")

    y_tr = torch.from_numpy(np.load(os.path.join(ART, "train_oracle.npz"))["oracle_action"])
    y_va = torch.from_numpy(np.load(os.path.join(ART, "validation_oracle.npz"))["oracle_action"])
    log(f"  oracle labels: train {tuple(y_tr.shape)}, validation {tuple(y_va.shape)}")

    summary = {}
    for seed in POLICY_SEEDS:
        variants, init_state = build_matched_variants(seed)
        counts = {v: count_trainable(m) for v, m in variants.items()}
        log(f"seed {seed}: identical initial state_dict, trainable params {counts}")
        assert len(set(counts.values())) == 1

        for variant in VARIANTS:
            model = variants[variant]
            log(f"  training {variant} (seed {seed})")
            tl = train_policy(
                model, tr["u"], tr[variant], y_tr, va["u"], va[variant], y_va,
                seed=seed, log_fn=log)
            path = os.path.join(CKPT_DIR, f"{variant}_seed{seed}.pt")
            torch.save({"variant": variant, "seed": seed,
                        "state_dict": model.state_dict(),
                        "best_epoch": tl.best_epoch,
                        "best_val_loss": tl.best_val_loss}, path)
            val_pred = predict_actions(model, va["u"], va[variant])
            val_acc = float((val_pred == y_va.numpy()).mean())
            summary[f"{variant}_seed{seed}"] = {
                "variant": variant, "seed": seed,
                "best_epoch": tl.best_epoch,
                "best_val_ce": tl.best_val_loss,
                "final_train_ce": tl.train_loss[-1] if tl.train_loss else None,
                "validation_action_accuracy": val_acc,
                "trainable_params": tl.trainable_params,
            }
            log(f"    best epoch {tl.best_epoch}, val CE {tl.best_val_loss:.5f}, "
                f"val action acc {val_acc:.4f}")

    with open(os.path.join(ART, "training_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    with open(MARKER, "w") as fh:
        json.dump({
            "training_complete": True,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoints": sorted(os.listdir(CKPT_DIR)),
            "note": "TEST oracle access is gated on this marker; model selection used "
                    "VALIDATION cross entropy only.",
        }, fh, indent=2)
    log(f"training complete; marker written ({len(summary)} checkpoints)")
    log("TEST oracle may now be computed for descriptive reporting only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
