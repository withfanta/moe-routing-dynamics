"""Extract Layer-4 source features ONCE, for reuse by every horizon worker.

Produces b_i = [h_i ; g_i] and r_i for the fit and test splits. No target, no
regret, no probe, no research statistic here.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from drev import SOURCE_LAYER, SOURCE_LAYER_HUMAN
from drev.data import rebuild_from_manifest
from drev.source_features import (
    assert_frozen,
    check_context_len,
    load_model,
    native_forward,
    rejected_evidence,
    verify_config,
)

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "DREV_P0")
DATASET_DIR = os.environ.get("DREV_DATASET_DIR")
MODEL_DIR = os.environ.get("DREV_MODEL_DIR")
MICROBATCH = int(os.environ.get("DREV_MICROBATCH", "8"))


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [source] {msg}"
    print(line, flush=True)
    with open(os.path.join(ART, "run.log"), "a") as fh:
        fh.write(line + "\n")


def extract(model, contexts, tag):
    """Layer-4 h, g, r plus the block target token, in microbatches."""
    store = {"h": [], "g": [], "r": [], "target": []}
    n, mb, done, t0 = len(contexts), MICROBATCH, 0, time.time()

    while done < n:
        chunk = contexts[done : done + mb]
        ids = torch.tensor([c.context for c in chunk]).cuda()
        tgt = torch.tensor([c.target for c in chunk]).cuda()
        check_context_len(ids)
        try:
            with torch.inference_mode():
                cap = native_forward(model, ids, tgt, [SOURCE_LAYER])
                block = model.model.layers[SOURCE_LAYER].mlp
                r = rejected_evidence(block, cap.x[SOURCE_LAYER])
                store["h"].append(cap.h[SOURCE_LAYER].float().cpu().numpy())
                store["g"].append(cap.g[SOURCE_LAYER].float().cpu().numpy())
                store["r"].append(r.float().cpu().numpy())
                store["target"].append(tgt.cpu().numpy())
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
            log(f"  {tag}: {done}/{n}, {el:.0f}s elapsed, mb={mb}")

    return {k: np.concatenate(v, axis=0) for k, v in store.items()}


def main():
    os.makedirs(ART, exist_ok=True)
    log(f"=== Layer-{SOURCE_LAYER_HUMAN} source extraction (code index {SOURCE_LAYER}) ===")
    log(f"GPU {torch.cuda.get_device_name(0)}")

    model, tok = load_model(model_dir=MODEL_DIR)
    model.eval()
    assert_frozen(model)
    verify_config(model)
    log("model frozen, config verified")

    manifest = os.path.join(ART, "data_manifest.json")
    fit, test = rebuild_from_manifest(manifest, tok, model.config.eos_token_id,
                                      dataset_dir=DATASET_DIR)
    log(f"manifest rebuilt and fingerprints verified: {len(fit)} fit, {len(test)} test")

    for tag, contexts in (("fit", fit), ("test", test)):
        d = extract(model, contexts, tag)
        path = os.path.join(ART, f"source_{tag}.npz")
        np.savez_compressed(path, **d)
        log(f"source_{tag}.npz written: h{d['h'].shape} g{d['g'].shape} r{d['r'].shape}")

    log("source extraction complete; no research statistic computed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
