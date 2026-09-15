"""Shared horizon-worker logic for DREV-P0.

A worker computes routing regret G_m at its assigned target layers, fits the two
Ridge probes per horizon on the cached Layer-4 source features, and writes one
independent horizon_<delta>.json each. No worker writes results.json.

Both workers run identical code, manifest, source features, model revision, and
routing semantics; they differ only in which target layers they own.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from drev import HORIZONS, MODEL_REVISION, SOURCE_LAYER
from drev.counterfactual import regret_descriptives, routing_regret
from drev.data import rebuild_from_manifest
from drev.probes import run_probe_pair
from drev.projection import (
    FrozenProjection,
    apply_shuffle,
    baseline_matrix,
    build_shuffle_permutations,
)
from drev.source_features import (
    assert_frozen,
    check_context_len,
    load_model,
    native_forward,
    probs_from_logits,
    verify_config,
)

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "DREV_P0")
DATASET_DIR = os.environ.get("DREV_DATASET_DIR")
MODEL_DIR = os.environ.get("DREV_MODEL_DIR")
MICROBATCH = int(os.environ.get("DREV_MICROBATCH", "8"))


def make_log(worker):
    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{worker}] {msg}"
        print(line, flush=True)
        with open(os.path.join(ART, "run.log"), "a") as fh:
            fh.write(line + "\n")
    return log


def extract_regret(model, contexts, target_layer, log, tag):
    """G_m for every context: native NLL vs the best of four equal-compute alts."""
    out = {"G": [], "native_nll": []}
    n, mb, done, t0 = len(contexts), MICROBATCH, 0, time.time()

    while done < n:
        chunk = contexts[done : done + mb]
        ids = torch.tensor([c.context for c in chunk]).cuda()
        tgt = torch.tensor([c.target for c in chunk]).cuda()
        check_context_len(ids)
        try:
            with torch.inference_mode():
                cap = native_forward(model, ids, tgt, [target_layer])
                probs = probs_from_logits(cap.g[target_layer])
                res = routing_regret(model, ids, tgt, target_layer, cap.token_nll, probs)
                out["G"].append(res.regret)
                out["native_nll"].append(res.native_nll)
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
            log(f"    {tag}: {done}/{n}, {el:.0f}s elapsed, mb={mb}")

    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def run_worker(worker_name, distances):
    """Run the horizons owned by this worker."""
    log = make_log(worker_name)
    log(f"=== worker {worker_name}: distances {list(distances)} ===")

    if not torch.cuda.is_available():
        raise RuntimeError("TECHNICAL_BLOCKER: CUDA unavailable")
    gpu = torch.cuda.get_device_name(0)
    log(f"GPU: {gpu} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    if "V100" not in gpu:
        raise RuntimeError(f"TECHNICAL_BLOCKER: expected Tesla V100, got {gpu}")

    model, tok = load_model(model_dir=MODEL_DIR)
    model.eval()
    assert_frozen(model)
    verify_config(model)
    log(f"model frozen FP16, revision {MODEL_REVISION}")

    manifest = os.path.join(ART, "data_manifest.json")
    fit_ctx, test_ctx = rebuild_from_manifest(manifest, tok, model.config.eos_token_id,
                                              dataset_dir=DATASET_DIR)

    # Cached Layer-4 source features, identical for every horizon.
    src_fit = np.load(os.path.join(ART, "source_fit.npz"))
    src_test = np.load(os.path.join(ART, "source_test.npz"))
    log(f"source features loaded: fit r{src_fit['r'].shape}, test r{src_test['r'].shape}")

    # Frozen compression, fit on the fit split only.
    b_fit = baseline_matrix(src_fit["h"], src_fit["g"])
    b_test = baseline_matrix(src_test["h"], src_test["g"])
    proj_b = FrozenProjection.fit(b_fit)
    proj_r = FrozenProjection.fit(src_fit["r"])
    z_b_fit, z_b_test = proj_b.transform(b_fit), proj_b.transform(b_test)
    z_r_fit, z_r_test = proj_r.transform(src_fit["r"]), proj_r.transform(src_test["r"])
    log(f"PCA fit on fit split only: b evr {proj_b.explained_variance_ratio_sum:.4f}, "
        f"r evr {proj_r.explained_variance_ratio_sum:.4f}")

    # The one fixed control, shared by all horizons.
    perm_fit, perm_test = build_shuffle_permutations(len(z_r_fit), len(z_r_test))
    z_r_fit_shuf = apply_shuffle(z_r_fit, perm_fit)
    z_r_test_shuf = apply_shuffle(z_r_test, perm_test)
    log("shuffle control built (seed 271828), zero fixed points, shared across horizons")

    for delta in distances:
        target_human, target_code = HORIZONS[delta]
        log(f"  horizon delta={delta}: target Layer {target_human} (code {target_code})")

        reg_fit = extract_regret(model, fit_ctx, target_code, log, f"d{delta} fit")
        reg_test = extract_regret(model, test_ctx, target_code, log, f"d{delta} test")

        probe = run_probe_pair(
            z_b_fit, z_r_fit, z_r_fit_shuf, reg_fit["G"],
            z_b_test, z_r_test, z_r_test_shuf, reg_test["G"],
        )
        desc = regret_descriptives(reg_test["G"])

        payload = {
            "experiment": "DREV-P0",
            "worker": worker_name,
            "gpu": gpu,
            "distance": delta,
            "target_layer": target_human,
            "target_layer_code": target_code,
            "source_layer": 4,
            "source_layer_code": SOURCE_LAYER,
            **probe,
            **desc,
            "mean_G_fit": float(np.mean(reg_fit["G"])),
            "model_revision": MODEL_REVISION,
        }
        path = os.path.join(ART, f"horizon_{delta}.json")
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        log(f"  horizon_{delta}.json written (target Layer {target_human})")

    log(f"worker {worker_name} complete")
    return 0
