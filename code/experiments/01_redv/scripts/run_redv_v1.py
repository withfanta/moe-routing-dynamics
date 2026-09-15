"""REDV-V1 formal run.

Extracts, for each of the three frozen layer transitions and each of the 512
validation and 512 test contexts:

    h_{l,t}, g_{l,t}   -> baseline feature b
    r_{l,t}            -> rejected evidence
    G_{l+1,t}          -> counterfactual routing regret

then fits the two Ridge probes per transition and applies the pre-registered
judgement rule. Test results are not inspected until all three transitions and
both probe fits are complete.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from redv import N_TEST, N_VALIDATION, TRANSITIONS
from redv.analysis import summarize
from redv.counterfactual import regret_descriptives, routing_regret
from redv.data import build_frozen_sample, manifest_payload, write_manifest
from redv.olmoe_hooks import assert_frozen, check_context_len, load_model, native_forward
from redv.probes import run_probe_pair
from redv.rejected import probs_from_logits, rejected_evidence

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "REDV_V1")
DATASET_DIR = os.environ.get("REDV_DATASET_DIR")
MODEL_DIR = os.environ.get("REDV_MODEL_DIR")
MICROBATCH = int(os.environ.get("REDV_MICROBATCH", "8"))


class Log:
    def __init__(self, path):
        self.fh = open(path, "a")

    def __call__(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()


def extract_split(model, blocks, log, tag):
    """Extract features and regret for one split, in microbatches.

    Returns ``{(src,tgt): {"h","g","r","G"}}`` plus native NLL bookkeeping.
    """
    capture_layers = sorted({l for pair in TRANSITIONS for l in pair})
    store = {pair: {"h": [], "g": [], "r": [], "G": [], "native_nll": [], "best_alt": []}
             for pair in TRANSITIONS}

    n = len(blocks)
    mb = MICROBATCH
    t0 = time.time()
    done = 0
    while done < n:
        chunk = blocks[done : done + mb]
        ids = torch.tensor([b.context for b in chunk]).cuda()
        targets = torch.tensor([b.target for b in chunk]).cuda()
        check_context_len(ids)

        try:
            cap = native_forward(model, ids, targets, capture_layers)

            for pair in TRANSITIONS:
                src, tgt = pair
                block = model.model.layers[src].mlp
                r = rejected_evidence(block, cap.x[src])
                probs_tgt = probs_from_logits(cap.g[tgt])
                res = routing_regret(model, ids, targets, tgt, cap.token_nll, probs_tgt)

                store[pair]["h"].append(cap.h[src].float().cpu().numpy())
                store[pair]["g"].append(cap.g[src].float().cpu().numpy())
                store[pair]["r"].append(r.float().cpu().numpy())
                store[pair]["G"].append(res.regret)
                store[pair]["native_nll"].append(res.native_nll)
                store[pair]["best_alt"].append(res.best_alt_nll)
        except torch.cuda.OutOfMemoryError:
            # Preregistered fallback: reduce microbatch, then go sequential.
            torch.cuda.empty_cache()
            if mb == 1:
                raise
            mb = max(1, mb // 2)
            log(f"  OOM -> reducing microbatch to {mb} (sample count unchanged)")
            continue

        done += len(chunk)
        if done % 64 == 0 or done == n:
            el = time.time() - t0
            log(f"  {tag}: {done}/{n} contexts, {el:.0f}s elapsed, "
                f"{el/done*(n-done):.0f}s remaining, mb={mb}")

    out = {}
    for pair in TRANSITIONS:
        out[pair] = {k: np.concatenate(v, axis=0) for k, v in store[pair].items()}
    return out


def main():
    os.makedirs(ART, exist_ok=True)
    log = Log(os.path.join(ART, "run.log"))
    log("=== REDV-V1 formal run ===")

    model, tok = load_model(model_dir=MODEL_DIR)
    assert_frozen(model)
    log(f"model loaded frozen, dtype {next(model.parameters()).dtype}")

    log("building frozen data sample (seed 20260914)")
    sample = build_frozen_sample(tok, model.config.eos_token_id, dataset_dir=DATASET_DIR)
    for split in ("validation", "test"):
        s = sample[split]
        log(f"  {split}: {s['n_blocks_available']} blocks available, "
            f"{s['n_sampled']} sampled, fingerprint {s['fingerprint'][:16]}")
    assert sample["validation"]["n_sampled"] == N_VALIDATION
    assert sample["test"]["n_sampled"] == N_TEST

    payload = manifest_payload(sample, extra={"tokenizer_class": tok.__class__.__name__})
    manifest_hash = write_manifest(os.path.join(ART, "data_manifest.json"), payload)
    log(f"data_manifest.json written, sha256 {manifest_hash}")

    features = {}
    for split in ("validation", "test"):
        log(f"extracting {split} ({sample[split]['n_sampled']} contexts)")
        features[split] = extract_split(model, sample[split]["blocks"], log, split)

    for split in ("validation", "test"):
        arrays = {}
        for (src, tgt), d in features[split].items():
            for k, v in d.items():
                arrays[f"L{src+1}_{tgt+1}_{k}"] = v
        np.savez_compressed(os.path.join(ART, f"{split}_features.npz"), **arrays)
        log(f"{split}_features.npz written")

    # Fit all probes for all transitions before inspecting any test result.
    log("fitting probes (all transitions) before inspecting test results")
    per_transition = []
    for src, tgt in TRANSITIONS:
        val = features["validation"][(src, tgt)]
        test = features["test"][(src, tgt)]
        res = run_probe_pair(val, test)
        desc = regret_descriptives(test["G"])
        per_transition.append({
            "transition_human": f"{src+1} -> {tgt+1}",
            "transition_code": [src, tgt],
            "n_validation": int(len(val["G"])),
            "n_test": int(len(test["G"])),
            "r2_baseline": res["r2_baseline"],
            "r2_augmented": res["r2_augmented"],
            "delta_r2": res["delta_r2"],
            "baseline_dim": res["baseline_dim"],
            "augmented_dim": res["augmented_dim"],
            "mean_G": desc["mean_G"],
            "median_G": desc["median_G"],
            "proportion_G_positive": desc["proportion_G_positive"],
            "mean_G_validation": float(np.mean(val["G"])),
        })
        log(f"  transition {src+1}->{tgt+1} probes fitted")

    deltas = [t["delta_r2"] for t in per_transition]
    results = summarize(per_transition, deltas)
    results["model_id"] = "allenai/OLMoE-1B-7B-0125"
    results["model_revision"] = "9b0c1aa87e34a20052389dce1f0cf01da783f654"
    results["data_manifest_sha256"] = manifest_hash
    results["probe"] = "sklearn.linear_model.Ridge(alpha=1.0)"
    results["rescue_modifications"] = "none"

    with open(os.path.join(ART, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    log("--- results ---")
    for t in per_transition:
        log(f"  {t['transition_human']}: R2_base {t['r2_baseline']:+.5f}  "
            f"R2_aug {t['r2_augmented']:+.5f}  Delta_R2 {t['delta_r2']:+.5f}  "
            f"meanG {t['mean_G']:.5f} medG {t['median_G']:.5f} propG+ {t['proportion_G_positive']:.4f}")
    log(f"  mean Delta_R2 {results['mean_delta_r2']:+.5f}")
    log(f"  VERDICT {results['verdict']}")
    log(results["stopping_note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
