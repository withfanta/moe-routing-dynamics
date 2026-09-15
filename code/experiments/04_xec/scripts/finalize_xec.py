"""XEC-P0 finalizer: apply policies at Layer 12, measure true NLL, apply the frozen rule.

Runs after all nine checkpoints exist. For each checkpoint it predicts one action per TEST
sample, executes exactly that action at Layer 12, runs Layers 13-16 natively, and records
the true next-token NLL. Only afterwards is the TEST oracle computed, for descriptive
headroom.
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
    MODEL_ID,
    MODEL_REVISION,
    N_ACTIONS,
    POLICY_SEEDS,
    TARGET_LAYER,
    TARGET_LAYER_HUMAN,
    VARIANTS,
)
from xec.analysis import (
    build_differences,
    expert_beats_fused_all_seeds,
    paired_bootstrap_ci,
    per_seed_means,
    summarize,
)
from xec.cache_features import (
    build_projection,
    current_context,
    expert_cache_items,
    fused_cache_items,
    projection_hash,
)
from xec.counterfactual import all_action_nlls, selected_action_nll
from xec.data import rebuild_from_manifest
from xec.olmoe import assert_frozen, load_model, verify_config
from xec.oracle import action_distribution, oracle_action, oracle_nll
from xec.policy import RoutingPolicy
from xec.training import predict_actions

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "XEC_P0")
CKPT_DIR = os.path.join(ART, "checkpoints")
MARKER = os.path.join(ART, "training_complete.json")
DATASET_DIR = os.environ.get("XEC_DATASET_DIR")
MODEL_DIR = os.environ.get("XEC_MODEL_DIR")
MICROBATCH = int(os.environ.get("XEC_MICROBATCH", "8"))


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [finalize] {msg}"
    print(line, flush=True)
    with open(os.path.join(ART, "run.log"), "a") as fh:
        fh.write(line + "\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def apply_actions(model, contexts, top12_ids_np, targets_np, actions, log, tag):
    """Execute each sample's predicted action at Layer 12 and return true NLL."""
    n = len(contexts)
    out = np.zeros(n, dtype=np.float64)
    mb, done, t0 = MICROBATCH, 0, time.time()
    while done < n:
        sl = slice(done, done + mb)
        chunk = contexts[sl]
        ids = torch.tensor([c.context for c in chunk]).cuda()
        tgt = torch.tensor(targets_np[sl]).cuda()
        t12 = torch.tensor(top12_ids_np[sl].astype(np.int64)).cuda()
        try:
            with torch.inference_mode():
                out[sl] = selected_action_nll(model, ids, tgt, t12, actions[sl])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if mb == 1:
                raise RuntimeError("TECHNICAL_BLOCKER: OOM at microbatch 1")
            mb = max(1, mb // 2)
            log(f"    OOM -> microbatch {mb}")
            continue
        done += len(chunk)
        if done % 512 == 0 or done == n:
            log(f"    {tag}: {done}/{n}, {time.time()-t0:.0f}s, mb={mb}")
    return out


def main():
    if not os.path.exists(MARKER):
        log("MISSING training-completion marker; run train_policies.py first")
        return 1
    log("=== XEC-P0 finalize ===")

    f = np.load(os.path.join(ART, "test_features.npz"))
    u_test = torch.from_numpy(current_context(f["x12"], f["g12"]).astype(np.float32))
    items = {
        "EXPERT_CACHE": torch.from_numpy(expert_cache_items(f["s"], f["hist_ids"])),
        "FUSED_CACHE": torch.from_numpy(fused_cache_items(f["s"])),
        "CURRENT_ONLY": None,
    }
    native = f["native_nll"].astype(np.float64)
    top12_ids = f["top12_ids"]
    targets = f["target"]
    n_test = u_test.shape[0]
    log(f"TEST n={n_test}, native mean NLL {native.mean():.6f}")

    # Predict every checkpoint's actions before touching the frozen model.
    predictions = {v: {} for v in VARIANTS}
    for variant in VARIANTS:
        for seed in POLICY_SEEDS:
            ck = torch.load(os.path.join(CKPT_DIR, f"{variant}_seed{seed}.pt"),
                            map_location="cpu")
            m = RoutingPolicy(variant)
            m.load_state_dict(ck["state_dict"])
            predictions[variant][seed] = predict_actions(m, u_test, items[variant])
            dist = action_distribution(predictions[variant][seed])
            log(f"  {variant} seed {seed}: predicted actions {dist['counts']} "
                f"(native {dist['fraction_native_action']:.4f})")

    model, tok = load_model(model_dir=MODEL_DIR)
    model.eval()
    assert_frozen(model)
    verify_config(model)
    log(f"frozen model loaded on {torch.cuda.get_device_name(0)}, no OLMoE gradients")
    log(f"projection sha256 {projection_hash(build_projection())}")

    contexts = rebuild_from_manifest(
        os.path.join(ART, "data_manifest.json"), tok, model.config.eos_token_id,
        dataset_dir=DATASET_DIR)["test"]
    assert len(contexts) == n_test
    assert np.array_equal(np.array([c.target for c in contexts]), targets)
    log("TEST contexts rebuilt and fingerprint-verified")

    per_seed = {v: {} for v in VARIANTS}
    for variant in VARIANTS:
        for seed in POLICY_SEEDS:
            log(f"  applying {variant} seed {seed} at Layer {TARGET_LAYER_HUMAN}")
            per_seed[variant][seed] = apply_actions(
                model, contexts, top12_ids, targets, predictions[variant][seed],
                log, f"{variant}/{seed}")
            log(f"    mean TEST NLL {per_seed[variant][seed].mean():.6f}")

    res_path = os.path.join(ART, "test_policy_results.npz")
    np.savez_compressed(
        res_path, native_nll=native,
        **{f"{v}_seed{s}_nll": per_seed[v][s] for v in VARIANTS for s in POLICY_SEEDS},
        **{f"{v}_seed{s}_action": predictions[v][s] for v in VARIANTS for s in POLICY_SEEDS})
    log(f"test_policy_results.npz written, sha256 {sha256_file(res_path)}")

    # Only now, with all policy evaluation complete, compute the TEST oracle.
    log("policy evaluation complete; computing TEST oracle for descriptive headroom only")
    rows, mb, done = [], MICROBATCH, 0
    while done < n_test:
        sl = slice(done, done + mb)
        chunk = contexts[sl]
        ids = torch.tensor([c.context for c in chunk]).cuda()
        tgt = torch.tensor(targets[sl]).cuda()
        t12 = torch.tensor(top12_ids[sl].astype(np.int64)).cuda()
        try:
            with torch.inference_mode():
                rows.append(all_action_nlls(model, ids, tgt, t12))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if mb == 1:
                raise RuntimeError("TECHNICAL_BLOCKER: OOM at microbatch 1")
            mb = max(1, mb // 2)
            continue
        done += len(chunk)
        if done % 512 == 0 or done == n_test:
            log(f"    test oracle: {done}/{n_test}, mb={mb}")
    A_test = np.concatenate(rows, axis=0)
    a_star = oracle_action(A_test)
    orc_path = os.path.join(ART, "test_oracle_descriptive.npz")
    np.savez_compressed(orc_path, action_nlls=A_test.astype(np.float64),
                        oracle_action=a_star.astype(np.int64))
    log(f"test_oracle_descriptive.npz written, sha256 {sha256_file(orc_path)}")

    # Primary statistics.
    diffs = build_differences(native, per_seed)
    stats = {k: paired_bootstrap_ci(v) for k, v in diffs.items()}
    seed_check = expert_beats_fused_all_seeds(per_seed)
    nll_means = per_seed_means(per_seed)

    expert_avg = np.stack([per_seed["EXPERT_CACHE"][s] for s in POLICY_SEEDS]).mean(axis=0)
    oracle_mean = float(oracle_nll(A_test).mean())
    acc = {}
    for v in VARIANTS:
        acc[v] = float(np.mean([
            (predictions[v][s] == a_star).mean() for s in POLICY_SEEDS]))

    descriptive = {
        "native_mean_nll": float(native.mean()),
        "five_action_oracle_mean_nll": oracle_mean,
        "oracle_headroom_vs_native": oracle_mean - float(native.mean()),
        "test_oracle_action_distribution": action_distribution(a_star),
        "action_accuracy_vs_test_oracle": acc,
        "predicted_action_distribution": {
            v: action_distribution(np.concatenate([predictions[v][s] for s in POLICY_SEEDS]))
            for v in VARIANTS},
        "expert_cache_fraction_improved_vs_native": float(np.mean(expert_avg < native)),
        "expert_cache_fraction_worsened_vs_native": float(np.mean(expert_avg > native)),
        "expert_cache_fraction_equal_vs_native": float(np.mean(expert_avg == native)),
    }

    results = summarize(stats, seed_check, nll_means, descriptive)
    with open(os.path.join(ART, "data_manifest.json"), "rb") as fh:
        man_sha = hashlib.sha256(fh.read()).hexdigest()
    with open(os.path.join(ART, "data_manifest.json")) as fh:
        man = json.load(fh)

    results.update({
        "experiment": "XEC-P0",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "target_layer": TARGET_LAYER_HUMAN,
        "target_layer_code": TARGET_LAYER,
        "n_test": int(n_test),
        "n_train": man["train"]["n"],
        "n_validation": man["validation"]["n"],
        "data_manifest_sha256": man_sha,
        "fingerprints": {r: man[r]["fingerprint"] for r in ("train", "validation", "test")},
        "eipc_exclusion": man["eipc_exclusion"],
        "projection_sha256": projection_hash(build_projection()),
        "policy_seeds": list(POLICY_SEEDS),
        "olmoe_receives_gradients": False,
        "artifact_sha256": {
            "test_policy_results.npz": sha256_file(res_path),
            "test_oracle_descriptive.npz": sha256_file(orc_path),
            "train_features.npz": sha256_file(os.path.join(ART, "train_features.npz")),
            "validation_features.npz": sha256_file(os.path.join(ART, "validation_features.npz")),
            "test_features.npz": sha256_file(os.path.join(ART, "test_features.npz")),
            "train_oracle.npz": sha256_file(os.path.join(ART, "train_oracle.npz")),
            "validation_oracle.npz": sha256_file(os.path.join(ART, "validation_oracle.npz")),
        },
        "rescue_modifications": "none",
        "prior_projects_status": "closed, read-only, results unmodified",
    })

    with open(os.path.join(ART, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    log("--- results ---")
    log(f"  native mean NLL            {native.mean():.6f}")
    for v in VARIANTS:
        row = " ".join(f"seed{s} {nll_means[v][str(s)]:.6f}" for s in POLICY_SEEDS)
        log(f"  {v:14s} {row}  mean {nll_means[v]['mean']:.6f}")
    log(f"  five-action oracle mean NLL {oracle_mean:.6f}")
    for k in ("d_native", "d_current", "d_fused"):
        st = stats[k]
        log(f"  {k:10s} mean {st['mean']:+.6f}  95% CI [{st['ci_low']:+.6f}, {st['ci_high']:+.6f}]")
    for c in results["conditions"]:
        log(f"    condition {c['condition']}: {'PASS' if c['passed'] else 'FAIL'} — {c['observed']}")
    log(f"  VERDICT {results['final_verdict']}")
    log(results["stopping_note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
