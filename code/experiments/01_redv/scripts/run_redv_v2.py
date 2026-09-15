"""REDV-V2 formal run: dimension-matched rejected-evidence test.

Order is fixed by the preregistration: verify the V100, load the pinned FP16
checkpoint, freeze the untouched-block manifest, extract ALL fit-set and ALL
test-set (b, r, G), build the one fixed shuffled control, fit all nine probes,
and only then compute held-out R² and apply the rule mechanically.

Reuses REDV-V1's validated hooks, rejected-evidence, and counterfactual code
unchanged. REDV-V1 artifacts are never written.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from redv import MODEL_ID, MODEL_REVISION, TRANSITIONS
from redv.counterfactual import regret_descriptives, routing_regret
from redv.data import build_token_stream, load_split_texts, make_blocks
from redv.olmoe_hooks import assert_frozen, check_context_len, load_model, native_forward
from redv.probes import ALPHA
from redv.rejected import probs_from_logits, rejected_evidence
from redv.v2_probes import (
    failed_conditions,
    run_v2_probe_triple,
    v2_decision_sentence,
    v2_stopping_note,
    v2_verdict,
)
from redv.v2_sampling import (
    N_FIT,
    N_TEST,
    SHUFFLE_SEED,
    V2_SEED,
    build_control_permutations,
    draw_v2_sample,
    load_v1_block_ids,
    untouched_pool,
    v2_manifest_payload,
    write_v2_manifest,
)

ART_V1 = os.path.join(os.path.dirname(__file__), "..", "artifacts", "REDV_V1")
ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "REDV_V2")
DATASET_DIR = os.environ.get("REDV_DATASET_DIR")
MODEL_DIR = os.environ.get("REDV_MODEL_DIR")
MICROBATCH = int(os.environ.get("REDV_MICROBATCH", "8"))
PHYSICAL_GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "unknown")


class Log:
    def __init__(self, path):
        self.fh = open(path, "a")

    def __call__(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()


def verify_device(log):
    """Record and verify the compute device before any formal extraction."""
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("TECHNICAL_BLOCKER: CUDA unavailable; OLMoE must not run on CPU")

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    log(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    log(f"torch.cuda.device_count()  = {torch.cuda.device_count()}")
    log(f"torch.cuda.get_device_name(0) = {name}")
    log(f"torch.cuda.get_device_properties(0) = {props}")
    log(f"torch {torch.__version__} / CUDA {torch.version.cuda} / transformers {transformers.__version__}")
    log(f"CUDA_VISIBLE_DEVICES = {PHYSICAL_GPU} (visible as cuda:0)")

    if "V100" not in name:
        raise RuntimeError(f"TECHNICAL_BLOCKER: expected a Tesla V100-class device, got {name}")

    env = {
        "experiment": "REDV-V2",
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "device_name_0": name,
        "device_properties_0": {
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "multi_processor_count": props.multi_processor_count,
            "capability": f"{props.major}.{props.minor}",
        },
        "cuda_visible_devices": PHYSICAL_GPU,
        "physical_gpu_index": PHYSICAL_GPU,
        "visible_as": "cuda:0",
        "device_selection_note": (
            "Protocol names CUDA_VISIBLE_DEVICES=0. Physical GPU 0 on this shared "
            "DGX-1 was occupied by another user's long-running job with ~2 GB free, "
            "insufficient for OLMoE FP16 (~14 GB). All eight GPUs are identical "
            "Tesla V100-SXM2-32GB, so the V100-only requirement holds. With user "
            "authorization a free V100 was pinned so torch sees it as its sole "
            "device cuda:0."
        ),
        "packages": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
        "dtype": "float16",
        "quantized": False,
        "inference_mode": True,
    }
    try:
        import scipy, sklearn

        env["packages"]["scipy"] = scipy.__version__
        env["packages"]["scikit-learn"] = sklearn.__version__
    except Exception:
        pass
    with open(os.path.join(ART, "environment.json"), "w") as fh:
        json.dump(env, fh, indent=2)
    return env


def build_v2_manifest(tok, eos_token_id, log):
    """Freeze the V2 sample from blocks REDV-V1 never observed."""
    v1_ids = load_v1_block_ids(os.path.join(ART_V1, "data_manifest.json"))
    log(f"REDV-V1 blocks excluded: validation {len(v1_ids['validation'])}, test {len(v1_ids['test'])}")

    blocks_by_split = {}
    for split in ("validation", "test"):
        texts = load_split_texts(split, dataset_dir=DATASET_DIR)
        stream = build_token_stream(texts, tok, eos_token_id)
        blocks_by_split[split] = make_blocks(stream)
        log(f"  {split}: {len(blocks_by_split[split])} blocks total")

    pool = untouched_pool(blocks_by_split, v1_ids)
    log(f"untouched pool: {len(pool)} blocks (need {N_FIT + N_TEST})")
    if len(pool) < N_FIT + N_TEST:
        raise RuntimeError(
            f"STOP: only {len(pool)} untouched blocks available, need {N_FIT + N_TEST}"
        )

    fit, test = draw_v2_sample(pool, seed=V2_SEED)
    log(f"drew {len(fit)} fit + {len(test)} test contexts (seed {V2_SEED})")

    payload = v2_manifest_payload(
        fit, test, len(pool), v1_ids,
        extra={"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
               "tokenizer_class": tok.__class__.__name__},
    )
    h = write_v2_manifest(os.path.join(ART, "data_manifest.json"), payload)
    log(f"data_manifest.json written, sha256 {h}")
    log(f"  fit fingerprint  {payload['fit']['fingerprint'][:16]}")
    log(f"  test fingerprint {payload['test']['fingerprint'][:16]}")
    return fit, test, h, payload


def extract(model, contexts, log, tag):
    """Extract b (h, g), r, and G for every context. Microbatch 8 -> 4 -> 2 -> 1."""
    capture_layers = sorted({l for pair in TRANSITIONS for l in pair})
    store = {pair: {"h": [], "g": [], "r": [], "G": [], "native_nll": []} for pair in TRANSITIONS}

    n = len(contexts)
    mb = MICROBATCH
    t0 = time.time()
    done = 0
    while done < n:
        chunk = contexts[done : done + mb]
        ids = torch.tensor([c.context for c in chunk]).cuda()
        targets = torch.tensor([c.target for c in chunk]).cuda()
        check_context_len(ids)

        try:
            with torch.inference_mode():
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
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if mb == 1:
                raise RuntimeError("TECHNICAL_BLOCKER: OOM at microbatch 1 on the V100")
            mb = max(1, mb // 2)
            log(f"  OOM -> microbatch {mb} (sample count and context length unchanged)")
            continue

        done += len(chunk)
        if done % 128 == 0 or done == n:
            el = time.time() - t0
            log(f"  {tag}: {done}/{n}, {el:.0f}s elapsed, {el/done*(n-done):.0f}s remaining, mb={mb}")

    return {pair: {k: np.concatenate(v, axis=0) for k, v in d.items()} for pair, d in store.items()}


def main():
    os.makedirs(ART, exist_ok=True)
    log = Log(os.path.join(ART, "run.log"))
    log("=== REDV-V2 formal run: dimension-matched rejected-evidence test ===")

    verify_device(log)

    model, tok = load_model(model_dir=MODEL_DIR)
    model.eval()
    assert_frozen(model)
    log(f"model loaded frozen, dtype {next(model.parameters()).dtype}, revision pinned")

    cfg = model.config
    assert cfg.architectures == ["OlmoeForCausalLM"] and cfg.hidden_size == 2048
    assert cfg.num_hidden_layers == 16 and cfg.num_experts == 64
    assert cfg.num_experts_per_tok == 8 and cfg.norm_topk_prob is False
    log("config verified identical to REDV-V1")

    with open(os.path.join(ART, "model_provenance.json"), "w") as fh:
        json.dump({
            "experiment": "REDV-V2",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "instruct_checkpoint": False,
            "architecture": cfg.architectures,
            "hidden_size": cfg.hidden_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_experts": cfg.num_experts,
            "num_experts_per_tok": cfg.num_experts_per_tok,
            "norm_topk_prob": cfg.norm_topk_prob,
            "dtype": "float16",
            "identical_to_v1": True,
        }, fh, indent=2)

    fit_ctx, test_ctx, manifest_hash, manifest = build_v2_manifest(
        tok, model.config.eos_token_id, log)

    log(f"extracting fit set ({len(fit_ctx)} contexts)")
    feats_fit = extract(model, fit_ctx, log, "fit")
    log(f"extracting final test set ({len(test_ctx)} contexts)")
    feats_test = extract(model, test_ctx, log, "test")

    for tag, feats in (("fit", feats_fit), ("test", feats_test)):
        arrays = {}
        for (src, tgt), d in feats.items():
            for k, v in d.items():
                arrays[f"L{src+1}_{tgt+1}_{k}"] = v
        np.savez_compressed(os.path.join(ART, f"{tag}_features.npz"), **arrays)
        log(f"{tag}_features.npz written")

    log(f"building the one fixed shuffled control (seed {SHUFFLE_SEED})")
    perms = build_control_permutations(len(fit_ctx), len(test_ctx))
    for pair in TRANSITIONS:
        for role in ("fit", "test"):
            p = perms[pair][role]
            assert not np.any(p == np.arange(len(p))), "derangement violated"
    log("  six derangements verified, no fixed points")

    log("fitting all nine probes before computing any R2")
    per_transition = []
    for src, tgt in TRANSITIONS:
        pair = (src, tgt)
        out = run_v2_probe_triple(
            feats_fit[pair], feats_test[pair], perms[pair]["fit"], perms[pair]["test"])
        desc = regret_descriptives(feats_test[pair]["G"])
        per_transition.append({
            "transition_human": f"{src+1} -> {tgt+1}",
            "transition_code": [src, tgt],
            "n_fit": int(len(feats_fit[pair]["G"])),
            "n_test": int(len(feats_test[pair]["G"])),
            "R2_baseline": out["r2_baseline"],
            "R2_real": out["r2_real"],
            "R2_shuffle": out["r2_shuffle"],
            "D": out["D"],
            "gain_over_baseline": out["gain_over_baseline"],
            "baseline_dim": out["baseline_dim"],
            "augmented_dim": out["augmented_dim"],
            "mean_G": desc["mean_G"],
            "median_G": desc["median_G"],
            "proportion_G_positive": desc["proportion_G_positive"],
        })
        log(f"  transition {src+1}->{tgt+1}: three probes fitted")

    r2_reals = [t["R2_real"] for t in per_transition]
    ds = [t["D"] for t in per_transition]
    mean_d = float(np.mean(ds))
    verdict = v2_verdict(r2_reals, ds)

    results = {
        "experiment": "REDV-V2",
        "transitions": per_transition,
        "mean_D": mean_d,
        "final_verdict": verdict,
        "decision_sentence": v2_decision_sentence(verdict),
        "stopping_note": v2_stopping_note(verdict),
        "failed_conditions": failed_conditions(r2_reals, ds),
        "judgement_rule": {
            "condition_1": "R2_real > 0 for all three transitions",
            "condition_2": "D > 0 for all three transitions",
            "condition_3": "mean(D) >= +0.02",
            "any_failure": "NOT_SUPPORTED",
            "frozen_before_results": True,
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "data_manifest_sha256": manifest_hash,
        "fit_fingerprint": manifest["fit"]["fingerprint"],
        "test_fingerprint": manifest["test"]["fingerprint"],
        "probe": f"sklearn.linear_model.Ridge(alpha={ALPHA})",
        "shuffle_seed": SHUFFLE_SEED,
        "v2_seed": V2_SEED,
        "gpu": torch.cuda.get_device_name(0),
        "rescue_modifications": "none",
        "redv_v1_verdict_unchanged": "SUPPORTED",
    }
    with open(os.path.join(ART, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    log("--- results ---")
    for t in per_transition:
        log(f"  {t['transition_human']}: R2_base {t['R2_baseline']:+.5f}  "
            f"R2_real {t['R2_real']:+.5f}  R2_shuf {t['R2_shuffle']:+.5f}  "
            f"D {t['D']:+.5f}  real-base {t['gain_over_baseline']:+.5f}")
        log(f"      meanG {t['mean_G']:.5f} medG {t['median_G']:.5f} propG+ {t['proportion_G_positive']:.4f}")
    log(f"  mean D {mean_d:+.5f}")
    log(f"  VERDICT {verdict}")
    for c in results["failed_conditions"]:
        log(f"    {c}")
    log(results["stopping_note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
