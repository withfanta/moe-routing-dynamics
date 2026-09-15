"""EIPC-P0 finalizer. Runs ONCE on CPU after both extraction workers finish.

1. builds FUSED / EXPERT_IDENTITY / SHUFFLED_IDENTITY;
2. fits FIT-only preprocessing;
3. fits exactly six probes (3 representations x 2 targets);
4. computes TEST R2 only after all probes are fitted;
5. applies the frozen decision rule;
6. writes the final result.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eipc import (
    ALL_HISTORY_CODE,
    MODEL_ID,
    MODEL_REVISION,
    PCA_COMPONENTS,
    PROJ_DIM,
    PROJECTION_SEED,
    REPRESENTATIONS,
    SHUFFLE_SEED,
    TARGETS,
)
from eipc.analysis import summarize
from eipc.expert_states import build_projection, projection_hash
from eipc.probes import ALPHA, run_target_probes
from eipc.representations import (
    FrozenPipeline,
    TargetScaler,
    build_fused,
    build_identity,
    build_shifts,
    build_shuffled,
    center_logits,
    raw_dimension,
)

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "EIPC_P0")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [finalize] {msg}"
    print(line, flush=True)
    with open(os.path.join(ART, "run.log"), "a") as fh:
        fh.write(line + "\n")


def load_split(role):
    path = os.path.join(ART, f"{role}_raw.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing {path}; both workers must finish first")
    d = np.load(path)
    s_by_layer = {li: d[f"L{li+1}_s"] for li in ALL_HISTORY_CODE}
    ids_by_layer = {li: d[f"L{li+1}_ids"] for li in ALL_HISTORY_CODE}
    logits = {m: d[f"L{m}_logits"] for m in TARGETS}
    return s_by_layer, ids_by_layer, logits


def main():
    log("=== EIPC-P0 finalize (CPU) ===")

    s_fit, ids_fit, logits_fit = load_split("fit")
    s_test, ids_test, logits_test = load_split("test")
    n_fit = s_fit[ALL_HISTORY_CODE[0]].shape[0]
    n_test = s_test[ALL_HISTORY_CODE[0]].shape[0]
    log(f"loaded FIT n={n_fit}, TEST n={n_test}")

    # Build all representations and fit all probes before reading any R2.
    per_target, target_meta, diagnostics = {}, {}, {}

    for target_human in sorted(TARGETS):
        spec = TARGETS[target_human]
        hist = spec["history_code"]
        log(f"target Layer {target_human} (code {spec['code']}), history Layers "
            f"{[li+1 for li in hist]}")

        shifts_fit = build_shifts(n_fit, len(hist), "fit")
        shifts_test = build_shifts(n_test, len(hist), "test")
        assert shifts_fit.min() >= 1 and shifts_fit.max() <= 63
        assert shifts_test.min() >= 1 and shifts_test.max() <= 63

        raw_fit = {
            "FUSED": build_fused(s_fit, hist),
            "EXPERT_IDENTITY": build_identity(s_fit, ids_fit, hist),
            "SHUFFLED_IDENTITY": build_shuffled(s_fit, ids_fit, hist, shifts_fit),
        }
        raw_test = {
            "FUSED": build_fused(s_test, hist),
            "EXPERT_IDENTITY": build_identity(s_test, ids_test, hist),
            "SHUFFLED_IDENTITY": build_shuffled(s_test, ids_test, hist, shifts_test),
        }
        for rep in REPRESENTATIONS:
            expected = raw_dimension(target_human, rep)
            assert raw_fit[rep].shape[1] == expected, (rep, raw_fit[rep].shape, expected)
            assert raw_test[rep].shape[1] == expected
            log(f"  raw {rep}: {raw_fit[rep].shape[1]} dims")

        # FIT-only compression, applied unchanged to TEST.
        X_fit, X_test, evr = {}, {}, {}
        for rep in REPRESENTATIONS:
            pipe = FrozenPipeline.fit(raw_fit[rep])
            X_fit[rep] = pipe.transform(raw_fit[rep])
            X_test[rep] = pipe.transform(raw_test[rep])
            evr[rep] = pipe.explained_variance_ratio_sum
            assert X_fit[rep].shape[1] == PCA_COMPONENTS
            assert X_test[rep].shape[1] == PCA_COMPONENTS
            log(f"  {rep} -> {PCA_COMPONENTS} dims, PCA evr {evr[rep]:.4f}")

        # Target: centred router logits, standardized with FIT statistics only.
        q_fit = center_logits(logits_fit[target_human])
        q_test = center_logits(logits_test[target_human])
        tsc = TargetScaler.fit(q_fit)
        Y_fit, Y_test = tsc.transform(q_fit), tsc.transform(q_test)
        log(f"  target centred and FIT-standardized: {Y_fit.shape[1]} dims")

        out = run_target_probes(X_fit, X_test, Y_fit, Y_test)
        per_target[target_human] = {
            "R2_fused": out["R2_fused"],
            "R2_identity": out["R2_identity"],
            "R2_shuffled": out["R2_shuffled"],
            "A": out["A"],
            "B": out["B"],
        }
        target_meta[target_human] = {
            "target_layer": target_human,
            "target_layer_code": spec["code"],
            "history_layers": list(spec["history_human"]),
            "n_fit": out["n_fit"],
            "n_test": out["n_test"],
            "input_dim": out["input_dim"],
            "target_dim": out["target_dim"],
            "raw_dims": {rep: raw_dimension(target_human, rep) for rep in REPRESENTATIONS},
        }
        diagnostics[target_human] = {"pca_evr": evr}
        log(f"  three probes fitted for Layer {target_human}")

    with open(os.path.join(ART, "data_manifest.json")) as fh:
        man = json.load(fh)
    with open(os.path.join(ART, "data_manifest.json"), "rb") as fh:
        man_sha = hashlib.sha256(fh.read()).hexdigest()

    results = summarize(per_target, target_meta)
    results["experiment"] = "EIPC-P0"
    results["model_id"] = MODEL_ID
    results["model_revision"] = MODEL_REVISION
    results["dataset_split_used"] = man["split_used"]
    results["data_manifest_sha256"] = man_sha
    results["fit_fingerprint"] = man["fit"]["fingerprint"]
    results["test_fingerprint"] = man["test"]["fingerprint"]
    results["projection"] = {
        "seed": PROJECTION_SEED,
        "out_dim": PROJ_DIM,
        "sha256": projection_hash(build_projection()),
    }
    results["compression"] = (
        f"StandardScaler -> PCA(n_components={PCA_COMPONENTS}, svd_solver='randomized') "
        "-> StandardScaler, fit on FIT only"
    )
    results["probe"] = f"sklearn.linear_model.Ridge(alpha={ALPHA}), multi-output"
    results["shuffle_seed"] = SHUFFLE_SEED
    results["diagnostics"] = {str(k): v for k, v in diagnostics.items()}
    results["rescue_modifications"] = "none"
    results["prior_projects_status"] = "closed, read-only, results unmodified"

    with open(os.path.join(ART, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    log("--- results ---")
    for m in sorted(TARGETS):
        p = per_target[m]
        log(f"  Layer {m}: R2_fused {p['R2_fused']:+.5f}  R2_identity {p['R2_identity']:+.5f}  "
            f"R2_shuffled {p['R2_shuffled']:+.5f}  A {p['A']:+.5f}  B {p['B']:+.5f}")
    log(f"  mean_A {results['mean_A']:+.5f}   mean_B {results['mean_B']:+.5f}")
    for c in results["conditions"]:
        log(f"    condition {c['condition']}: {'PASS' if c['passed'] else 'FAIL'} — {c['observed']}")
    log(f"  VERDICT {results['final_verdict']}")
    log(results["stopping_note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
