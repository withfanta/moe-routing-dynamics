"""Collect the four horizon files, apply the frozen rule, write results.

Runs once on CPU after both GPU workers finish. This is the only writer of
results.json.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from drev import (
    DELAYED_DISTANCES,
    HORIZONS,
    IMMEDIATE_DISTANCE,
    MODEL_ID,
    MODEL_REVISION,
    PCA_COMPONENTS,
    PROBE_DIM,
    SHUFFLE_SEED,
)
from drev.analysis import summarize
from drev.probes import ALPHA

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "DREV_P0")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [finalize] {msg}"
    print(line, flush=True)
    with open(os.path.join(ART, "run.log"), "a") as fh:
        fh.write(line + "\n")


def main():
    per_distance, meta = {}, {}
    for delta in sorted(HORIZONS):
        path = os.path.join(ART, f"horizon_{delta}.json")
        if not os.path.exists(path):
            log(f"MISSING {path}; both workers must finish before finalizing")
            return 1
        with open(path) as fh:
            d = json.load(fh)
        per_distance[delta] = {
            "R2_real": d["R2_real"],
            "R2_shuffle": d["R2_shuffle"],
            "D": d["D"],
            "mean_G": d["mean_G"],
            "median_G": d["median_G"],
            "proportion_G_positive": d["proportion_G_positive"],
        }
        meta[delta] = {
            "distance": delta,
            "target_layer": d["target_layer"],
            "target_layer_code": d["target_layer_code"],
            "n_fit": d["n_fit"],
            "n_test": d["n_test"],
            "worker": d["worker"],
            "gpu": d["gpu"],
            "probe_dim": d["probe_dim"],
        }

    with open(os.path.join(ART, "data_manifest.json")) as fh:
        man = json.load(fh)

    results = summarize(per_distance, meta)
    results["experiment"] = "DREV-P0"
    results["source_layer"] = 4
    results["source_layer_code"] = 3
    results["model_id"] = MODEL_ID
    results["model_revision"] = MODEL_REVISION
    results["data_manifest_sha256"] = None  # filled below
    results["fit_fingerprint"] = man["fit"]["fingerprint"]
    results["test_fingerprint"] = man["test"]["fingerprint"]
    results["untouched_pool_size"] = man["untouched_pool_size"]
    results["prior_blocks_excluded"] = man["prior_blocks_excluded"]
    results["probe"] = f"sklearn.linear_model.Ridge(alpha={ALPHA})"
    results["compression"] = (
        f"StandardScaler + PCA(n_components={PCA_COMPONENTS}, svd_solver='randomized') "
        f"per block, fit split only; probe dim {PROBE_DIM}"
    )
    results["shuffle_seed"] = SHUFFLE_SEED
    results["delayed_distances"] = list(DELAYED_DISTANCES)
    results["immediate_distance"] = IMMEDIATE_DISTANCE
    results["rescue_modifications"] = "none"
    results["redv_project_status"] = "closed, read-only, result unmodified"

    import hashlib

    with open(os.path.join(ART, "data_manifest.json"), "rb") as fh:
        results["data_manifest_sha256"] = hashlib.sha256(fh.read()).hexdigest()

    with open(os.path.join(ART, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    log("--- results ---")
    for delta in sorted(HORIZONS):
        p, m = per_distance[delta], meta[delta]
        log(f"  delta={delta} (Layer {m['target_layer']}): R2_real {p['R2_real']:+.5f}  "
            f"R2_shuf {p['R2_shuffle']:+.5f}  D {p['D']:+.5f}  "
            f"meanG {p['mean_G']:.5f} medG {p['median_G']:.5f} propG+ {p['proportion_G_positive']:.4f}")
    log(f"  mean_D_delayed (delta 2,4,8) {results['mean_D_delayed']:+.5f}")
    log(f"  D_immediate (delta 1)        {results['D_immediate']:+.5f}")
    for c in results["conditions"]:
        log(f"    condition {c['condition']}: {'PASS' if c['passed'] else 'FAIL'} — {c['observed']}")
    log(f"  VERDICT {results['final_verdict']}")
    log(results["stopping_note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
