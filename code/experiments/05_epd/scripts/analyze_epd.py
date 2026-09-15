"""EPD-P0 analysis: four probes, layerwise curves, descriptive transitions, pattern.

Runs once on CPU after the shards are merged. Fits all preprocessing on FIT only, evaluates
TEST once, then classifies the decomposition pattern descriptively. No verdict is produced.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from epd import (
    HISTORY_HUMAN,
    LAYERWISE_PCA_COMPONENTS,
    MODEL_ID,
    MODEL_REVISION,
    N_FIT,
    N_HISTORY_LAYERS,
    N_TEST,
    NONTRIVIAL_GAP,
    PCA_COMPONENTS,
    RAW_DIMS,
    REPRESENTATIONS,
    TARGET_LAYER_HUMAN,
)
from epd.analysis import (
    classify_pattern,
    co_selection_enrichment,
    layerwise_summary,
    pattern_interpretation,
)
from epd.extraction import load_eipc_projection
from epd.probes import ALPHA, descriptive_gaps, fit_and_score, run_main_probes
from epd.representations import (
    FrozenPipeline,
    TargetScaler,
    build_all,
    build_content_rank_layer,
    build_id_path_layer,
    center_logits,
    compress,
)

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "EPD_P0")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [analyze] {msg}"
    print(line, flush=True)
    with open(os.path.join(ART, "run.log"), "a") as fh:
        fh.write(line + "\n")


def main():
    log("=== EPD-P0 analysis (CPU) ===")

    d = np.load(os.path.join(ART, "merged_raw.npz"))
    is_test = d["is_test"].astype(bool)
    s, hist_ids, g12, top8_12 = d["s"], d["hist_ids"], d["g12"], d["top8_12"]
    log(f"loaded merged: n={s.shape[0]} (fit {int((~is_test).sum())}, test {int(is_test.sum())})")
    log(f"  reconstruction from extraction: abs {float(d['recon_max_abs'][0]):.3e}, "
        f"relative {float(d['recon_max_rel'][0]):.3e}")

    _, proj_meta = load_eipc_projection()
    log(f"reused EIPC projection hash re-verified: {proj_meta['sha256']}")

    fit_i, test_i = ~is_test, is_test
    assert int(fit_i.sum()) == N_FIT and int(test_i.sum()) == N_TEST

    # Build all four representations from the same captured quantities.
    reps_fit = build_all(s[fit_i], hist_ids[fit_i])
    reps_test = build_all(s[test_i], hist_ids[test_i])
    for name in REPRESENTATIONS:
        log(f"  raw {name:16s} {reps_fit[name].shape[1]:6d} dims (expect {RAW_DIMS[name]})")

    # Target: centred Layer-12 logits, standardized with FIT statistics only.
    q_fit, q_test = center_logits(g12[fit_i]), center_logits(g12[test_i])
    tsc = TargetScaler.fit(q_fit)
    Y_fit, Y_test = tsc.transform(q_fit), tsc.transform(q_test)
    log(f"target Layer {TARGET_LAYER_HUMAN}: centred, FIT-standardized, {Y_fit.shape[1]} dims")

    # Shared 32-dim budget, fit on FIT only.
    X_fit, X_test, evr = compress(reps_fit, reps_test, PCA_COMPONENTS)
    for name in REPRESENTATIONS:
        log(f"  {name:16s} -> {PCA_COMPONENTS} dims, PCA evr {evr[name]:.4f}")

    r2 = run_main_probes(X_fit, X_test, Y_fit, Y_test)
    gaps = descriptive_gaps(r2)
    log("main probes fitted and evaluated once on TEST")

    # Layerwise exploratory curves at 16 dims.
    log(f"layerwise probes ({LAYERWISE_PCA_COMPONENTS} dims each)")
    r2_id_layer, r2_content_layer = {}, {}
    for pos in range(N_HISTORY_LAYERS):
        human = HISTORY_HUMAN[pos]
        id_fit = build_id_path_layer(hist_ids[fit_i], pos)
        id_test = build_id_path_layer(hist_ids[test_i], pos)
        ct_fit = build_content_rank_layer(s[fit_i], pos)
        ct_test = build_content_rank_layer(s[test_i], pos)

        pid = FrozenPipeline.fit(id_fit, LAYERWISE_PCA_COMPONENTS)
        pct = FrozenPipeline.fit(ct_fit, LAYERWISE_PCA_COMPONENTS)
        a, _ = fit_and_score(pid.transform(id_fit), Y_fit, pid.transform(id_test), Y_test)
        b, _ = fit_and_score(pct.transform(ct_fit), Y_fit, pct.transform(ct_test), Y_test)
        r2_id_layer[human], r2_content_layer[human] = a, b
        log(f"  Layer {human:2d}: ID_PATH R2 {a:+.5f}   CONTENT_RANK R2 {b:+.5f}")

    lw = layerwise_summary(r2_id_layer, r2_content_layer)

    # Descriptive transition summary over all samples.
    log("descriptive co-selection enrichment (no statistical testing)")
    trans = co_selection_enrichment(hist_ids, top8_12)

    pattern = classify_pattern(r2, NONTRIVIAL_GAP)

    with open(os.path.join(ART, "data_manifest.json")) as fh:
        man = json.load(fh)
    with open(os.path.join(ART, "data_manifest.json"), "rb") as fh:
        man_sha = hashlib.sha256(fh.read()).hexdigest()

    results = {
        "experiment": "EPD-P0",
        "exploratory": True,
        "no_formal_verdict": True,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "target_layer": TARGET_LAYER_HUMAN,
        "history_layers": list(HISTORY_HUMAN),
        "n_fit": N_FIT,
        "n_test": N_TEST,
        "data_manifest_sha256": man_sha,
        "fingerprints": {r: man[r]["fingerprint"] for r in ("fit", "test")},
        "prior_exclusion": man["prior_exclusion"],
        "reused_eipc_projection": proj_meta,
        "probe": f"sklearn.linear_model.Ridge(alpha={ALPHA}), multi-output",
        "compression": (f"StandardScaler -> PCA({PCA_COMPONENTS}, randomized) -> "
                        "StandardScaler, fit on FIT only"),
        "raw_dims": RAW_DIMS,
        "pca_explained_variance_ratio": evr,
        "main": {
            "R2_fused": r2["FUSED"],
            "R2_id": r2["ID_PATH"],
            "R2_content": r2["CONTENT_RANK"],
            "R2_full": r2["FULL_PROVENANCE"],
            **gaps,
        },
        "layerwise": {
            "R2_id_layer": {str(k): v for k, v in r2_id_layer.items()},
            "R2_content_layer": {str(k): v for k, v in r2_content_layer.items()},
            "summary": lw,
            "n_components": LAYERWISE_PCA_COMPONENTS,
        },
        "transitions": trans,
        "decomposition_pattern": pattern["pattern"],
        "pattern_detail": pattern,
        "pattern_interpretation": pattern_interpretation(pattern["pattern"]),
        "reconstruction": {
            "max_abs": float(d["recon_max_abs"][0]),
            "max_relative": float(d["recon_max_rel"][0]),
        },
        "architecture_built": "none",
        "rescue_modifications": "none",
        "prior_projects_status": "closed, read-only, results unmodified",
    }

    with open(os.path.join(ART, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    log("--- main results ---")
    log(f"  R2_fused   {r2['FUSED']:+.5f}")
    log(f"  R2_id      {r2['ID_PATH']:+.5f}")
    log(f"  R2_content {r2['CONTENT_RANK']:+.5f}")
    log(f"  R2_full    {r2['FULL_PROVENANCE']:+.5f}")
    log(f"  G_full_vs_fused          {gaps['G_full_vs_fused']:+.5f}")
    log(f"  G_identity_given_content {gaps['G_identity_given_content']:+.5f}")
    log(f"  G_content_given_identity {gaps['G_content_given_identity']:+.5f}")
    log(f"  layerwise: best ID layer {lw['best_id_layer']}, best CONTENT layer "
        f"{lw['best_content_layer']}, mean ID {lw['mean_id']:+.5f}, "
        f"mean CONTENT {lw['mean_content']:+.5f}")
    log(f"  PATTERN {pattern['pattern']}")
    log(f"  {results['pattern_interpretation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
