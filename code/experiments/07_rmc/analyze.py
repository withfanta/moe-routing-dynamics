"""RMC-P0 analysis: eight probes, paired bootstrap, mechanical verdict.

For each target block in {12, 20} and each history depth k in {1, 2, 4, 8}, fits
Ridge(alpha=1.0) from a 16-d routing-history feature to the 8-d centred standardized MLP
router logits. The recent-layer scaler is fitted once per target and reused for every k;
only the older-history block changes.

Runs once on CPU after merging. Gates itself on the protocol checks (test P).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from rmc import (
    ALPHA, ART, BOOTSTRAP_SEED, DELTA4_THRESHOLD, EXPERIMENT, FEATURE_DIM, HERE, K_VALUES,
    MODEL_ID, MODEL_REVISION, NOT_REPLICATED, N_BOOTSTRAP, N_FIT, N_LAYERS, NUM_EXPERTS,
    N_TEST, PCA_COMPONENTS, PCA_SEED, PRIMARY_K, RECENT_DIM, REPLICATED, TARGET_LAYERS,
    TOP_K, older_layers, sha256_file,
)


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [rmc] {msg}"
    print(line, flush=True)
    with open(os.path.join(ART, "analyze.log"), "a") as fh:
        fh.write(line + "\n")


# --------------------------------------------------------------------- representations


def selection_state(ids: np.ndarray, layers_human: Sequence[int]) -> np.ndarray:
    """S_l concatenated over the requested human-readable layers.

    S_l[e] = 1 if MLP expert e is in the native Top-2 at layer l, else 0. Identities only:
    no rank, no score, no probability. Result is (n, 8 * len(layers)).
    """
    n = ids.shape[0]
    pos = [int(l) - 1 for l in layers_human]
    if any(p < 0 or p >= N_LAYERS for p in pos):
        raise ValueError(f"layers {list(layers_human)} outside 1..{N_LAYERS}")
    sel = ids[:, pos, :]
    out = np.zeros((n, len(pos), NUM_EXPERTS), dtype=np.float64)
    rows = np.arange(n)[:, None, None]
    slots = np.arange(len(pos))[None, :, None]
    out[rows, slots, sel] = 1.0
    return out.reshape(n, len(pos) * NUM_EXPERTS)


def center_logits(g: np.ndarray) -> np.ndarray:
    """q = g - mean(g) over the 8 expert dimensions, per sample."""
    if g.shape[-1] != NUM_EXPERTS:
        raise ValueError(f"expected {NUM_EXPERTS} logits, got {g.shape[-1]}")
    return g - g.mean(axis=-1, keepdims=True)


@dataclass
class OlderBlock:
    """StandardScaler -> PCA(8, randomized) -> StandardScaler, fit on FIT only."""

    scaler_in: StandardScaler
    pca: PCA
    scaler_out: StandardScaler

    @classmethod
    def fit(cls, X_fit: np.ndarray) -> "OlderBlock":
        if X_fit.shape[1] < PCA_COMPONENTS:
            raise AssertionError(
                f"{X_fit.shape[1]} raw dims cannot yield {PCA_COMPONENTS} components")
        scaler_in = StandardScaler()
        Xs = scaler_in.fit_transform(X_fit)
        pca = PCA(n_components=PCA_COMPONENTS, svd_solver="randomized",
                  random_state=PCA_SEED)
        Z = pca.fit_transform(Xs)
        return cls(scaler_in=scaler_in, pca=pca, scaler_out=StandardScaler().fit(Z))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler_out.transform(self.pca.transform(self.scaler_in.transform(X)))

    @property
    def evr(self) -> float:
        return float(self.pca.explained_variance_ratio_.sum())


def feature(z_recent: np.ndarray, z_old: np.ndarray | None) -> np.ndarray:
    """[z_recent ; z_old], or [z_recent ; zeros(8)] for k=1. Always 16 dims."""
    block = np.zeros_like(z_recent) if z_old is None else z_old
    X = np.hstack([z_recent, block])
    if X.shape[1] != FEATURE_DIM:
        raise AssertionError(f"feature has {X.shape[1]} dims, expected {FEATURE_DIM}")
    return X


# ------------------------------------------------------------------------------ probe


def fit_probe(X_fit, Y_fit) -> Ridge:
    model = Ridge(alpha=ALPHA)
    if model.alpha != ALPHA:
        raise AssertionError(f"alpha is {model.alpha}, expected {ALPHA}")
    model.fit(X_fit, Y_fit)
    return model


def score(Y_true, Y_pred) -> float:
    return float(r2_score(Y_true, Y_pred, multioutput="uniform_average"))


def paired_bootstrap(Y_test, pred_1, pred_k, n_resamples: int = N_BOOTSTRAP,
                     seed: int = BOOTSTRAP_SEED) -> Dict[str, float]:
    """95% percentile CI for R2(k) - R2(1), resampling TEST sample indices jointly.

    Both models are scored on the SAME resampled index set every draw, so the paired
    structure is preserved and the CI is about the difference, not about two independent
    estimates.
    """
    rng = np.random.default_rng(seed)
    n = Y_test.shape[0]
    deltas = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)              # jointly resampled indices
        yt = Y_test[idx]
        deltas[b] = score(yt, pred_k[idx]) - score(yt, pred_1[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"ci_low": float(lo), "ci_high": float(hi),
            "mean": float(deltas.mean()), "median": float(np.median(deltas)),
            "frac_above_zero": float((deltas > 0).mean()),
            "n_resamples": int(n_resamples), "seed": int(seed)}


# ------------------------------------------------------------------------------- run


def verdict_conditions(per_target: Dict[int, Dict]) -> Tuple[Dict[str, bool], str]:
    """The six preregistered conditions and the resulting verdict, applied mechanically.

    Three per target: R2 at k=1 positive, Delta4 at least the frozen threshold, and the
    paired-bootstrap CI lower bound above zero. REPLICATED requires all six. There is no
    INCONCLUSIVE outcome for ordinary numerical results.
    """
    conditions: Dict[str, bool] = {}
    for m in TARGET_LAYERS:
        t = per_target[m]
        conditions[f"L{m}_R2_k1_positive"] = bool(t["R2"]["k1"] > 0)
        conditions[f"L{m}_Delta4_at_least_threshold"] = bool(t["Delta4"] >= DELTA4_THRESHOLD)
        conditions[f"L{m}_bootstrap_ci_low_above_zero"] = bool(
            t["bootstrap_Delta4"]["ci_low"] > 0)
    return conditions, (REPLICATED if all(conditions.values()) else NOT_REPLICATED)


def require_checks_passed() -> str:
    """Test P: no formal R2 before every implementation check passes."""
    log("test P gate: running implementation checks before any statistic")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", os.path.join(HERE, "test_rmc.py"), "-q"],
        capture_output=True, text=True, cwd=HERE)
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit("implementation checks failed; no statistics computed")
    log(f"  checks passed: {tail}")
    return tail


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    os.makedirs(ART, exist_ok=True)
    log(f"=== {EXPERIMENT} analysis (CPU) ===")
    gate = require_checks_passed()

    d = np.load(os.path.join(ART, "merged.npz"))
    ids, logits, is_test = d["ids"].astype(np.int64), d["logits"].astype(np.float64), \
        d["is_test"].astype(bool)
    assert int((~is_test).sum()) == N_FIT and int(is_test.sum()) == N_TEST
    fit_i, test_i = ~is_test, is_test
    log(f"merged loaded: n={ids.shape[0]} (FIT {N_FIT}, TEST {N_TEST}), "
        f"{N_LAYERS} layers, {NUM_EXPERTS} experts, top-{TOP_K}")

    per_target: Dict[int, Dict] = {}
    for m in TARGET_LAYERS:
        log(f"--- target Layer {m} ---")

        # Target: native MLP router logits at m, centred, FIT-standardized.
        g = logits[:, m - 1, :]
        q_fit, q_test = center_logits(g[fit_i]), center_logits(g[test_i])
        tsc = StandardScaler().fit(q_fit)                     # FIT only
        Y_fit, Y_test = tsc.transform(q_fit), tsc.transform(q_test)

        # z_recent: S_{m-1}, scaler fitted ONCE per target and reused for every k.
        s_recent_fit = selection_state(ids[fit_i], (m - 1,))
        s_recent_test = selection_state(ids[test_i], (m - 1,))
        rsc = StandardScaler().fit(s_recent_fit)              # fitted once
        z_recent_fit, z_recent_test = rsc.transform(s_recent_fit), rsc.transform(s_recent_test)
        assert z_recent_fit.shape[1] == RECENT_DIM
        log(f"  z_recent from Layer {m - 1}, {RECENT_DIM} dims, one scaler reused for all k")

        r2: Dict[int, float] = {}
        preds: Dict[int, np.ndarray] = {}
        evr: Dict[int, float] = {}
        raw_dims: Dict[int, int] = {}
        for k in K_VALUES:
            old = older_layers(m, k)
            if old:
                Xo_fit = selection_state(ids[fit_i], old)
                Xo_test = selection_state(ids[test_i], old)
                block = OlderBlock.fit(Xo_fit)                # FIT only
                zo_fit, zo_test = block.transform(Xo_fit), block.transform(Xo_test)
                evr[k], raw_dims[k] = block.evr, Xo_fit.shape[1]
            else:
                zo_fit = zo_test = None
                evr[k], raw_dims[k] = float("nan"), 0

            Xf = feature(z_recent_fit, zo_fit)
            Xt = feature(z_recent_test, zo_test)
            model = fit_probe(Xf, Y_fit)
            preds[k] = model.predict(Xt)
            r2[k] = score(Y_test, preds[k])
            log(f"  k={k}: older layers {list(old)} raw {raw_dims[k]:2d} -> "
                f"{PCA_COMPONENTS} dims, feature {Xf.shape[1]}, TEST R2 {r2[k]:+.5f}")

        delta4 = r2[PRIMARY_K] - r2[1]
        boot = paired_bootstrap(Y_test, preds[1], preds[PRIMARY_K])
        log(f"  Delta4 {delta4:+.5f}  95% CI [{boot['ci_low']:+.5f}, {boot['ci_high']:+.5f}]")

        per_target[m] = {
            "R2": {f"k{k}": r2[k] for k in K_VALUES},
            "Delta4": delta4,
            "Delta2": r2[2] - r2[1],
            "Delta8": r2[8] - r2[1],
            "Step2": r2[2] - r2[1],
            "Step4": r2[4] - r2[2],
            "Step8": r2[8] - r2[4],
            "bootstrap_Delta4": boot,
            "older_raw_dims": {f"k{k}": raw_dims[k] for k in K_VALUES},
            "pca_evr": {f"k{k}": evr[k] for k in K_VALUES},
            "history_layers": {f"k{k}": list(range(m - k, m)) for k in K_VALUES},
        }

    conditions, verdict = verdict_conditions(per_target)

    man_sha = sha256_file(os.path.join(ART, "data_manifest.json"))
    with open(os.path.join(ART, "data_manifest.json")) as fh:
        man = json.load(fh)
    with open(os.path.join(ART, "model_provenance.json")) as fh:
        prov = json.load(fh)

    results = {
        "experiment": EXPERIMENT,
        "title": "Routing Memory Cross-Model Pilot",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "implementation": prov["implementation"],
        "implementation_version": prov["implementation_version"],
        "implementation_file_sha256": prov["implementation_file_sha256"],
        "router_studied": prov["router_studied"],
        "moa_router_used": False,
        "dataset_revision": man["dataset_revision"],
        "data_manifest_sha256": man_sha,
        "fingerprints": {"fit": man["fit_fingerprint"], "test": man["test_fingerprint"]},
        "shard_sha256": {f"shard_{s}": sha256_file(os.path.join(ART, f"shard_{s}.npz"))
                         for s in range(4)},
        "n_fit": N_FIT,
        "n_test": N_TEST,
        "target_layers": list(TARGET_LAYERS),
        "k_values": list(K_VALUES),
        "primary_comparison": "k=1 vs k=4",
        "probe": f"sklearn.linear_model.Ridge(alpha={ALPHA}), multi-output",
        "feature_dim": FEATURE_DIM,
        "compression": (f"StandardScaler -> PCA({PCA_COMPONENTS}, randomized, "
                        f"random_state={PCA_SEED}) -> StandardScaler, fit on FIT only"),
        "target": "native MLP router logits, per-sample centred, FIT-standardized",
        "input": "8-d binary native Top-2 MLP selection indicator per layer, identities only",
        "delta4_threshold": DELTA4_THRESHOLD,
        "per_target": {str(m): per_target[m] for m in TARGET_LAYERS},
        "conditions": conditions,
        "verdict": verdict,
        "checks": gate,
        "architecture_built": "none",
        "hyperparameter_search": "none",
        "additional_model": "none",
        "additional_dataset": "none",
        "rescue_modifications": "none",
        "prior_projects_status": "closed, read-only, results unmodified",
        "git_head_at_run": git_head(),
    }
    with open(os.path.join(ART, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    log("--- results ---")
    for m in TARGET_LAYERS:
        t = per_target[m]
        log(f"  Layer {m}: " + "  ".join(f"k={k} {t['R2'][f'k{k}']:+.5f}" for k in K_VALUES))
        log(f"    Delta4 {t['Delta4']:+.5f}  CI [{t['bootstrap_Delta4']['ci_low']:+.5f}, "
            f"{t['bootstrap_Delta4']['ci_high']:+.5f}]  "
            f"Delta2 {t['Delta2']:+.5f}  Delta8 {t['Delta8']:+.5f}")
    for name, ok in conditions.items():
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    log(f"  VERDICT {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
