"""RMO-P0 — Routing Markov-Order Pilot.

Reads the closed EPD-P0 artifact READ-ONLY and asks one question: once the Layer-11
expert-selection pattern is known, does earlier routing history still improve prediction
of Layer-12 router logits?

Five Ridge(alpha=1.0) probes, all on exactly 32 input dimensions, all sharing one fixed
16-d Layer-11 representation. Only the amount of earlier history varies. CPU only; no
model is loaded, no GPU is touched.

Scope is frozen by protocol.md. Run once; do not modify after seeing R² values.
"""

from __future__ import annotations

import hashlib
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

# ------------------------------------------------------------------ frozen constants

EXPERIMENT = "RMO-P0"

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")

# Read-only source. EPD-P0 is CLOSED and is never written to.
EPD_DIR = "/home/h-li/work/expert_provenance_decomposition/artifacts/EPD_P0"
EPD_MANIFEST = os.path.join(EPD_DIR, "data_manifest.json")
EPD_MERGED = os.path.join(EPD_DIR, "merged_raw.npz")
EPD_MANIFEST_SHA256 = "bbbe06fcb451b217de3b03cf75d37921a4e19f555d95d92302b8898d8a3466ba"
EPD_SOURCE_COMMIT = "995d23e"

MODEL_ID = "allenai/OLMoE-1B-7B-0125"
MODEL_REVISION = "9b0c1aa87e34a20052389dce1f0cf01da783f654"

N_FIT = 512
N_TEST = 256
NUM_EXPERTS = 64
TOP_K = 8
N_HISTORY_LAYERS = 11
HISTORY_HUMAN = tuple(range(1, 12))   # Layers 1..11 -> hist_ids positions 0..10
TARGET_LAYER_HUMAN = 12
RECENT_LAYER_HUMAN = 11

PCA_COMPONENTS = 16                   # per block: z_recent and each z_old(k)
PCA_SEED = 20260920
PROBE_DIM = 2 * PCA_COMPONENTS        # every model sees exactly 32 dimensions
ALPHA = 1.0

# Cumulative older-history windows, in human layer numbers. Layer 11 is excluded from
# every window: it is always supplied separately as z_recent.
WINDOWS: Dict[int, Tuple[int, ...]] = {
    2: (10,),
    4: (8, 9, 10),
    8: (4, 5, 6, 7, 8, 9, 10),
    11: tuple(range(1, 11)),
}
K_ORDER = (1, 2, 4, 8, 11)
HISTORY_LABEL = {
    1: "L11 only",
    2: "L10-11",
    4: "L8-11",
    8: "L4-11",
    11: "L1-11",
}

# Descriptive only. Never a preregistered significance threshold.
NOTICEABLE = 0.02

LOCAL_DOMINANT = "LOCAL-DOMINANT"
SHORT_HISTORY = "SHORT-HISTORY"
LONGER_HISTORY_CANDIDATE = "LONGER-HISTORY-CANDIDATE"
NO_CLEAR_PATTERN = "NO-CLEAR-PATTERN"

DATA_NOT_AVAILABLE = "DATA_NOT_AVAILABLE"

REQUIRED_KEYS = ("hist_ids", "g12", "is_test")


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [rmo] {msg}"
    print(line, flush=True)
    with open(os.path.join(ART, "run.log"), "a") as fh:
        fh.write(line + "\n")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------- source loading


def verify_source() -> Dict[str, str]:
    """Test A: the source manifest must hash to EPD-P0's recorded value.

    Also records the hashes of every source file actually read, so the provenance of this
    analysis is pinned to specific bytes rather than to a path.
    """
    if not os.path.isdir(EPD_DIR):
        raise SystemExit(f"{DATA_NOT_AVAILABLE}: {EPD_DIR} missing")
    man_sha = sha256_file(EPD_MANIFEST)
    if man_sha != EPD_MANIFEST_SHA256:
        raise AssertionError(
            f"EPD-P0 manifest hash mismatch: got {man_sha}, expected {EPD_MANIFEST_SHA256}")
    hashes = {"data_manifest.json": man_sha, "merged_raw.npz": sha256_file(EPD_MERGED)}
    for i in range(4):
        p = os.path.join(EPD_DIR, f"shard_{i}.npz")
        if os.path.exists(p):
            hashes[f"shard_{i}.npz"] = sha256_file(p)
    return hashes


@dataclass
class Source:
    hist_ids: np.ndarray   # (768, 11, 8) int, native Top-8 identities, Layers 1..11
    g12: np.ndarray        # (768, 64) float, native Layer-12 router logits
    is_test: np.ndarray    # (768,) bool, EPD-P0's own split
    hashes: Dict[str, str]


def load_source() -> Source:
    """Load the merged EPD-P0 artifact. Stops with DATA_NOT_AVAILABLE if anything is
    missing; never re-extracts."""
    hashes = verify_source()
    d = np.load(EPD_MERGED)
    missing = [k for k in REQUIRED_KEYS if k not in d.files]
    if missing:
        raise SystemExit(f"{DATA_NOT_AVAILABLE}: merged_raw.npz lacks {missing}")

    hist_ids = d["hist_ids"].astype(np.int64)
    g12 = d["g12"].astype(np.float64)
    is_test = d["is_test"].astype(bool)

    if hist_ids.shape[1:] != (N_HISTORY_LAYERS, TOP_K):
        raise SystemExit(f"{DATA_NOT_AVAILABLE}: hist_ids shape {hist_ids.shape}")
    if g12.shape[1] != NUM_EXPERTS:
        raise SystemExit(f"{DATA_NOT_AVAILABLE}: g12 shape {g12.shape}")

    # Test B: the split is EPD-P0's, unchanged and unresampled.
    n_fit, n_test = int((~is_test).sum()), int(is_test.sum())
    if (n_fit, n_test) != (N_FIT, N_TEST):
        raise AssertionError(f"split is {n_fit}/{n_test}, expected {N_FIT}/{N_TEST}")

    return Source(hist_ids=hist_ids, g12=g12, is_test=is_test, hashes=hashes)


# ---------------------------------------------------------------- representations


def selection_indicator(hist_ids: np.ndarray, layers_human: Sequence[int]) -> np.ndarray:
    """EPD-P0's binary selection path, restricted to the requested layers.

    E_l[e] = 1 if expert e is in the native Top-8 at layer l, else 0. No probabilities,
    no logits, no activation content. Result is (n, 64 * len(layers)).
    """
    n = hist_ids.shape[0]
    pos = [int(l) - 1 for l in layers_human]
    if any(p < 0 or p >= N_HISTORY_LAYERS for p in pos):
        raise ValueError(f"layers {list(layers_human)} outside 1..{N_HISTORY_LAYERS}")
    ids = hist_ids[:, pos, :]
    out = np.zeros((n, len(pos), NUM_EXPERTS), dtype=np.float64)
    rows = np.arange(n)[:, None, None]
    slots = np.arange(len(pos))[None, :, None]
    out[rows, slots, ids] = 1.0
    return out.reshape(n, len(pos) * NUM_EXPERTS)


def center_logits(g: np.ndarray) -> np.ndarray:
    """q = g - mean(g) over the 64 expert dimensions, per sample. EPD-P0's transform."""
    if g.shape[-1] != NUM_EXPERTS:
        raise ValueError(f"expected {NUM_EXPERTS} logits, got {g.shape[-1]}")
    return g - g.mean(axis=-1, keepdims=True)


@dataclass
class FrozenBlock:
    """StandardScaler -> PCA(16, randomized) -> StandardScaler, fit on FIT only.

    fit() takes features alone, so no target information and no TEST statistic can enter.
    """

    scaler_in: StandardScaler
    pca: PCA
    scaler_out: StandardScaler

    @classmethod
    def fit(cls, X_fit: np.ndarray) -> "FrozenBlock":
        if X_fit.shape[1] < PCA_COMPONENTS:
            raise AssertionError(
                f"{X_fit.shape[1]} raw dims cannot yield {PCA_COMPONENTS} components")
        scaler_in = StandardScaler()
        Xs = scaler_in.fit_transform(X_fit)
        pca = PCA(n_components=PCA_COMPONENTS, svd_solver="randomized",
                  random_state=PCA_SEED)
        Z = pca.fit_transform(Xs)
        scaler_out = StandardScaler().fit(Z)
        return cls(scaler_in=scaler_in, pca=pca, scaler_out=scaler_out)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler_out.transform(self.pca.transform(self.scaler_in.transform(X)))

    @property
    def evr(self) -> float:
        return float(self.pca.explained_variance_ratio_.sum())


# --------------------------------------------------------------------------- probes


def probe_input(z_recent: np.ndarray, z_old: np.ndarray | None) -> np.ndarray:
    """[z_recent ; z_old] or [z_recent ; zeros(16)] for the k=1 baseline.

    The zero block holds dimensionality at exactly 32 for every model, so the baseline is
    neither advantaged nor penalised by input width.
    """
    block = np.zeros_like(z_recent) if z_old is None else z_old
    X = np.hstack([z_recent, block])
    if X.shape[1] != PROBE_DIM:                       # test G
        raise AssertionError(f"probe input has {X.shape[1]} dims, expected {PROBE_DIM}")
    return X


def fit_and_score(X_fit, Y_fit, X_test, Y_test) -> float:
    """Ridge(alpha=1.0), multi-output, scored once on TEST."""
    model = Ridge(alpha=ALPHA)
    if model.alpha != ALPHA:                          # test I
        raise AssertionError(f"alpha is {model.alpha}, expected {ALPHA}")
    model.fit(X_fit, Y_fit)
    return float(r2_score(Y_test, model.predict(X_test), multioutput="uniform_average"))


def cumulative_gains(r2: Dict[int, float]) -> Dict[str, float]:
    base = r2[1]
    return {f"G_{k}": r2[k] - base for k in (2, 4, 8, 11)}


def stepwise_gains(r2: Dict[int, float]) -> Dict[str, float]:
    return {
        "S_2": r2[2] - r2[1],
        "S_4": r2[4] - r2[2],
        "S_8": r2[8] - r2[4],
        "S_11": r2[11] - r2[8],
    }


def classify(r2: Dict[int, float], noticeable: float = NOTICEABLE) -> Dict[str, object]:
    """Assign one exploratory label. Descriptive; no formal Markov order is claimed.

    The four labels are checked in an order that reflects how much the curve has to show:
    LOCAL-DOMINANT needs the whole curve flat; SHORT-HISTORY needs recent history to help
    and older history to add nothing beyond it; LONGER-HISTORY-CANDIDATE needs the late
    windows to keep improving on the recent ones. Anything else is NO-CLEAR-PATTERN.
    """
    G = cumulative_gains(r2)
    S = stepwise_gains(r2)
    best_recent = max(r2[2], r2[4])
    late_over_recent = max(r2[8], r2[11]) - best_recent

    # The windows are nested, so a later model always has strictly more history available
    # than an earlier one. A material DROP therefore cannot be read as "older history adds
    # little": it means the fixed 16-dim compression is retaining different things at
    # different window sizes, so the curve is not interpretable as a history-depth curve.
    drops = [r2[a] - r2[b] for a, b in zip(K_ORDER, K_ORDER[1:]) if r2[b] < r2[a]]
    max_drop = max(drops) if drops else 0.0
    irregular = max_drop > noticeable

    if irregular:
        label = NO_CLEAR_PATTERN
    # Flat everywhere: no window beats the baseline noticeably.
    elif max(G.values()) < noticeable:
        label = LOCAL_DOMINANT
    # Recent history clearly helps; the later windows add nothing noticeable beyond it.
    elif max(G["G_2"], G["G_4"]) >= noticeable and late_over_recent < noticeable:
        label = SHORT_HISTORY
    # The later windows keep improving materially on the best recent window.
    elif late_over_recent >= noticeable:
        label = LONGER_HISTORY_CANDIDATE
    # Gains exist but no single comparison is noticeable: do not force a direction.
    else:
        label = NO_CLEAR_PATTERN

    return {
        "label": label,
        "noticeable": noticeable,
        "max_cumulative_gain": max(G.values()),
        "best_recent_r2": best_recent,
        "late_over_best_recent": late_over_recent,
        "max_drop_along_curve": max_drop,
        "curve_irregular": irregular,
        "cumulative": G,
        "stepwise": S,
    }


INTERPRETATION = {
    LOCAL_DOMINANT: (
        "Adding routing history older than Layer 11 produces little or no consistent "
        "improvement under this frozen probe. Consistent with the EPD-P0 ID_PATH signal "
        "being largely local routing persistence."),
    SHORT_HISTORY: (
        "Layer 10 and/or Layers 8-10 improve prediction beyond Layer 11 alone, but older "
        "history adds little after that, under this frozen probe."),
    LONGER_HISTORY_CANDIDATE: (
        "Prediction keeps improving materially as Layers 4-10 and 1-10 are added, so "
        "expert-routing trajectories show predictive structure beyond the immediately "
        "preceding layer under this frozen probe. This is not causal memory, a formal "
        "Markov order, or evidence that storing old paths would improve NLL."),
    NO_CLEAR_PATTERN: (
        "The history curve is irregular or the differences are too small to read a "
        "direction from under this frozen probe."),
}


# ----------------------------------------------------------------------------- run


def require_tests_passed() -> str:
    """Test J: no formal statistic is computed until the implementation checks pass.

    The gate runs the test module itself, so it cannot be satisfied by a stale marker.
    """
    log("test J gate: running implementation checks before any statistic")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", os.path.join(HERE, "test_analysis.py"), "-q"],
        capture_output=True, text=True, cwd=HERE,
        env={**os.environ, "RMO_IN_TEST_GATE": "1"})
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
    log(f"=== {EXPERIMENT} — Routing Markov-Order Pilot (CPU only) ===")
    log("no model is loaded, no GPU is used, EPD-P0 is read-only")

    gate = require_tests_passed()

    src = load_source()
    log(f"source verified: EPD-P0 manifest sha256 {src.hashes['data_manifest.json']}")
    log(f"  merged_raw.npz sha256 {src.hashes['merged_raw.npz']}")
    log(f"  FIT {N_FIT} / TEST {N_TEST}, EPD-P0's own split, not resampled")

    fit_i, test_i = ~src.is_test, src.is_test

    # Target: EPD-P0's transform. Centre per sample, standardize with FIT statistics only.
    q_fit = center_logits(src.g12[fit_i])
    q_test = center_logits(src.g12[test_i])
    tsc = StandardScaler().fit(q_fit)                 # test H: FIT only
    Y_fit, Y_test = tsc.transform(q_fit), tsc.transform(q_test)
    log(f"target: native Layer {TARGET_LAYER_HUMAN} router logits, centred, "
        f"FIT-standardized, {Y_fit.shape[1]} dims")

    # z_recent: Layer 11 alone, fitted ONCE and reused by every model.
    e11_fit = selection_indicator(src.hist_ids[fit_i], (RECENT_LAYER_HUMAN,))
    e11_test = selection_indicator(src.hist_ids[test_i], (RECENT_LAYER_HUMAN,))
    recent_block = FrozenBlock.fit(e11_fit)           # fitted once, never refitted
    z_recent_fit = recent_block.transform(e11_fit)
    z_recent_test = recent_block.transform(e11_test)
    log(f"z_recent: Layer {RECENT_LAYER_HUMAN} only, {e11_fit.shape[1]} raw dims -> "
        f"{PCA_COMPONENTS} dims, PCA evr {recent_block.evr:.4f}, fitted once on FIT")

    # z_old(k): each older-history window compressed independently, FIT only.
    z_old_fit: Dict[int, np.ndarray] = {}
    z_old_test: Dict[int, np.ndarray] = {}
    old_evr: Dict[int, float] = {}
    old_raw: Dict[int, int] = {}
    for k, layers in WINDOWS.items():
        Xf = selection_indicator(src.hist_ids[fit_i], layers)
        Xt = selection_indicator(src.hist_ids[test_i], layers)
        block = FrozenBlock.fit(Xf)                   # test F: FIT only
        z_old_fit[k], z_old_test[k] = block.transform(Xf), block.transform(Xt)
        old_evr[k], old_raw[k] = block.evr, Xf.shape[1]
        log(f"z_old({k:2d}): Layers {list(layers)}, {Xf.shape[1]:4d} raw dims -> "
            f"{PCA_COMPONENTS} dims, PCA evr {block.evr:.4f}")

    # Five probes, each on exactly 32 dimensions, identical except for older history.
    r2: Dict[int, float] = {}
    for k in K_ORDER:
        Xf = probe_input(z_recent_fit, None if k == 1 else z_old_fit[k])
        Xt = probe_input(z_recent_test, None if k == 1 else z_old_test[k])
        r2[k] = fit_and_score(Xf, Y_fit, Xt, Y_test)
        log(f"  k={k:2d} ({HISTORY_LABEL[k]:9s}) dim {Xf.shape[1]} "
            f"TEST R2 {r2[k]:+.5f}")

    G, S = cumulative_gains(r2), stepwise_gains(r2)
    detail = classify(r2)
    label = detail["label"]

    results = {
        "experiment": EXPERIMENT,
        "title": "Routing Markov-Order Pilot",
        "exploratory": True,
        "no_formal_verdict": True,
        "post_hoc_on_already_observed_data": True,
        "independent_confirmation": False,
        "new_model_inference_run": False,
        "gpu_used": False,
        "source": {
            "project": "/home/h-li/work/expert_provenance_decomposition",
            "artifact_dir": EPD_DIR,
            "epd_commit": EPD_SOURCE_COMMIT,
            "manifest_sha256_expected": EPD_MANIFEST_SHA256,
            "file_sha256": src.hashes,
            "status": "closed, read-only, unmodified",
        },
        "underlying_capture": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "note": "captured by EPD-P0; no inference was run by RMO-P0",
        },
        "n_fit": N_FIT,
        "n_test": N_TEST,
        "resampled": False,
        "target": (f"native Layer {TARGET_LAYER_HUMAN} router logits, per-sample centred, "
                   "FIT-standardized"),
        "input": "64-d binary native Top-8 selection indicator per layer, identities only",
        "recent_layer": RECENT_LAYER_HUMAN,
        "recent_fitted_once": True,
        "compression": (f"StandardScaler -> PCA({PCA_COMPONENTS}, randomized, "
                        f"random_state={PCA_SEED}) -> StandardScaler, fit on FIT only"),
        "probe": f"sklearn.linear_model.Ridge(alpha={ALPHA}), multi-output",
        "probe_input_dim": PROBE_DIM,
        "windows": {str(k): list(v) for k, v in WINDOWS.items()},
        "older_raw_dims": {str(k): v for k, v in old_raw.items()},
        "pca_explained_variance_ratio": {
            "z_recent": recent_block.evr,
            **{f"z_old_{k}": v for k, v in old_evr.items()},
        },
        "r2_test": {f"R2_{k}": r2[k] for k in K_ORDER},
        "history_label": {str(k): HISTORY_LABEL[k] for k in K_ORDER},
        "cumulative_gains": G,
        "stepwise_gains": S,
        "pattern": label,
        "pattern_detail": detail,
        "interpretation": INTERPRETATION[label],
        "scientific_limit": (
            "Tests predictive dependence beyond Layer 11 under a fixed linear "
            "low-dimensional probe. Does not prove causal memory, a formal Markov order, "
            "that a recurrent router will help, that storing old paths improves NLL, or "
            "that routing should use long-term memory."),
        "checks": gate,
        "architecture_built": "none",
        "hyperparameter_search": "none",
        "rescue_modifications": "none",
        "prior_projects_status": "closed, read-only, results unmodified",
        "git_head_at_run": git_head(),
    }

    with open(os.path.join(ART, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    log("--- results ---")
    for k in K_ORDER:
        log(f"  {HISTORY_LABEL[k]:9s} dim {PROBE_DIM}  R2 {r2[k]:+.5f}")
    log(f"  cumulative  G_2 {G['G_2']:+.5f}  G_4 {G['G_4']:+.5f}  "
        f"G_8 {G['G_8']:+.5f}  G_11 {G['G_11']:+.5f}")
    log(f"  stepwise    S_2 {S['S_2']:+.5f}  S_4 {S['S_4']:+.5f}  "
        f"S_8 {S['S_8']:+.5f}  S_11 {S['S_11']:+.5f}")
    log(f"  PATTERN {label}")
    log(f"  {INTERPRETATION[label]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
