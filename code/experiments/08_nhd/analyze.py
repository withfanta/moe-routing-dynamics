"""NHD-P0 analysis: Part A nonlinear comparison, Part B cross-fitted residual test.

Reads the frozen RMC-P0 arrays read-only, builds raw uncompressed R and H, trains the five
Part-A models per target over three fixed seeds, then runs the cross-fitted residual probe.
Gates itself on the protocol checks (test S) before any TEST statistic is computed.

CPU only. No new model inference.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from nhd import (
    ALPHA, ART, CROSSFIT_SEED, EXPERIMENT, FIT_SPLIT_SEED, FULL_DIM, HERE, HIDDEN,
    HISTORY_DIM, HISTORY_OFFSETS, INIT_SEEDS, JOINT_ONLY, LR, MATCHED_WIDTH, MAX_EPOCHS,
    NEAR_ZERO, NONLINEAR_ACCESSIBILITY, NO_CLEAR_PATTERN, NUM_EXPERTS, N_FIT, N_FOLD,
    N_LAYERS, N_TEST, N_TRAIN, N_VAL, PATIENCE, PERMUTE_SEED, RECENT_DIM,
    RESIDUAL_HISTORY_VALUE, SOURCE_COMMIT, SOURCE_EXPERIMENT, TARGET_LAYERS, TOP_K,
    WEIGHT_DECAY, history_layers, load_source, matched_width, recent_layer, verify_source,
)


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [nhd] {msg}"
    print(line, flush=True)
    with open(os.path.join(ART, "analyze.log"), "a") as fh:
        fh.write(line + "\n")


# --------------------------------------------------------------------- representations


def selection_state(ids: np.ndarray, layers_human: Sequence[int]) -> np.ndarray:
    """Raw binary Top-2 MLP selection indicator, concatenated over the given layers.

    Identities only: no gate value, no probability, no rank order. No compression.
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
    """q = g - mean(g) over the 8 expert dimensions, per sample. Same as RMC-P0."""
    if g.shape[-1] != NUM_EXPERTS:
        raise ValueError(f"expected {NUM_EXPERTS} logits, got {g.shape[-1]}")
    return g - g.mean(axis=-1, keepdims=True)


def build_variables(ids: np.ndarray, logits: np.ndarray, is_test: np.ndarray,
                    target: int) -> Dict[str, np.ndarray]:
    """R, H, [H;R] and the FIT-standardized centred target, for one target layer."""
    fit_i, test_i = ~is_test, is_test
    m1 = recent_layer(target)
    hl = history_layers(target)

    R_fit = selection_state(ids[fit_i], (m1,))
    R_test = selection_state(ids[test_i], (m1,))
    H_fit = selection_state(ids[fit_i], hl)
    H_test = selection_state(ids[test_i], hl)
    assert R_fit.shape[1] == RECENT_DIM and H_fit.shape[1] == HISTORY_DIM

    g = logits[:, target - 1, :]
    tsc = StandardScaler().fit(center_logits(g[fit_i]))          # FIT only
    Y_fit = tsc.transform(center_logits(g[fit_i]))
    Y_test = tsc.transform(center_logits(g[test_i]))

    full_fit = np.hstack([H_fit, R_fit])
    full_test = np.hstack([H_test, R_test])
    assert full_fit.shape[1] == FULL_DIM == 32

    return {"R_fit": R_fit, "R_test": R_test, "H_fit": H_fit, "H_test": H_test,
            "full_fit": full_fit, "full_test": full_test,
            "Y_fit": Y_fit, "Y_test": Y_test,
            "recent_layer": m1, "history_layers": list(hl)}


def score(Y_true, Y_pred) -> float:
    return float(r2_score(Y_true, Y_pred, multioutput="uniform_average"))


# ------------------------------------------------------------------------------- models


def make_mlp(in_dim: int, hidden: int, out_dim: int, seed: int) -> nn.Module:
    """The one frozen architecture family: Linear -> GELU -> Linear, biases on."""
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fit_split(n: int, seed: int, n_train: int) -> Tuple[np.ndarray, np.ndarray]:
    """One deterministic split, frozen by seed. Used for early stopping only."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return perm[:n_train], perm[n_train:]


def train_mlp(X: np.ndarray, Y: np.ndarray, in_dim: int, hidden: int, out_dim: int,
              seed: int, split_seed: int = FIT_SPLIT_SEED,
              n_train: int | None = None) -> Tuple[nn.Module, Dict]:
    """Train on a FIT-side array with early stopping on a held-in validation slice.

    TEST never enters: the validation slice is carved out of the array passed in, which is
    always FIT or a FIT fold. Restores the best-validation checkpoint.
    """
    n = X.shape[0]
    if n_train is None:
        n_train = int(round(n * N_TRAIN / (N_TRAIN + N_VAL)))
    tr, va = fit_split(n, split_seed, n_train)

    Xt = torch.tensor(X[tr], dtype=torch.float32)
    Yt = torch.tensor(Y[tr], dtype=torch.float32)
    Xv = torch.tensor(X[va], dtype=torch.float32)
    Yv = torch.tensor(Y[va], dtype=torch.float32)

    model = make_mlp(in_dim, hidden, out_dim, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best_val, best_state, best_epoch, since = float("inf"), None, -1, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        loss_fn(model(Xt), Yt).backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(Xv), Yv))
        if val < best_val:
            best_val, best_epoch, since = val, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val_mse": best_val, "best_epoch": best_epoch,
                   "epochs_run": epoch + 1, "n_train": int(len(tr)), "n_val": int(len(va)),
                   "n_params": n_params(model)}


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy().astype(np.float64)


def ridge(X_fit, Y_fit, X_test) -> np.ndarray:
    model = Ridge(alpha=ALPHA)
    model.fit(X_fit, Y_fit)
    return model.predict(X_test)


# --------------------------------------------------------------------- Part A per target


def part_a(v: Dict[str, np.ndarray]) -> Dict:
    """Five models: two Ridge references and three MLPs over three fixed seeds."""
    out: Dict = {"linear": {}, "nonlinear": {}, "training": {}}

    out["linear"]["LINEAR_RECENT"] = score(
        v["Y_test"], ridge(v["R_fit"], v["Y_fit"], v["R_test"]))
    out["linear"]["LINEAR_HISTORY"] = score(
        v["Y_test"], ridge(v["full_fit"], v["Y_fit"], v["full_test"]))
    out["linear"]["linear_gain"] = (out["linear"]["LINEAR_HISTORY"]
                                    - out["linear"]["LINEAR_RECENT"])
    log(f"    LINEAR_RECENT  {out['linear']['LINEAR_RECENT']:+.5f}   "
        f"LINEAR_HISTORY {out['linear']['LINEAR_HISTORY']:+.5f}   "
        f"gain {out['linear']['linear_gain']:+.5f}")

    specs = {
        "MLP_RECENT": ("R_fit", "R_test", RECENT_DIM, HIDDEN),
        "MLP_RECENT_MATCHED": ("R_fit", "R_test", RECENT_DIM, MATCHED_WIDTH),
        "MLP_HISTORY": ("full_fit", "full_test", FULL_DIM, HIDDEN),
    }
    for name, (fk, tk, in_dim, hidden) in specs.items():
        out["nonlinear"][name] = {}
        out["training"][name] = {}
        for seed in INIT_SEEDS:
            model, info = train_mlp(v[fk], v["Y_fit"], in_dim, hidden, NUM_EXPERTS, seed)
            out["nonlinear"][name][str(seed)] = score(v["Y_test"], predict(model, v[tk]))
            out["training"][name][str(seed)] = info
        vals = [out["nonlinear"][name][str(s)] for s in INIT_SEEDS]
        out["nonlinear"][name]["mean"] = float(np.mean(vals))
        out["nonlinear"][name]["std"] = float(np.std(vals))
        log(f"    {name:<20} " + "  ".join(f"seed {s} {out['nonlinear'][name][str(s)]:+.5f}"
                                           for s in INIT_SEEDS)
            + f"  mean {out['nonlinear'][name]['mean']:+.5f}"
            + f"  ({out['training'][name][str(INIT_SEEDS[0])]['n_params']} params)")

    nl = out["nonlinear"]
    out["A"] = nl["MLP_RECENT"]["mean"] - out["linear"]["LINEAR_RECENT"]
    out["B"] = nl["MLP_HISTORY"]["mean"] - nl["MLP_RECENT"]["mean"]
    out["B_matched"] = nl["MLP_HISTORY"]["mean"] - nl["MLP_RECENT_MATCHED"]["mean"]
    out["A_per_seed"] = {str(s): nl["MLP_RECENT"][str(s)] - out["linear"]["LINEAR_RECENT"]
                         for s in INIT_SEEDS}
    out["B_per_seed"] = {str(s): nl["MLP_HISTORY"][str(s)] - nl["MLP_RECENT"][str(s)]
                         for s in INIT_SEEDS}
    out["B_matched_per_seed"] = {
        str(s): nl["MLP_HISTORY"][str(s)] - nl["MLP_RECENT_MATCHED"][str(s)]
        for s in INIT_SEEDS}
    log(f"    A {out['A']:+.5f}   B {out['B']:+.5f}   B_matched {out['B_matched']:+.5f}")
    return out


# --------------------------------------------------- Part B cross-fitted residual test


def crossfit_residuals(X: np.ndarray, Y: np.ndarray, out_dim: int,
                       seed: int = INIT_SEEDS[0]) -> Tuple[np.ndarray, Dict]:
    """Two-fold cross-fitted residuals: each sample is predicted by the other fold's model.

    No sample is ever residualized by a model that saw it in training.
    """
    n = X.shape[0]
    rng = np.random.default_rng(CROSSFIT_SEED)
    perm = rng.permutation(n)
    fold_a, fold_b = perm[:N_FOLD], perm[N_FOLD:]
    assert len(fold_a) == len(fold_b) == N_FOLD
    assert set(fold_a.tolist()).isdisjoint(fold_b.tolist())

    resid = np.zeros((n, out_dim), dtype=np.float64)
    info = {}
    for label, train_idx, apply_idx in (("f_A", fold_a, fold_b), ("f_B", fold_b, fold_a)):
        model, meta = train_mlp(X[train_idx], Y[train_idx], X.shape[1], HIDDEN, out_dim,
                                seed, n_train=int(round(N_FOLD * N_TRAIN
                                                        / (N_TRAIN + N_VAL))))
        resid[apply_idx] = Y[apply_idx] - predict(model, X[apply_idx])
        info[label] = {**meta, "trained_on": label[-1],
                       "n_applied": int(len(apply_idx))}
    return resid, {"folds": {"a": fold_a.tolist(), "b": fold_b.tolist()}, "models": info}


def part_b(v: Dict[str, np.ndarray]) -> Dict:
    """Cross-fitted FIT residual probe, evaluated on TEST residuals from FIT-only models."""
    eps_Y_fit, info_y = crossfit_residuals(v["R_fit"], v["Y_fit"], NUM_EXPERTS)
    eps_H_fit, info_h = crossfit_residuals(v["R_fit"], v["H_fit"], HISTORY_DIM)
    assert eps_Y_fit.shape == (N_FIT, NUM_EXPERTS)
    assert eps_H_fit.shape == (N_FIT, HISTORY_DIM)

    probe = Ridge(alpha=ALPHA)
    probe.fit(eps_H_fit, eps_Y_fit)          # cross-fitted FIT residuals only

    # TEST residuals come only from models trained on all of FIT.
    f_full, meta_f = train_mlp(v["R_fit"], v["Y_fit"], RECENT_DIM, HIDDEN, NUM_EXPERTS,
                               INIT_SEEDS[0])
    g_full, meta_g = train_mlp(v["R_fit"], v["H_fit"], RECENT_DIM, HIDDEN, HISTORY_DIM,
                               INIT_SEEDS[0])
    eps_Y_test = v["Y_test"] - predict(f_full, v["R_test"])
    eps_H_test = v["H_test"] - predict(g_full, v["R_test"])

    C = score(eps_Y_test, probe.predict(eps_H_test))

    # Descriptive permutation control, never tuned against.
    rng = np.random.default_rng(PERMUTE_SEED)
    perm = rng.permutation(eps_H_test.shape[0])
    C_perm = score(eps_Y_test, probe.predict(eps_H_test[perm]))

    log(f"    residual TEST R2 {C:+.5f}   permuted-history {C_perm:+.5f}")
    return {"residual_R2": C, "permuted_residual_R2": C_perm,
            "crossfit_seed": CROSSFIT_SEED, "permute_seed": PERMUTE_SEED,
            "eps_Y_fit_std": float(eps_Y_fit.std()),
            "eps_H_fit_std": float(eps_H_fit.std()),
            "f_full": meta_f, "g_full": meta_g,
            "crossfit_Y": info_y["models"], "crossfit_H": info_h["models"],
            "folds_disjoint": True}


# ---------------------------------------------------------------------- classification


def classify(per_target: Dict[int, Dict]) -> Tuple[str, Dict]:
    """The one descriptive label, on criteria frozen in protocol.md section 10.

    Read of the prose, fixed before results: "near zero" is |value| < 0.02; "greatly
    improves" is A >= 0.02; "consistently exceeds" is B and B_matched both >= 0.02 in the
    mean AND positive for all three seeds; "stays positive" is residual R2 > 0. The two
    targets must agree, and a target whose seeds disagree in sign on B is unstable.
    """
    facts = {}
    labels = []
    for m in TARGET_LAYERS:
        t = per_target[m]
        a = t["part_a"]
        A, B, Bm = a["A"], a["B"], a["B_matched"]
        C = t["part_b"]["residual_R2"]
        b_seeds = list(a["B_per_seed"].values())
        bm_seeds = list(a["B_matched_per_seed"].values())

        f = {
            "A": A, "B": B, "B_matched": Bm, "residual_R2": C,
            "nonlinear_improves": bool(A >= NEAR_ZERO),
            "B_near_zero": bool(abs(B) < NEAR_ZERO),
            "residual_near_zero": bool(abs(C) < NEAR_ZERO),
            "residual_positive": bool(C > 0),
            "B_seeds_all_positive": bool(all(x > 0 for x in b_seeds)),
            "B_matched_seeds_all_positive": bool(all(x > 0 for x in bm_seeds)),
            "B_seeds_agree_in_sign": bool(all(x > 0 for x in b_seeds)
                                          or all(x < 0 for x in b_seeds)),
            "consistently_exceeds": bool(B >= NEAR_ZERO and Bm >= NEAR_ZERO
                                         and all(x > 0 for x in b_seeds)
                                         and all(x > 0 for x in bm_seeds)),
        }

        if not f["B_seeds_agree_in_sign"]:
            label = NO_CLEAR_PATTERN
        elif f["nonlinear_improves"] and f["B_near_zero"] and f["residual_near_zero"]:
            label = NONLINEAR_ACCESSIBILITY
        elif f["consistently_exceeds"] and f["residual_positive"] \
                and not f["residual_near_zero"]:
            label = RESIDUAL_HISTORY_VALUE
        elif B >= NEAR_ZERO and f["residual_near_zero"]:
            label = JOINT_ONLY
        else:
            label = NO_CLEAR_PATTERN

        f["label"] = label
        facts[f"L{m}"] = f
        labels.append(label)

    overall = labels[0] if len(set(labels)) == 1 else NO_CLEAR_PATTERN
    facts["targets_agree"] = bool(len(set(labels)) == 1)
    facts["per_target_labels"] = {f"L{m}": l for m, l in zip(TARGET_LAYERS, labels)}
    return overall, facts


# ------------------------------------------------------------------------------- run


def require_checks_passed() -> str:
    """Test S: no TEST metric before every implementation check passes."""
    log("test S gate: running implementation checks before any TEST statistic")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", os.path.join(HERE, "test_nhd.py"), "-q"],
        capture_output=True, text=True, cwd=HERE)
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit("implementation checks failed; no TEST statistics computed")
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
    log(f"=== {EXPERIMENT} analysis (CPU, no new inference) ===")
    gate = require_checks_passed()

    src = verify_source()
    log(f"source {SOURCE_EXPERIMENT} clean at {src['source_head']}, hashes match pins")
    ids, logits, is_test = load_source()
    torch.set_num_threads(4)

    w, mp, hp = matched_width()
    log(f"MLP_RECENT_MATCHED width {w} ({mp} params vs MLP_HISTORY {hp})")

    per_target: Dict[int, Dict] = {}
    for m in TARGET_LAYERS:
        v = build_variables(ids, logits, is_test, m)
        log(f"--- target Layer {m}: R = S_{v['recent_layer']}, "
            f"H = S_{v['history_layers']}, [H;R] dim {v['full_fit'].shape[1]} ---")
        a = part_a(v)
        b = part_b(v)
        per_target[m] = {"part_a": a, "part_b": b,
                         "recent_layer": v["recent_layer"],
                         "history_layers": v["history_layers"],
                         "dims": {"R": RECENT_DIM, "H": HISTORY_DIM, "full": FULL_DIM}}

    label, facts = classify(per_target)

    results = {
        "experiment": EXPERIMENT,
        "title": "Nonlinear History Decoding Pilot",
        "source_experiment": SOURCE_EXPERIMENT,
        "source_commit_pinned": SOURCE_COMMIT,
        "source_verification": src,
        "new_model_inference": False,
        "gpu_used": False,
        "target_layers": list(TARGET_LAYERS),
        "variables": {"R": "S_{m-1} raw binary Top-2 MLP selection, 8 dims",
                      "H": "[S_{m-4}, S_{m-3}, S_{m-2}] raw, 24 dims",
                      "full": "[H;R], 32 dims",
                      "pca_used": False, "compression_used": False,
                      "moa_router_used": False,
                      "router_probabilities_used": False,
                      "expert_activations_used": False},
        "target": "native MLP router logits, per-sample centred, FIT-standardized (as RMC-P0)",
        "n_fit": N_FIT, "n_test": N_TEST,
        "training": {"optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY,
                     "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                     "fit_split": {"train": N_TRAIN, "val": N_VAL, "seed": FIT_SPLIT_SEED},
                     "init_seeds": list(INIT_SEEDS),
                     "early_stopping_on": "validation MSE inside FIT",
                     "test_used_for_early_stopping": False},
        "architectures": {
            "MLP_RECENT": f"Linear({RECENT_DIM},{HIDDEN}) -> GELU -> Linear({HIDDEN},{NUM_EXPERTS})",
            "MLP_HISTORY": f"Linear({FULL_DIM},{HIDDEN}) -> GELU -> Linear({HIDDEN},{NUM_EXPERTS})",
            "MLP_RECENT_MATCHED": f"Linear({RECENT_DIM},{w}) -> GELU -> Linear({w},{NUM_EXPERTS})",
            "residual_H_model": f"Linear({RECENT_DIM},{HIDDEN}) -> GELU -> Linear({HIDDEN},{HISTORY_DIM})",
            "matched_width": w, "matched_params": mp, "history_params": hp,
            "width_chosen_mechanically": True, "width_sweep": False},
        "probe": f"sklearn.linear_model.Ridge(alpha={ALPHA})",
        "near_zero_threshold": NEAR_ZERO,
        "per_target": {str(m): per_target[m] for m in TARGET_LAYERS},
        "classification": label,
        "classification_facts": facts,
        "checks": gate,
        "rescue_modifications": "none",
        "prior_projects_status": "closed, read-only, results unmodified",
        "git_head_at_run": git_head(),
    }
    with open(os.path.join(ART, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    log("--- summary ---")
    for m in TARGET_LAYERS:
        t = per_target[m]
        a, b = t["part_a"], t["part_b"]
        log(f"  Layer {m}: linear recent {a['linear']['LINEAR_RECENT']:+.5f} "
            f"history {a['linear']['LINEAR_HISTORY']:+.5f} "
            f"gain {a['linear']['linear_gain']:+.5f}")
        log(f"    MLP means: recent {a['nonlinear']['MLP_RECENT']['mean']:+.5f}  "
            f"matched {a['nonlinear']['MLP_RECENT_MATCHED']['mean']:+.5f}  "
            f"history {a['nonlinear']['MLP_HISTORY']['mean']:+.5f}")
        log(f"    A {a['A']:+.5f}  B {a['B']:+.5f}  B_matched {a['B_matched']:+.5f}  "
            f"C {b['residual_R2']:+.5f}  (permuted {b['permuted_residual_R2']:+.5f})")
        log(f"    label {facts[f'L{m}']['label']}")
    log(f"  CLASSIFICATION {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
