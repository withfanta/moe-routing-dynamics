"""The three EIPC-P0 probes per target. All Ridge(alpha=1.0), multi-output.

Each probe maps a 64-dimensional compressed history to the 64-dimensional future
router state. Ridge supports multi-output regression natively.

No alpha search, no MLP, no nonlinear model, no classifier, no other probe.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from . import PCA_COMPONENTS

ALPHA = 1.0


def fit_and_evaluate(
    X_fit: np.ndarray, Y_fit: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Fit on FIT, evaluate once on TEST with uniform-average multi-output R²."""
    if X_fit.shape[1] != PCA_COMPONENTS:
        raise AssertionError(f"probe input must be {PCA_COMPONENTS}-d, got {X_fit.shape[1]}")
    if X_test.shape[1] != PCA_COMPONENTS:
        raise AssertionError(f"probe input must be {PCA_COMPONENTS}-d, got {X_test.shape[1]}")

    model = Ridge(alpha=ALPHA)
    model.fit(X_fit, Y_fit)
    pred = model.predict(X_test)
    score = float(r2_score(Y_test, pred, multioutput="uniform_average"))
    return score, pred


def run_target_probes(
    X_fit: Dict[str, np.ndarray],
    X_test: Dict[str, np.ndarray],
    Y_fit: np.ndarray,
    Y_test: np.ndarray,
) -> Dict[str, float]:
    """Fit the three probes for one target layer and return R², A and B.

    ``X_fit``/``X_test`` are keyed by representation name and each already
    compressed to 64 dimensions.
    """
    r2 = {}
    for rep in ("FUSED", "EXPERT_IDENTITY", "SHUFFLED_IDENTITY"):
        r2[rep], _ = fit_and_evaluate(X_fit[rep], Y_fit, X_test[rep], Y_test)

    return {
        "R2_fused": r2["FUSED"],
        "R2_identity": r2["EXPERT_IDENTITY"],
        "R2_shuffled": r2["SHUFFLED_IDENTITY"],
        "A": r2["EXPERT_IDENTITY"] - r2["FUSED"],
        "B": r2["EXPERT_IDENTITY"] - r2["SHUFFLED_IDENTITY"],
        "alpha": ALPHA,
        "input_dim": int(X_fit["FUSED"].shape[1]),
        "target_dim": int(Y_fit.shape[1]),
        "n_fit": int(X_fit["FUSED"].shape[0]),
        "n_test": int(X_test["FUSED"].shape[0]),
    }
