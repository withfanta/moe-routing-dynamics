"""The two DREV-P0 probes per horizon. Both Ridge(alpha=1.0).

    REAL:     [z_b ; z_r]           128 dims
    SHUFFLED: [z_b ; z_r_shuffled]  128 dims

Same family, same alpha, same preprocessing, same target. They differ only in
whether each rejected vector belongs to its own sample.

No alpha tuning, no MLP, no nonlinear predictor, no other probe.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import Ridge

from . import PROBE_DIM

ALPHA = 1.0


def probe_input(z_b: np.ndarray, z_r: np.ndarray) -> np.ndarray:
    """Concatenate the two 64-dim projected blocks into a 128-dim input."""
    X = np.concatenate([z_b, z_r], axis=1)
    if X.shape[1] != PROBE_DIM:
        raise AssertionError(f"probe input must be {PROBE_DIM} dims, got {X.shape[1]}")
    return X


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination against the test-set mean."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def fit_and_evaluate(
    X_fit: np.ndarray, y_fit: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Fit Ridge(alpha=1.0) on the fit split, evaluate once on the test split.

    The projected features are already centred and scaled by the frozen PCA
    pipeline, so no further standardization is applied here. The target is not
    standardized.
    """
    model = Ridge(alpha=ALPHA)
    model.fit(X_fit, y_fit)
    pred = model.predict(X_test)
    return r2_score(y_test, pred), pred


def run_probe_pair(
    z_b_fit: np.ndarray,
    z_r_fit: np.ndarray,
    z_r_fit_shuf: np.ndarray,
    G_fit: np.ndarray,
    z_b_test: np.ndarray,
    z_r_test: np.ndarray,
    z_r_test_shuf: np.ndarray,
    G_test: np.ndarray,
) -> Dict[str, float]:
    """Fit the REAL and SHUFFLED probes for one horizon."""
    X_real_fit = probe_input(z_b_fit, z_r_fit)
    X_real_test = probe_input(z_b_test, z_r_test)
    X_shuf_fit = probe_input(z_b_fit, z_r_fit_shuf)
    X_shuf_test = probe_input(z_b_test, z_r_test_shuf)

    assert X_real_fit.shape == X_shuf_fit.shape
    assert X_real_test.shape == X_shuf_test.shape

    r2_real, _ = fit_and_evaluate(X_real_fit, G_fit, X_real_test, G_test)
    r2_shuf, _ = fit_and_evaluate(X_shuf_fit, G_fit, X_shuf_test, G_test)

    return {
        "R2_real": r2_real,
        "R2_shuffle": r2_shuf,
        "D": r2_real - r2_shuf,
        "probe_dim": int(X_real_fit.shape[1]),
        "n_fit": int(X_real_fit.shape[0]),
        "n_test": int(X_real_test.shape[0]),
        "alpha": ALPHA,
    }
