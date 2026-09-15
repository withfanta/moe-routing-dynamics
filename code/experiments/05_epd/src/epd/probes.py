"""The four EPD-P0 probes and the layerwise probes. All Ridge(alpha=1.0), multi-output.

Each probe maps a compressed historical representation to the 64-dimensional future
Layer-12 router state. No alpha search, no nonlinear model, no neural network.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

ALPHA = 1.0


def fit_and_score(X_fit: np.ndarray, Y_fit: np.ndarray,
                  X_test: np.ndarray, Y_test: np.ndarray) -> Tuple[float, np.ndarray]:
    """Fit on FIT, evaluate once on TEST with uniform-average multi-output R²."""
    model = Ridge(alpha=ALPHA)
    model.fit(X_fit, Y_fit)
    pred = model.predict(X_test)
    return float(r2_score(Y_test, pred, multioutput="uniform_average")), pred


def run_main_probes(X_fit: Dict[str, np.ndarray], X_test: Dict[str, np.ndarray],
                    Y_fit: np.ndarray, Y_test: np.ndarray) -> Dict[str, float]:
    """One probe per representation. Returns R² keyed by representation name."""
    out = {}
    for name in X_fit:
        out[name], _ = fit_and_score(X_fit[name], Y_fit, X_test[name], Y_test)
    return out


def descriptive_gaps(r2: Dict[str, float]) -> Dict[str, float]:
    """Descriptive gaps. These are NOT causal effects.

    G_identity_given_content: how much FULL adds over rank-ordered content alone.
    G_content_given_identity: how much FULL adds over the bare selection path alone.
    """
    return {
        "G_full_vs_fused": r2["FULL_PROVENANCE"] - r2["FUSED"],
        "G_identity_given_content": r2["FULL_PROVENANCE"] - r2["CONTENT_RANK"],
        "G_content_given_identity": r2["FULL_PROVENANCE"] - r2["ID_PATH"],
    }
