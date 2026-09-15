"""The two REDV-V1 probes. Exactly two, both Ridge(alpha=1.0).

Probe A (baseline):   input b_{l,t} = [h_{l,t} ; g_{l,t}]            dim 2112
Probe B (augmented):  input [b_{l,t} ; r_{l,t}]                      dim 4160

Target for both: G_{l+1,t}.

Features are standardized with validation-set mean and standard deviation only.
Zero-variance dimensions are left at zero. The target is not standardized.

No MLP, no nonlinear model, no hyperparameter search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import Ridge

ALPHA = 1.0


def baseline_features(h: np.ndarray, g: np.ndarray) -> np.ndarray:
    """b = [h ; g]. Dimension 2048 + 64 = 2112."""
    return np.concatenate([h, g], axis=1)


def augmented_features(h: np.ndarray, g: np.ndarray, r: np.ndarray) -> np.ndarray:
    """[b ; r]. Differs from the baseline ONLY by access to rejected evidence."""
    return np.concatenate([h, g, r], axis=1)


@dataclass
class Standardizer:
    """Validation-set standardizer. Zero-variance dims stay at zero."""

    mean: np.ndarray
    std: np.ndarray
    nonzero: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        nonzero = std > 0.0
        return cls(mean=mean, std=std, nonzero=nonzero)

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros_like(X, dtype=np.float64)
        cols = self.nonzero
        out[:, cols] = (X[:, cols] - self.mean[cols]) / self.std[cols]
        return out


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Standard coefficient of determination against the test-set mean."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def fit_and_evaluate(
    X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Standardize on validation, fit Ridge(alpha=1.0), evaluate once on test."""
    scaler = Standardizer.fit(X_val)
    model = Ridge(alpha=ALPHA)
    model.fit(scaler.transform(X_val), y_val)
    pred = model.predict(scaler.transform(X_test))
    return r2_score(y_test, pred), pred


def run_probe_pair(
    val: Dict[str, np.ndarray], test: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """Fit both probes for one transition and return held-out R² and Delta_R2.

    ``val``/``test`` each carry ``h``, ``g``, ``r`` and target ``G``.
    """
    Xb_val = baseline_features(val["h"], val["g"])
    Xb_test = baseline_features(test["h"], test["g"])
    Xa_val = augmented_features(val["h"], val["g"], val["r"])
    Xa_test = augmented_features(test["h"], test["g"], test["r"])

    r2_base, _ = fit_and_evaluate(Xb_val, val["G"], Xb_test, test["G"])
    r2_aug, _ = fit_and_evaluate(Xa_val, val["G"], Xa_test, test["G"])

    return {
        "r2_baseline": r2_base,
        "r2_augmented": r2_aug,
        "delta_r2": r2_aug - r2_base,
        "baseline_dim": int(Xb_val.shape[1]),
        "augmented_dim": int(Xa_val.shape[1]),
    }
