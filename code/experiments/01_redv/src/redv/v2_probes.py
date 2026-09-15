"""REDV-V2 matched-control probes and the final judgement rule.

Three probes per transition, all Ridge(alpha=1.0), all standardized with fit-set
statistics only:

    A baseline   b                    2112 dims
    B real       [b ; r]              4160 dims
    C shuffled   [b ; r_perm(i)]      4160 dims

B and C are dimension-matched and share preprocessing logic exactly, so the only
difference between them is whether each rejected vector belongs to its own sample.

    D = R2_real - R2_shuffle     (primary matched-control statistic)

SUPPORTED requires R2_real > 0 for all three transitions, D > 0 for all three, and
mean(D) >= +0.02. Anything else is NOT_SUPPORTED.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .probes import ALPHA, augmented_features, baseline_features, fit_and_evaluate

SUPPORTED = "SUPPORTED"
NOT_SUPPORTED = "NOT_SUPPORTED"
INCONCLUSIVE = "INCONCLUSIVE"

MEAN_D_THRESHOLD = 0.02


def shuffled_rejected(r: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Row-permute the rejected matrix. Exactly a reordering, no other change."""
    if perm.shape[0] != r.shape[0]:
        raise ValueError(f"permutation length {perm.shape[0]} != n rows {r.shape[0]}")
    if np.any(perm == np.arange(len(perm))):
        raise AssertionError("control permutation must be a derangement")
    return r[perm]


def run_v2_probe_triple(
    fit: Dict[str, np.ndarray],
    test: Dict[str, np.ndarray],
    perm_fit: np.ndarray,
    perm_test: np.ndarray,
) -> Dict[str, float]:
    """Fit the three probes for one transition and return held-out R² values.

    ``fit``/``test`` each carry ``h``, ``g``, ``r`` and target ``G``.
    """
    Xb_fit = baseline_features(fit["h"], fit["g"])
    Xb_test = baseline_features(test["h"], test["g"])

    Xr_fit = augmented_features(fit["h"], fit["g"], fit["r"])
    Xr_test = augmented_features(test["h"], test["g"], test["r"])

    Xs_fit = augmented_features(fit["h"], fit["g"], shuffled_rejected(fit["r"], perm_fit))
    Xs_test = augmented_features(test["h"], test["g"], shuffled_rejected(test["r"], perm_test))

    # Equal dimension is a protocol requirement, not an incidental property.
    assert Xr_fit.shape == Xs_fit.shape and Xr_fit.shape[1] == 4160
    assert Xr_test.shape == Xs_test.shape and Xr_test.shape[1] == 4160

    r2_base, _ = fit_and_evaluate(Xb_fit, fit["G"], Xb_test, test["G"])
    r2_real, _ = fit_and_evaluate(Xr_fit, fit["G"], Xr_test, test["G"])
    r2_shuf, _ = fit_and_evaluate(Xs_fit, fit["G"], Xs_test, test["G"])

    return {
        "r2_baseline": r2_base,
        "r2_real": r2_real,
        "r2_shuffle": r2_shuf,
        "D": r2_real - r2_shuf,
        "gain_over_baseline": r2_real - r2_base,
        "baseline_dim": int(Xb_fit.shape[1]),
        "augmented_dim": int(Xr_fit.shape[1]),
        "alpha": ALPHA,
    }


def v2_verdict(r2_reals: Sequence[float], ds: Sequence[float]) -> str:
    """Mechanical application of the pre-registered rule."""
    if len(r2_reals) != 3 or len(ds) != 3:
        raise ValueError("expected exactly 3 transitions")
    if not all(v > 0.0 for v in r2_reals):
        return NOT_SUPPORTED
    if not all(d > 0.0 for d in ds):
        return NOT_SUPPORTED
    if (sum(ds) / 3.0) < MEAN_D_THRESHOLD:
        return NOT_SUPPORTED
    return SUPPORTED


def failed_conditions(r2_reals: Sequence[float], ds: Sequence[float]) -> List[str]:
    """Which of the three conditions failed. Reporting aid, not a second rule."""
    out = []
    if not all(v > 0.0 for v in r2_reals):
        bad = [i for i, v in enumerate(r2_reals) if v <= 0.0]
        out.append(f"condition 1 failed: R2_real <= 0 at transition index {bad}")
    if not all(d > 0.0 for d in ds):
        bad = [i for i, d in enumerate(ds) if d <= 0.0]
        out.append(f"condition 2 failed: D <= 0 at transition index {bad}")
    mean_d = sum(ds) / 3.0
    if mean_d < MEAN_D_THRESHOLD:
        out.append(f"condition 3 failed: mean(D) = {mean_d:+.5f} < +0.02")
    return out


def v2_decision_sentence(verdict: str) -> str:
    if verdict == SUPPORTED:
        return (
            "The prerequisite phenomenon is supported under the frozen "
            "OLMoE/WikiText setting using a dimension-matched shuffled control."
        )
    if verdict == NOT_SUPPORTED:
        return (
            "REDV-V2 does not support sample-specific delayed predictive value of "
            "rejected near-miss expert evidence under the frozen OLMoE/WikiText "
            "setting with a dimension-matched shuffled control. The "
            "rejected-expert delayed-value direction is stopped."
        )
    return "REDV-V2 is inconclusive due to a technical failure."


def v2_stopping_note(verdict: str) -> str:
    if verdict == SUPPORTED:
        return (
            "REDV-V2 stops here. No attention, memory, reconsideration, or Routing "
            "Surprise mechanism is implemented; those require explicit user "
            "authorization."
        )
    return (
        "The entire rejected-expert delayed-value direction is stopped. No REDV-V3. "
        "No sample-size increase, no new dataset or model, no alpha change, no PCA, "
        "no MLP, no rank change, no added layers, no additional shuffle seeds, no "
        "subgroup reinterpretation. The negative result is preserved."
    )
