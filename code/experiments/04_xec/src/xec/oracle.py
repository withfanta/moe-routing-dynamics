"""Oracle action labels for XEC-P0.

    a_i* = argmin_a L_i(a)

Ties break to the lowest action index, so native action 0 wins exact ties.

The oracle is a supervised signal for TRAIN and VALIDATION only. The TEST oracle may be
computed solely after all policy training is frozen, and only for descriptive headroom
reporting; it must never affect model selection. ``load_test_oracle_guarded`` enforces
that ordering at the file level.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

from . import N_ACTIONS


def oracle_action(action_nlls: np.ndarray) -> np.ndarray:
    """Lowest-index argmin over the five action NLLs."""
    if action_nlls.ndim != 2 or action_nlls.shape[1] != N_ACTIONS:
        raise ValueError(f"expected (n, {N_ACTIONS}), got {action_nlls.shape}")
    return np.argmin(action_nlls, axis=1).astype(np.int64)


def oracle_nll(action_nlls: np.ndarray) -> np.ndarray:
    """NLL achieved by the oracle action."""
    return action_nlls.min(axis=1)


def action_distribution(actions: np.ndarray) -> Dict[str, float]:
    counts = np.bincount(actions, minlength=N_ACTIONS)
    total = int(counts.sum())
    return {
        "counts": [int(c) for c in counts],
        "fractions": [float(c / total) for c in counts],
        "fraction_native_action": float(counts[0] / total),
        "fraction_swap_actions": float(counts[1:].sum() / total),
    }


def load_test_oracle_guarded(path: str, training_complete_marker: str) -> np.ndarray:
    """Load the TEST oracle only once policy training is provably finished.

    The marker file is written by the training script after all nine checkpoints and
    their validation-based selection are complete, which makes the protocol ordering a
    filesystem fact rather than a convention.
    """
    if not os.path.exists(training_complete_marker):
        raise RuntimeError(
            "TEST oracle is unavailable until policy training is frozen; "
            f"missing marker {training_complete_marker}")
    d = np.load(path)
    return d["oracle_action"]
