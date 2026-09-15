"""Five-action NLL evaluation at Layer 12 for XEC-P0.

Each of the five actions executes exactly eight experts, so compute is identical.
Only the Layer-12 routing decision for the experimental token is overridden; all other
tokens and layers stay native, and Layers 13-16 run normally afterwards.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from . import N_ACTIONS
from .olmoe import action_nll, native_forward


@torch.no_grad()
def all_action_nlls(model, input_ids: torch.Tensor, targets: torch.Tensor,
                    top12_ids: torch.Tensor) -> np.ndarray:
    """NLL under every one of the five actions. Returns (n, 5)."""
    cols = []
    for a in range(N_ACTIONS):
        cols.append(action_nll(model, input_ids, targets, top12_ids, a).float().cpu().numpy())
    return np.stack(cols, axis=1)


@torch.no_grad()
def selected_action_nll(model, input_ids: torch.Tensor, targets: torch.Tensor,
                        top12_ids: torch.Tensor, actions: np.ndarray) -> np.ndarray:
    """NLL when each sample uses its own predicted action.

    Samples are grouped by action so each distinct action costs one forward pass.
    """
    n = input_ids.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for a in np.unique(actions):
        rows = np.nonzero(actions == a)[0]
        idx = torch.as_tensor(rows, device=input_ids.device, dtype=torch.long)
        nll = action_nll(model, input_ids[idx], targets[idx], top12_ids[idx], int(a))
        out[rows] = nll.float().cpu().numpy()
    return out


def native_nll_check(action_matrix: np.ndarray, native: np.ndarray, atol: float = 1e-4):
    """Action 0 must reproduce the plain native NLL. Returns the max absolute error."""
    err = float(np.max(np.abs(action_matrix[:, 0] - native)))
    if err > atol:
        raise AssertionError(f"action 0 differs from native NLL by {err}")
    return err
