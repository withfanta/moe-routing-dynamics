"""Routing regret at a future target layer, for DREV-P0.

Copied from the validated REDV implementation (`src/redv/counterfactual.py`, REDV
commit 6d1c57f). Unchanged definition.

At target layer m, for the experimental token, four equal-compute alternatives keep
native ranks 1-7 and replace native rank 8 with rank 9, 10, 11, or 12. Every route
executes exactly eight experts. Only the target layer's routing decision for that
one token is overridden; every intermediate layer runs natively, and after the
intervention all later layers continue normally.

    G_m(i) = max(0, L_native(m,i) - L_best_alt(m,i))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from . import REJECTED_RANKS
from .source_features import build_override, flat_index, forced_forward_nll


@dataclass
class RegretResult:
    native_nll: np.ndarray
    alt_nll: np.ndarray  # (n, 4), columns ordered by replacement rank 9..12
    best_alt_nll: np.ndarray
    regret: np.ndarray


@torch.no_grad()
def routing_regret(
    model, input_ids, targets, target_layer: int, native_nll, full_probs
) -> RegretResult:
    """Compute G at ``target_layer`` for every context in the batch."""
    n, seq_len = input_ids.shape
    flat = [flat_index(i, seq_len) for i in range(n)]

    cols = []
    for rank in REJECTED_RANKS:
        override = build_override(full_probs, flat, [rank] * n)
        cols.append(forced_forward_nll(model, input_ids, targets, target_layer, override)
                    .float().cpu().numpy())

    alt = np.stack(cols, axis=1)
    nat = native_nll.float().cpu().numpy()
    best = alt.min(axis=1)
    return RegretResult(native_nll=nat, alt_nll=alt, best_alt_nll=best,
                        regret=np.maximum(0.0, nat - best))


@torch.no_grad()
def native_route_equivalence_nll(model, input_ids, targets, target_layer: int, full_probs):
    """Force exactly the native top-8 identities; must reproduce the stock NLL."""
    from .source_features import native_top_identities

    n, seq_len = input_ids.shape
    ids = native_top_identities(full_probs)
    override = {flat_index(i, seq_len): ids[i] for i in range(n)}
    return forced_forward_nll(model, input_ids, targets, target_layer, override)


def regret_descriptives(regret: np.ndarray) -> Dict[str, float]:
    """Descriptive only; these never affect the verdict."""
    return {
        "mean_G": float(np.mean(regret)),
        "median_G": float(np.median(regret)),
        "proportion_G_positive": float(np.mean(regret > 0.0)),
    }
