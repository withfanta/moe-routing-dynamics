"""Counterfactual routing regret G_{l+1,t} for REDV-V1.

At target layer l+1 and the experimental token t, four equal-compute
alternatives are constructed. Each keeps native ranks 1-7 and replaces native
rank 8 with one of ranks 9, 10, 11, 12:

    native:  [1,2,3,4,5,6,7,8]
    alt A:   [1,2,3,4,5,6,7,9]
    alt B:   [1,2,3,4,5,6,7,10]
    alt C:   [1,2,3,4,5,6,7,11]
    alt D:   [1,2,3,4,5,6,7,12]

Every route executes exactly eight experts, so compute is identical. Only this
one MoE routing decision, for this one token, is overridden; all other tokens,
layers, and later routing decisions stay native, and the rest of the frozen
model continues normally.

    G_{l+1,t} = max(0, L_native - min_a L_alt_a)

G = 0 means no tested equal-compute alternative improves the native route.
G > 0 means at least one does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

from . import REJECTED_RANKS
from .olmoe_hooks import build_override, flat_index, forced_forward_nll


ALT_LABELS = ("A", "B", "C", "D")


@dataclass
class RegretResult:
    """Regret for one batch of contexts at one target layer."""

    native_nll: np.ndarray
    alt_nll: np.ndarray  # (n_contexts, 4), columns ordered by replacement rank
    best_alt_nll: np.ndarray
    regret: np.ndarray
    alt_ranks: List[int] = field(default_factory=lambda: list(REJECTED_RANKS))


@torch.no_grad()
def routing_regret(
    model,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    target_layer: int,
    native_nll: torch.Tensor,
    full_probs: torch.Tensor,
) -> RegretResult:
    """Compute G at ``target_layer`` for every context in the batch.

    ``full_probs`` is the native 64-way router probability matrix at
    ``target_layer`` for the experimental tokens, shape ``(n_contexts, 64)``.
    ``native_nll`` is the already-computed native next-token NLL.
    """
    n, seq_len = input_ids.shape
    flat = [flat_index(i, seq_len) for i in range(n)]

    alt_cols = []
    for rank in REJECTED_RANKS:
        override = build_override(full_probs, flat, [rank] * n)
        nll = forced_forward_nll(model, input_ids, targets, target_layer, override)
        alt_cols.append(nll.float().cpu().numpy())

    alt = np.stack(alt_cols, axis=1)
    nat = native_nll.float().cpu().numpy()
    best = alt.min(axis=1)
    regret = np.maximum(0.0, nat - best)

    return RegretResult(native_nll=nat, alt_nll=alt, best_alt_nll=best, regret=regret)


@torch.no_grad()
def native_route_equivalence_nll(
    model,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    target_layer: int,
    full_probs: torch.Tensor,
) -> torch.Tensor:
    """Force exactly the native top-8 identities. Test A/F support.

    Should reproduce the ordinary model NLL within FP16 tolerance, since the
    forced path differs from native only in how the eight identities were chosen.
    """
    from .olmoe_hooks import native_top_identities

    n, seq_len = input_ids.shape
    flat = [flat_index(i, seq_len) for i in range(n)]
    ids = native_top_identities(full_probs)
    override = {int(f): ids[i] for i, f in enumerate(flat)}
    return forced_forward_nll(model, input_ids, targets, target_layer, override)


def regret_descriptives(regret: np.ndarray) -> Dict[str, float]:
    """Descriptive only. These values never change the verdict."""
    return {
        "mean_G": float(np.mean(regret)),
        "median_G": float(np.median(regret)),
        "proportion_G_positive": float(np.mean(regret > 0.0)),
    }
