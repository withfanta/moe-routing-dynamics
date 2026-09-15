"""The XEC-P0 routing policy. One module class, three variants.

    q = W_q(u)                      u is 2112-d, q is 64-d
    k_j = W_k(z_j),  v_j = W_v(z_j) z_j is a 139-d cache item
    score_j = q^T k_j / sqrt(64)
    alpha = softmax(scores)
    m = sum_j alpha_j v_j
    logits = W_policy([q ; m])      128-d -> 5 actions

No hidden MLP, no multi-head attention, no residual stack, no LayerNorm with trainable
affine parameters.

All three variants instantiate the SAME class and therefore hold W_q, W_k, W_v and
W_policy with identical shapes, so trainable parameter counts match exactly.
CURRENT_ONLY forces m = 0 and never reads the cache, but still carries W_k and W_v so the
comparison is parameter-matched.

Only these modules train. OLMoE features arrive as detached arrays, so no gradient can
reach the frozen model.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ATTN_DIM, CURRENT_DIM, ITEM_DIM, N_ACTIONS, POLICY_IN_DIM, VARIANTS


class RoutingPolicy(nn.Module):
    """Single-head attention over cache items with a linear 5-way action head."""

    def __init__(self, variant: str):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        self.variant = variant

        # Present in every variant so parameter counts are identical.
        self.W_q = nn.Linear(CURRENT_DIM, ATTN_DIM, bias=False)
        self.W_k = nn.Linear(ITEM_DIM, ATTN_DIM, bias=False)
        self.W_v = nn.Linear(ITEM_DIM, ATTN_DIM, bias=False)
        self.W_policy = nn.Linear(POLICY_IN_DIM, N_ACTIONS, bias=False)

    @property
    def uses_cache(self) -> bool:
        return self.variant in ("FUSED_CACHE", "EXPERT_CACHE")

    def forward(self, u: torch.Tensor, items: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``u`` is (n, 2112); ``items`` is (n, n_items, 139) or None for CURRENT_ONLY."""
        q = self.W_q(u)  # (n, 64)

        if not self.uses_cache:
            # CURRENT_ONLY: memory is forcibly the zero vector.
            m = torch.zeros_like(q)
        else:
            if items is None:
                raise ValueError(f"{self.variant} requires cache items")
            k = self.W_k(items)  # (n, J, 64)
            v = self.W_v(items)  # (n, J, 64)
            scores = torch.einsum("nd,njd->nj", q, k) / math.sqrt(ATTN_DIM)
            alpha = F.softmax(scores, dim=-1)
            m = torch.einsum("nj,njd->nd", alpha, v)

        return self.W_policy(torch.cat([q, m], dim=-1))

    @torch.no_grad()
    def attention_weights(self, u: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Attention distribution, for inspection only. Not a verdict statistic."""
        if not self.uses_cache:
            raise ValueError("CURRENT_ONLY has no attention distribution")
        q = self.W_q(u)
        k = self.W_k(items)
        scores = torch.einsum("nd,njd->nj", q, k) / math.sqrt(ATTN_DIM)
        return F.softmax(scores, dim=-1)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_matched_variants(seed: int):
    """All three variants starting from the identical initial state_dict.

    A reference module is initialized once under ``seed``; every variant then loads that
    exact state, so for a given seed the three runs differ only in memory contents and
    subsequent training.
    """
    torch.manual_seed(seed)
    reference = RoutingPolicy("EXPERT_CACHE")
    init_state = {k: v.detach().clone() for k, v in reference.state_dict().items()}

    out = {}
    for variant in VARIANTS:
        m = RoutingPolicy(variant)
        m.load_state_dict(init_state)
        out[variant] = m

    counts = {v: count_trainable(m) for v, m in out.items()}
    if len(set(counts.values())) != 1:
        raise AssertionError(f"trainable parameter counts differ: {counts}")
    return out, init_state
