"""Fixed cache projection and cache-item construction for XEC-P0.

P: R^2048 -> R^64 is generated once from numpy.random.default_rng(20260918) with entries
Normal(0, 1/sqrt(64)). It is fixed, non-trainable, and shared across layers, experts and
splits.

Cache items are 139-dimensional:

    z = [LayerNorm(s) 64 ; layer one-hot 11 ; expert one-hot 64]

EXPERT_CACHE contributes 11 * 8 = 88 items per sample, one per selected expert.
FUSED_CACHE contributes 11 items, one per layer, each holding the per-layer sum of the
same projected contributions with a zero expert-ID block.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from . import (
    HIDDEN_SIZE,
    ITEM_DIM,
    N_EXPERT_ITEMS,
    N_FUSED_ITEMS,
    N_HISTORY_LAYERS,
    NUM_EXPERTS,
    PROJ_DIM,
    PROJECTION_SEED,
    TOP_K,
)


def build_projection(seed: int = PROJECTION_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0 / np.sqrt(PROJ_DIM), size=(HIDDEN_SIZE, PROJ_DIM))


def projection_hash(P: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(P, dtype=np.float64).tobytes()).hexdigest()


def projection_metadata(P: np.ndarray) -> Dict:
    return {
        "seed": PROJECTION_SEED,
        "shape": list(P.shape),
        "in_dim": HIDDEN_SIZE,
        "out_dim": PROJ_DIM,
        "distribution": "Normal(0, 1/sqrt(64))",
        "dtype": str(P.dtype),
        "sha256": projection_hash(P),
        "trainable": False,
        "learned": False,
        "shared_across_layers_experts_splits": True,
    }


def write_projection(path: str, P: np.ndarray) -> str:
    meta = projection_metadata(P)
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta["sha256"]


def layer_norm_np(v: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Elementwise-affine=False LayerNorm over the last dimension."""
    mu = v.mean(axis=-1, keepdims=True)
    var = v.var(axis=-1, keepdims=True)
    return (v - mu) / np.sqrt(var + eps)


def center_logits(g: np.ndarray) -> np.ndarray:
    """centered(g) = g - mean(g) over the 64 expert dimensions."""
    return g - g.mean(axis=-1, keepdims=True)


def current_context(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """u = [LayerNorm(x) ; centered(g)], dimension 2048 + 64 = 2112."""
    u = np.concatenate([layer_norm_np(x), center_logits(g)], axis=-1)
    return u


def _one_hot(idx: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(idx.shape + (n,), dtype=np.float32)
    np.put_along_axis(out, idx[..., None].astype(np.int64), 1.0, axis=-1)
    return out


def expert_cache_items(s: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """EXPERT_CACHE items.

    ``s`` is (n, 11, 8, 64) projected weighted contributions and ``ids`` is (n, 11, 8)
    the true native selected expert identities. Returns (n, 88, 139).
    """
    n, n_layers, k, d = s.shape
    if (n_layers, k, d) != (N_HISTORY_LAYERS, TOP_K, PROJ_DIM):
        raise ValueError(f"expected (n, {N_HISTORY_LAYERS}, {TOP_K}, {PROJ_DIM}), got {s.shape}")

    content = layer_norm_np(s.astype(np.float32))
    layer_idx = np.broadcast_to(np.arange(n_layers)[None, :, None], (n, n_layers, k))
    layer_oh = _one_hot(np.ascontiguousarray(layer_idx), N_HISTORY_LAYERS)
    expert_oh = _one_hot(ids.astype(np.int64), NUM_EXPERTS)

    items = np.concatenate([content, layer_oh, expert_oh], axis=-1)
    items = items.reshape(n, n_layers * k, ITEM_DIM)
    assert items.shape[1] == N_EXPERT_ITEMS and items.shape[2] == ITEM_DIM
    return items


def fused_cache_items(s: np.ndarray) -> np.ndarray:
    """FUSED_CACHE items, derived from exactly the same projected contributions.

    Per layer, f = sum over selected experts of s. The expert-ID block is zero.
    Returns (n, 11, 139).
    """
    n, n_layers, k, d = s.shape
    if (n_layers, k, d) != (N_HISTORY_LAYERS, TOP_K, PROJ_DIM):
        raise ValueError(f"expected (n, {N_HISTORY_LAYERS}, {TOP_K}, {PROJ_DIM}), got {s.shape}")

    fused = s.astype(np.float32).sum(axis=2)  # (n, 11, 64)
    content = layer_norm_np(fused)
    layer_idx = np.broadcast_to(np.arange(n_layers)[None, :], (n, n_layers))
    layer_oh = _one_hot(np.ascontiguousarray(layer_idx), N_HISTORY_LAYERS)
    expert_zero = np.zeros((n, n_layers, NUM_EXPERTS), dtype=np.float32)

    items = np.concatenate([content, layer_oh, expert_zero], axis=-1)
    assert items.shape[1] == N_FUSED_ITEMS and items.shape[2] == ITEM_DIM
    return items


@torch.no_grad()
def project_contributions(P_t: torch.Tensor, contrib: torch.Tensor) -> torch.Tensor:
    """Apply the fixed projection to captured contributions, in float32."""
    return contrib.float() @ P_t
