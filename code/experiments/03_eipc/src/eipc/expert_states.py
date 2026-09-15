"""Fixed random projection P: R^2048 -> R^32 and per-sample expert state extraction.

P is generated once from ``numpy.random.default_rng(20260917)`` with entries
Normal(0, 1/sqrt(32)). It is fixed, non-trainable, identical across layers, across
experts, and across FIT and TEST.

Because P is linear, P(sum_e c_e) == sum_e P(c_e), so the FUSED representation can be
built from the same projected per-expert contributions as EXPERT_IDENTITY. That
identity is unit-tested numerically.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Tuple

import numpy as np
import torch

from . import HIDDEN_SIZE, PROJ_DIM, PROJECTION_SEED, TOP_K


def build_projection(seed: int = PROJECTION_SEED) -> np.ndarray:
    """The fixed projection matrix, shape (2048, 32), float64 for determinism."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0 / np.sqrt(PROJ_DIM),
                      size=(HIDDEN_SIZE, PROJ_DIM))


def projection_hash(P: np.ndarray) -> str:
    """Content hash of the projection matrix, for provenance."""
    return hashlib.sha256(np.ascontiguousarray(P, dtype=np.float64).tobytes()).hexdigest()


def projection_metadata(P: np.ndarray) -> Dict:
    return {
        "seed": PROJECTION_SEED,
        "shape": list(P.shape),
        "in_dim": HIDDEN_SIZE,
        "out_dim": PROJ_DIM,
        "distribution": "Normal(0, 1/sqrt(32))",
        "dtype": str(P.dtype),
        "sha256": projection_hash(P),
        "trainable": False,
        "shared_across_layers": True,
        "shared_across_experts": True,
        "shared_across_fit_and_test": True,
    }


def write_projection(path: str, P: np.ndarray) -> str:
    meta = projection_metadata(P)
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta["sha256"]


def project(P: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Apply P to contributions with a trailing hidden dimension."""
    if c.shape[-1] != HIDDEN_SIZE:
        raise ValueError(f"expected trailing dim {HIDDEN_SIZE}, got {c.shape[-1]}")
    return c @ P


def project_torch(P_t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Torch-side projection, computed in float32 on the captured contributions."""
    return c.float() @ P_t


@torch.no_grad()
def extract_layer_states(block, x: torch.Tensor, logits: torch.Tensor, P_t: torch.Tensor):
    """Per-sample selected-expert states at one historical layer.

    Returns a dict with the native Top-8 identities, the native Top-8
    probabilities, and the projected weighted contributions ``s`` of shape
    ``(n, 8, 32)``. Full 2048-d expert outputs are projected immediately and not
    retained, which keeps artifacts small.
    """
    from .olmoe import selected_expert_contributions

    identities, probs, contrib = selected_expert_contributions(block, x, logits)
    s = project_torch(P_t, contrib)  # (n, 8, 32)
    return {
        "topk_ids": identities.to(torch.int16).cpu().numpy(),
        "topk_probs": probs.float().cpu().numpy(),
        "s": s.float().cpu().numpy(),
    }


def fused_projected(s: np.ndarray) -> np.ndarray:
    """f = sum over the eight selected slots of the projected contributions.

    Equals P(y) up to float error, since P is linear and y is the sum of the same
    contributions.
    """
    if s.ndim != 3 or s.shape[1] != TOP_K or s.shape[2] != PROJ_DIM:
        raise ValueError(f"expected (n, {TOP_K}, {PROJ_DIM}), got {s.shape}")
    return s.sum(axis=1)


def identity_slots(s: np.ndarray, topk_ids: np.ndarray, num_experts: int = 64) -> np.ndarray:
    """Scatter projected contributions into 64 slots indexed by expert identity.

    Slot e holds s for expert e when e was selected, and a 32-d zero vector
    otherwise. Returns ``(n, 64, 32)``. The slot index IS the expert identity.
    """
    n = s.shape[0]
    out = np.zeros((n, num_experts, PROJ_DIM), dtype=s.dtype)
    rows = np.arange(n)[:, None]
    out[rows, topk_ids.astype(np.int64)] = s
    return out


def cyclic_shifts(n_samples: int, n_layers: int, seed: int) -> np.ndarray:
    """Non-zero cyclic shift per (sample, layer), drawn in {1, ..., 63}.

    A non-zero shift guarantees no expert remains in its own identity slot, so
    expert identity is destroyed while every contribution vector, its layer, the
    number of selected experts, and all numerical values are preserved.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(low=1, high=64, size=(n_samples, n_layers), dtype=np.int64)


def apply_cyclic_shift(slots: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Cyclically shift the 64 expert slots per sample.

    ``slots`` is ``(n, 64, 32)`` and ``shifts`` is ``(n,)`` with values in 1..63.
    """
    if slots.shape[1] != 64:
        raise ValueError(f"expected 64 expert slots, got {slots.shape[1]}")
    if np.any(shifts == 0) or np.any(shifts > 63) or np.any(shifts < 1):
        raise AssertionError("cyclic shifts must lie in {1, ..., 63}")
    out = np.empty_like(slots)
    for i in range(slots.shape[0]):
        out[i] = np.roll(slots[i], int(shifts[i]), axis=0)
    return out
