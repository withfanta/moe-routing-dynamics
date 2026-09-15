"""Build the three EIPC-P0 representations and the future-routing target.

All three are constructed from exactly the same projected selected-expert
contributions ``s`` — the fused representation is their per-layer sum, so no
representation sees information the others do not. They differ only in whether, and
how faithfully, expert identity survives.

Raw dimensions before compression:

| target | history layers | FUSED | EXPERT_IDENTITY / SHUFFLED_IDENTITY |
|---|---|---|---|
| Layer 8 | 1..7 | 7 * 32 = 224 | 7 * 64 * 32 = 14336 |
| Layer 12 | 1..11 | 11 * 32 = 352 | 11 * 64 * 32 = 22528 |

Every representation is then compressed to exactly 64 dimensions with a pipeline fit
on FIT data only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import NUM_EXPERTS, PCA_COMPONENTS, PCA_SEED, PROJ_DIM, SHUFFLE_SEED, TARGETS
from .expert_states import apply_cyclic_shift, cyclic_shifts, fused_projected, identity_slots


def build_fused(s_by_layer: Dict[int, np.ndarray], history_code: Sequence[int]) -> np.ndarray:
    """FUSED: per-layer projected fused contribution, concatenated over history."""
    cols = [fused_projected(s_by_layer[li]) for li in history_code]
    return np.concatenate(cols, axis=1)


def build_identity(
    s_by_layer: Dict[int, np.ndarray],
    ids_by_layer: Dict[int, np.ndarray],
    history_code: Sequence[int],
) -> np.ndarray:
    """EXPERT_IDENTITY: 64 identity-indexed slots per layer, concatenated."""
    cols = []
    for li in history_code:
        slots = identity_slots(s_by_layer[li], ids_by_layer[li], NUM_EXPERTS)
        cols.append(slots.reshape(slots.shape[0], -1))
    return np.concatenate(cols, axis=1)


def build_shuffled(
    s_by_layer: Dict[int, np.ndarray],
    ids_by_layer: Dict[int, np.ndarray],
    history_code: Sequence[int],
    shifts: np.ndarray,
) -> np.ndarray:
    """SHUFFLED_IDENTITY: identical slots, cyclically shifted per (sample, layer).

    ``shifts`` is ``(n_samples, len(history_code))`` with values in 1..63.
    """
    cols = []
    for pos, li in enumerate(history_code):
        slots = identity_slots(s_by_layer[li], ids_by_layer[li], NUM_EXPERTS)
        shifted = apply_cyclic_shift(slots, shifts[:, pos])
        cols.append(shifted.reshape(shifted.shape[0], -1))
    return np.concatenate(cols, axis=1)


def build_shifts(n_samples: int, n_layers: int, role: str) -> np.ndarray:
    """Deterministic non-zero cyclic shifts, generated independently for FIT and TEST.

    One seed (314159) with a role-specific offset so the two splits get independent
    draws without introducing a second seed choice.
    """
    if role not in ("fit", "test"):
        raise ValueError(f"role must be fit or test, got {role!r}")
    offset = 0 if role == "fit" else 1
    return cyclic_shifts(n_samples, n_layers, SHUFFLE_SEED + offset)


# ------------------------------------------------------------------ compression


@dataclass
class FrozenPipeline:
    """StandardScaler -> PCA(64) -> StandardScaler, fit on FIT only.

    No target information enters: fit() takes features alone.
    """

    scaler_in: StandardScaler
    pca: PCA
    scaler_out: StandardScaler

    @classmethod
    def fit(cls, X_fit: np.ndarray) -> "FrozenPipeline":
        scaler_in = StandardScaler()
        Xs = scaler_in.fit_transform(X_fit)
        pca = PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", random_state=PCA_SEED)
        Z = pca.fit_transform(Xs)
        scaler_out = StandardScaler()
        scaler_out.fit(Z)
        return cls(scaler_in=scaler_in, pca=pca, scaler_out=scaler_out)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler_out.transform(self.pca.transform(self.scaler_in.transform(X)))

    @property
    def explained_variance_ratio_sum(self) -> float:
        return float(self.pca.explained_variance_ratio_.sum())


# ---------------------------------------------------------------------- target


def center_logits(g: np.ndarray) -> np.ndarray:
    """q = g - mean(g) over the 64 expert dimensions, per sample.

    Softmax routing is invariant to adding a constant to all logits, so centring
    removes a direction that carries no routing information.
    """
    if g.shape[-1] != NUM_EXPERTS:
        raise ValueError(f"expected {NUM_EXPERTS} logits, got {g.shape[-1]}")
    return g - g.mean(axis=-1, keepdims=True)


@dataclass
class TargetScaler:
    """Per-dimension standardization of the centred target, fit on FIT only."""

    scaler: StandardScaler

    @classmethod
    def fit(cls, q_fit: np.ndarray) -> "TargetScaler":
        sc = StandardScaler()
        sc.fit(q_fit)
        return cls(scaler=sc)

    def transform(self, q: np.ndarray) -> np.ndarray:
        return self.scaler.transform(q)


def raw_dimension(target_human: int, representation: str) -> int:
    """Expected raw feature dimension, for assertions and documentation."""
    n_layers = len(TARGETS[target_human]["history_code"])
    if representation == "FUSED":
        return n_layers * PROJ_DIM
    if representation in ("EXPERT_IDENTITY", "SHUFFLED_IDENTITY"):
        return n_layers * NUM_EXPERTS * PROJ_DIM
    raise ValueError(f"unknown representation {representation!r}")
