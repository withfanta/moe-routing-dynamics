"""Frozen unsupervised compression and the matched shuffle control.

DREV-P0 does not repeat REDV's p >> n Ridge setup. Baseline b (2112) and rejected
evidence r (2048) are each independently standardized and reduced to exactly 64 PCA
components, so the probe input is 128 dimensions against n_fit = 512.

Preprocessing is fit on the 512 fit samples ONLY. No target information enters the
scaler or the PCA: both see features alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import PCA_COMPONENTS, PCA_SEED, SHUFFLE_SEED


@dataclass
class FrozenProjection:
    """StandardScaler + PCA(64), fit on the fit split only."""

    scaler: StandardScaler
    pca: PCA

    @classmethod
    def fit(cls, X_fit: np.ndarray) -> "FrozenProjection":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_fit)
        pca = PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", random_state=PCA_SEED)
        pca.fit(Xs)
        return cls(scaler=scaler, pca=pca)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(X))

    @property
    def explained_variance_ratio_sum(self) -> float:
        return float(self.pca.explained_variance_ratio_.sum())


def baseline_matrix(h: np.ndarray, g: np.ndarray) -> np.ndarray:
    """b = [h ; g], 2048 + 64 = 2112."""
    return np.concatenate([h, g], axis=1)


def derange(n: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation of 0..n-1 with no fixed point.

    Copied from the validated REDV implementation (`src/redv/v2_sampling.py`, REDV
    commit f3dc29b). Draw ``rng.permutation(n)``, then scan i ascending and whenever
    ``p[i] == i`` swap ``p[i]`` with ``p[(i + 1) % n]``. One forward pass suffices.
    The seed is never changed to obtain a derangement.
    """
    if n < 2:
        raise ValueError("derangement requires n >= 2")
    p = rng.permutation(n)
    for i in range(n):
        if p[i] == i:
            j = (i + 1) % n
            p[i], p[j] = p[j], p[i]
    if np.any(p == np.arange(n)):  # pragma: no cover
        raise AssertionError("derangement failed")
    return p


def build_shuffle_permutations(n_fit: int, n_test: int) -> Tuple[np.ndarray, np.ndarray]:
    """The one fixed control: one fit permutation, one independent test permutation.

    Both come from a single generator seeded with 271828, drawn fit-then-test. The
    SAME pair is reused for all four horizons, so no horizon gets its own favorable
    shuffle.
    """
    rng = np.random.default_rng(SHUFFLE_SEED)
    return derange(n_fit, rng), derange(n_test, rng)


def apply_shuffle(z_r: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """z_r_shuffled(i) = z_r(perm(i)). Exactly a row permutation."""
    if perm.shape[0] != z_r.shape[0]:
        raise ValueError(f"permutation length {perm.shape[0]} != n rows {z_r.shape[0]}")
    if np.any(perm == np.arange(len(perm))):
        raise AssertionError("control permutation must be a derangement")
    return z_r[perm]
