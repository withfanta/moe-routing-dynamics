"""The four EPD-P0 historical representations and the shared compression.

All four are built from the same captured quantities, so no representation sees information
the others could not have. They differ only in which aspect of provenance survives:

| representation | raw dim | keeps |
|---|---|---|
| FUSED | 11 * 32 = 352 | per-layer sum only |
| ID_PATH | 11 * 64 = 704 | binary selection path, no content |
| CONTENT_RANK | 11 * 8 * 32 = 2816 | rank-ordered content, no expert ID |
| FULL_PROVENANCE | 11 * 64 * 32 = 22528 | expert ID and content together |

Each is compressed to exactly 32 dimensions with a pipeline fit on FIT only, so the
comparison holds the dimensional budget constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import (
    LAYERWISE_PCA_COMPONENTS,
    N_HISTORY_LAYERS,
    NUM_EXPERTS,
    PCA_COMPONENTS,
    PCA_SEED,
    PROJ_DIM,
    RAW_DIMS,
    TOP_K,
)


def build_fused(s: np.ndarray) -> np.ndarray:
    """FUSED: per-layer sum of the projected contributions, concatenated.

    ``s`` is (n, 11, 8, 32). Result is (n, 352).
    """
    _check_s(s)
    return s.sum(axis=2).reshape(s.shape[0], -1)


def build_id_path(ids: np.ndarray) -> np.ndarray:
    """ID_PATH: per-layer 64-d binary selection indicator, concatenated.

    ``ids`` is (n, 11, 8) of native expert identities. Result is (n, 704), values in {0, 1}.
    No router probabilities, no logits, no activation content.
    """
    n, n_layers, k = ids.shape
    if (n_layers, k) != (N_HISTORY_LAYERS, TOP_K):
        raise ValueError(f"expected (n, {N_HISTORY_LAYERS}, {TOP_K}), got {ids.shape}")
    out = np.zeros((n, n_layers, NUM_EXPERTS), dtype=np.float32)
    rows = np.arange(n)[:, None, None]
    layers = np.arange(n_layers)[None, :, None]
    out[rows, layers, ids.astype(np.int64)] = 1.0
    return out.reshape(n, -1)


def build_content_rank(s: np.ndarray) -> np.ndarray:
    """CONTENT_RANK: the eight projected contributions in native rank order.

    ``s`` slot k already holds the rank-(k+1) expert, so ordering is inherited from capture.
    Result is (n, 2816) and carries no absolute expert identity.
    """
    _check_s(s)
    return s.reshape(s.shape[0], -1)


def build_full_provenance(s: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """FULL_PROVENANCE: 64 identity-indexed slots per layer, zero where unselected.

    Result is (n, 22528). The slot index IS the expert identity.
    """
    _check_s(s)
    n, n_layers, k, d = s.shape
    out = np.zeros((n, n_layers, NUM_EXPERTS, d), dtype=np.float32)
    rows = np.arange(n)[:, None, None]
    layers = np.arange(n_layers)[None, :, None]
    out[rows, layers, ids.astype(np.int64)] = s.astype(np.float32)
    return out.reshape(n, -1)


def build_all(s: np.ndarray, ids: np.ndarray) -> Dict[str, np.ndarray]:
    reps = {
        "FUSED": build_fused(s),
        "ID_PATH": build_id_path(ids),
        "CONTENT_RANK": build_content_rank(s),
        "FULL_PROVENANCE": build_full_provenance(s, ids),
    }
    for name, X in reps.items():
        if X.shape[1] != RAW_DIMS[name]:
            raise AssertionError(f"{name} raw dim {X.shape[1]} != {RAW_DIMS[name]}")
    return reps


# ------------------------------------------------------------ single-layer views


def build_id_path_layer(ids: np.ndarray, layer_pos: int) -> np.ndarray:
    """ID_PATH restricted to one historical layer. Result is (n, 64)."""
    n = ids.shape[0]
    out = np.zeros((n, NUM_EXPERTS), dtype=np.float32)
    rows = np.arange(n)[:, None]
    out[rows, ids[:, layer_pos, :].astype(np.int64)] = 1.0
    return out


def build_content_rank_layer(s: np.ndarray, layer_pos: int) -> np.ndarray:
    """CONTENT_RANK restricted to one historical layer. Result is (n, 256)."""
    _check_s(s)
    return s[:, layer_pos, :, :].reshape(s.shape[0], -1)


# ------------------------------------------------------------------ compression


@dataclass
class FrozenPipeline:
    """StandardScaler -> PCA(k) -> StandardScaler, fit on FIT only.

    fit() takes features alone, so no target information can enter.
    """

    scaler_in: StandardScaler
    pca: PCA
    scaler_out: StandardScaler
    n_components: int

    @classmethod
    def fit(cls, X_fit: np.ndarray, n_components: int = PCA_COMPONENTS) -> "FrozenPipeline":
        # PCA cannot return more components than min(n_samples, n_features).
        k = min(n_components, min(X_fit.shape))
        if k != n_components:
            raise AssertionError(
                f"requested {n_components} components but input is {X_fit.shape}")
        scaler_in = StandardScaler()
        Xs = scaler_in.fit_transform(X_fit)
        pca = PCA(n_components=k, svd_solver="randomized", random_state=PCA_SEED)
        Z = pca.fit_transform(Xs)
        scaler_out = StandardScaler()
        scaler_out.fit(Z)
        return cls(scaler_in=scaler_in, pca=pca, scaler_out=scaler_out, n_components=k)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler_out.transform(self.pca.transform(self.scaler_in.transform(X)))

    @property
    def explained_variance_ratio_sum(self) -> float:
        return float(self.pca.explained_variance_ratio_.sum())


def compress(reps_fit: Dict[str, np.ndarray], reps_test: Dict[str, np.ndarray],
             n_components: int = PCA_COMPONENTS):
    """Fit one pipeline per representation on FIT, apply unchanged to TEST."""
    X_fit, X_test, evr = {}, {}, {}
    for name in reps_fit:
        pipe = FrozenPipeline.fit(reps_fit[name], n_components)
        X_fit[name] = pipe.transform(reps_fit[name])
        X_test[name] = pipe.transform(reps_test[name])
        evr[name] = pipe.explained_variance_ratio_sum
        assert X_fit[name].shape[1] == n_components
        assert X_test[name].shape[1] == n_components
    return X_fit, X_test, evr


# ---------------------------------------------------------------------- target


def center_logits(g: np.ndarray) -> np.ndarray:
    """q = g - mean(g) over the 64 expert dimensions, per sample."""
    if g.shape[-1] != NUM_EXPERTS:
        raise ValueError(f"expected {NUM_EXPERTS} logits, got {g.shape[-1]}")
    return g - g.mean(axis=-1, keepdims=True)


@dataclass
class TargetScaler:
    scaler: StandardScaler

    @classmethod
    def fit(cls, q_fit: np.ndarray) -> "TargetScaler":
        sc = StandardScaler()
        sc.fit(q_fit)
        return cls(scaler=sc)

    def transform(self, q: np.ndarray) -> np.ndarray:
        return self.scaler.transform(q)


def _check_s(s: np.ndarray) -> None:
    if s.ndim != 4 or s.shape[1:] != (N_HISTORY_LAYERS, TOP_K, PROJ_DIM):
        raise ValueError(
            f"expected (n, {N_HISTORY_LAYERS}, {TOP_K}, {PROJ_DIM}), got {s.shape}")
