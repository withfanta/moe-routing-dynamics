"""REDV-V2 specific tests G-K.

These verify sampling hygiene, the derangement, distribution identity, equal
dimension, and probe identity. They must NOT inspect whether D is positive.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from redv import TRANSITIONS
from redv.v2_probes import MEAN_D_THRESHOLD, NOT_SUPPORTED, SUPPORTED, shuffled_rejected, v2_verdict
from redv.v2_sampling import (
    N_FIT,
    N_TEST,
    SHUFFLE_SEED,
    V2_SEED,
    build_control_permutations,
    context_key,
    derange,
    load_v1_block_ids,
)

ART_V1 = os.path.join(os.path.dirname(__file__), "..", "artifacts", "REDV_V1")
ART_V2 = os.path.join(os.path.dirname(__file__), "..", "artifacts", "REDV_V2")


# ------------------------------------------------------------------ test H


def test_H_derangement_no_fixed_points():
    """perm(i) != i for every sample, across many sizes."""
    for n in (2, 3, 5, 17, 64, 1024):
        rng = np.random.default_rng(SHUFFLE_SEED)
        p = derange(n, rng)
        assert sorted(p.tolist()) == list(range(n)), "must be a permutation"
        assert not np.any(p == np.arange(n)), f"fixed point at n={n}"


def test_H_derangement_deterministic():
    """Same seed reproduces the same derangement."""
    a = derange(1024, np.random.default_rng(SHUFFLE_SEED))
    b = derange(1024, np.random.default_rng(SHUFFLE_SEED))
    assert np.array_equal(a, b)


def test_H_derangement_handles_forced_fixed_points():
    """The repair pass removes fixed points from an identity-like draw."""

    class IdentityRng:
        def permutation(self, n):
            return np.arange(n)

    p = derange(64, IdentityRng())
    assert sorted(p.tolist()) == list(range(64))
    assert not np.any(p == np.arange(64))


def test_H_all_control_permutations_are_derangements():
    """Every one of the six control permutations is a derangement."""
    perms = build_control_permutations(N_FIT, N_TEST)
    assert len(perms) == 3
    for pair in TRANSITIONS:
        for role, n in (("fit", N_FIT), ("test", N_TEST)):
            p = perms[pair][role]
            assert p.shape == (n,)
            assert not np.any(p == np.arange(n)), f"{pair} {role} has a fixed point"


def test_H_permutations_differ_across_transitions_and_roles():
    """One seed, but independent draws per transition and role."""
    perms = build_control_permutations(N_FIT, N_TEST)
    seen = [perms[p][r].tobytes() for p in TRANSITIONS for r in ("fit", "test")]
    assert len(set(seen)) == 6, "all six permutations should be distinct draws"


def test_H_shuffled_rejected_rejects_identity_permutation():
    r = np.random.RandomState(0).randn(10, 4)
    with pytest.raises(AssertionError):
        shuffled_rejected(r, np.arange(10))


# ------------------------------------------------------------------ test I


def test_I_shuffle_is_exact_row_permutation():
    """Sorted row hashes of the shuffled matrix match the real matrix exactly."""
    rng = np.random.RandomState(3)
    r = rng.randn(256, 64)
    perm = derange(256, np.random.default_rng(SHUFFLE_SEED))
    s = shuffled_rejected(r, perm)

    def row_hashes(m):
        return sorted(hashlib.sha256(row.tobytes()).hexdigest() for row in m)

    assert row_hashes(r) == row_hashes(s), "must be exactly a row permutation"
    # Distributional invariants the control is required to preserve.
    assert np.allclose(np.sort(r, axis=0), np.sort(s, axis=0))
    assert np.allclose(r.mean(axis=0), s.mean(axis=0))
    assert np.allclose(r.std(axis=0), s.std(axis=0))
    # And the correspondence is genuinely broken.
    assert not np.allclose(r, s)


# ------------------------------------------------------------------ test J


def test_J_equal_dimension():
    """X_real and X_shuffle have identical shape, both exactly 4160 columns."""
    from redv.probes import augmented_features

    n = 32
    h = np.random.RandomState(0).randn(n, 2048)
    g = np.random.RandomState(1).randn(n, 64)
    r = np.random.RandomState(2).randn(n, 2048)
    perm = derange(n, np.random.default_rng(SHUFFLE_SEED))

    X_real = augmented_features(h, g, r)
    X_shuf = augmented_features(h, g, shuffled_rejected(r, perm))

    assert X_real.shape == X_shuf.shape
    assert X_real.shape[1] == 4160
    assert X_shuf.shape[1] == 4160


# ------------------------------------------------------------------ test K


def test_K_probe_identity_real_vs_shuffled():
    """REAL and SHUFFLED differ only by row order of the rejected block.

    Same Ridge alpha, same standardization code path, same target, same n. Feeding
    the SAME rejected matrix through both paths must give identical R².
    """
    from redv.v2_probes import run_v2_probe_triple

    n_fit, n_test = 64, 48
    rs = np.random.RandomState(7)
    fit = {"h": rs.randn(n_fit, 2048), "g": rs.randn(n_fit, 64),
           "r": rs.randn(n_fit, 2048), "G": np.abs(rs.randn(n_fit)) * 0.05}
    test = {"h": rs.randn(n_test, 2048), "g": rs.randn(n_test, 64),
            "r": rs.randn(n_test, 2048), "G": np.abs(rs.randn(n_test)) * 0.05}

    pf = derange(n_fit, np.random.default_rng(SHUFFLE_SEED))
    pt = derange(n_test, np.random.default_rng(SHUFFLE_SEED))
    out = run_v2_probe_triple(fit, test, pf, pt)

    assert out["alpha"] == 1.0
    assert out["baseline_dim"] == 2112
    assert out["augmented_dim"] == 4160
    assert out["D"] == pytest.approx(out["r2_real"] - out["r2_shuffle"])
    assert out["gain_over_baseline"] == pytest.approx(out["r2_real"] - out["r2_baseline"])

    # Identity check: if the "shuffled" rows are already the real rows in a
    # permutation applied to BOTH the features and the target, the pipelines are
    # equivalent up to sample order, so R² is unchanged. Here a cheaper
    # equivalence: running the triple with r replaced by its own permutation
    # yields a shuffled R² equal to the real R² of the permuted data.
    perm_fit_data = {**fit, "r": fit["r"][pf]}
    perm_test_data = {**test, "r": test["r"][pt]}
    out2 = run_v2_probe_triple(perm_fit_data, perm_test_data, pf, pt)
    assert out2["r2_real"] == pytest.approx(out["r2_shuffle"], abs=1e-12)


def test_K_verdict_rule_mechanics():
    """The rule is applied mechanically; no INCONCLUSIVE for numeric outcomes."""
    assert v2_verdict([0.05, 0.05, 0.05], [0.03, 0.03, 0.03]) == SUPPORTED
    # Condition 1: R2_real must be > 0 everywhere.
    assert v2_verdict([-0.01, 0.05, 0.05], [0.03, 0.03, 0.03]) == NOT_SUPPORTED
    assert v2_verdict([0.0, 0.05, 0.05], [0.03, 0.03, 0.03]) == NOT_SUPPORTED
    # Condition 2: D must be > 0 everywhere.
    assert v2_verdict([0.05, 0.05, 0.05], [0.03, -0.01, 0.03]) == NOT_SUPPORTED
    assert v2_verdict([0.05, 0.05, 0.05], [0.03, 0.0, 0.03]) == NOT_SUPPORTED
    # Condition 3: mean(D) >= 0.02.
    assert v2_verdict([0.05, 0.05, 0.05], [0.01, 0.01, 0.01]) == NOT_SUPPORTED
    assert v2_verdict([0.05, 0.05, 0.05], [0.02, 0.02, 0.02]) == SUPPORTED
    assert MEAN_D_THRESHOLD == 0.02
    with pytest.raises(ValueError):
        v2_verdict([0.05, 0.05], [0.03, 0.03])


# ------------------------------------------------------------------ test G


def test_G_v1_manifest_readable_and_unchanged():
    """The V1 manifest is present and used read-only."""
    p = os.path.join(ART_V1, "data_manifest.json")
    assert os.path.exists(p)
    ids = load_v1_block_ids(p)
    assert len(ids["validation"]) == 512
    assert len(ids["test"]) == 512


@pytest.mark.skipif(
    not os.path.exists(os.path.join(ART_V2, "data_manifest.json")),
    reason="V2 manifest not built yet",
)
def test_G_no_sample_overlap():
    """V2 fit ∩ V2 test = empty, and V2 all ∩ REDV-V1 all = empty."""
    with open(os.path.join(ART_V2, "data_manifest.json")) as fh:
        m2 = json.load(fh)
    v1 = load_v1_block_ids(os.path.join(ART_V1, "data_manifest.json"))

    def keys(role):
        d = m2[role]
        return set(zip(d["source_splits"], d["block_ids"]))

    fit_keys, test_keys = keys("fit"), keys("test")
    assert len(fit_keys) == N_FIT == m2["fit"]["n"]
    assert len(test_keys) == N_TEST == m2["test"]["n"]
    assert not (fit_keys & test_keys), "V2 fit and test must be disjoint"

    v1_keys = {("validation", b) for b in v1["validation"]} | {("test", b) for b in v1["test"]}
    assert not ((fit_keys | test_keys) & v1_keys), "V2 must not reuse any REDV-V1 block"
    assert m2["v2_seed"] == V2_SEED
    assert m2["shuffle_seed"] == SHUFFLE_SEED
    assert m2["train_split_used"] is False


def test_G_pool_exclusion_logic():
    """untouched_pool drops exactly the V1 ids and keeps the rest in order."""
    from redv.v2_sampling import untouched_pool

    class B:
        def __init__(self, i):
            self.block_id = i
            self.stream_start = i * 129
            self.tokens = tuple(range(129))

    blocks = {"validation": [B(i) for i in range(10)], "test": [B(i) for i in range(10)]}
    v1 = {"validation": {0, 1, 2}, "test": {7, 8, 9}}
    pool = untouched_pool(blocks, v1)

    assert [(s, b.block_id) for s, b in pool[:7]] == [
        ("validation", i) for i in range(3, 10)
    ]
    assert [(s, b.block_id) for s, b in pool[7:]] == [("test", i) for i in range(7)]
    assert all(b.block_id not in v1[s] for s, b in pool)


def test_G_insufficient_pool_raises():
    """A pool smaller than 2048 stops rather than silently shrinking n."""
    from redv.v2_sampling import draw_v2_sample

    with pytest.raises(RuntimeError, match="insufficient untouched blocks"):
        draw_v2_sample([("validation", object())] * 100)
