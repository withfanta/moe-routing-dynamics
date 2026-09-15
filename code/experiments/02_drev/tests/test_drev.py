"""DREV-P0 tests A-L. Implementation correctness only; no verdict statistic."""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from drev import (
    HORIZONS,
    N_FIT,
    N_TEST,
    PCA_COMPONENTS,
    PROBE_DIM,
    REJECTED_RANKS,
    SHUFFLE_SEED,
    SOURCE_LAYER,
    TOP_K,
)
from drev.counterfactual import native_route_equivalence_nll, routing_regret
from drev.data import load_prior_used_keys, make_blocks
from drev.projection import (
    FrozenProjection,
    apply_shuffle,
    baseline_matrix,
    build_shuffle_permutations,
    derange,
)
from drev.source_features import (
    _generic_moe_forward,
    assert_frozen,
    build_override,
    flat_index,
    forced_route,
    load_model,
    native_forward,
    native_top_identities,
    next_token_nll,
    probs_from_logits,
    rejected_evidence,
    rejected_evidence_reference,
    rejected_identities,
    router_probs,
    verify_config,
)

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "DREV_P0")
SEED = 20260916
HIDDEN_ATOL = 2e-2
NLL_ATOL = 5e-3


@pytest.fixture(scope="module")
def model_and_tok():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return load_model(model_dir=os.environ.get("DREV_MODEL_DIR"))


@pytest.fixture(scope="module")
def batch():
    g = torch.Generator().manual_seed(SEED)
    ids = torch.randint(0, 50000, (4, 128), generator=g).cuda()
    targets = torch.randint(0, 50000, (4,), generator=g).cuda()
    return ids, targets


# ------------------------------------------------------------------ test A


def test_A_model_revision_and_config(model_and_tok):
    model, _ = model_and_tok
    verify_config(model)
    assert not model.training
    assert_frozen(model)
    assert all(not p.requires_grad for p in model.parameters())
    assert next(model.parameters()).dtype == torch.float16


def test_A_provenance_recorded():
    p = os.path.join(ART, "model_provenance.json")
    if not os.path.exists(p):
        pytest.skip("provenance not yet written")
    d = json.load(open(p))
    assert d["model_revision_resolved"] == "9b0c1aa87e34a20052389dce1f0cf01da783f654"
    assert d["config_mismatches"] == []
    assert d["quantized"] is False


# ------------------------------------------------------------------ test B


def test_B_native_forced_route_reproduces_stock(model_and_tok):
    """Forcing the stock-selected eight experts reproduces the stock MoE output."""
    model, _ = model_and_tok
    for layer_idx in [SOURCE_LAYER] + [c for _, c in HORIZONS.values()]:
        block = model.model.layers[layer_idx].mlp
        x = torch.randn(6, 1, 2048,
                        generator=torch.Generator().manual_seed(SEED + layer_idx)).half().cuda()
        stock_out, stock_logits = block(x)

        probs = probs_from_logits(block.gate(x.view(-1, 2048)))
        ids = native_top_identities(probs)
        forced_out, forced_logits = _generic_moe_forward(
            block, x, {i: ids[i] for i in range(x.shape[0])})

        assert torch.allclose(stock_logits, forced_logits, atol=HIDDEN_ATOL)
        assert torch.allclose(stock_out, forced_out, atol=HIDDEN_ATOL), \
            f"layer {layer_idx} max diff {(stock_out - forced_out).abs().max()}"


def test_B_no_override_is_bitwise_native(model_and_tok):
    model, _ = model_and_tok
    block = model.model.layers[SOURCE_LAYER].mlp
    x = torch.randn(3, 5, 2048, generator=torch.Generator().manual_seed(SEED)).half().cuda()
    a, la = block(x)
    b, lb = _generic_moe_forward(block, x, None)
    assert torch.equal(la, lb) and torch.equal(a, b)


def test_B_native_loss_equivalence(model_and_tok, batch):
    """The non-intervened path reproduces the ordinary next-token loss."""
    model, _ = model_and_tok
    ids, targets = batch
    for _, target_code in HORIZONS.values():
        with torch.inference_mode():
            cap = native_forward(model, ids, targets, [target_code])
            forced = native_route_equivalence_nll(
                model, ids, targets, target_code, probs_from_logits(cap.g[target_code]))
        assert torch.allclose(cap.token_nll, forced, atol=NLL_ATOL), \
            f"layer {target_code} max diff {(cap.token_nll - forced).abs().max()}"


# ------------------------------------------------------------------ test C


def test_C_rejected_ranks_are_9_to_12_unrenormalized(model_and_tok):
    model, _ = model_and_tok
    block = model.model.layers[SOURCE_LAYER].mlp
    x = torch.randn(5, 2048, generator=torch.Generator().manual_seed(SEED)).half().cuda()
    probs = router_probs(block, x)
    ids, p = rejected_identities(probs)
    ranked = torch.topk(probs, 12, dim=-1)

    assert ids.shape == (5, 4)
    for row in range(5):
        for slot, rank in enumerate(REJECTED_RANKS):
            assert ids[row, slot] == ranked.indices[row, rank - 1]
            assert torch.isclose(p[row, slot], probs[row, ids[row, slot]])
        assert p[row].sum() < 1.0, "weak experts must stay weak"
        native = set(native_top_identities(probs)[row].tolist())
        assert not (set(ids[row].tolist()) & native), "ranks 9-12 disjoint from top-8"


def test_C_rejected_evidence_matches_reference(model_and_tok):
    model, _ = model_and_tok
    block = model.model.layers[SOURCE_LAYER].mlp
    x = torch.randn(7, 2048, generator=torch.Generator().manual_seed(SEED)).half().cuda()
    with torch.inference_mode():
        fast = rejected_evidence(block, x)
        ref = rejected_evidence_reference(block, x)
    assert fast.shape == (7, 2048)
    assert torch.allclose(fast, ref, atol=1e-4)


def test_C_expert_matches_model_module(model_and_tok):
    model, _ = model_and_tok
    block = model.model.layers[SOURCE_LAYER].mlp
    x = torch.randn(3, 2048, generator=torch.Generator().manual_seed(SEED)).half().cuda()
    for j in (0, 31, 63):
        e = block.experts[j]
        manual = e.down_proj(e.act_fn(e.gate_proj(x)) * e.up_proj(x))
        assert torch.equal(block.experts[j](x), manual)


# ------------------------------------------------------------------ test D


def test_D_intervention_changes_only_target_token(model_and_tok):
    """Overriding one token changes that token's route and no other token's.

    Isolation is asserted on the routing decisions (exact) and on other tokens'
    outputs (FP16 tolerance). Bitwise equality is deliberately NOT required for
    non-target tokens: the MoE loops over experts and evaluates each on the batch
    of tokens routed to it, so moving the target token between experts changes
    those experts' batch composition and hence the cuBLAS GEMM reduction order for
    tokens sharing them. That perturbs unrelated tokens by about one FP16 ULP
    (~6e-5 here) with identical routing and identical mathematics. Forcing the
    target's own native identities is bitwise identical everywhere, confirming the
    effect comes from batch composition and not from routing leakage.
    """
    model, _ = model_and_tok
    block = model.model.layers[HORIZONS[4][1]].mlp
    x = torch.randn(1, 6, 2048, generator=torch.Generator().manual_seed(SEED)).half().cuda()

    base, _ = _generic_moe_forward(block, x, None)
    probs = router_probs(block, x.view(-1, 2048))
    tok = 3

    # A semantic no-op route must be bitwise identical for every token.
    noop = {tok: native_top_identities(probs)[tok]}
    same, _ = _generic_moe_forward(block, x, noop)
    for t in range(6):
        assert torch.equal(base[0, t], same[0, t]), f"no-op route perturbed token {t}"

    ov = build_override(probs[tok : tok + 1], [tok], [9])
    pert, _ = _generic_moe_forward(block, x, ov)

    native_ids = native_top_identities(probs)
    for t in range(6):
        if t == tok:
            # The target must change materially, well beyond FP16 noise.
            assert (base[0, t] - pert[0, t]).abs().max() > 1e-3
        else:
            # Routing is exactly unchanged, and the output moves at most by
            # accumulation-order noise.
            assert torch.equal(native_ids[t], native_top_identities(probs)[t])
            assert torch.allclose(base[0, t], pert[0, t], atol=1e-3), \
                f"token {t} moved beyond FP16 accumulation noise"


def test_D_route_keeps_ranks_1_7_and_swaps_rank_8(model_and_tok):
    """Routes compared as sets: FP16 router probabilities can tie exactly, so
    topk order is not canonical while the selected experts are."""
    model, _ = model_and_tok
    block = model.model.layers[HORIZONS[2][1]].mlp
    x = torch.randn(5, 2048, generator=torch.Generator().manual_seed(SEED)).half().cuda()
    probs = router_probs(block, x)
    ranked12 = torch.topk(probs, 12, dim=-1).indices

    for rank in REJECTED_RANKS:
        ov = build_override(probs, list(range(5)), [rank] * 5)
        for row in range(5):
            forced = ov[row]
            keep7 = set(ranked12[row, :7].tolist())
            assert len(forced) == TOP_K
            assert len(set(forced.tolist())) == TOP_K
            assert set(forced[:7].tolist()) == keep7
            assert forced[7] == ranked12[row, rank - 1]
            assert int(forced[7]) not in keep7


# ------------------------------------------------------------------ test E


def test_E_intermediate_layers_remain_native(model_and_tok, batch):
    """A forced route at a distant target leaves every earlier layer identical."""
    model, _ = model_and_tok
    ids, targets = batch
    target_code = HORIZONS[8][1]  # Layer 12, code 11
    intermediates = [SOURCE_LAYER, 4, 5, 6, 7, 8, 9, 10]

    with torch.inference_mode():
        cap = native_forward(model, ids, targets, intermediates + [target_code])
        ov = build_override(
            probs_from_logits(cap.g[target_code]),
            [flat_index(i, ids.shape[1]) for i in range(ids.shape[0])],
            [9] * ids.shape[0])
        with forced_route(model, target_code, ov):
            cap2 = native_forward(model, ids, targets, intermediates + [target_code])

    for li in intermediates:
        assert torch.equal(cap.h[li], cap2.h[li]), f"layer {li} hidden state changed"
        assert torch.equal(cap.g[li], cap2.g[li]), f"layer {li} router logits changed"
    # Router logits at the target are computed before the override.
    assert torch.equal(cap.g[target_code], cap2.g[target_code])
    # The target's post-layer hidden state must change.
    assert not torch.equal(cap.h[target_code], cap2.h[target_code])


# ------------------------------------------------------------------ test F


def test_F_source_features_independent_of_horizon(model_and_tok, batch):
    """Layer-4 b and r do not depend on which horizon layers are also captured."""
    model, _ = model_and_tok
    ids, targets = batch

    with torch.inference_mode():
        cap_a = native_forward(model, ids, targets, [SOURCE_LAYER, HORIZONS[1][1], HORIZONS[2][1]])
        r_a = rejected_evidence(model.model.layers[SOURCE_LAYER].mlp, cap_a.x[SOURCE_LAYER])
        cap_b = native_forward(model, ids, targets, [SOURCE_LAYER, HORIZONS[4][1], HORIZONS[8][1]])
        r_b = rejected_evidence(model.model.layers[SOURCE_LAYER].mlp, cap_b.x[SOURCE_LAYER])
        cap_c = native_forward(model, ids, targets, [SOURCE_LAYER])
        r_c = rejected_evidence(model.model.layers[SOURCE_LAYER].mlp, cap_c.x[SOURCE_LAYER])

    for name, cap, r in (("A", cap_a, r_a), ("B", cap_b, r_b)):
        assert torch.equal(cap.h[SOURCE_LAYER], cap_c.h[SOURCE_LAYER]), f"h differs for {name}"
        assert torch.equal(cap.g[SOURCE_LAYER], cap_c.g[SOURCE_LAYER]), f"g differs for {name}"
        assert torch.equal(r, r_c), f"r differs for {name}"


# ------------------------------------------------------------------ test G


def test_G_zero_overlap_with_prior_projects():
    """REDV-V1 / REDV-V2 / DREV manifests have zero overlap."""
    used, counts = load_prior_used_keys()
    assert counts["redv_v1"] == 1024
    assert counts["redv_v2"] == 2048
    assert counts["union"] == 3072, "V1 and V2 must not overlap each other"

    p = os.path.join(ART, "data_manifest.json")
    if not os.path.exists(p):
        pytest.skip("DREV manifest not built yet")
    m = json.load(open(p))
    fit = set(zip(m["fit"]["source_splits"], m["fit"]["block_ids"]))
    test = set(zip(m["test"]["source_splits"], m["test"]["block_ids"]))

    assert len(fit) == N_FIT and len(test) == N_TEST
    assert not (fit & test), "DREV fit and test must be disjoint"
    assert not ((fit | test) & used), "DREV must not reuse any REDV block"
    assert m["train_split_used"] is False


def test_G_blocks_non_overlapping_129():
    blocks = make_blocks(list(range(129 * 8 + 5)))
    assert len(blocks) == 8
    starts = [b.stream_start for b in blocks]
    assert all(starts[i + 1] - starts[i] == 129 for i in range(len(starts) - 1))
    assert all(len(b.tokens) == 129 for b in blocks)


# ------------------------------------------------------------------ test H


def test_H_pca_fit_on_fit_set_only():
    """The projection is fit on fit data alone; test data never informs it."""
    rs = np.random.RandomState(0)
    X_fit = rs.randn(200, 300)
    X_test = rs.randn(120, 300) * 5.0 + 10.0  # deliberately different distribution

    proj = FrozenProjection.fit(X_fit)
    mean_before = proj.scaler.mean_.copy()
    comp_before = proj.pca.components_.copy()

    _ = proj.transform(X_test)
    assert np.array_equal(proj.scaler.mean_, mean_before), "scaler must not refit"
    assert np.array_equal(proj.pca.components_, comp_before), "PCA must not refit"

    # Fitting on fit-only differs from fitting on the union: the union would leak.
    proj_leaky = FrozenProjection.fit(np.vstack([X_fit, X_test]))
    assert not np.allclose(proj.pca.components_, proj_leaky.pca.components_)

    assert proj.transform(X_fit).shape == (200, PCA_COMPONENTS)
    assert proj.transform(X_test).shape == (120, PCA_COMPONENTS)


def test_H_pca_is_unsupervised():
    """PCA sees features only; the target is never passed in."""
    import inspect

    src = inspect.getsource(FrozenProjection.fit)
    assert "y" not in inspect.signature(FrozenProjection.fit).parameters
    assert "G" not in src


def test_H_baseline_matrix_dimension():
    h = np.zeros((10, 2048))
    g = np.zeros((10, 64))
    assert baseline_matrix(h, g).shape == (10, 2112)


# ------------------------------------------------------------------ test I


def test_I_real_and_shuffled_are_128_dimensional():
    from drev.probes import probe_input

    n = 40
    rs = np.random.RandomState(1)
    z_b = rs.randn(n, PCA_COMPONENTS)
    z_r = rs.randn(n, PCA_COMPONENTS)
    perm = derange(n, np.random.default_rng(SHUFFLE_SEED))

    X_real = probe_input(z_b, z_r)
    X_shuf = probe_input(z_b, apply_shuffle(z_r, perm))
    assert X_real.shape == X_shuf.shape == (n, PROBE_DIM)
    assert PROBE_DIM == 128


# ------------------------------------------------------------------ test J


def test_J_shuffle_has_zero_fixed_points():
    for n in (2, 3, 17, 512):
        p = derange(n, np.random.default_rng(SHUFFLE_SEED))
        assert sorted(p.tolist()) == list(range(n))
        assert not np.any(p == np.arange(n)), f"fixed point at n={n}"


def test_J_shuffle_permutations_deterministic_and_shared():
    """One fixed pair, reused for every horizon."""
    a_fit, a_test = build_shuffle_permutations(N_FIT, N_TEST)
    b_fit, b_test = build_shuffle_permutations(N_FIT, N_TEST)
    assert np.array_equal(a_fit, b_fit) and np.array_equal(a_test, b_test)
    assert not np.any(a_fit == np.arange(N_FIT))
    assert not np.any(a_test == np.arange(N_TEST))
    assert not np.array_equal(a_fit, a_test), "fit and test draws are independent"


def test_J_apply_shuffle_rejects_identity():
    z = np.random.RandomState(0).randn(10, 4)
    with pytest.raises(AssertionError):
        apply_shuffle(z, np.arange(10))


# ------------------------------------------------------------------ test K


def test_K_shuffle_is_exact_row_permutation():
    rs = np.random.RandomState(5)
    z_r = rs.randn(256, PCA_COMPONENTS)
    perm = derange(256, np.random.default_rng(SHUFFLE_SEED))
    s = apply_shuffle(z_r, perm)

    def row_hashes(m):
        return sorted(hashlib.sha256(row.tobytes()).hexdigest() for row in m)

    assert row_hashes(z_r) == row_hashes(s)
    assert np.allclose(np.sort(z_r, axis=0), np.sort(s, axis=0))
    assert np.allclose(z_r.mean(axis=0), s.mean(axis=0))
    assert np.allclose(z_r.std(axis=0), s.std(axis=0))
    assert not np.allclose(z_r, s), "correspondence must actually be broken"


# ------------------------------------------------------------------ test L


def test_L_smoke_produces_no_verdict_statistic():
    """The smoke script must not fit PCA/probes or compute R2 or D.

    Checks executable code only: comments and docstrings are stripped first, so
    prose that merely mentions a forbidden name does not trip the check.
    """
    import ast

    p = os.path.join(os.path.dirname(__file__), "..", "scripts", "smoke_drev.py")
    tree = ast.parse(open(p).read())

    # Collect every referenced name, attribute, and imported symbol.
    referenced = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            referenced.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            referenced.update(a.name for a in node.names)

    for forbidden in ("FrozenProjection", "run_probe_pair", "r2_score", "verdict",
                      "summarize", "fit_and_evaluate", "Ridge", "PCA",
                      "build_shuffle_permutations", "apply_shuffle"):
        assert forbidden not in referenced, \
            f"smoke script must not use {forbidden} in executable code"


def test_L_frozen_constants():
    assert SOURCE_LAYER == 3
    assert HORIZONS == {1: (5, 4), 2: (6, 5), 4: (8, 7), 8: (12, 11)}
    assert REJECTED_RANKS == (9, 10, 11, 12)
    assert PCA_COMPONENTS == 64 and PROBE_DIM == 128
    assert N_FIT == 512 and N_TEST == 512
    assert SHUFFLE_SEED == 271828


def test_L_pilot_rule_mechanics():
    from drev.analysis import NOT_PROMISING, PROMISING, verdict

    def mk(d1, delayed_ds, delayed_reals):
        per = {1: {"R2_real": 0.05, "D": d1}}
        for dist, d, real in zip((2, 4, 8), delayed_ds, delayed_reals):
            per[dist] = {"R2_real": real, "D": d}
        return per

    # All conditions satisfied.
    assert verdict(mk(0.0, [0.03, 0.03, 0.03], [0.05, 0.05, 0.05])) == PROMISING
    # Condition 1: fewer than two delayed R2_real > 0.
    assert verdict(mk(0.0, [0.03, 0.03, 0.03], [-0.01, -0.01, 0.05])) == NOT_PROMISING
    # Condition 2: fewer than two delayed D > 0.
    assert verdict(mk(0.0, [-0.01, -0.01, 0.30], [0.05, 0.05, 0.05])) == NOT_PROMISING
    # Condition 3: mean delayed D below threshold.
    assert verdict(mk(0.0, [0.01, 0.01, 0.01], [0.05, 0.05, 0.05])) == NOT_PROMISING
    # Condition 4: delayed mean must exceed the immediate D.
    assert verdict(mk(0.50, [0.03, 0.03, 0.03], [0.05, 0.05, 0.05])) == NOT_PROMISING
    # Exactly two positives is enough for conditions 1 and 2.
    assert verdict(mk(0.0, [0.05, 0.05, -0.01], [0.05, 0.05, -0.01])) == PROMISING
    with pytest.raises(ValueError):
        verdict({1: {"R2_real": 0.0, "D": 0.0}})
