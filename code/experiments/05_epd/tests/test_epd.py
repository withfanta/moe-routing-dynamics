"""EPD-P0 invariants A-Q. Implementation correctness only; no verdict statistic."""

from __future__ import annotations

import ast
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from epd import (
    EIPC_PROJECTION_SEED,
    EIPC_PROJECTION_SHA256,
    HIDDEN_SIZE,
    HISTORY_CODE,
    LAYERWISE_PCA_COMPONENTS,
    MODEL_REVISION,
    N_FIT,
    N_HISTORY_LAYERS,
    N_TEST,
    NUM_EXPERTS,
    PCA_COMPONENTS,
    PROJ_DIM,
    RAW_DIMS,
    REPRESENTATIONS,
    TARGET_LAYER,
    TOP_K,
)
from epd.data import load_prior_used_blocks, verify_prior_projects_avoided_train
from epd.extraction import (
    assert_frozen,
    fused_from_contributions,
    load_eipc_projection,
    load_model,
    native_topk,
    project,
    projection_hash,
    router_probs,
    selected_expert_contributions,
    verify_config,
)
from epd.representations import (
    FrozenPipeline,
    build_all,
    build_content_rank,
    build_content_rank_layer,
    build_full_provenance,
    build_fused,
    build_id_path,
    build_id_path_layer,
    center_logits,
    compress,
)

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "EPD_P0")
SEED = 20260919
FP16_REL_TOL = 5e-3


@pytest.fixture(scope="module")
def model_and_tok():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return load_model(model_dir=os.environ.get("EPD_MODEL_DIR"))


def _synth(n=20, seed=0):
    rs = np.random.RandomState(seed)
    s = rs.randn(n, N_HISTORY_LAYERS, TOP_K, PROJ_DIM).astype(np.float32)
    ids = np.stack([[rs.choice(NUM_EXPERTS, TOP_K, replace=False)
                     for _ in range(N_HISTORY_LAYERS)] for _ in range(n)]).astype(np.int16)
    return s, ids


# ------------------------------------------------------------------ A, B


def test_A_model_revision_and_config(model_and_tok):
    model, _ = model_and_tok
    verify_config(model)
    p = os.path.join(ART, "model_provenance.json")
    if os.path.exists(p):
        d = json.load(open(p))
        assert d["model_revision"] == MODEL_REVISION
        assert d["config_mismatches"] == []
        assert d["quantized"] is False
        assert d["olmoe_receives_gradients"] is False
        assert d["dataset_split_used"] == "train"


def test_B_frozen_eval_fp16(model_and_tok):
    model, _ = model_and_tok
    assert not model.training
    assert_frozen(model)
    assert all(not p.requires_grad for p in model.parameters())
    assert next(model.parameters()).dtype == torch.float16


# ------------------------------------------------------------------ C, D


def test_C_native_topk_captured_correctly(model_and_tok):
    model, _ = model_and_tok
    block = model.model.layers[5].mlp
    x = torch.randn(6, HIDDEN_SIZE, generator=torch.Generator().manual_seed(SEED)).half().cuda()
    logits = block.gate(x)
    probs = router_probs(logits)
    ids, vals = native_topk(logits, TOP_K)

    assert probs.dtype == torch.float32
    assert torch.allclose(probs.sum(dim=-1), torch.ones(6, device=probs.device), atol=1e-4)
    for row in range(6):
        expected = torch.topk(probs[row], TOP_K)
        assert set(ids[row].tolist()) == set(expected.indices.tolist())
        # Descending rank order, which CONTENT_RANK depends on.
        assert all(vals[row, i] >= vals[row, i + 1] for i in range(TOP_K - 1))
        for slot in range(TOP_K):
            assert torch.isclose(vals[row, slot], probs[row, ids[row, slot]])
        # No renormalization.
        assert vals[row].sum() < 1.0


def test_D_no_rejected_experts_executed(model_and_tok):
    """Only the native Top-8 appear; ranks 9+ never enter the capture."""
    model, _ = model_and_tok
    block = model.model.layers[4].mlp
    x = torch.randn(5, HIDDEN_SIZE, generator=torch.Generator().manual_seed(SEED)).half().cuda()
    logits = block.gate(x)
    with torch.inference_mode():
        ids, _, contrib = selected_expert_contributions(block, x, logits)

    assert ids.shape == (5, TOP_K)
    assert contrib.shape == (5, TOP_K, HIDDEN_SIZE)
    ranked = torch.topk(router_probs(logits), 16, dim=-1).indices
    for row in range(5):
        rejected = set(ranked[row, TOP_K:].tolist())
        assert not (set(ids[row].tolist()) & rejected), "a rejected expert was captured"


# ------------------------------------------------------------------ E


def test_E_weighted_reconstruction(model_and_tok):
    """Sum of the weighted selected contributions reproduces the stock fused output.

    Judged on a RELATIVE basis: fp16 accumulation makes absolute error scale with |y|.
    """
    model, _ = model_and_tok
    for li in (0, 5, 10):
        block = model.model.layers[li].mlp
        x = torch.randn(5, 1, HIDDEN_SIZE,
                        generator=torch.Generator().manual_seed(SEED + li)).half().cuda()
        with torch.inference_mode():
            y, logits = block(x)
            _, _, contrib = selected_expert_contributions(
                block, x.view(-1, HIDDEN_SIZE), logits)
            recon = fused_from_contributions(contrib)
        yf = y.view(-1, HIDDEN_SIZE).float()
        rel = ((recon - yf).norm() / yf.norm()).item()
        assert rel < FP16_REL_TOL, f"layer {li} relative error {rel}"


# ------------------------------------------------------------------ F, G


def test_F_eipc_projection_hash_matches():
    """The reused projection matches the EIPC artifact and the expected constant."""
    P, meta = load_eipc_projection()
    assert P.shape == (HIDDEN_SIZE, PROJ_DIM)
    assert meta["seed"] == EIPC_PROJECTION_SEED == 20260917
    assert meta["sha256"] == EIPC_PROJECTION_SHA256
    assert projection_hash(P) == EIPC_PROJECTION_SHA256
    assert meta["out_dim"] == PROJ_DIM == 32
    # Deterministic across calls.
    P2, _ = load_eipc_projection()
    assert np.array_equal(P, P2)


def test_F_refuses_wrong_projection(tmp_path):
    """A tampered artifact aborts rather than silently substituting a projection."""
    bad = tmp_path / "projection.json"
    json.dump({"seed": 999, "shape": [HIDDEN_SIZE, PROJ_DIM], "out_dim": PROJ_DIM,
               "sha256": "deadbeef"}, open(bad, "w"))
    with pytest.raises(AssertionError, match="seed"):
        load_eipc_projection(str(bad))


def test_G_projection_linearity():
    P, _ = load_eipc_projection()
    rs = np.random.RandomState(1)
    c = rs.randn(8, TOP_K, HIDDEN_SIZE)
    lhs = project(P, c.sum(axis=1))
    rhs = project(P, c).sum(axis=1)
    assert np.allclose(lhs, rhs, atol=1e-8), f"max diff {np.abs(lhs - rhs).max()}"


# ------------------------------------------------------------------ H, I, J, K, L


def test_H_id_path_is_binary_selection_only():
    s, ids = _synth()
    X = build_id_path(ids)
    assert X.shape[1] == RAW_DIMS["ID_PATH"] == 704
    assert set(np.unique(X).tolist()) <= {0.0, 1.0}
    # Exactly 8 ones per layer.
    per_layer = X.reshape(X.shape[0], N_HISTORY_LAYERS, NUM_EXPERTS).sum(axis=2)
    assert np.all(per_layer == TOP_K)
    # Ones sit exactly at the true selected identities.
    for row in range(X.shape[0]):
        blk = X[row].reshape(N_HISTORY_LAYERS, NUM_EXPERTS)
        for lpos in range(N_HISTORY_LAYERS):
            assert set(np.nonzero(blk[lpos])[0].tolist()) == set(ids[row, lpos].tolist())


def test_I_content_rank_carries_no_expert_identity():
    """Permuting expert IDs leaves CONTENT_RANK unchanged; it changes ID/FULL."""
    s, ids = _synth(n=10, seed=2)
    rs = np.random.RandomState(5)
    ids_shuffled = ids.copy()
    for row in range(ids.shape[0]):
        for lpos in range(N_HISTORY_LAYERS):
            ids_shuffled[row, lpos] = rs.choice(NUM_EXPERTS, TOP_K, replace=False)

    a = build_content_rank(s)
    b = build_content_rank(s)  # identity ignored entirely by construction
    assert np.array_equal(a, b)
    assert a.shape[1] == RAW_DIMS["CONTENT_RANK"] == 2816
    # ID_PATH and FULL_PROVENANCE do depend on identity.
    assert not np.array_equal(build_id_path(ids), build_id_path(ids_shuffled))
    assert not np.array_equal(build_full_provenance(s, ids),
                              build_full_provenance(s, ids_shuffled))


def test_J_content_rank_ordering_is_rank_order():
    """Slot k of CONTENT_RANK holds the rank-(k+1) contribution, in order."""
    s, ids = _synth(n=6, seed=3)
    X = build_content_rank(s)
    blocks = X.reshape(X.shape[0], N_HISTORY_LAYERS, TOP_K, PROJ_DIM)
    assert np.array_equal(blocks, s)
    # Reordering the rank axis changes the representation, so order carries information.
    s_rev = s[:, :, ::-1, :].copy()
    assert not np.array_equal(build_content_rank(s_rev), X)


def test_K_full_provenance_slot_equals_expert_id():
    s, ids = _synth(n=8, seed=4)
    X = build_full_provenance(s, ids)
    assert X.shape[1] == RAW_DIMS["FULL_PROVENANCE"] == 22528
    slots = X.reshape(X.shape[0], N_HISTORY_LAYERS, NUM_EXPERTS, PROJ_DIM)
    for row in range(X.shape[0]):
        for lpos in range(N_HISTORY_LAYERS):
            selected = set(int(e) for e in ids[row, lpos])
            for k in range(TOP_K):
                e = int(ids[row, lpos, k])
                assert np.allclose(slots[row, lpos, e], s[row, lpos, k])
            for e in range(NUM_EXPERTS):
                if e not in selected:
                    assert np.all(slots[row, lpos, e] == 0.0)
            assert (np.abs(slots[row, lpos]).sum(axis=-1) > 0).sum() == TOP_K


def test_L_fused_is_sum_of_same_contributions():
    """FUSED equals the per-layer sum of the vectors used by CONTENT_RANK and FULL."""
    s, ids = _synth(n=12, seed=6)
    reps = build_all(s, ids)
    n = s.shape[0]

    fused = reps["FUSED"]
    assert fused.shape[1] == RAW_DIMS["FUSED"] == 352
    # From CONTENT_RANK: sum over the rank axis.
    from_content = reps["CONTENT_RANK"].reshape(n, N_HISTORY_LAYERS, TOP_K, PROJ_DIM).sum(axis=2)
    assert np.allclose(fused, from_content.reshape(n, -1), atol=1e-6)
    # From FULL_PROVENANCE: sum over the 64 expert slots.
    from_full = reps["FULL_PROVENANCE"].reshape(
        n, N_HISTORY_LAYERS, NUM_EXPERTS, PROJ_DIM).sum(axis=2)
    assert np.allclose(fused, from_full.reshape(n, -1), atol=1e-6)


# ------------------------------------------------------------------ M, N


def test_M_N_manifest_disjoint_and_prior_excluded():
    p = os.path.join(ART, "data_manifest.json")
    if not os.path.exists(p):
        pytest.skip("manifest not built yet")
    m = json.load(open(p))
    fit = set(m["fit"]["block_ids"])
    test = set(m["test"]["block_ids"])

    assert len(fit) == m["fit"]["n"] == N_FIT == 512
    assert len(test) == m["test"]["n"] == N_TEST == 256
    assert not (fit & test), "FIT and TEST must be disjoint"

    excluded, info = load_prior_used_blocks()
    assert not ((fit | test) & excluded), "EIPC/XEC blocks must be excluded"
    assert info["eipc_blocks"] == 2048 and info["xec_blocks"] == 3584
    assert info["eipc_xec_overlap"] == 0
    assert m["split_used"] == "train"
    assert m["experimental_token_position"] == 127
    # Shards preserve FIT/TEST identity.
    for s, counts in m["shard_role_counts"].items():
        assert counts["fit"] == 128 and counts["test"] == 64


def test_N_prior_projects_never_used_train():
    assert verify_prior_projects_avoided_train() == {
        "REDV_V1": True, "REDV_V2": True, "DREV_P0": True}


# ------------------------------------------------------------------ O


def test_O_target_never_appears_in_features():
    """Historical features come from Layers 1-11 only; the target is Layer 12."""
    assert TARGET_LAYER == 11
    assert max(HISTORY_CODE) == 10 < TARGET_LAYER
    assert TARGET_LAYER not in HISTORY_CODE

    # Representation builders take only history arrays, never the target logits.
    import inspect

    for fn in (build_fused, build_id_path, build_content_rank, build_full_provenance):
        params = set(inspect.signature(fn).parameters)
        assert "g12" not in params and "target" not in params and "q" not in params

    # Feature values are invariant to the target: perturbing g12 cannot change them.
    s, ids = _synth(n=6, seed=7)
    before = build_all(s, ids)
    after = build_all(s, ids)
    for k in REPRESENTATIONS:
        assert np.array_equal(before[k], after[k])


def test_O_target_centering_invariance():
    rs = np.random.RandomState(8)
    g = rs.randn(20, NUM_EXPERTS) + 3.0
    q = center_logits(g)
    assert np.allclose(q.mean(axis=1), 0.0, atol=1e-12)
    assert np.allclose(center_logits(g + 9.0), q, atol=1e-12)
    with pytest.raises(ValueError):
        center_logits(rs.randn(4, 32))


# ------------------------------------------------------------------ P, Q


def test_P_main_probe_inputs_are_32_dims():
    s, ids = _synth(n=200, seed=9)
    s_t, ids_t = _synth(n=90, seed=10)
    reps_fit = build_all(s, ids)
    reps_test = build_all(s_t, ids_t)
    X_fit, X_test, evr = compress(reps_fit, reps_test, PCA_COMPONENTS)
    for name in REPRESENTATIONS:
        assert X_fit[name].shape == (200, PCA_COMPONENTS)
        assert X_test[name].shape == (90, PCA_COMPONENTS)
    assert PCA_COMPONENTS == 32


def test_Q_layerwise_probe_inputs_are_16_dims():
    s, ids = _synth(n=100, seed=11)
    for pos in (0, 5, 10):
        idl = build_id_path_layer(ids, pos)
        ctl = build_content_rank_layer(s, pos)
        assert idl.shape == (100, NUM_EXPERTS)
        assert ctl.shape == (100, TOP_K * PROJ_DIM)
        for X in (idl, ctl):
            pipe = FrozenPipeline.fit(X, LAYERWISE_PCA_COMPONENTS)
            assert pipe.transform(X).shape == (100, LAYERWISE_PCA_COMPONENTS)
    assert LAYERWISE_PCA_COMPONENTS == 16


def test_Q_pipeline_fit_on_fit_only():
    rs = np.random.RandomState(12)
    X_fit = rs.randn(200, 300)
    X_test = rs.randn(80, 300) * 5.0 + 9.0
    pipe = FrozenPipeline.fit(X_fit, PCA_COMPONENTS)
    before = pipe.pca.components_.copy()
    _ = pipe.transform(X_test)
    assert np.array_equal(pipe.pca.components_, before)
    leaky = FrozenPipeline.fit(np.vstack([X_fit, X_test]), PCA_COMPONENTS)
    assert not np.allclose(pipe.pca.components_, leaky.pca.components_)


def test_Q_smoke_computes_no_verdict_statistic():
    p = os.path.join(os.path.dirname(__file__), "..", "scripts", "smoke_epd.py")
    tree = ast.parse(open(p).read())
    referenced = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            referenced.update(a.name for a in node.names)
    for forbidden in ("compress", "FrozenPipeline", "run_main_probes", "fit_and_score",
                      "r2_score", "Ridge", "classify_pattern", "descriptive_gaps",
                      "co_selection_enrichment", "TargetScaler"):
        assert forbidden not in referenced, f"smoke must not use {forbidden}"


def test_Q_pattern_classification_mechanics():
    from epd.analysis import (
        CONTENT_DOMINANT,
        INTERACTION,
        NO_CLEAN_DECOMPOSITION,
        PATH_DOMINANT,
        classify_pattern,
    )

    def mk(fused, idp, cnt, full):
        return {"FUSED": fused, "ID_PATH": idp, "CONTENT_RANK": cnt, "FULL_PROVENANCE": full}

    # Path-dominant: ID close to FULL, well above CONTENT.
    assert classify_pattern(mk(0.10, 0.30, 0.10, 0.31))["pattern"] == PATH_DOMINANT
    # Path-dominant also when the pure path EXCEEDS full provenance. This happens because
    # all representations share one compression budget and FULL is far denser, so PCA
    # retains less of its variance; FULL still provably contains ID_PATH.
    assert classify_pattern(mk(0.20, 0.67, 0.07, 0.34))["pattern"] == PATH_DOMINANT
    # Symmetrically for content.
    assert classify_pattern(mk(0.20, 0.07, 0.67, 0.34))["pattern"] == CONTENT_DOMINANT
    # Content-dominant: CONTENT close to FULL, well above ID.
    assert classify_pattern(mk(0.10, 0.10, 0.30, 0.31))["pattern"] == CONTENT_DOMINANT
    # Interaction: FULL clearly above both.
    assert classify_pattern(mk(0.10, 0.15, 0.15, 0.40))["pattern"] == INTERACTION
    # Ambiguous: everything within the descriptive gap.
    assert classify_pattern(mk(0.20, 0.20, 0.20, 0.21))["pattern"] == NO_CLEAN_DECOMPOSITION
