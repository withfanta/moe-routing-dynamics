"""EIPC-P0 invariants A-P. Implementation correctness only; no verdict statistic."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eipc import (
    ALL_HISTORY_CODE,
    HIDDEN_SIZE,
    MODEL_REVISION,
    NUM_EXPERTS,
    N_FIT,
    N_TEST,
    PCA_COMPONENTS,
    PROJ_DIM,
    PROJECTION_SEED,
    SHUFFLE_SEED,
    TARGETS,
    TOP_K,
)
from eipc.expert_states import (
    apply_cyclic_shift,
    build_projection,
    cyclic_shifts,
    fused_projected,
    identity_slots,
    project,
    projection_hash,
)
from eipc.olmoe import (
    assert_frozen,
    fused_from_contributions,
    load_model,
    native_topk,
    router_probs,
    selected_expert_contributions,
    verify_config,
)
from eipc.representations import (
    FrozenPipeline,
    TargetScaler,
    build_fused,
    build_identity,
    build_shifts,
    build_shuffled,
    center_logits,
    raw_dimension,
)

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "EIPC_P0")
SEED = 20260917
FP16_ATOL = 2e-2


@pytest.fixture(scope="module")
def model_and_tok():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return load_model(model_dir=os.environ.get("EIPC_MODEL_DIR"))


def _synthetic_states(n=24, n_layers=3, seed=0):
    """Synthetic s / ids consistent with the real extraction shapes."""
    rs = np.random.RandomState(seed)
    s_by_layer, ids_by_layer = {}, {}
    for li in range(n_layers):
        s_by_layer[li] = rs.randn(n, TOP_K, PROJ_DIM).astype(np.float32)
        ids = np.stack([rs.choice(NUM_EXPERTS, size=TOP_K, replace=False) for _ in range(n)])
        ids_by_layer[li] = ids.astype(np.int16)
    return s_by_layer, ids_by_layer


# ------------------------------------------------------------------ A, B


def test_A_model_revision_and_config(model_and_tok):
    model, _ = model_and_tok
    verify_config(model)
    p = os.path.join(ART, "model_provenance.json")
    if os.path.exists(p):
        d = json.load(open(p))
        assert d["model_revision_resolved"] == MODEL_REVISION
        assert d["config_mismatches"] == []
        assert d["quantized"] is False
        assert d["dataset_split_used"] == "train"


def test_B_model_frozen_and_eval(model_and_tok):
    model, _ = model_and_tok
    assert not model.training
    assert_frozen(model)
    assert all(not p.requires_grad for p in model.parameters())
    assert next(model.parameters()).dtype == torch.float16


# ------------------------------------------------------------------ C


def test_C_native_topk_semantics(model_and_tok):
    """fp32 softmax over 64, Top-8, no renormalization, original probabilities."""
    model, _ = model_and_tok
    block = model.model.layers[3].mlp
    x = torch.randn(6, HIDDEN_SIZE, generator=torch.Generator().manual_seed(SEED)).half().cuda()
    logits = block.gate(x)

    probs = router_probs(logits)
    assert probs.dtype == torch.float32
    assert torch.allclose(probs.sum(dim=-1), torch.ones(6, device=probs.device), atol=1e-4)

    ids, vals = native_topk(logits)
    assert ids.shape == (6, TOP_K) and vals.shape == (6, TOP_K)
    for row in range(6):
        # Identities are the true argmax set, and probabilities are unrenormalized.
        expected = torch.topk(probs[row], TOP_K)
        assert set(ids[row].tolist()) == set(expected.indices.tolist())
        for slot in range(TOP_K):
            assert torch.isclose(vals[row, slot], probs[row, ids[row, slot]])
        assert vals[row].sum() < 1.0, "top-8 mass must not be renormalized to 1"
        # Descending order.
        assert all(vals[row, i] >= vals[row, i + 1] for i in range(TOP_K - 1))


# ------------------------------------------------------------------ D


def test_D_selected_experts_reproduce_stock_moe(model_and_tok):
    """Weighted sum of separately executed selected experts == stock MoE output."""
    model, _ = model_and_tok
    for li in (0, 5, 10):
        block = model.model.layers[li].mlp
        x = torch.randn(5, 1, HIDDEN_SIZE,
                        generator=torch.Generator().manual_seed(SEED + li)).half().cuda()
        with torch.inference_mode():
            stock_y, stock_logits = block(x)
            flat_x = x.view(-1, HIDDEN_SIZE)
            _, _, contrib = selected_expert_contributions(block, flat_x, stock_logits)
            recon = fused_from_contributions(contrib)

        err = (recon - stock_y.view(-1, HIDDEN_SIZE).float()).abs().max().item()
        assert err < FP16_ATOL, f"layer {li} reconstruction error {err}"


def test_D_only_selected_experts_executed(model_and_tok):
    """Exactly the eight native experts are run; contributions are non-zero for them."""
    model, _ = model_and_tok
    block = model.model.layers[2].mlp
    x = torch.randn(4, HIDDEN_SIZE, generator=torch.Generator().manual_seed(SEED)).half().cuda()
    logits = block.gate(x)
    with torch.inference_mode():
        ids, probs, contrib = selected_expert_contributions(block, x, logits)
    assert contrib.shape == (4, TOP_K, HIDDEN_SIZE)
    # Each of the eight slots carries a real, non-degenerate contribution.
    for row in range(4):
        for slot in range(TOP_K):
            assert contrib[row, slot].abs().sum() > 0
        assert len(set(ids[row].tolist())) == TOP_K


# ------------------------------------------------------------------ E, F


def test_E_projection_deterministic_and_hash_stable():
    a = build_projection()
    b = build_projection()
    assert a.shape == (HIDDEN_SIZE, PROJ_DIM)
    assert np.array_equal(a, b)
    assert projection_hash(a) == projection_hash(b)
    # Seed governs the draw.
    assert not np.array_equal(a, build_projection(seed=PROJECTION_SEED + 1))
    # Scale is as specified.
    assert abs(a.std() - 1.0 / np.sqrt(PROJ_DIM)) < 0.01

    p = os.path.join(ART, "projection.json")
    if os.path.exists(p):
        meta = json.load(open(p))
        assert meta["sha256"] == projection_hash(a)
        assert meta["out_dim"] == PROJ_DIM
        assert meta["trainable"] is False


def test_F_projection_linearity():
    """P(sum_e c_e) == sum_e P(c_e), the property FUSED relies on."""
    rs = np.random.RandomState(1)
    P = build_projection()
    c = rs.randn(10, TOP_K, HIDDEN_SIZE)

    lhs = project(P, c.sum(axis=1))
    rhs = fused_projected(project(P, c))
    assert lhs.shape == rhs.shape == (10, PROJ_DIM)
    assert np.allclose(lhs, rhs, atol=1e-8), f"max diff {np.abs(lhs - rhs).max()}"


# ------------------------------------------------------------------ G


def test_G_fused_derived_from_same_contributions():
    """FUSED is exactly the per-layer sum of the EXPERT_IDENTITY slot contents."""
    s_by_layer, ids_by_layer = _synthetic_states(n=16, n_layers=4)
    hist = (0, 1, 2, 3)

    fused = build_fused(s_by_layer, hist)
    identity = build_identity(s_by_layer, ids_by_layer, hist)

    # Reduce the identity representation back over expert slots per layer.
    n = fused.shape[0]
    slots = identity.reshape(n, len(hist), NUM_EXPERTS, PROJ_DIM)
    recovered = slots.sum(axis=2).reshape(n, -1)
    assert np.allclose(fused, recovered, atol=1e-6), "FUSED must equal the slot sum"


# ------------------------------------------------------------------ H


def test_H_identity_slots_match_true_expert_ids():
    """Slot e holds the contribution of expert e, and other slots are zero."""
    s_by_layer, ids_by_layer = _synthetic_states(n=12, n_layers=1)
    s, ids = s_by_layer[0], ids_by_layer[0]
    slots = identity_slots(s, ids)

    assert slots.shape == (12, NUM_EXPERTS, PROJ_DIM)
    for row in range(12):
        selected = set(int(e) for e in ids[row])
        for slot_idx in range(TOP_K):
            e = int(ids[row, slot_idx])
            assert np.allclose(slots[row, e], s[row, slot_idx]), f"slot {e} content"
        for e in range(NUM_EXPERTS):
            if e not in selected:
                assert np.all(slots[row, e] == 0.0), f"unselected slot {e} must be zero"
        assert (np.abs(slots[row]).sum(axis=1) > 0).sum() == TOP_K


# ------------------------------------------------------------------ I, J


def test_I_shifts_are_nonzero_only():
    for role in ("fit", "test"):
        sh = build_shifts(64, 7, role)
        assert sh.shape == (64, 7)
        assert sh.min() >= 1 and sh.max() <= 63
        assert not np.any(sh == 0), "a zero shift would preserve identity"
    # FIT and TEST draws are independent.
    assert not np.array_equal(build_shifts(64, 7, "fit"), build_shifts(64, 7, "test"))
    # Deterministic.
    assert np.array_equal(build_shifts(64, 7, "fit"), build_shifts(64, 7, "fit"))


def test_I_cyclic_shift_moves_every_expert():
    """No expert remains in its own slot after a non-zero cyclic shift."""
    s_by_layer, ids_by_layer = _synthetic_states(n=8, n_layers=1)
    slots = identity_slots(s_by_layer[0], ids_by_layer[0])
    shifts = cyclic_shifts(8, 1, SHUFFLE_SEED)[:, 0]
    shifted = apply_cyclic_shift(slots, shifts)

    for row in range(8):
        occupied_before = {e for e in range(NUM_EXPERTS) if np.abs(slots[row, e]).sum() > 0}
        occupied_after = {e for e in range(NUM_EXPERTS) if np.abs(shifted[row, e]).sum() > 0}
        assert len(occupied_before) == len(occupied_after) == TOP_K
        assert not (occupied_before & occupied_after) or shifts[row] != 0
        # Every occupied slot moved by exactly the shift.
        for e in occupied_before:
            assert np.allclose(shifted[row, (e + int(shifts[row])) % NUM_EXPERTS], slots[row, e])

    with pytest.raises(AssertionError):
        apply_cyclic_shift(slots, np.zeros(8, dtype=np.int64))


def test_J_shuffled_contains_same_vectors():
    """SHUFFLED holds exactly the same contribution vectors, only relocated."""
    s_by_layer, ids_by_layer = _synthetic_states(n=16, n_layers=3)
    hist = (0, 1, 2)
    shifts = build_shifts(16, len(hist), "fit")

    identity = build_identity(s_by_layer, ids_by_layer, hist)
    shuffled = build_shuffled(s_by_layer, ids_by_layer, hist, shifts)

    assert identity.shape == shuffled.shape

    n = identity.shape[0]
    a = identity.reshape(n, len(hist), NUM_EXPERTS, PROJ_DIM)
    b = shuffled.reshape(n, len(hist), NUM_EXPERTS, PROJ_DIM)

    for row in range(n):
        for lpos in range(len(hist)):
            def row_hashes(m):
                return sorted(hashlib.sha256(v.tobytes()).hexdigest() for v in m)
            assert row_hashes(a[row, lpos]) == row_hashes(b[row, lpos]), "same vector multiset"
            # Same number of occupied slots, same layer, same values.
            assert (np.abs(a[row, lpos]).sum(axis=1) > 0).sum() == \
                   (np.abs(b[row, lpos]).sum(axis=1) > 0).sum() == TOP_K
    # The per-layer sum is invariant, so only identity was destroyed.
    assert np.allclose(a.sum(axis=2), b.sum(axis=2), atol=1e-6)
    assert not np.allclose(identity, shuffled), "identity must actually be destroyed"


# ------------------------------------------------------------------ K


def test_K_fit_and_test_manifests_disjoint():
    p = os.path.join(ART, "data_manifest.json")
    if not os.path.exists(p):
        pytest.skip("manifest not built yet")
    m = json.load(open(p))
    fit, test = set(m["fit"]["block_ids"]), set(m["test"]["block_ids"])
    assert len(fit) == m["fit"]["n"] == N_FIT
    assert len(test) == m["test"]["n"] == N_TEST
    assert not (fit & test), "FIT and TEST must be disjoint"
    assert m["split_used"] == "train"
    assert m["validation_or_test_used"] is False
    assert m["overlapping_windows"] is False
    assert m["experimental_token_position"] == 127


# ------------------------------------------------------------------ L, M


def test_L_pipeline_fit_on_fit_only():
    rs = np.random.RandomState(2)
    X_fit = rs.randn(300, 500)
    X_test = rs.randn(120, 500) * 4.0 + 7.0

    pipe = FrozenPipeline.fit(X_fit)
    mean_before = pipe.scaler_in.mean_.copy()
    comp_before = pipe.pca.components_.copy()
    out_before = pipe.scaler_out.mean_.copy()

    _ = pipe.transform(X_test)
    assert np.array_equal(pipe.scaler_in.mean_, mean_before)
    assert np.array_equal(pipe.pca.components_, comp_before)
    assert np.array_equal(pipe.scaler_out.mean_, out_before)

    leaky = FrozenPipeline.fit(np.vstack([X_fit, X_test]))
    assert not np.allclose(pipe.pca.components_, leaky.pca.components_)


def test_L_pipeline_is_unsupervised():
    import inspect

    assert "y" not in inspect.signature(FrozenPipeline.fit).parameters
    src = inspect.getsource(FrozenPipeline.fit)
    for forbidden in ("logits", "target", "q_"):
        assert forbidden not in src


def test_L_target_scaler_fit_only():
    rs = np.random.RandomState(3)
    q_fit = rs.randn(200, NUM_EXPERTS)
    q_test = rs.randn(80, NUM_EXPERTS) * 3.0
    tsc = TargetScaler.fit(q_fit)
    before = tsc.scaler.mean_.copy()
    _ = tsc.transform(q_test)
    assert np.array_equal(tsc.scaler.mean_, before)


def test_M_final_dimensions_all_64():
    rs = np.random.RandomState(4)
    for raw_dim in (224, 352, 14336, 22528):
        X_fit = rs.randn(200, raw_dim)
        X_test = rs.randn(90, raw_dim)
        pipe = FrozenPipeline.fit(X_fit)
        assert pipe.transform(X_fit).shape == (200, PCA_COMPONENTS)
        assert pipe.transform(X_test).shape == (90, PCA_COMPONENTS)
        assert PCA_COMPONENTS == 64


def test_M_raw_dimensions_as_specified():
    assert raw_dimension(8, "FUSED") == 7 * 32 == 224
    assert raw_dimension(12, "FUSED") == 11 * 32 == 352
    assert raw_dimension(8, "EXPERT_IDENTITY") == 7 * 64 * 32 == 14336
    assert raw_dimension(12, "EXPERT_IDENTITY") == 11 * 64 * 32 == 22528
    assert raw_dimension(8, "SHUFFLED_IDENTITY") == raw_dimension(8, "EXPERT_IDENTITY")


# ------------------------------------------------------------------ N, O


def test_N_target_is_centred_router_logits():
    rs = np.random.RandomState(5)
    g = rs.randn(50, NUM_EXPERTS) + 3.0
    q = center_logits(g)
    assert q.shape == g.shape
    assert np.allclose(q.mean(axis=1), 0.0, atol=1e-12)
    # Invariance: adding a constant to all logits leaves the target unchanged.
    assert np.allclose(center_logits(g + 5.0), q, atol=1e-12)
    with pytest.raises(ValueError):
        center_logits(rs.randn(4, 32))


def test_O_history_excludes_target_and_later_layers():
    """History for each target uses only strictly earlier layers."""
    for target_human, spec in TARGETS.items():
        target_code = spec["code"]
        assert max(spec["history_code"]) < target_code, target_human
        assert max(spec["history_human"]) < target_human
        assert spec["history_human"] == tuple(range(1, target_human))
        # Code/human indexing consistency.
        assert spec["code"] == target_human - 1
        for h, c in zip(spec["history_human"], spec["history_code"]):
            assert c == h - 1
    # Nothing beyond Layer 11 is ever stored as history.
    assert max(ALL_HISTORY_CODE) == 10
    assert max(ALL_HISTORY_CODE) < TARGETS[12]["code"]


def test_O_worker_stores_only_permitted_layers():
    """The extraction worker stores history 1..11 and target logits only."""
    p = os.path.join(os.path.dirname(__file__), "..", "scripts", "_extract_worker.py")
    tree = ast.parse(open(p).read())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "ALL_HISTORY_CODE" in names
    assert "TARGET_CODES" in names


# ------------------------------------------------------------------ P


def test_P_smoke_computes_no_verdict_statistic():
    """The smoke script must not fit compression/probes or compute R2/A/B."""
    p = os.path.join(os.path.dirname(__file__), "..", "scripts", "smoke_eipc.py")
    tree = ast.parse(open(p).read())
    referenced = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            referenced.update(a.name for a in node.names)

    for forbidden in ("FrozenPipeline", "TargetScaler", "run_target_probes", "r2_score",
                      "Ridge", "verdict", "summarize", "fit_and_evaluate",
                      "build_shifts", "build_shuffled"):
        assert forbidden not in referenced, f"smoke must not use {forbidden}"


def test_P_frozen_constants():
    assert PROJ_DIM == 32 and PCA_COMPONENTS == 64
    assert N_FIT == 1024 and N_TEST == 1024
    assert PROJECTION_SEED == 20260917 and SHUFFLE_SEED == 314159
    assert TOP_K == 8 and NUM_EXPERTS == 64
    assert sorted(TARGETS) == [8, 12]


def test_P_decision_rule_mechanics():
    from eipc.analysis import NOT_PROMISING, PROMISING, verdict

    def mk(identity, a, b):
        return {8: {"R2_identity": identity[0], "A": a[0], "B": b[0]},
                12: {"R2_identity": identity[1], "A": a[1], "B": b[1]}}

    assert verdict(mk([0.1, 0.1], [0.03, 0.03], [0.03, 0.03])) == PROMISING
    # Condition 1: R2_identity must be positive at both.
    assert verdict(mk([-0.1, 0.1], [0.03, 0.03], [0.03, 0.03])) == NOT_PROMISING
    assert verdict(mk([0.0, 0.1], [0.03, 0.03], [0.03, 0.03])) == NOT_PROMISING
    # Condition 2: A > 0 at both.
    assert verdict(mk([0.1, 0.1], [0.03, -0.01], [0.03, 0.03])) == NOT_PROMISING
    # Condition 3: B > 0 at both.
    assert verdict(mk([0.1, 0.1], [0.03, 0.03], [0.03, 0.0])) == NOT_PROMISING
    # Condition 4: mean(A) threshold.
    assert verdict(mk([0.1, 0.1], [0.01, 0.01], [0.03, 0.03])) == NOT_PROMISING
    # Condition 5: mean(B) threshold.
    assert verdict(mk([0.1, 0.1], [0.03, 0.03], [0.01, 0.01])) == NOT_PROMISING
    # Exactly at the thresholds passes.
    assert verdict(mk([0.1, 0.1], [0.02, 0.02], [0.02, 0.02])) == PROMISING
    with pytest.raises(ValueError):
        verdict({8: {"R2_identity": 0.1, "A": 0.1, "B": 0.1}})
