"""XEC-P0 invariants A-Q. Implementation correctness only; no verdict statistic."""

from __future__ import annotations

import ast
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from xec import (
    ATTN_DIM,
    CURRENT_DIM,
    HIDDEN_SIZE,
    HISTORY_CODE,
    ITEM_DIM,
    MAX_RANK,
    MODEL_REVISION,
    N_ACTIONS,
    N_EXPERT_ITEMS,
    N_FUSED_ITEMS,
    N_HISTORY_LAYERS,
    N_TEST,
    N_TRAIN,
    N_VALIDATION,
    NUM_EXPERTS,
    POLICY_SEEDS,
    PROJ_DIM,
    SWAP_RANKS,
    TARGET_LAYER,
    TOP_K,
    VARIANTS,
)
from xec.cache_features import (
    build_projection,
    current_context,
    expert_cache_items,
    fused_cache_items,
    layer_norm_np,
    projection_hash,
)
from xec.counterfactual import native_nll_check
from xec.data import load_eipc_used_blocks, verify_prior_projects_avoided_train
from xec.olmoe import (
    _moe_forward_with_override,
    action_identities,
    action_nll,
    assert_frozen,
    flat_index,
    forced_route,
    fused_from_contributions,
    load_model,
    native_forward,
    native_topk,
    router_probs,
    selected_expert_contributions,
    verify_config,
)
from xec.oracle import load_test_oracle_guarded, oracle_action
from xec.policy import RoutingPolicy, build_matched_variants, count_trainable

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "XEC_P0")
SEED = 20260918
FP16_ATOL = 2e-2
NLL_ATOL = 5e-3


@pytest.fixture(scope="module")
def model_and_tok():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return load_model(model_dir=os.environ.get("XEC_MODEL_DIR"))


@pytest.fixture(scope="module")
def batch():
    g = torch.Generator().manual_seed(SEED)
    ids = torch.randint(0, 50000, (4, 128), generator=g).cuda()
    tgt = torch.randint(0, 50000, (4,), generator=g).cuda()
    return ids, tgt


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
        assert d["olmoe_receives_gradients"] is False


def test_B_frozen_eval_fp16_no_grads(model_and_tok):
    model, _ = model_and_tok
    assert not model.training
    assert_frozen(model)
    assert all(not p.requires_grad for p in model.parameters())
    assert next(model.parameters()).dtype == torch.float16


# ------------------------------------------------------------------ C, G


def test_C_native_topk_semantics(model_and_tok):
    model, _ = model_and_tok
    block = model.model.layers[TARGET_LAYER].mlp
    x = torch.randn(6, HIDDEN_SIZE, generator=torch.Generator().manual_seed(SEED)).half().cuda()
    logits = block.gate(x)
    probs = router_probs(logits)

    assert probs.dtype == torch.float32
    assert torch.allclose(probs.sum(dim=-1), torch.ones(6, device=probs.device), atol=1e-4)

    ids, vals = native_topk(logits, TOP_K)
    for row in range(6):
        expected = torch.topk(probs[row], TOP_K)
        assert set(ids[row].tolist()) == set(expected.indices.tolist())
        for slot in range(TOP_K):
            assert torch.isclose(vals[row, slot], probs[row, ids[row, slot]])
        assert all(vals[row, i] >= vals[row, i + 1] for i in range(TOP_K - 1))


def test_G_no_topk_renormalization(model_and_tok):
    """Top-8 mass stays unnormalized, and the config flag is false."""
    model, _ = model_and_tok
    assert model.config.norm_topk_prob is False
    for li in (0, TARGET_LAYER):
        block = model.model.layers[li].mlp
        assert block.norm_topk_prob is False
        x = torch.randn(8, HIDDEN_SIZE,
                        generator=torch.Generator().manual_seed(SEED + li)).half().cuda()
        _, vals = native_topk(block.gate(x), TOP_K)
        assert torch.all(vals.sum(dim=-1) < 1.0), "top-8 probabilities must not sum to 1"


# ------------------------------------------------------------------ D, H, I


def test_D_selected_experts_reproduce_stock_moe(model_and_tok):
    """Weighted sum of separately executed selected experts == stock MoE output."""
    model, _ = model_and_tok
    for li in (0, 5, 10, TARGET_LAYER):
        block = model.model.layers[li].mlp
        x = torch.randn(5, 1, HIDDEN_SIZE,
                        generator=torch.Generator().manual_seed(SEED + li)).half().cuda()
        with torch.inference_mode():
            stock_y, stock_logits = block(x)
            _, _, contrib = selected_expert_contributions(
                block, x.view(-1, HIDDEN_SIZE), stock_logits)
            recon = fused_from_contributions(contrib)
        err = (recon - stock_y.view(-1, HIDDEN_SIZE).float()).abs().max().item()
        assert err < FP16_ATOL, f"layer {li} reconstruction error {err}"


def test_H_cache_uses_selected_experts_only(model_and_tok):
    """Cached identities are exactly the native Top-8; no rejected expert appears."""
    model, _ = model_and_tok
    block = model.model.layers[3].mlp
    x = torch.randn(6, HIDDEN_SIZE, generator=torch.Generator().manual_seed(SEED)).half().cuda()
    logits = block.gate(x)
    with torch.inference_mode():
        ids, probs, contrib = selected_expert_contributions(block, x, logits)

    native_ids, _ = native_topk(logits, TOP_K)
    assert ids.shape == (6, TOP_K)
    assert contrib.shape == (6, TOP_K, HIDDEN_SIZE)
    ranked12 = torch.topk(router_probs(logits), MAX_RANK, dim=-1).indices
    for row in range(6):
        assert set(ids[row].tolist()) == set(native_ids[row].tolist())
        # Ranks 9-12 must never enter the cache.
        rejected = set(ranked12[row, TOP_K:].tolist())
        assert not (set(ids[row].tolist()) & rejected), "rejected expert leaked into cache"


def test_I_cached_expert_ids_match_true_native(model_and_tok):
    """Item expert one-hot blocks decode back to the true native selected IDs."""
    model, _ = model_and_tok
    block = model.model.layers[4].mlp
    x = torch.randn(3, HIDDEN_SIZE, generator=torch.Generator().manual_seed(SEED)).half().cuda()
    logits = block.gate(x)
    with torch.inference_mode():
        ids, _, contrib = selected_expert_contributions(block, x, logits)

    P = build_projection()
    s = (contrib.float().cpu().numpy() @ P)[:, None, :, :]
    s = np.repeat(s, N_HISTORY_LAYERS, axis=1)
    ids_np = np.repeat(ids.cpu().numpy()[:, None, :], N_HISTORY_LAYERS, axis=1)

    items = expert_cache_items(s, ids_np)
    assert items.shape == (3, N_EXPERT_ITEMS, ITEM_DIM)
    expert_block = items[..., PROJ_DIM + N_HISTORY_LAYERS :]
    decoded = expert_block.argmax(axis=-1).reshape(3, N_HISTORY_LAYERS, TOP_K)
    assert np.array_equal(decoded, ids_np.astype(np.int64))
    # Layer one-hot decodes to the true layer position.
    layer_block = items[..., PROJ_DIM : PROJ_DIM + N_HISTORY_LAYERS]
    dl = layer_block.argmax(axis=-1).reshape(3, N_HISTORY_LAYERS, TOP_K)
    assert np.array_equal(dl, np.broadcast_to(
        np.arange(N_HISTORY_LAYERS)[None, :, None], dl.shape))


# ------------------------------------------------------------------ E, F


def test_E_action_zero_reproduces_native(model_and_tok, batch):
    """Action 0 must reproduce the plain native NLL."""
    model, _ = model_and_tok
    ids, tgt = batch
    with torch.inference_mode():
        cap = native_forward(model, ids, tgt, [TARGET_LAYER])
        t12, _ = native_topk(cap.g[TARGET_LAYER], MAX_RANK)
        a0 = action_nll(model, ids, tgt, t12, 0)
    err = (a0 - cap.nll).abs().max().item()
    assert err < NLL_ATOL, f"action 0 differs from native by {err}"
    # And the array-level checker agrees.
    A = np.stack([a0.float().cpu().numpy()] * N_ACTIONS, axis=1)
    native_nll_check(A, cap.nll.float().cpu().numpy())


def test_E_action_identities_structure(model_and_tok):
    """Each action executes exactly 8 experts; 1-4 swap only rank 8."""
    model, _ = model_and_tok
    block = model.model.layers[TARGET_LAYER].mlp
    x = torch.randn(5, HIDDEN_SIZE, generator=torch.Generator().manual_seed(SEED)).half().cuda()
    t12, _ = native_topk(block.gate(x), MAX_RANK)

    a0 = action_identities(t12, 0)
    assert a0.shape == (5, TOP_K)
    for a in range(1, N_ACTIONS):
        ids = action_identities(t12, a)
        assert ids.shape == (5, TOP_K)
        for row in range(5):
            assert len(set(ids[row].tolist())) == TOP_K, "no duplicate experts"
            assert set(ids[row, :7].tolist()) == set(t12[row, :7].tolist()), "ranks 1-7 kept"
            assert ids[row, 7] == t12[row, SWAP_RANKS[a - 1] - 1], "8th slot is the swap rank"
    with pytest.raises(ValueError):
        action_identities(t12, 5)


def test_F_actions_differ_only_at_target_layer(model_and_tok, batch):
    """A forced action leaves earlier layers and other tokens untouched."""
    model, _ = model_and_tok
    ids, tgt = batch
    earlier = list(HISTORY_CODE)

    with torch.inference_mode():
        cap = native_forward(model, ids, tgt, earlier + [TARGET_LAYER])
        t12, _ = native_topk(cap.g[TARGET_LAYER], MAX_RANK)
        override = {flat_index(i, ids.shape[1]): action_identities(t12, 1)[i]
                    for i in range(ids.shape[0])}
        with forced_route(model, TARGET_LAYER, override):
            cap2 = native_forward(model, ids, tgt, earlier + [TARGET_LAYER])

    for li in earlier:
        assert torch.equal(cap.x[li], cap2.x[li]), f"layer {li} input changed"
        assert torch.equal(cap.g[li], cap2.g[li]), f"layer {li} router logits changed"
        assert torch.equal(cap.y[li], cap2.y[li]), f"layer {li} fused output changed"
    # Router logits at the target are computed before the override.
    assert torch.equal(cap.g[TARGET_LAYER], cap2.g[TARGET_LAYER])
    # The target's fused output must change.
    assert not torch.equal(cap.y[TARGET_LAYER], cap2.y[TARGET_LAYER])


def test_F_other_tokens_keep_routing(model_and_tok):
    """Overriding one token changes no other token's route."""
    model, _ = model_and_tok
    block = model.model.layers[TARGET_LAYER].mlp
    x = torch.randn(1, 6, HIDDEN_SIZE,
                    generator=torch.Generator().manual_seed(SEED)).half().cuda()
    base, logits = _moe_forward_with_override(block, x, None)
    probs = router_probs(logits)
    native = native_topk(logits, TOP_K)[0]

    tok = 3
    t12 = torch.topk(probs, MAX_RANK, dim=-1).indices
    pert, _ = _moe_forward_with_override(
        block, x, {tok: action_identities(t12[tok : tok + 1], 1)[0]})

    for t in range(6):
        if t == tok:
            assert (base[0, t] - pert[0, t]).abs().max() > 1e-3
        else:
            # Routing unchanged; only accumulation-order noise may differ.
            assert torch.equal(native[t], native_topk(logits, TOP_K)[0][t])
            assert torch.allclose(base[0, t], pert[0, t], atol=1e-3)


# ------------------------------------------------------------------ J, K


def test_J_fused_derived_from_same_projected_items():
    """FUSED content is the per-layer sum of the SAME projected contributions."""
    rs = np.random.RandomState(0)
    s = rs.randn(12, N_HISTORY_LAYERS, TOP_K, PROJ_DIM).astype(np.float32)
    ids = np.stack([[rs.choice(NUM_EXPERTS, TOP_K, replace=False)
                     for _ in range(N_HISTORY_LAYERS)] for _ in range(12)]).astype(np.int16)

    fused = fused_cache_items(s)
    expert = expert_cache_items(s, ids)
    assert fused.shape == (12, N_FUSED_ITEMS, ITEM_DIM)
    assert expert.shape == (12, N_EXPERT_ITEMS, ITEM_DIM)

    # Fused content equals LayerNorm(sum of the same s vectors).
    assert np.allclose(fused[..., :PROJ_DIM], layer_norm_np(s.sum(axis=2)), atol=1e-5)
    # Expert-ID block is zero for fused items.
    assert np.all(fused[..., PROJ_DIM + N_HISTORY_LAYERS :] == 0.0)
    # Both carry the same layer one-hot structure.
    assert fused[..., PROJ_DIM : PROJ_DIM + N_HISTORY_LAYERS].sum() == 12 * N_HISTORY_LAYERS


def test_K_projection_deterministic_and_hashed():
    a = build_projection()
    b = build_projection()
    assert a.shape == (HIDDEN_SIZE, PROJ_DIM)
    assert np.array_equal(a, b)
    assert projection_hash(a) == projection_hash(b)
    assert abs(a.std() - 1.0 / np.sqrt(PROJ_DIM)) < 0.01
    p = os.path.join(ART, "projection.json")
    if os.path.exists(p):
        meta = json.load(open(p))
        assert meta["sha256"] == projection_hash(a)
        assert meta["trainable"] is False and meta["out_dim"] == PROJ_DIM


def test_K_current_context_shape_and_centering():
    rs = np.random.RandomState(1)
    x = rs.randn(9, HIDDEN_SIZE)
    g = rs.randn(9, NUM_EXPERTS) + 4.0
    u = current_context(x, g)
    assert u.shape == (9, CURRENT_DIM) == (9, 2112)
    # LayerNorm part is zero-mean unit-var; logits part is centred.
    assert np.allclose(u[:, :HIDDEN_SIZE].mean(axis=1), 0.0, atol=1e-6)
    assert np.allclose(u[:, HIDDEN_SIZE:].mean(axis=1), 0.0, atol=1e-10)
    # Constant logit shift is invariant.
    assert np.allclose(current_context(x, g + 7.0), u, atol=1e-9)


# ------------------------------------------------------------------ L, M


def test_L_M_manifest_disjoint_and_eipc_excluded():
    p = os.path.join(ART, "data_manifest.json")
    if not os.path.exists(p):
        pytest.skip("manifest not built yet")
    m = json.load(open(p))
    tr = set(m["train"]["block_ids"])
    va = set(m["validation"]["block_ids"])
    te = set(m["test"]["block_ids"])

    assert len(tr) == m["train"]["n"] == N_TRAIN
    assert len(va) == m["validation"]["n"] == N_VALIDATION
    assert len(te) == m["test"]["n"] == N_TEST
    assert not (tr & va) and not (tr & te) and not (va & te), "splits must be disjoint"

    excluded, _ = load_eipc_used_blocks()
    assert not ((tr | va | te) & excluded), "EIPC-P0 blocks must be excluded"
    assert m["split_used"] == "train"
    assert m["eipc_exclusion"]["eipc_blocks_excluded"] == 2048
    assert m["experimental_token_position"] == 127


def test_M_prior_projects_never_used_train():
    assert verify_prior_projects_avoided_train() == {
        "REDV_V1": True, "REDV_V2": True, "DREV_P0": True}


# ------------------------------------------------------------------ N, O


def test_N_identical_trainable_parameter_counts():
    counts = {}
    for variant in VARIANTS:
        m = RoutingPolicy(variant)
        counts[variant] = count_trainable(m)
        # Every variant carries all four matrices.
        names = {n for n, _ in m.named_parameters()}
        assert names == {"W_q.weight", "W_k.weight", "W_v.weight", "W_policy.weight"}
    assert len(set(counts.values())) == 1, counts
    expected = (CURRENT_DIM * ATTN_DIM + ITEM_DIM * ATTN_DIM * 2 + 2 * ATTN_DIM * N_ACTIONS)
    assert set(counts.values()) == {expected}, (counts, expected)


def test_O_same_seed_identical_initial_state():
    for seed in POLICY_SEEDS:
        variants, init_state = build_matched_variants(seed)
        for variant, m in variants.items():
            sd = m.state_dict()
            for k, v in init_state.items():
                assert torch.equal(sd[k], v), f"{variant} differs at {k} for seed {seed}"
    # Different seeds give different initializations.
    a, _ = build_matched_variants(POLICY_SEEDS[0])
    b, _ = build_matched_variants(POLICY_SEEDS[1])
    assert not torch.equal(a["EXPERT_CACHE"].W_q.weight, b["EXPERT_CACHE"].W_q.weight)


def test_O_current_only_forces_zero_memory():
    """CURRENT_ONLY ignores cache contents entirely."""
    torch.manual_seed(SEED)
    m = RoutingPolicy("CURRENT_ONLY")
    u = torch.randn(5, CURRENT_DIM)
    out_a = m(u, None)
    # Passing items must not change anything for CURRENT_ONLY.
    items = torch.randn(5, N_EXPERT_ITEMS, ITEM_DIM)
    out_b = m(u, items)
    assert torch.equal(out_a, out_b)
    assert out_a.shape == (5, N_ACTIONS)
    assert not m.uses_cache


def test_O_cache_variants_use_memory():
    torch.manual_seed(SEED)
    m = RoutingPolicy("EXPERT_CACHE")
    u = torch.randn(4, CURRENT_DIM)
    i1 = torch.randn(4, N_EXPERT_ITEMS, ITEM_DIM)
    i2 = torch.randn(4, N_EXPERT_ITEMS, ITEM_DIM)
    assert m.uses_cache
    assert not torch.allclose(m(u, i1), m(u, i2)), "memory contents must matter"
    with pytest.raises(ValueError):
        m(u, None)


# ------------------------------------------------------------------ P


def test_P_test_oracle_gated_behind_training_marker(tmp_path):
    """The TEST oracle refuses to load until training is provably complete."""
    fake_oracle = tmp_path / "test_oracle_descriptive.npz"
    np.savez(fake_oracle, oracle_action=np.zeros(3, dtype=np.int64))
    missing_marker = str(tmp_path / "training_complete.json")

    with pytest.raises(RuntimeError, match="unavailable until policy training is frozen"):
        load_test_oracle_guarded(str(fake_oracle), missing_marker)

    with open(missing_marker, "w") as fh:
        json.dump({"training_complete": True}, fh)
    assert load_test_oracle_guarded(str(fake_oracle), missing_marker).shape == (3,)


def test_P_training_script_never_reads_test_oracle():
    """train_policies.py must not reference TEST oracle or TEST features."""
    p = os.path.join(os.path.dirname(__file__), "..", "scripts", "train_policies.py")
    src = open(p).read()
    tree = ast.parse(src)
    consts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
              and isinstance(n.value, str)]
    for c in consts:
        assert "test_oracle" not in c, f"training references {c}"
        assert "test_features" not in c, f"training references {c}"
    assert "test_policy_results" not in src


def test_P_oracle_tie_breaks_to_lowest_index():
    """Exact ties resolve to action 0, so native wins."""
    A = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0],   # all tie -> 0
        [2.0, 1.0, 1.0, 3.0, 4.0],   # tie between 1 and 2 -> 1
        [5.0, 4.0, 3.0, 2.0, 1.0],   # strict min -> 4
    ])
    assert np.array_equal(oracle_action(A), np.array([0, 1, 4]))
    with pytest.raises(ValueError):
        oracle_action(np.zeros((3, 4)))


# ------------------------------------------------------------------ Q


def test_Q_smoke_computes_no_verdict_statistic():
    p = os.path.join(os.path.dirname(__file__), "..", "scripts", "smoke_xec.py")
    tree = ast.parse(open(p).read())
    referenced = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            referenced.update(a.name for a in node.names)

    for forbidden in ("paired_bootstrap_ci", "build_differences", "verdict", "summarize",
                      "train_policy", "oracle_action", "expert_beats_fused_all_seeds",
                      "per_seed_means"):
        assert forbidden not in referenced, f"smoke must not use {forbidden}"


def test_Q_frozen_constants():
    assert TARGET_LAYER == 11 and N_ACTIONS == 5
    assert SWAP_RANKS == (9, 10, 11, 12) and MAX_RANK == 12
    assert HISTORY_CODE == tuple(range(11)) and N_HISTORY_LAYERS == 11
    assert PROJ_DIM == 64 and ITEM_DIM == 139
    assert N_EXPERT_ITEMS == 88 and N_FUSED_ITEMS == 11
    assert CURRENT_DIM == 2112 and ATTN_DIM == 64
    assert POLICY_SEEDS == (42, 123, 2026)
    assert (N_TRAIN, N_VALIDATION, N_TEST) == (2048, 512, 1024)


def test_Q_decision_rule_mechanics():
    from xec.analysis import NOT_PROMISING, PROMISING, verdict

    def stats(mean, hi):
        return {k: {"mean": mean, "ci_low": mean - 0.01, "ci_high": hi}
                for k in ("d_native", "d_current", "d_fused")}

    win = {"all_seeds_expert_lower": True,
           "per_seed": {"42": {}, "123": {}, "2026": {}}}
    lose = {"all_seeds_expert_lower": False,
            "per_seed": {"42": {}, "123": {}, "2026": {}}}

    assert verdict(stats(-0.02, -0.01), win) == PROMISING
    # Mean not negative enough.
    assert verdict(stats(-0.004, -0.001), win) == NOT_PROMISING
    # CI upper bound touches zero.
    assert verdict(stats(-0.02, 0.0), win) == NOT_PROMISING
    assert verdict(stats(-0.02, 0.005), win) == NOT_PROMISING
    # Seed consistency fails.
    assert verdict(stats(-0.02, -0.01), lose) == NOT_PROMISING
    # Exactly at the threshold passes.
    assert verdict(stats(-0.005, -0.001), win) == PROMISING
    # One comparison failing is enough.
    s = stats(-0.02, -0.01)
    s["d_fused"] = {"mean": -0.001, "ci_low": -0.01, "ci_high": -0.0005}
    assert verdict(s, win) == NOT_PROMISING


def test_Q_bootstrap_is_deterministic_and_paired():
    from xec.analysis import paired_bootstrap_ci

    rs = np.random.RandomState(3)
    d = rs.randn(200) * 0.01 - 0.02
    a = paired_bootstrap_ci(d)
    b = paired_bootstrap_ci(d)
    assert a == b, "bootstrap must be deterministic under the fixed seed"
    assert a["n_bootstrap"] == 10000 and a["bootstrap_seed"] == 314159
    assert a["ci_low"] < a["mean"] < a["ci_high"]
