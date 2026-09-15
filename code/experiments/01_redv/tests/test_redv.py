"""REDV-V1 implementation-correctness tests (A-F from the preregistration).

These verify plumbing only. They must not compute or inspect Delta_R2, research
correlations, success/failure, or subgroup performance.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from redv import CONTEXT_LEN, REJECTED_RANKS, TOP_K, TRANSITIONS
from redv.counterfactual import native_route_equivalence_nll, routing_regret
from redv.olmoe_hooks import (
    _generic_moe_forward,
    assert_frozen,
    build_override,
    flat_index,
    load_model,
    native_forward,
    native_top_identities,
    next_token_nll,
)
from redv.rejected import (
    expert_output,
    probs_from_logits,
    rejected_evidence,
    rejected_evidence_reference,
    rejected_identities,
    router_probs,
)

SEED = 20260914
# FP16 tolerances: OLMoE hidden states are O(1e1), NLL is O(1e0).
HIDDEN_ATOL = 2e-2
NLL_ATOL = 5e-3


@pytest.fixture(scope="module")
def model_and_tok():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model, tok = load_model()
    return model, tok


@pytest.fixture(scope="module")
def batch(model_and_tok):
    """A small deterministic random-token batch for plumbing checks."""
    model, tok = model_and_tok
    g = torch.Generator().manual_seed(SEED)
    ids = torch.randint(0, 50000, (4, CONTEXT_LEN), generator=g).cuda()
    targets = torch.randint(0, 50000, (4,), generator=g).cuda()
    return ids, targets


# ------------------------------------------------------------------ test D


def test_D_frozen_model_no_grad(model_and_tok):
    """All parameters requires_grad=False, model in eval()."""
    model, _ = model_and_tok
    assert not model.training
    assert_frozen(model)
    assert all(not p.requires_grad for p in model.parameters())


def test_D_config_matches_preregistration(model_and_tok):
    model, _ = model_and_tok
    cfg = model.config
    assert cfg.architectures == ["OlmoeForCausalLM"]
    assert cfg.hidden_size == 2048
    assert cfg.num_hidden_layers == 16
    assert cfg.num_experts == 64
    assert cfg.num_experts_per_tok == TOP_K == 8
    assert cfg.norm_topk_prob is False


# ------------------------------------------------------------------ test A


def test_A_native_route_equivalence_moe_block(model_and_tok):
    """Forcing the stock-selected eight experts reproduces the stock MoE output.

    Checked directly at the MoE block on several random tokens.
    """
    model, _ = model_and_tok
    torch.manual_seed(SEED)

    for layer_idx in (3, 7, 11):
        block = model.model.layers[layer_idx].mlp
        x = (torch.randn(6, 1, 2048, generator=torch.Generator().manual_seed(SEED + layer_idx))
             .half().cuda())

        stock_out, stock_logits = block(x)

        probs = probs_from_logits(block.gate(x.view(-1, 2048)))
        ids = native_top_identities(probs)
        override = {i: ids[i] for i in range(x.shape[0])}
        forced_out, forced_logits = _generic_moe_forward(block, x, override)

        assert torch.allclose(stock_logits, forced_logits, atol=HIDDEN_ATOL)
        assert torch.allclose(stock_out, forced_out, atol=HIDDEN_ATOL), (
            f"layer {layer_idx} max diff {(stock_out - forced_out).abs().max()}"
        )


def test_A_no_override_is_exactly_native(model_and_tok):
    """With override=None the re-implementation matches the stock block."""
    model, _ = model_and_tok
    block = model.model.layers[7].mlp
    x = torch.randn(3, 5, 2048, generator=torch.Generator().manual_seed(SEED)).half().cuda()
    a, la = block(x)
    b, lb = _generic_moe_forward(block, x, None)
    assert torch.equal(la, lb)
    assert torch.equal(a, b)


# ------------------------------------------------------------------ test F


def test_F_native_loss_equivalence_full_model(model_and_tok, batch):
    """Forcing native identities reproduces the ordinary next-token loss."""
    model, _ = model_and_tok
    ids, targets = batch

    cap = native_forward(model, ids, targets, capture_layers=[4])
    probs = probs_from_logits(cap.g[4])
    forced = native_route_equivalence_nll(model, ids, targets, 4, probs)

    assert torch.allclose(cap.token_nll, forced, atol=NLL_ATOL), (
        f"max NLL diff {(cap.token_nll - forced).abs().max()}"
    )


def test_F_regret_native_column_matches(model_and_tok, batch):
    """The regret runner's native NLL equals a plain forward pass NLL."""
    model, _ = model_and_tok
    ids, targets = batch

    cap = native_forward(model, ids, targets, capture_layers=[8])
    out = model(input_ids=ids, use_cache=False)
    plain = next_token_nll(out.logits, targets, ids.shape[1])

    assert torch.allclose(cap.token_nll, plain, atol=1e-6)


# ------------------------------------------------------------------ test B


def test_B_override_keeps_ranks_1_to_7(model_and_tok):
    """Forced routes keep native ranks 1-7 and swap only rank 8.

    Routes are compared as SETS of expert identities. The MoE output is an
    order-invariant probability-weighted sum, and router probabilities can tie
    exactly in FP16, in which case topk(8) and topk(12) may order the tied pair
    differently while selecting the same experts. ``build_override`` derives both
    the kept seven and the replacement from one consistent topk-12 ranking, so
    rank assignment is internally consistent.
    """
    model, _ = model_and_tok
    block = model.model.layers[4].mlp
    x = torch.randn(5, 2048, generator=torch.Generator().manual_seed(SEED)).half().cuda()
    probs = router_probs(block, x)

    native = native_top_identities(probs)
    ranked12 = torch.topk(probs, 12, dim=-1).indices

    for rank in REJECTED_RANKS:
        ov = build_override(probs, list(range(5)), [rank] * 5)
        for row in range(5):
            forced = ov[row]
            keep7 = set(ranked12[row, :7].tolist())

            assert len(forced) == TOP_K, "every route executes exactly 8 experts"
            assert len(set(forced.tolist())) == TOP_K, "no duplicate experts"
            assert set(forced[:7].tolist()) == keep7, "ranks 1-7 unchanged"
            assert forced[7] == ranked12[row, rank - 1], "8th slot is the replacement rank"
            assert int(forced[7]) not in keep7, "replacement is not already routed"

            # The replacement comes from beyond the native top-8, so the forced
            # route differs from the native route unless the two experts carry
            # exactly equal router probability.
            native_set = set(native[row].tolist())
            if set(forced.tolist()) == native_set:
                swapped_out = (native_set - set(forced[:7].tolist())).pop()
                assert probs[row, int(forced[7])] == probs[row, swapped_out], (
                    "forced route may only coincide with native under an exact "
                    "probability tie"
                )


def test_B_expert_isolation_unrelated_tokens(model_and_tok):
    """Overriding one token leaves other tokens' MoE outputs bit-identical."""
    model, _ = model_and_tok
    block = model.model.layers[4].mlp
    x = torch.randn(1, 6, 2048, generator=torch.Generator().manual_seed(SEED)).half().cuda()

    base, _ = _generic_moe_forward(block, x, None)
    probs = router_probs(block, x.view(-1, 2048))
    target_tok = 3
    ov = build_override(probs[target_tok : target_tok + 1], [target_tok], [9])
    pert, _ = _generic_moe_forward(block, x, ov)

    for tok in range(6):
        if tok == target_tok:
            assert not torch.equal(base[0, tok], pert[0, tok]), "target token must change"
        else:
            assert torch.equal(base[0, tok], pert[0, tok]), f"token {tok} must be untouched"


def test_B_expert_isolation_unrelated_layers(model_and_tok, batch):
    """A forced route at layer l+1 leaves earlier layers' hidden states identical."""
    model, _ = model_and_tok
    ids, targets = batch
    src, tgt = 3, 4

    cap = native_forward(model, ids, targets, capture_layers=[src, tgt])
    probs = probs_from_logits(cap.g[tgt])
    ov = build_override(probs, [flat_index(i, ids.shape[1]) for i in range(ids.shape[0])],
                        [9] * ids.shape[0])

    from redv.olmoe_hooks import forced_route

    with forced_route(model, tgt, ov):
        cap2 = native_forward(model, ids, targets, capture_layers=[src, tgt])

    # Layer src precedes the intervention: identical.
    assert torch.equal(cap.h[src], cap2.h[src])
    assert torch.equal(cap.g[src], cap2.g[src])
    # Router logits at tgt are computed before the override: also identical.
    assert torch.equal(cap.g[tgt], cap2.g[tgt])
    # The post-layer hidden state at tgt must change.
    assert not torch.equal(cap.h[tgt], cap2.h[tgt])


def test_B_different_replacement_ranks_differ(model_and_tok, batch):
    """The four alternatives are genuinely distinct routes."""
    model, _ = model_and_tok
    ids, targets = batch
    cap = native_forward(model, ids, targets, capture_layers=[8])
    probs = probs_from_logits(cap.g[8])
    res = routing_regret(model, ids, targets, 8, cap.token_nll, probs)
    assert res.alt_nll.shape == (ids.shape[0], 4)
    # Alternatives should not all collapse to one value.
    assert np.unique(np.round(res.alt_nll, 6)).size > 1


# ------------------------------------------------------------------ test C


def test_C_rejected_expert_matches_model_expert(model_and_tok):
    """E_{l,j}(x) matches the model's own expert module computation."""
    model, _ = model_and_tok
    block = model.model.layers[7].mlp
    x = torch.randn(3, 2048, generator=torch.Generator().manual_seed(SEED)).half().cuda()

    for j in (0, 17, 63):
        mine = expert_output(block, j, x)
        expert = block.experts[j]
        manual = expert.down_proj(expert.act_fn(expert.gate_proj(x)) * expert.up_proj(x))
        assert torch.equal(mine, manual)


def test_C_rejected_evidence_matches_reference(model_and_tok):
    """Grouped rejected-evidence equals the straight-line reference."""
    model, _ = model_and_tok
    block = model.model.layers[11].mlp
    x = torch.randn(7, 2048, generator=torch.Generator().manual_seed(SEED)).half().cuda()

    fast = rejected_evidence(block, x)
    ref = rejected_evidence_reference(block, x)
    assert fast.shape == (7, 2048)
    assert torch.allclose(fast, ref, atol=1e-4), f"max diff {(fast - ref).abs().max()}"


def test_C_rejected_uses_ranks_9_to_12_unrenormalized(model_and_tok):
    """Rejected identities are ranks 9-12 and probabilities are not renormalized."""
    model, _ = model_and_tok
    block = model.model.layers[3].mlp
    x = torch.randn(4, 2048, generator=torch.Generator().manual_seed(SEED)).half().cuda()
    probs = router_probs(block, x)

    ids, p = rejected_identities(probs)
    ranked = torch.topk(probs, 12, dim=-1)

    assert ids.shape == (4, 4) and p.shape == (4, 4)
    for row in range(4):
        for slot, rank in enumerate(REJECTED_RANKS):
            assert ids[row, slot] == ranked.indices[row, rank - 1]
            # Original probability, not renormalized.
            assert torch.isclose(p[row, slot], probs[row, ids[row, slot]])
        # Rejected mass must stay well below 1: weak experts stay weak.
        assert p[row].sum() < 1.0
        # Ranks 9-12 are disjoint from the native top-8.
        native = set(native_top_identities(probs)[row].tolist())
        assert not (set(ids[row].tolist()) & native)


def test_C_rejected_evidence_is_probability_weighted_sum(model_and_tok):
    """r equals sum_j p_j * E_j(x), with the original p_j."""
    model, _ = model_and_tok
    block = model.model.layers[7].mlp
    x = torch.randn(2, 2048, generator=torch.Generator().manual_seed(SEED + 1)).half().cuda()
    probs = router_probs(block, x)
    ids, p = rejected_identities(probs)

    manual = torch.zeros(2, 2048, dtype=torch.float32, device=x.device)
    for row in range(2):
        for slot in range(4):
            manual[row] += expert_output(block, int(ids[row, slot]), x[row:row+1])[0].float() * p[row, slot]

    got = rejected_evidence(block, x, probs)
    assert torch.allclose(got, manual, atol=1e-4)


# ------------------------------------------------------------------ test E


def test_E_manifest_determinism():
    """Rebuilding with seed 20260914 reproduces the same block ids and ranges."""
    from redv.data import Block, blocks_fingerprint, make_blocks, sample_blocks

    stream = list(range(200_000))
    blocks = make_blocks(stream)

    a = sample_blocks(blocks, 512, SEED, 1)
    b = sample_blocks(blocks, 512, SEED, 1)
    assert [x.block_id for x in a] == [x.block_id for x in b]
    assert [x.stream_start for x in a] == [x.stream_start for x in b]
    assert blocks_fingerprint(a) == blocks_fingerprint(b)

    # Validation and test draws are independent.
    c = sample_blocks(blocks, 512, SEED, 2)
    assert [x.block_id for x in a] != [x.block_id for x in c]

    # Sampling is without replacement.
    assert len({x.block_id for x in a}) == 512


def test_E_blocks_are_non_overlapping_129():
    from redv.data import make_blocks

    stream = list(range(129 * 10 + 7))
    blocks = make_blocks(stream)
    assert len(blocks) == 10
    for i, b in enumerate(blocks):
        assert len(b.tokens) == 129
        assert b.stream_start == i * 129
        assert len(b.context) == CONTEXT_LEN
        assert b.target == b.tokens[128]
    # Non-overlapping: consecutive starts differ by exactly the block length.
    starts = [b.stream_start for b in blocks]
    assert all(starts[i + 1] - starts[i] == 129 for i in range(len(starts) - 1))


def test_E_train_split_refused():
    from redv.data import load_split_texts

    with pytest.raises(ValueError, match="validation and test only"):
        load_split_texts("train")


# ------------------------------------------------------- frozen-constant guards


def test_frozen_constants():
    assert CONTEXT_LEN == 128
    assert TRANSITIONS == ((3, 4), (7, 8), (11, 12))
    assert REJECTED_RANKS == (9, 10, 11, 12)
    assert SEED == 20260914


def test_probe_dimensions_and_standardizer():
    from redv.probes import ALPHA, Standardizer, augmented_features, baseline_features

    assert ALPHA == 1.0
    h = np.random.RandomState(0).randn(10, 2048)
    g = np.random.RandomState(1).randn(10, 64)
    r = np.random.RandomState(2).randn(10, 2048)
    assert baseline_features(h, g).shape == (10, 2112)
    assert augmented_features(h, g, r).shape == (10, 4160)

    # Zero-variance dimension stays at zero.
    X = np.ones((5, 3))
    X[:, 1] = [1.0, 2.0, 3.0, 4.0, 5.0]
    sc = Standardizer.fit(X)
    out = sc.transform(X)
    assert np.all(out[:, 0] == 0.0)
    assert np.all(out[:, 2] == 0.0)
    assert abs(out[:, 1].mean()) < 1e-12


def test_judgement_rule():
    from redv.analysis import INCONCLUSIVE, NOT_SUPPORTED, SUPPORTED, verdict

    assert verdict([0.03, 0.03, 0.03]) == SUPPORTED
    assert verdict([0.05, 0.001, 0.01]) == SUPPORTED  # mean .0203, all > 0
    assert verdict([0.0, 0.0, 0.0]) == NOT_SUPPORTED
    assert verdict([-0.01, -0.01, 0.5]) == NOT_SUPPORTED  # two <= 0
    assert verdict([0.005, 0.005, 0.005]) == NOT_SUPPORTED  # mean < .01
    assert verdict([0.05, 0.01, -0.01]) == INCONCLUSIVE  # mean .0167, one <= 0
    assert verdict([0.02, 0.01, 0.012]) == INCONCLUSIVE  # mean .014, all > 0
    with pytest.raises(ValueError):
        verdict([0.01, 0.02])
