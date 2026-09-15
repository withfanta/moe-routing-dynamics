"""Frozen OLMoE loading, native capture, and forced Layer-12 routes for XEC-P0.

Router semantics follow the installed `transformers 4.45.1`
`OlmoeSparseMoeBlock.forward`: router logits from a bias-free gate, fp32 softmax over
all 64 experts, Top-8, no Top-k renormalization when `norm_topk_prob` is false, and
each expert output weighted by its own original probability. Forced routes apply the
override before the renormalization check, so the only intended difference from native
execution is the identity of the eighth expert.

No OLMoE parameter ever receives gradients: parameters are frozen at load and all
frozen-model work runs under `torch.inference_mode()`.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from . import (
    CONTEXT_LEN,
    HIDDEN_SIZE,
    MAX_RANK,
    MODEL_ID,
    MODEL_REVISION,
    NUM_EXPERTS,
    SWAP_RANKS,
    TARGET_LAYER,
    TOP_K,
)


def load_model(dtype=torch.float16, device: str = "cuda", model_dir: Optional[str] = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = model_dir if model_dir is not None else MODEL_ID
    kwargs = {} if model_dir is not None else {"revision": MODEL_REVISION}

    tokenizer = AutoTokenizer.from_pretrained(src, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=dtype, **kwargs)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def assert_frozen(model) -> None:
    """No OLMoE parameter receives gradients, and the model is in eval mode."""
    if model.training:
        raise AssertionError("model must be in eval() mode")
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if trainable:
        raise AssertionError(f"{len(trainable)} OLMoE parameters require grad")


def verify_config(model) -> None:
    cfg = model.config
    assert cfg.architectures == ["OlmoeForCausalLM"], cfg.architectures
    assert cfg.hidden_size == HIDDEN_SIZE == 2048
    assert cfg.num_hidden_layers == 16
    assert cfg.num_experts == NUM_EXPERTS == 64
    assert cfg.num_experts_per_tok == TOP_K == 8
    assert cfg.norm_topk_prob is False


def check_context_len(input_ids: torch.Tensor) -> None:
    if input_ids.shape[1] != CONTEXT_LEN:
        raise AssertionError(f"context length frozen at {CONTEXT_LEN}, got {input_ids.shape[1]}")


def experimental_position(seq_len: int) -> int:
    return seq_len - 1


def flat_index(batch_pos: int, seq_len: int) -> int:
    return batch_pos * seq_len + experimental_position(seq_len)


def router_probs(logits: torch.Tensor) -> torch.Tensor:
    """fp32 softmax over all 64 experts, before Top-k masking."""
    return F.softmax(logits, dim=-1, dtype=torch.float)


def native_topk(logits: torch.Tensor, k: int = TOP_K):
    probs = router_probs(logits)
    top = torch.topk(probs, k, dim=-1)
    return top.indices, top.values


def action_identities(top12_ids: torch.Tensor, action: int) -> torch.Tensor:
    """The eight expert identities executed by ``action``.

    Action 0 is native ranks 1-8. Actions 1-4 keep ranks 1-7 and replace rank 8 with
    rank 9, 10, 11 or 12. Every action executes exactly eight experts.
    """
    if action == 0:
        return top12_ids[..., :TOP_K]
    if not 1 <= action <= len(SWAP_RANKS):
        raise ValueError(f"action must be 0..{len(SWAP_RANKS)}, got {action}")
    rank = SWAP_RANKS[action - 1]
    keep = top12_ids[..., : TOP_K - 1]
    swap = top12_ids[..., rank - 1 : rank]
    return torch.cat([keep, swap], dim=-1)


def _moe_forward_with_override(block, hidden_states: torch.Tensor,
                               override: Optional[Dict[int, torch.Tensor]]):
    """The installed OlmoeSparseMoeBlock.forward, with an optional route override.

    Forced experts keep their ORIGINAL 64-way probabilities; the override is applied
    before the ``norm_topk_prob`` check so weighting semantics are identical.
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = block.gate(hidden_states)

    full_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(full_weights, block.top_k, dim=-1)

    if override:
        for flat_idx, identities in override.items():
            selected_experts[flat_idx] = identities
            routing_weights[flat_idx] = full_weights[flat_idx, identities]

    if block.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype, device=hidden_states.device)
    expert_mask = torch.nn.functional.one_hot(
        selected_experts, num_classes=block.num_experts).permute(2, 1, 0)

    for expert_idx in range(block.num_experts):
        expert_layer = block.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits


@contextlib.contextmanager
def forced_route(model, layer_idx: int, override: Dict[int, torch.Tensor]):
    """Force one layer's routing for specific flattened token indices."""
    block = model.model.layers[layer_idx].mlp
    original = block.forward

    def patched(hidden_states):
        return _moe_forward_with_override(block, hidden_states, override)

    block.forward = patched
    try:
        yield
    finally:
        block.forward = original


@dataclass
class Capture:
    """Per-layer native quantities at the experimental token."""

    x: Dict[int, torch.Tensor]
    g: Dict[int, torch.Tensor]
    y: Dict[int, torch.Tensor]
    nll: torch.Tensor


def next_token_nll(logits: torch.Tensor, targets: torch.Tensor, seq_len: int) -> torch.Tensor:
    pos = experimental_position(seq_len)
    return F.cross_entropy(logits[:, pos, :].float(), targets, reduction="none")


@torch.no_grad()
def native_forward(model, input_ids: torch.Tensor, targets: torch.Tensor,
                   capture_layers: Sequence[int]) -> Capture:
    """One unmodified native pass, capturing x, g, fused y, and the true next-token NLL."""
    seq_len = input_ids.shape[1]
    pos = experimental_position(seq_len)
    x_store: Dict[int, torch.Tensor] = {}
    g_store: Dict[int, torch.Tensor] = {}
    y_store: Dict[int, torch.Tensor] = {}
    handles = []

    def mk_x(li):
        def hook(_m, args):
            x_store[li] = args[0][:, pos, :].detach().clone()
        return hook

    def mk_g(li):
        def hook(_m, _a, output):
            g_store[li] = output.view(input_ids.shape[0], seq_len, -1)[:, pos, :].detach().clone()
        return hook

    def mk_y(li):
        def hook(_m, _a, output):
            hs = output[0] if isinstance(output, tuple) else output
            y_store[li] = hs[:, pos, :].detach().clone()
        return hook

    for li in capture_layers:
        block = model.model.layers[li].mlp
        handles.append(block.register_forward_pre_hook(mk_x(li)))
        handles.append(block.gate.register_forward_hook(mk_g(li)))
        handles.append(block.register_forward_hook(mk_y(li)))

    try:
        out = model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return Capture(x=x_store, g=g_store, y=y_store,
                   nll=next_token_nll(out.logits, targets, seq_len))


@torch.no_grad()
def action_nll(model, input_ids: torch.Tensor, targets: torch.Tensor,
               top12_ids: torch.Tensor, action: int) -> torch.Tensor:
    """True next-token NLL after applying ``action`` at Layer 12.

    Only the target layer's decision for the experimental token changes; every other
    token, layer, and later routing decision stays native, and Layers 13-16 run
    normally after the intervention.
    """
    n, seq_len = input_ids.shape
    ids = action_identities(top12_ids, action)
    override = {flat_index(i, seq_len): ids[i] for i in range(n)}
    with forced_route(model, TARGET_LAYER, override):
        out = model(input_ids=input_ids, use_cache=False)
    return next_token_nll(out.logits, targets, seq_len)


@torch.no_grad()
def selected_expert_contributions(block, x: torch.Tensor, logits: torch.Tensor):
    """Weighted contributions c = p * o for the native Top-8 SELECTED experts only.

    Returns ``(identities, probs, contributions)``. No rejected expert is executed.
    """
    identities, probs = native_topk(logits, TOP_K)
    n, hidden = x.shape
    contrib = torch.zeros(n, TOP_K, hidden, dtype=torch.float32, device=x.device)
    for slot in range(TOP_K):
        col_ids = identities[:, slot]
        col_p = probs[:, slot]
        for e in torch.unique(col_ids).tolist():
            rows = torch.nonzero(col_ids == e, as_tuple=True)[0]
            contrib[rows, slot] = block.experts[int(e)](x[rows]).float() * col_p[rows, None]
    return identities, probs, contrib


def fused_from_contributions(contrib: torch.Tensor) -> torch.Tensor:
    """Sum over the eight selected experts: the native fused MoE contribution."""
    return contrib.sum(dim=1)
