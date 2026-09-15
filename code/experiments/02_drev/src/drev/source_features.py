"""OLMoE interception and Layer-4 source features for DREV-P0.

Copied from the validated REDV implementation (`src/redv/olmoe_hooks.py` and
`src/redv/rejected.py`, REDV commit 6d1c57f), which reproduces the installed
`transformers 4.45.1` `OlmoeSparseMoeBlock.forward` exactly: router logits from a
bias-free gate, fp32 softmax over all 64 experts, top-8 selection, no top-k
renormalization when `norm_topk_prob` is false, then each expert output scaled by
its own original router probability.

Source features are computed ONCE at Layer 4 and reused by every horizon worker:

    b_i = [h_i ; g_i]                                   2048 + 64 = 2112
    r_i = sum_{j in ranks 9..12} p_ij * E_j(x_i)         2048
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from . import CONTEXT_LEN, MODEL_ID, MODEL_REVISION, REJECTED_RANKS, TOP_K


def load_model(dtype=torch.float16, device: str = "cuda", model_dir: Optional[str] = None):
    """Load the pinned checkpoint frozen, in eval mode, FP16."""
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
    if model.training:
        raise AssertionError("model must be in eval() mode")
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if trainable:
        raise AssertionError(f"{len(trainable)} parameters require grad")


def verify_config(model) -> None:
    cfg = model.config
    assert cfg.architectures == ["OlmoeForCausalLM"], cfg.architectures
    assert cfg.hidden_size == 2048
    assert cfg.num_hidden_layers == 16
    assert cfg.num_experts == 64
    assert cfg.num_experts_per_tok == TOP_K == 8
    assert cfg.norm_topk_prob is False


def check_context_len(input_ids: torch.Tensor) -> None:
    if input_ids.shape[1] != CONTEXT_LEN:
        raise AssertionError(f"context length frozen at {CONTEXT_LEN}, got {input_ids.shape[1]}")


# ------------------------------------------------------------ MoE re-implementation


def _generic_moe_forward(block, hidden_states: torch.Tensor, override: Optional[Dict[int, torch.Tensor]]):
    """The installed OlmoeSparseMoeBlock.forward, with optional route override.

    ``override`` maps a flattened token index to eight forced expert identities.
    Forced experts keep their ORIGINAL 64-way router probabilities; the override is
    applied before the ``norm_topk_prob`` check so forced routes obey identical
    weighting semantics.
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
        dtype=hidden_states.dtype, device=hidden_states.device,
    )
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=block.num_experts).permute(2, 1, 0)

    for expert_idx in range(block.num_experts):
        expert_layer = block.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits


@contextlib.contextmanager
def forced_route(model, layer_idx: int, override: Dict[int, torch.Tensor]):
    """Force one layer's routing for specific tokens. All other layers stay native."""
    block = model.model.layers[layer_idx].mlp
    original = block.forward

    def patched(hidden_states):
        return _generic_moe_forward(block, hidden_states, override)

    block.forward = patched
    try:
        yield
    finally:
        block.forward = original


def probs_from_logits(g: torch.Tensor) -> torch.Tensor:
    """p = softmax(g) over all 64 experts, fp32, pre-masking."""
    return F.softmax(g, dim=-1, dtype=torch.float)


def router_probs(block, x: torch.Tensor) -> torch.Tensor:
    return F.softmax(block.gate(x), dim=-1, dtype=torch.float)


def native_top_identities(full_probs: torch.Tensor) -> torch.Tensor:
    return torch.topk(full_probs, TOP_K, dim=-1).indices


def build_override(
    full_probs: torch.Tensor, flat_indices: Sequence[int], replacement_ranks: Sequence[int]
) -> Dict[int, torch.Tensor]:
    """Native ranks 1-7 plus one replacement rank. Always exactly 8 experts."""
    max_rank = max(replacement_ranks)
    ranked = torch.topk(full_probs, max_rank, dim=-1).indices
    override: Dict[int, torch.Tensor] = {}
    for row, (flat_idx, rank) in enumerate(zip(flat_indices, replacement_ranks)):
        keep = ranked[row, : TOP_K - 1]
        swap = ranked[row, rank - 1 : rank]
        override[int(flat_idx)] = torch.cat([keep, swap])
    return override


# ------------------------------------------------------------ rejected evidence


def rejected_identities(full_probs: torch.Tensor):
    """Identities and ORIGINAL probabilities at ranks 9-12, in rank order."""
    max_rank = max(REJECTED_RANKS)
    top = torch.topk(full_probs, max_rank, dim=-1)
    cols = [r - 1 for r in REJECTED_RANKS]
    return top.indices[:, cols], top.values[:, cols]


def expert_output(block, expert_idx: int, x: torch.Tensor) -> torch.Tensor:
    """E_j(x): the model's own expert module, so it matches by construction."""
    return block.experts[expert_idx](x)


@torch.no_grad()
def rejected_evidence(block, x: torch.Tensor, full_probs: torch.Tensor | None = None) -> torch.Tensor:
    """r = sum_{j in ranks 9..12} p_j * E_j(x). Probabilities NOT renormalized."""
    if full_probs is None:
        full_probs = router_probs(block, x)

    identities, probs = rejected_identities(full_probs)
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)

    for slot in range(identities.shape[1]):
        col_ids = identities[:, slot]
        col_p = probs[:, slot]
        for expert_idx in torch.unique(col_ids).tolist():
            rows = torch.nonzero(col_ids == expert_idx, as_tuple=True)[0]
            out[rows] += expert_output(block, int(expert_idx), x[rows]).float() * col_p[rows, None]
    return out


@torch.no_grad()
def rejected_evidence_reference(block, x: torch.Tensor) -> torch.Tensor:
    """Straight-line reference, one token and expert at a time. For tests."""
    full_probs = router_probs(block, x)
    identities, probs = rejected_identities(full_probs)
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
    for row in range(x.shape[0]):
        for slot in range(identities.shape[1]):
            j = int(identities[row, slot])
            out[row] += expert_output(block, j, x[row : row + 1])[0].float() * probs[row, slot]
    return out


# ------------------------------------------------------------------- capture


@dataclass
class Capture:
    x: Dict[int, torch.Tensor]
    g: Dict[int, torch.Tensor]
    h: Dict[int, torch.Tensor]
    token_nll: torch.Tensor


def _target_position(seq_len: int) -> int:
    """The experimental token: final position of the 128-token context."""
    return seq_len - 1


def next_token_nll(logits: torch.Tensor, targets: torch.Tensor, seq_len: int) -> torch.Tensor:
    pos = _target_position(seq_len)
    return F.cross_entropy(logits[:, pos, :].float(), targets, reduction="none")


def flat_index(batch_pos: int, seq_len: int) -> int:
    return batch_pos * seq_len + _target_position(seq_len)


@torch.no_grad()
def native_forward(model, input_ids, targets, capture_layers: Sequence[int]) -> Capture:
    """One native pass, capturing x, g, h at the requested layers."""
    seq_len = input_ids.shape[1]
    pos = _target_position(seq_len)
    x_store: Dict[int, torch.Tensor] = {}
    g_store: Dict[int, torch.Tensor] = {}
    h_store: Dict[int, torch.Tensor] = {}
    handles = []

    def mk_x(li):
        def hook(_m, args):
            x_store[li] = args[0][:, pos, :].detach().clone()
        return hook

    def mk_g(li):
        def hook(_m, _a, output):
            g_store[li] = output.view(input_ids.shape[0], seq_len, -1)[:, pos, :].detach().clone()
        return hook

    def mk_h(li):
        def hook(_m, _a, output):
            hs = output[0] if isinstance(output, tuple) else output
            h_store[li] = hs[:, pos, :].detach().clone()
        return hook

    for li in capture_layers:
        layer = model.model.layers[li]
        handles.append(layer.mlp.register_forward_pre_hook(mk_x(li)))
        handles.append(layer.mlp.gate.register_forward_hook(mk_g(li)))
        handles.append(layer.register_forward_hook(mk_h(li)))

    try:
        out = model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return Capture(x=x_store, g=g_store, h=h_store,
                   token_nll=next_token_nll(out.logits, targets, seq_len))


@torch.no_grad()
def forced_forward_nll(model, input_ids, targets, layer_idx: int, override) -> torch.Tensor:
    """Next-token NLL with one layer's route forced for the experimental token."""
    with forced_route(model, layer_idx, override):
        out = model(input_ids=input_ids, use_cache=False)
    return next_token_nll(out.logits, targets, input_ids.shape[1])
