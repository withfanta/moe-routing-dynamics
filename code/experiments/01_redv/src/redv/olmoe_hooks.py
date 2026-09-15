"""Frozen-model plumbing for REDV-V1.

Two things happen here:

1. A native forward pass that captures, for the experimental token only, the
   MoE-block input ``x``, the router logits ``g``, and the post-layer hidden
   state ``h``.
2. A forced-route forward pass that overrides exactly one MoE routing decision
   (one layer, one token) and otherwise runs the frozen model normally.

Routing semantics are copied from the installed
``transformers.models.olmoe.modeling_olmoe.OlmoeSparseMoeBlock.forward``:
softmax over all 64 experts in fp32, top-k, renormalize only when
``norm_topk_prob`` is true (it is false for this checkpoint), cast to input
dtype, then scale each expert output by its own router probability. The forced
path applies the override *before* the ``norm_topk_prob`` check so that forced
routes obey identical weighting semantics; the only intended difference from
native execution is the identity of the eighth expert.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from . import CONTEXT_LEN, MODEL_ID, MODEL_REVISION, TOP_K


def load_model(dtype=torch.float16, device: str = "cuda", model_dir: Optional[str] = None):
    """Load the pinned OLMoE checkpoint frozen and in eval mode."""
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


def moe_blocks(model) -> List[torch.nn.Module]:
    return [layer.mlp for layer in model.model.layers]


# ---------------------------------------------------------------- forced routing


def _generic_moe_forward(block, hidden_states: torch.Tensor, override: Optional[Dict[int, torch.Tensor]]):
    """Re-implementation of the installed OlmoeSparseMoeBlock.forward.

    With ``override=None`` this is the native computation. ``override`` maps a
    flattened token index to the eight forced expert identities for that token;
    those experts keep their ORIGINAL 64-way router probabilities.
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
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=block.num_experts).permute(2, 1, 0)

    for expert_idx in range(block.num_experts):
        expert_layer = block.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


@contextlib.contextmanager
def forced_route(model, layer_idx: int, override: Dict[int, torch.Tensor]):
    """Temporarily force one MoE layer's routing for specific flattened tokens.

    Only ``layer_idx`` is affected; every other layer, token, and later routing
    decision runs natively.
    """
    block = model.model.layers[layer_idx].mlp
    original = block.forward

    def patched(hidden_states):
        return _generic_moe_forward(block, hidden_states, override)

    block.forward = patched
    try:
        yield
    finally:
        block.forward = original


def build_override(
    full_probs: torch.Tensor,
    flat_indices: Sequence[int],
    replacement_ranks: Sequence[int],
) -> Dict[int, torch.Tensor]:
    """Native ranks 1-7 plus one replacement rank, per token.

    ``full_probs`` is the (n_tokens, 64) native router probability matrix for the
    tokens named by ``flat_indices``. Every forced route executes exactly eight
    experts, so compute is unchanged.
    """
    max_rank = max(replacement_ranks)
    ranked = torch.topk(full_probs, max_rank, dim=-1).indices  # (n, max_rank)
    override: Dict[int, torch.Tensor] = {}
    for row, (flat_idx, rank) in enumerate(zip(flat_indices, replacement_ranks)):
        keep = ranked[row, : TOP_K - 1]  # native ranks 1-7
        swap = ranked[row, rank - 1 : rank]  # the replacement rank
        override[int(flat_idx)] = torch.cat([keep, swap])
    return override


def native_top_identities(full_probs: torch.Tensor) -> torch.Tensor:
    """Native top-8 expert identities in native topk order."""
    return torch.topk(full_probs, TOP_K, dim=-1).indices


# ------------------------------------------------------------------- capture


@dataclass
class NativeCapture:
    """Per-context native quantities at the experimental token.

    ``x``/``g``/``h`` are keyed by layer index. ``token_nll`` is the native
    next-token NLL for the block's true target.
    """

    x: Dict[int, torch.Tensor]
    g: Dict[int, torch.Tensor]
    h: Dict[int, torch.Tensor]
    token_nll: torch.Tensor


def _target_position(seq_len: int) -> int:
    """The experimental token: final position of the 128-token context."""
    return seq_len - 1


def next_token_nll(logits: torch.Tensor, targets: torch.Tensor, seq_len: int) -> torch.Tensor:
    """NLL of the true target token from the experimental token's position."""
    pos = _target_position(seq_len)
    step = logits[:, pos, :].float()
    return F.cross_entropy(step, targets, reduction="none")


@torch.no_grad()
def native_forward(
    model,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    capture_layers: Sequence[int],
) -> NativeCapture:
    """One native forward pass, capturing x, g, h at the requested layers."""
    seq_len = input_ids.shape[1]
    pos = _target_position(seq_len)
    x_store: Dict[int, torch.Tensor] = {}
    g_store: Dict[int, torch.Tensor] = {}
    h_store: Dict[int, torch.Tensor] = {}
    handles = []

    def make_x_hook(li):
        def hook(_module, args):
            # MoE block input: (batch, seq, hidden), pre-flatten.
            x_store[li] = args[0][:, pos, :].detach().clone()
        return hook

    def make_g_hook(li):
        def hook(_module, _args, output):
            # Router logits: (batch * seq, 64), exactly as the native block sees them.
            g_store[li] = output.view(input_ids.shape[0], seq_len, -1)[:, pos, :].detach().clone()
        return hook

    def make_h_hook(li):
        def hook(_module, _args, output):
            hs = output[0] if isinstance(output, tuple) else output
            h_store[li] = hs[:, pos, :].detach().clone()
        return hook

    for li in capture_layers:
        layer = model.model.layers[li]
        handles.append(layer.mlp.register_forward_pre_hook(make_x_hook(li)))
        handles.append(layer.mlp.gate.register_forward_hook(make_g_hook(li)))
        handles.append(layer.register_forward_hook(make_h_hook(li)))

    try:
        out = model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    nll = next_token_nll(out.logits, targets, seq_len)
    return NativeCapture(x=x_store, g=g_store, h=h_store, token_nll=nll)


@torch.no_grad()
def forced_forward_nll(
    model,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    layer_idx: int,
    override: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """Next-token NLL under a forced route at one layer for the experimental token."""
    seq_len = input_ids.shape[1]
    with forced_route(model, layer_idx, override):
        out = model(input_ids=input_ids, use_cache=False)
    return next_token_nll(out.logits, targets, seq_len)


def flat_index(batch_pos: int, seq_len: int) -> int:
    """Flattened index of the experimental token for batch row ``batch_pos``."""
    return batch_pos * seq_len + _target_position(seq_len)


def assert_frozen(model) -> None:
    """Test D support: model is in eval mode with no trainable parameters."""
    if model.training:
        raise AssertionError("model must be in eval() mode")
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if trainable:
        raise AssertionError(f"{len(trainable)} parameters still require grad, e.g. {trainable[:3]}")


def check_context_len(input_ids: torch.Tensor) -> None:
    if input_ids.shape[1] != CONTEXT_LEN:
        raise AssertionError(f"context length is frozen at {CONTEXT_LEN}, got {input_ids.shape[1]}")
