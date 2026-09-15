"""Frozen OLMoE loading and native capture for EIPC-P0.

Router semantics follow the installed `transformers 4.45.1`
`OlmoeSparseMoeBlock.forward`: router logits from a bias-free gate, fp32 softmax
over all 64 experts, Top-8 selection, no Top-k renormalization when
`norm_topk_prob` is false, and each expert output weighted by its original
probability.

EIPC-P0 executes ONLY the experts the native router already selected. No rejected
expert is ever run, and model execution is never modified: the re-execution of
selected experts happens in analysis code on captured inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from . import CONTEXT_LEN, MODEL_ID, MODEL_REVISION, NUM_EXPERTS, TOP_K


def load_model(dtype=torch.float16, device: str = "cuda", model_dir: Optional[str] = None):
    """Load the pinned checkpoint frozen, eval, FP16."""
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
    assert cfg.num_experts == NUM_EXPERTS == 64
    assert cfg.num_experts_per_tok == TOP_K == 8
    assert cfg.norm_topk_prob is False


def check_context_len(input_ids: torch.Tensor) -> None:
    if input_ids.shape[1] != CONTEXT_LEN:
        raise AssertionError(f"context length frozen at {CONTEXT_LEN}, got {input_ids.shape[1]}")


def experimental_position(seq_len: int) -> int:
    """The experimental token: final position of the 128-token context."""
    return seq_len - 1


def router_probs(logits: torch.Tensor) -> torch.Tensor:
    """p = softmax(logits) over all 64 experts, fp32, before Top-k masking."""
    return F.softmax(logits, dim=-1, dtype=torch.float)


def native_topk(logits: torch.Tensor):
    """Native Top-8 identities and their ORIGINAL probabilities, in topk order.

    No renormalization: this checkpoint has norm_topk_prob = false.
    """
    probs = router_probs(logits)
    top = torch.topk(probs, TOP_K, dim=-1)
    return top.indices, top.values


@dataclass
class NativeCapture:
    """Per-layer native quantities at the experimental token.

    ``x`` is the MoE-block input, ``g`` the router logits, ``y`` the stock fused MoE
    block output. All keyed by code layer index.
    """

    x: Dict[int, torch.Tensor]
    g: Dict[int, torch.Tensor]
    y: Dict[int, torch.Tensor]


@torch.no_grad()
def native_forward(model, input_ids: torch.Tensor, capture_layers: Sequence[int]) -> NativeCapture:
    """One unmodified native forward pass, capturing x, g and the fused output y."""
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
            # OlmoeSparseMoeBlock returns (final_hidden_states, router_logits).
            hs = output[0] if isinstance(output, tuple) else output
            y_store[li] = hs[:, pos, :].detach().clone()
        return hook

    for li in capture_layers:
        block = model.model.layers[li].mlp
        handles.append(block.register_forward_pre_hook(mk_x(li)))
        handles.append(block.gate.register_forward_hook(mk_g(li)))
        handles.append(block.register_forward_hook(mk_y(li)))

    try:
        model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return NativeCapture(x=x_store, g=g_store, y=y_store)


@torch.no_grad()
def selected_expert_contributions(block, x: torch.Tensor, logits: torch.Tensor):
    """Weighted contributions c_{i,e} = p_{i,e} * o_{i,e} for SELECTED experts only.

    Returns ``(identities, probs, contributions)`` with shapes ``(n, 8)``,
    ``(n, 8)`` and ``(n, 8, hidden)``. Only the eight experts the native router
    already chose are executed; unselected experts are never run.
    """
    identities, probs = native_topk(logits)
    n, hidden = x.shape
    contrib = torch.zeros(n, TOP_K, hidden, dtype=torch.float32, device=x.device)

    # Group by expert identity so each selected expert runs on one batch.
    for slot in range(TOP_K):
        col_ids = identities[:, slot]
        col_p = probs[:, slot]
        for e in torch.unique(col_ids).tolist():
            rows = torch.nonzero(col_ids == e, as_tuple=True)[0]
            out = block.experts[int(e)](x[rows])
            contrib[rows, slot] = out.float() * col_p[rows, None]

    return identities, probs, contrib


@torch.no_grad()
def fused_from_contributions(contrib: torch.Tensor) -> torch.Tensor:
    """y = sum over the eight selected experts. The ordinary fused MoE output."""
    return contrib.sum(dim=1)
