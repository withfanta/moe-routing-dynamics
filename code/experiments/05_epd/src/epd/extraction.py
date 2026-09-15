"""Frozen OLMoE capture and the reused EIPC projection for EPD-P0.

Native routing semantics follow the installed transformers 4.45.1
`OlmoeSparseMoeBlock.forward`: router logits from a bias-free gate, fp32 softmax over all
64 experts, Top-8, no Top-k renormalization, and each expert output weighted by its own
original probability.

Only the experts the native router already selected are executed. Nothing here trains, and
no OLMoE parameter receives gradients.

The projection is the EIPC-P0 matrix, regenerated from its recorded seed and then verified
against the recorded hash. If the hash does not match, extraction refuses to proceed rather
than silently using a different projection.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from . import (
    CONTEXT_LEN,
    EIPC_PROJECTION_JSON,
    EIPC_PROJECTION_SEED,
    EIPC_PROJECTION_SHA256,
    HIDDEN_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    NUM_EXPERTS,
    PROJ_DIM,
    TOP_K,
)


# ------------------------------------------------------------------ projection


def projection_hash(P: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(P, dtype=np.float64).tobytes()).hexdigest()


def load_eipc_projection(json_path: str = EIPC_PROJECTION_JSON) -> tuple:
    """Rebuild the EIPC-P0 projection and verify it against the recorded artifact.

    EIPC-P0 stored the projection's metadata and hash rather than the matrix itself, so the
    matrix is regenerated from the recorded seed and distribution and then hash-checked
    against both the artifact and the expected constant. Any mismatch raises.
    """
    with open(json_path) as fh:
        meta = json.load(fh)

    if meta["seed"] != EIPC_PROJECTION_SEED:
        raise AssertionError(f"EIPC projection seed is {meta['seed']}, expected {EIPC_PROJECTION_SEED}")
    if meta["out_dim"] != PROJ_DIM or meta["shape"] != [HIDDEN_SIZE, PROJ_DIM]:
        raise AssertionError(f"unexpected EIPC projection shape {meta['shape']}")

    rng = np.random.default_rng(meta["seed"])
    P = rng.normal(loc=0.0, scale=1.0 / np.sqrt(PROJ_DIM), size=(HIDDEN_SIZE, PROJ_DIM))

    h = projection_hash(P)
    if h != meta["sha256"]:
        raise AssertionError(
            f"regenerated projection hash {h} != EIPC artifact hash {meta['sha256']}")
    if h != EIPC_PROJECTION_SHA256:
        raise AssertionError(
            f"projection hash {h} != expected {EIPC_PROJECTION_SHA256}")
    return P, meta


def project(P: np.ndarray, c: np.ndarray) -> np.ndarray:
    if c.shape[-1] != HIDDEN_SIZE:
        raise ValueError(f"expected trailing dim {HIDDEN_SIZE}, got {c.shape[-1]}")
    return c @ P


@torch.no_grad()
def project_torch(P_t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return c.float() @ P_t


# ----------------------------------------------------------------------- model


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
    if model.training:
        raise AssertionError("model must be in eval() mode")
    bad = [n for n, p in model.named_parameters() if p.requires_grad]
    if bad:
        raise AssertionError(f"{len(bad)} OLMoE parameters require grad")


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


def router_probs(logits: torch.Tensor) -> torch.Tensor:
    """fp32 softmax over all 64 experts, before Top-k masking."""
    return F.softmax(logits, dim=-1, dtype=torch.float)


def native_topk(logits: torch.Tensor, k: int = TOP_K):
    """Top-k identities and their ORIGINAL probabilities, in descending rank order."""
    probs = router_probs(logits)
    top = torch.topk(probs, k, dim=-1)
    return top.indices, top.values


@dataclass
class Capture:
    x: Dict[int, torch.Tensor]
    g: Dict[int, torch.Tensor]
    y: Dict[int, torch.Tensor]


@torch.no_grad()
def native_forward(model, input_ids: torch.Tensor, capture_layers: Sequence[int]) -> Capture:
    """One unmodified native pass, capturing MoE input, router logits and fused output."""
    seq_len = input_ids.shape[1]
    pos = seq_len - 1  # the experimental token
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
        model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return Capture(x=x_store, g=g_store, y=y_store)


@torch.no_grad()
def selected_expert_contributions(block, x: torch.Tensor, logits: torch.Tensor):
    """c = p * o for the native Top-8 only, in native rank order.

    Returns ``(ids, probs, contrib)`` with contrib shaped (n, 8, hidden). Slot k holds the
    rank-(k+1) selected expert, which is what CONTENT_RANK relies on. Unselected experts
    are never executed.
    """
    ids, probs = native_topk(logits, TOP_K)
    n, hidden = x.shape
    contrib = torch.zeros(n, TOP_K, hidden, dtype=torch.float32, device=x.device)
    for slot in range(TOP_K):
        col_ids = ids[:, slot]
        col_p = probs[:, slot]
        for e in torch.unique(col_ids).tolist():
            rows = torch.nonzero(col_ids == e, as_tuple=True)[0]
            contrib[rows, slot] = block.experts[int(e)](x[rows]).float() * col_p[rows, None]
    return ids, probs, contrib


def fused_from_contributions(contrib: torch.Tensor) -> torch.Tensor:
    """Sum over the eight selected experts: the native fused MoE contribution."""
    return contrib.sum(dim=1)
