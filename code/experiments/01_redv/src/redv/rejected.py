"""Rejected-evidence vector r_{l,t} for REDV-V1.

For source layer l and the experimental token t, OLMoE normally executes only
ranks 1-8. For this analysis only, the four near-miss rejected experts at ranks
9, 10, 11, 12 are explicitly executed on the SAME expert input x_{l,t}:

    r_{l,t} = sum_{j in ranks 9..12}  p_{l,t,j} * E_{l,j}(x_{l,t})

``p_{l,t,j}`` is the ORIGINAL unmodified router probability of expert j. It is
deliberately NOT renormalized over ranks 9-12, so weakly scored rejected experts
stay weak. This is an oracle analysis: intentionally expensive, because the
question is whether rejected information exists at all.

No attention weights, no learned aggregator, no max/mean/concat/MLP variant.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from . import REJECTED_RANKS


def router_probs(block, x: torch.Tensor) -> torch.Tensor:
    """Native 64-way router probabilities for expert-block input ``x``.

    Matches the installed implementation: linear gate then fp32 softmax over all
    experts, before any top-k masking.
    """
    logits = block.gate(x)
    return F.softmax(logits, dim=-1, dtype=torch.float)


def probs_from_logits(g: torch.Tensor) -> torch.Tensor:
    """p = softmax(g) over all 64 experts, fp32, pre-masking."""
    return F.softmax(g, dim=-1, dtype=torch.float)


def rejected_identities(full_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expert identities and original probabilities at ranks 9-12.

    Returns ``(identities, probs)`` each shaped ``(n_tokens, 4)``, ordered by
    rank 9, 10, 11, 12.
    """
    max_rank = max(REJECTED_RANKS)
    top = torch.topk(full_probs, max_rank, dim=-1)
    cols = [r - 1 for r in REJECTED_RANKS]
    return top.indices[:, cols], top.values[:, cols]


def expert_output(block, expert_idx: int, x: torch.Tensor) -> torch.Tensor:
    """E_{l,j}(x): the full FFN output of expert j on input x.

    Delegates to the model's own expert module, so it is by construction the same
    computation the model would perform for that expert.
    """
    return block.experts[expert_idx](x)


@torch.no_grad()
def rejected_evidence(block, x: torch.Tensor, full_probs: torch.Tensor | None = None) -> torch.Tensor:
    """Compute r_{l,t} for a batch of expert inputs.

    ``x`` is ``(n_tokens, hidden)``. Returns ``(n_tokens, hidden)`` in fp32.
    """
    if full_probs is None:
        full_probs = router_probs(block, x)

    identities, probs = rejected_identities(full_probs)
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)

    # Group tokens by expert identity so each expert runs on a single batch.
    for slot in range(identities.shape[1]):
        col_ids = identities[:, slot]
        col_p = probs[:, slot]
        for expert_idx in torch.unique(col_ids).tolist():
            rows = torch.nonzero(col_ids == expert_idx, as_tuple=True)[0]
            e_out = expert_output(block, int(expert_idx), x[rows])
            out[rows] += e_out.float() * col_p[rows, None]

    return out


@torch.no_grad()
def rejected_evidence_reference(block, x: torch.Tensor) -> torch.Tensor:
    """Straight-line reference implementation, one token and one expert at a time.

    Used by the unit tests to confirm the grouped implementation above is
    numerically the same thing.
    """
    full_probs = router_probs(block, x)
    identities, probs = rejected_identities(full_probs)
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
    for row in range(x.shape[0]):
        for slot in range(identities.shape[1]):
            j = int(identities[row, slot])
            p = probs[row, slot]
            out[row] += expert_output(block, j, x[row : row + 1])[0].float() * p
    return out


def rejected_report(full_probs: torch.Tensor) -> Dict[str, object]:
    """Small descriptive dict for logging/manifest. No research statistic."""
    identities, probs = rejected_identities(full_probs)
    return {
        "ranks": list(REJECTED_RANKS),
        "renormalized": False,
        "example_identities": identities[0].tolist(),
        "example_probs": [float(v) for v in probs[0].tolist()],
    }
