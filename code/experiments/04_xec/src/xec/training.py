"""Policy training for XEC-P0.

Supervised 5-way cross entropy against frozen oracle labels. AdamW, lr 1e-3, wd 0.01,
batch 64, at most 20 epochs, no scheduler, no class weighting, no hyperparameter search.
Checkpoint selection is the lowest VALIDATION cross entropy; the TEST oracle is never
consulted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, WEIGHT_DECAY


@dataclass
class TrainLog:
    variant: str
    seed: int
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    trainable_params: int = 0


def _batches(n: int, batch_size: int, generator: torch.Generator):
    order = torch.randperm(n, generator=generator)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


@torch.no_grad()
def evaluate_loss(model: nn.Module, u: torch.Tensor, items: Optional[torch.Tensor],
                  y: torch.Tensor, batch_size: int = 256) -> float:
    model.eval()
    total, count = 0.0, 0
    for start in range(0, u.shape[0], batch_size):
        sl = slice(start, start + batch_size)
        it = None if items is None else items[sl]
        logits = model(u[sl], it)
        loss = F.cross_entropy(logits, y[sl], reduction="sum")
        total += float(loss)
        count += int(y[sl].shape[0])
    return total / count


@torch.no_grad()
def predict_actions(model: nn.Module, u: torch.Tensor, items: Optional[torch.Tensor],
                    batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, u.shape[0], batch_size):
        sl = slice(start, start + batch_size)
        it = None if items is None else items[sl]
        out.append(model(u[sl], it).argmax(dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.int64)


def train_policy(
    model: nn.Module,
    u_train: torch.Tensor,
    items_train: Optional[torch.Tensor],
    y_train: torch.Tensor,
    u_val: torch.Tensor,
    items_val: Optional[torch.Tensor],
    y_val: torch.Tensor,
    seed: int,
    log_fn=None,
) -> TrainLog:
    """Train one variant and return the best-validation state in ``model``."""
    from .policy import count_trainable

    tl = TrainLog(variant=getattr(model, "variant", "?"), seed=seed,
                  trainable_params=count_trainable(model))
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(seed)

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    tl.best_val_loss = evaluate_loss(model, u_val, items_val, y_val)
    tl.best_epoch = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running, seen = 0.0, 0
        for idx in _batches(u_train.shape[0], BATCH_SIZE, gen):
            it = None if items_train is None else items_train[idx]
            logits = model(u_train[idx], it)
            loss = F.cross_entropy(logits, y_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss) * int(idx.shape[0])
            seen += int(idx.shape[0])

        tr = running / seen
        va = evaluate_loss(model, u_val, items_val, y_val)
        tl.train_loss.append(tr)
        tl.val_loss.append(va)

        if va < tl.best_val_loss:
            tl.best_val_loss = va
            tl.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if log_fn is not None:
            log_fn(f"    epoch {epoch:2d}: train CE {tr:.5f}  val CE {va:.5f}"
                   f"{'  <- best' if tl.best_epoch == epoch else ''}")

    model.load_state_dict(best_state)
    return tl


def assert_no_frozen_model_grads(olmoe_model) -> None:
    """Guard: the policy optimizer must never touch OLMoE parameters."""
    bad = [n for n, p in olmoe_model.named_parameters() if p.requires_grad]
    if bad:
        raise AssertionError(f"OLMoE parameters require grad: {bad[:3]}")
