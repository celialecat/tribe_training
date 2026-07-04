"""Optimizer construction with sensible weight-decay handling.

Decoupled weight decay is not applied to biases, LayerNorm/BatchNorm parameters
or 1-D parameters — decaying those hurts more than it helps. We split parameters
into two groups accordingly, a small detail that matters for Transformer models.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def build_optimizer(model: nn.Module, *, cfg: Any) -> torch.optim.Optimizer:
    """Construct an optimiser from the training config mapping."""
    kind = str(cfg.get("optimizer", "adamw")).lower()
    lr = float(cfg.get("lr", 3e-4))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    betas = tuple(float(b) for b in cfg.get("betas", (0.9, 0.999)))

    decay, no_decay = _split_parameters(model)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    if kind == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=betas)
    if kind == "adam":
        return torch.optim.Adam(groups, lr=lr, betas=betas)
    if kind == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9, nesterov=True)
    raise ValueError(f"unknown optimizer: {kind!r}")


def _split_parameters(model: nn.Module) -> tuple[list, list]:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay
