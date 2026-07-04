"""Learning-rate schedulers.

Provides a linear-warmup → cosine-decay schedule (the workhorse for Transformer
training) and a validation-plateau schedule, selected by config. The cosine
schedule is step-based so warmup is precise regardless of dataset size.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler, ReduceLROnPlateau


def build_scheduler(
    optimizer: Optimizer,
    *,
    kind: str,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr: float = 1e-6,
    base_lr: float = 3e-4,
) -> LRScheduler | ReduceLROnPlateau | None:
    """Construct a scheduler by name.

    ``cosine`` and ``none`` are step-based (call ``.step()`` every optimiser
    step); ``plateau`` is epoch-based on a validation metric.
    """
    kind = kind.lower()
    if kind == "none":
        return None
    if kind == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=min_lr)
    if kind == "cosine":
        min_ratio = max(min_lr / base_lr, 0.0)

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        return LambdaLR(optimizer, lr_lambda)
    raise ValueError(f"unknown scheduler kind: {kind!r}")


def is_step_based(scheduler: object) -> bool:
    """True if the scheduler advances per optimiser step (not per epoch)."""
    return isinstance(scheduler, LambdaLR)
