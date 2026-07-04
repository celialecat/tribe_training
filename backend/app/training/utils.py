"""Small training utilities: seeding, metric meters, distributed helpers."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(slots=True)
class AverageMeter:
    """Tracks a running average of a scalar (e.g. loss over an epoch)."""

    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


@dataclass(slots=True)
class DistributedContext:
    """Snapshot of the distributed environment for the current process."""

    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = "nccl"
    extra: dict = field(default_factory=dict)

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def detect_distributed(requested: bool) -> DistributedContext:
    """Read torchrun env vars to decide whether/how to run distributed."""
    if not requested or "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return DistributedContext(enabled=False)
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size <= 1:
        return DistributedContext(enabled=False)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    return DistributedContext(
        enabled=True,
        rank=int(os.environ["RANK"]),
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        world_size=world_size,
        backend=backend,
    )
