"""Checkpointing: atomic saves, rolling retention and full-state resume.

A checkpoint captures everything needed to resume training bit-for-bit — model,
optimiser, scheduler, AMP scaler, RNG-independent counters, the fitted target
normaliser and the composed config. ``best.pt`` always points at the best model
by the monitored metric; ``last.pt`` at the most recent. Numbered epoch files
are retained up to ``keep_last_k``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class CheckpointState:
    """The full payload persisted to disk."""

    model: dict[str, Any]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any] | None
    scaler: dict[str, Any] | None
    epoch: int
    global_step: int
    best_metric: float | None
    normalizer: dict[str, Any]
    config: dict[str, Any]
    metrics: dict[str, float]

    def to_payload(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "scaler": self.scaler,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "normalizer": self.normalizer,
            "config": self.config,
            "metrics": self.metrics,
        }


class CheckpointManager:
    """Manage checkpoint files within a run directory."""

    def __init__(self, directory: str | Path, *, keep_last_k: int = 3) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_k = keep_last_k

    # ------------------------------------------------------------------ save --
    def save(self, state: CheckpointState, *, is_best: bool) -> Path:
        payload = state.to_payload()
        epoch_path = self.dir / f"epoch_{state.epoch:04d}.pt"
        self._atomic_save(payload, epoch_path)
        self._atomic_save(payload, self.dir / "last.pt")
        if is_best:
            self._atomic_save(payload, self.dir / "best.pt")
            logger.info("New best checkpoint at epoch %d -> best.pt", state.epoch)
        self._prune()
        return epoch_path

    @staticmethod
    def _atomic_save(payload: dict[str, Any], path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)  # atomic on POSIX; avoids half-written checkpoints

    def _prune(self) -> None:
        epochs = sorted(self.dir.glob("epoch_*.pt"))
        for stale in epochs[: -self.keep_last_k] if self.keep_last_k > 0 else []:
            stale.unlink(missing_ok=True)

    # ------------------------------------------------------------------ load --
    def load(self, path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        logger.info("Loaded checkpoint %s (epoch %s)", path, payload.get("epoch"))
        return payload

    def latest(self) -> Path | None:
        last = self.dir / "last.pt"
        return last if last.exists() else None

    def best(self) -> Path | None:
        best = self.dir / "best.pt"
        return best if best.exists() else None
