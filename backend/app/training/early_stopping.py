"""Early stopping on a monitored validation metric."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EarlyStopping:
    """Stop training when the monitored metric stops improving.

    ``mode="min"`` treats lower as better (e.g. loss); ``"max"`` the reverse.
    """

    patience: int = 12
    mode: str = "min"
    min_delta: float = 0.0
    best: float | None = None
    num_bad_epochs: int = 0

    def is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float) -> bool:
        """Record a metric; return True if this is a new best."""
        if self.is_improvement(value):
            self.best = value
            self.num_bad_epochs = 0
            return True
        self.num_bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.num_bad_epochs >= self.patience
