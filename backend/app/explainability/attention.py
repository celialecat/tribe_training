"""Temporal attention extraction.

The BrainEncoder's attention-pooling layer yields, per video, a weight over
timesteps indicating which moments most influenced the pooled representation.
We surface that as a time-aligned importance curve (weight per second), which
the dashboard plots beneath the video timeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from app.models.model import SuccessPredictor


@dataclass(slots=True)
class AttentionTimeline:
    """Per-timestep attention aligned to real video time."""

    weights: np.ndarray      # (T_valid,) attention weights summing to ~1
    times_seconds: np.ndarray  # (T_valid,) timestamp of each weight

    def peak_time(self) -> float:
        if self.weights.size == 0:
            return 0.0
        return float(self.times_seconds[int(np.argmax(self.weights))])

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "weights": self.weights.tolist(),
            "times_seconds": self.times_seconds.tolist(),
        }


@torch.no_grad()
def temporal_attention(
    model: SuccessPredictor,
    brain: torch.Tensor,      # (1, T, V)
    metadata: torch.Tensor,   # (1, F)
    *,
    pad_mask: torch.Tensor | None = None,
    tr_seconds: float = 1.49,
    hemodynamic_offset_seconds: float = 5.0,
) -> AttentionTimeline:
    """Extract the temporal attention timeline for a single sample."""
    model.eval()
    _, _, attention = model(brain, metadata, pad_mask, return_attention=True)
    weights = attention.squeeze(0).cpu().numpy()  # (T,)

    valid = weights.shape[0]
    if pad_mask is not None:
        valid = int((~pad_mask.squeeze(0).cpu()).sum())
        weights = weights[:valid]

    # Map timestep index -> stimulus time. TRIBE shifts predictions backward by
    # the haemodynamic offset, so we add it back to recover stimulus onset time.
    times = np.arange(valid, dtype=np.float32) * tr_seconds + hemodynamic_offset_seconds
    return AttentionTimeline(weights=weights.astype(np.float32), times_seconds=times)
