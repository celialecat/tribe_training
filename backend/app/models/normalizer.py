"""TargetNormalizer — per-target standardisation with mask awareness.

Targets span very different scales (log-views ~[0, 8] vs. engagement ~[0, 0.2]).
Standardising each target to zero mean / unit variance puts them on comparable
footing so the multitask loss is not dominated by the largest-magnitude target.

Statistics are fit on the *training split only* (respecting the observation
mask) and serialised with the checkpoint, so inference de-normalises with the
exact same constants — a common and easily-missed source of train/serve skew.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from app.core.constants import NUM_TARGETS, TARGET_ORDER


@dataclass(slots=True)
class TargetNormalizer:
    """Affine per-target standardiser: ``z = (y - mean) / std``."""

    mean: np.ndarray  # (NUM_TARGETS,)
    std: np.ndarray   # (NUM_TARGETS,)

    @classmethod
    def identity(cls) -> TargetNormalizer:
        return cls(mean=np.zeros(NUM_TARGETS, np.float32), std=np.ones(NUM_TARGETS, np.float32))

    @classmethod
    def fit(cls, values: np.ndarray, mask: np.ndarray, *, eps: float = 1e-6) -> TargetNormalizer:
        """Fit from ``(N, NUM_TARGETS)`` values + mask, ignoring masked entries."""
        values = np.asarray(values, dtype=np.float64)
        mask = np.asarray(mask, dtype=np.float64)
        count = mask.sum(axis=0)
        mean = np.where(count > 0, (values * mask).sum(0) / np.maximum(count, 1.0), 0.0)
        var = np.where(
            count > 0,
            ((values - mean) ** 2 * mask).sum(0) / np.maximum(count, 1.0),
            1.0,
        )
        std = np.sqrt(np.maximum(var, eps))
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    # ---------------------------------------------------------- transforms --
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        mean, std = self._tensors(y)
        return (y - mean) / std

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        mean, std = self._tensors(z)
        return z * std + mean

    def denormalize_std(self, sigma_z: torch.Tensor) -> torch.Tensor:
        """Map a std-dev from standardised space back to original units."""
        _, std = self._tensors(sigma_z)
        return sigma_z * std

    def _tensors(self, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.as_tensor(self.mean, dtype=ref.dtype, device=ref.device)
        std = torch.as_tensor(self.std, dtype=ref.dtype, device=ref.device)
        return mean, std

    # ------------------------------------------------------------- (de)ser --
    def to_dict(self) -> dict[str, list[float]]:
        return {
            "targets": [t.value for t in TARGET_ORDER],
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> TargetNormalizer:
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
        )
