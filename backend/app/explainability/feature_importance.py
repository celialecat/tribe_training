"""Permutation feature importance for the metadata features.

Model-agnostic and always available (no captum/shap needed): shuffle one
metadata feature across the batch and measure how much the prediction error
grows. A feature the model relies on causes a large degradation when scrambled.
Averaged over repeats for stability.
"""

from __future__ import annotations

import numpy as np
import torch

from app.dataset.features import FEATURE_NAMES
from app.models.model import SuccessPredictor
from app.models.normalizer import TargetNormalizer


@torch.no_grad()
def _masked_mse(
    model: SuccessPredictor,
    brain: torch.Tensor,
    metadata: torch.Tensor,
    pad_mask: torch.Tensor,
    target_z: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    mean, _ = model(brain, metadata, pad_mask)
    err = ((mean - target_z) ** 2 * mask).sum()
    return float(err / mask.sum().clamp_min(1.0))


@torch.no_grad()
def permutation_importance(
    model: SuccessPredictor,
    brain: torch.Tensor,      # (N, T, V)
    metadata: torch.Tensor,   # (N, F)
    pad_mask: torch.Tensor,   # (N, T)
    targets: torch.Tensor,    # (N, num_targets) raw units
    mask: torch.Tensor,       # (N, num_targets)
    normalizer: TargetNormalizer,
    *,
    n_repeats: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """Return ``{feature_name: mean error increase}`` from permutation."""
    model.eval()
    target_z = normalizer.normalize(targets)
    baseline = _masked_mse(model, brain, metadata, pad_mask, target_z, mask)

    generator = torch.Generator().manual_seed(seed)
    n, f = metadata.shape
    importances = np.zeros(f, dtype=np.float64)
    for j in range(f):
        deltas = []
        for _ in range(n_repeats):
            perm = torch.randperm(n, generator=generator)
            shuffled = metadata.clone()
            shuffled[:, j] = metadata[perm, j]
            score = _masked_mse(model, brain, shuffled, pad_mask, target_z, mask)
            deltas.append(score - baseline)
        importances[j] = float(np.mean(deltas))

    names = FEATURE_NAMES if f == len(FEATURE_NAMES) else tuple(f"feat_{i}" for i in range(f))
    return {name: float(importances[i]) for i, name in enumerate(names)}
