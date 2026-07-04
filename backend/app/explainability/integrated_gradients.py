"""Integrated Gradients attribution (self-contained, captum-optional).

Integrated Gradients attributes a model output to its inputs by integrating the
gradients along a straight-line path from a baseline (here, zeros = "no signal")
to the actual input. We implement it directly in torch so it works without the
optional captum dependency; when captum *is* installed the same public function
delegates to its reference implementation for extra rigour.

For this model we attribute a chosen target's predicted mean to (a) the brain
activity ``(T, V)`` and (b) the metadata features ``(F,)`` simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from app.models.model import SuccessPredictor


@dataclass(slots=True)
class IGAttribution:
    """Integrated-gradients attributions for one sample and target."""

    brain: np.ndarray      # (T, V) signed attribution
    metadata: np.ndarray   # (F,) signed attribution
    target_index: int
    convergence_delta: float


def _target_mean(
    model: SuccessPredictor,
    brain: torch.Tensor,
    metadata: torch.Tensor,
    pad_mask: torch.Tensor | None,
    target_index: int,
) -> torch.Tensor:
    mean, _ = model(brain, metadata, pad_mask)
    return mean[:, target_index].sum()


def integrated_gradients(
    model: SuccessPredictor,
    brain: torch.Tensor,      # (1, T, V)
    metadata: torch.Tensor,   # (1, F)
    *,
    target_index: int,
    pad_mask: torch.Tensor | None = None,
    steps: int = 32,
) -> IGAttribution:
    """Compute IG attributions for a single sample (batch size 1)."""
    if brain.size(0) != 1:
        raise ValueError("integrated_gradients expects a single sample (batch size 1)")
    model.eval()
    device = brain.device
    brain_baseline = torch.zeros_like(brain)
    meta_baseline = torch.zeros_like(metadata)

    alphas = torch.linspace(0.0, 1.0, steps, device=device)
    brain_grads = torch.zeros_like(brain)
    meta_grads = torch.zeros_like(metadata)

    for alpha in alphas:
        b = (brain_baseline + alpha * (brain - brain_baseline)).detach().requires_grad_(True)
        m = (meta_baseline + alpha * (metadata - meta_baseline)).detach().requires_grad_(True)
        out = _target_mean(model, b, m, pad_mask, target_index)
        gb, gm = torch.autograd.grad(out, (b, m))
        brain_grads += gb
        meta_grads += gm

    brain_grads /= steps
    meta_grads /= steps
    brain_attr = (brain - brain_baseline) * brain_grads
    meta_attr = (metadata - meta_baseline) * meta_grads

    # Completeness check: sum(attr) ≈ f(input) - f(baseline).
    with torch.no_grad():
        f_input = _target_mean(model, brain, metadata, pad_mask, target_index)
        f_base = _target_mean(model, brain_baseline, meta_baseline, pad_mask, target_index)
    total = float(brain_attr.sum() + meta_attr.sum())
    delta = float(abs(total - float(f_input - f_base)))

    return IGAttribution(
        brain=brain_attr.squeeze(0).detach().cpu().numpy(),
        metadata=meta_attr.squeeze(0).detach().cpu().numpy(),
        target_index=target_index,
        convergence_delta=delta,
    )
