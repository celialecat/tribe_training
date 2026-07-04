"""Multitask regression losses with masking and per-target weighting.

The primary objective is a **masked heteroscedastic Gaussian negative log-
likelihood**: for each target the model predicts a mean and a log-variance, and
the loss is the Gaussian NLL, evaluated only where the target is observed and
weighted per target. This jointly optimises accuracy and uncertainty calibration
— the variance term is penalised for both over- and under-confidence.

When ``heteroscedastic`` is disabled the loss degrades exactly to a masked,
weighted MSE (constant unit variance), so the same code path serves both modes.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from app.core.constants import TARGET_ORDER


class MaskedMultitaskLoss(nn.Module):
    """Masked, per-target-weighted Gaussian NLL (or MSE) over the target vector."""

    def __init__(
        self,
        target_weights: dict[str, float] | None = None,
        *,
        heteroscedastic: bool = True,
    ) -> None:
        super().__init__()
        weights = target_weights or {}
        weight_vec = torch.tensor(
            [float(weights.get(t.value, 1.0)) for t in TARGET_ORDER], dtype=torch.float32
        )
        self.register_buffer("weights", weight_vec)
        self.heteroscedastic = heteroscedastic
        self._log_2pi = math.log(2.0 * math.pi)

    def forward(
        self,
        mean: torch.Tensor,      # (B, T)
        log_var: torch.Tensor,   # (B, T)
        target: torch.Tensor,    # (B, T) — standardised
        mask: torch.Tensor,      # (B, T) — 1 where observed
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(scalar_loss, per_target_mse)`` where MSE is for logging."""
        squared_error = (mean - target) ** 2

        if self.heteroscedastic:
            precision = torch.exp(-log_var)
            per_element = 0.5 * (precision * squared_error + log_var + self._log_2pi)
        else:
            per_element = 0.5 * squared_error

        weighted = per_element * self.weights.to(mean.device)
        masked = weighted * mask
        denom = mask.sum().clamp_min(1.0)
        loss = masked.sum() / denom

        # Per-target MSE (in standardised space) for monitoring, mask-aware.
        with torch.no_grad():
            per_target_count = mask.sum(0).clamp_min(1.0)
            per_target_mse = (squared_error * mask).sum(0) / per_target_count
            logs = {t.value: per_target_mse[i] for i, t in enumerate(TARGET_ORDER)}
        return loss, logs
