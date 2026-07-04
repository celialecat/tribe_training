"""MultitaskPredictor — brain latent + metadata → success targets + uncertainty.

A shared-trunk multitask MLP. The trunk consumes the concatenated brain latent
and causal metadata features; two linear heads emit, per target, a **mean** and
a **log-variance**. Predicting variance (a *heteroscedastic* head) lets the model
express input-dependent uncertainty, which is exactly what calibrated confidence
intervals require — some videos are inherently more predictable than others.

All targets are modelled in the standardised space produced by
:class:`app.models.normalizer.TargetNormalizer`; de-normalisation and clamping of
bounded targets happen at the prediction boundary, not inside the network.
"""

from __future__ import annotations

import torch
from torch import nn

from app.models.config import PredictorConfig

# Clamp predicted log-variance for numerical stability (variance in ~[e^-7, e^3]).
_LOGVAR_MIN, _LOGVAR_MAX = -7.0, 3.0


class MultitaskPredictor(nn.Module):
    """Predict per-target mean and log-variance from latent + metadata."""

    def __init__(self, config: PredictorConfig | None = None) -> None:
        super().__init__()
        self.config = config or PredictorConfig()
        cfg = self.config

        in_dim = cfg.latent_dim + cfg.metadata_dim
        layers: list[nn.Module] = []
        prev = in_dim
        for hidden in cfg.hidden_dims:
            layers += [
                nn.Linear(prev, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            ]
            prev = hidden
        self.trunk = nn.Sequential(*layers)

        self.mean_head = nn.Linear(prev, cfg.num_targets)
        self.logvar_head = (
            nn.Linear(prev, cfg.num_targets) if cfg.heteroscedastic else None
        )

    def forward(
        self, latent: torch.Tensor, metadata: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, log_var)`` each of shape ``(B, num_targets)``.

        When the head is homoscedastic, ``log_var`` is returned as zeros (unit
        variance) so downstream code has a single, uniform interface.
        """
        features = torch.cat([latent, metadata], dim=-1)
        hidden = self.trunk(features)
        mean = self.mean_head(hidden)
        if self.logvar_head is not None:
            log_var = self.logvar_head(hidden).clamp(_LOGVAR_MIN, _LOGVAR_MAX)
        else:
            log_var = torch.zeros_like(mean)
        return mean, log_var
