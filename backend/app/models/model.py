"""SuccessPredictor — the end-to-end model: BrainEncoder + MultitaskPredictor.

``forward`` returns raw standardised ``(mean, log_var)`` for training. ``predict``
wraps a trained model with a :class:`TargetNormalizer` to return human-unit point
estimates, per-target std-devs and confidence intervals — the object the API and
dashboard consume.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from scipy.stats import norm
from torch import nn

from app.core.constants import BOUNDED_TARGETS, TARGET_ORDER, Target
from app.models.brain_encoder import BrainEncoder
from app.models.config import ModelConfig
from app.models.normalizer import TargetNormalizer
from app.models.predictor import MultitaskPredictor


@dataclass(slots=True)
class Prediction:
    """A decoded, human-unit prediction for one video."""

    values: dict[str, float]              # target -> point estimate
    intervals: dict[str, tuple[float, float]]  # target -> (low, high)
    uncertainty: dict[str, float]         # target -> std-dev (original units)
    attention: list[float] | None = None  # temporal attention weights, if any
    confidence_level: float = 0.95


class SuccessPredictor(nn.Module):
    """Compose the brain encoder and multitask predictor into one model."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.encoder = BrainEncoder(self.config.brain_encoder)
        self.predictor = MultitaskPredictor(self.config.predictor)

    def forward(
        self,
        brain: torch.Tensor,
        metadata: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(brain, pad_mask=pad_mask, return_attention=return_attention)
        if return_attention:
            latent, attention = encoded  # type: ignore[misc]
            mean, log_var = self.predictor(latent, metadata)
            return mean, log_var, attention
        latent = encoded  # type: ignore[assignment]
        mean, log_var = self.predictor(latent, metadata)
        return mean, log_var

    @torch.no_grad()
    def predict(
        self,
        brain: torch.Tensor,
        metadata: torch.Tensor,
        normalizer: TargetNormalizer,
        pad_mask: torch.Tensor | None = None,
        *,
        confidence_level: float = 0.95,
    ) -> list[Prediction]:
        """Decode a batch into calibrated, human-unit predictions."""
        self.eval()
        mean_z, log_var_z, attention = self.forward(
            brain, metadata, pad_mask=pad_mask, return_attention=True
        )
        # De-standardise mean and std-dev back to original target units.
        mean = normalizer.denormalize(mean_z)
        sigma_z = torch.exp(0.5 * log_var_z)
        sigma = normalizer.denormalize_std(sigma_z)

        z_score = float(norm.ppf(0.5 + confidence_level / 2.0))
        low = mean - z_score * sigma
        high = mean + z_score * sigma

        results: list[Prediction] = []
        for b in range(mean.size(0)):
            values, intervals, uncertainty = {}, {}, {}
            for i, target in enumerate(TARGET_ORDER):
                point = float(mean[b, i])
                lo, hi = float(low[b, i]), float(high[b, i])
                if target in BOUNDED_TARGETS:
                    point = _clip01(point)
                    lo, hi = _clip01(lo), _clip01(hi)
                values[target.value] = point
                intervals[target.value] = (lo, hi)
                uncertainty[target.value] = float(sigma[b, i])
            results.append(
                Prediction(
                    values=values,
                    intervals=intervals,
                    uncertainty=uncertainty,
                    attention=attention[b].tolist(),
                    confidence_level=confidence_level,
                )
            )
        return results

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def virality_score(prediction: Prediction) -> float:
    """A single 0-100 headline score blending virality and 30-day reach."""
    virality = prediction.values.get(Target.virality.value, 0.0)
    log_views_30d = prediction.values.get(Target.log_views_30d.value, 0.0)
    # Normalise log10(views) roughly into [0, 1] over 0..8 (1 to 100M views).
    reach = max(0.0, min(1.0, log_views_30d / 8.0))
    return round(100.0 * (0.6 * virality + 0.4 * reach), 1)
