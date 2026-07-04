"""Explainer — a single facade over all attribution methods.

Combines the per-sample explanations (temporal attention + Integrated Gradients
→ brain-region importance + metadata attributions) into one JSON-serialisable
object for the dashboard, and offers a dataset-level permutation feature
importance. Methods are chosen to always work without optional dependencies;
SHAP is available separately when installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from app.core.constants import TARGET_ORDER, Target
from app.explainability.attention import temporal_attention
from app.explainability.brain_regions import aggregate_region_importance
from app.explainability.feature_importance import permutation_importance
from app.explainability.integrated_gradients import integrated_gradients
from app.models.model import SuccessPredictor
from app.models.normalizer import TargetNormalizer


@dataclass(slots=True)
class SampleExplanation:
    """All per-sample attributions for one target."""

    target: str
    attention: dict[str, list[float]]
    brain_regions: dict[str, float]
    metadata_attribution: dict[str, float]
    ig_convergence_delta: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "attention": self.attention,
            "brain_regions": self.brain_regions,
            "metadata_attribution": self.metadata_attribution,
            "ig_convergence_delta": self.ig_convergence_delta,
            **self.extras,
        }


class Explainer:
    """Produce attributions for a trained :class:`SuccessPredictor`."""

    def __init__(
        self, model: SuccessPredictor, normalizer: TargetNormalizer, *, device: str = "cpu"
    ) -> None:
        self.model = model.to(device).eval()
        self.normalizer = normalizer
        self.device = torch.device(device)

    def explain_sample(
        self,
        brain: torch.Tensor,      # (1, T, V) or (T, V)
        metadata: torch.Tensor,   # (1, F) or (F,)
        *,
        pad_mask: torch.Tensor | None = None,
        target: Target = Target.log_views_30d,
        tr_seconds: float = 1.49,
        hemodynamic_offset_seconds: float = 5.0,
        region_bands: int = 4,
        ig_steps: int = 32,
    ) -> SampleExplanation:
        """Explain a single video's prediction for one target."""
        brain = self._ensure_batch(brain).to(self.device)
        metadata = self._ensure_batch(metadata).to(self.device)
        if pad_mask is not None:
            pad_mask = self._ensure_batch(pad_mask).to(self.device)
        target_index = TARGET_ORDER.index(target)

        timeline = temporal_attention(
            self.model, brain, metadata, pad_mask=pad_mask,
            tr_seconds=tr_seconds, hemodynamic_offset_seconds=hemodynamic_offset_seconds,
        )
        ig = integrated_gradients(
            self.model, brain, metadata,
            target_index=target_index, pad_mask=pad_mask, steps=ig_steps,
        )
        regions = aggregate_region_importance(ig.brain, bands=region_bands)
        meta_names = self._metadata_names(ig.metadata.shape[0])
        meta_attr = {name: float(ig.metadata[i]) for i, name in enumerate(meta_names)}

        return SampleExplanation(
            target=target.value,
            attention=timeline.as_dict(),
            brain_regions=regions,
            metadata_attribution=meta_attr,
            ig_convergence_delta=ig.convergence_delta,
            extras={"attention_peak_seconds": timeline.peak_time()},
        )

    def dataset_feature_importance(
        self,
        brain: torch.Tensor,
        metadata: torch.Tensor,
        pad_mask: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        *,
        n_repeats: int = 5,
    ) -> dict[str, float]:
        """Permutation importance of metadata features over a batch."""
        return permutation_importance(
            self.model, brain.to(self.device), metadata.to(self.device),
            pad_mask.to(self.device), targets.to(self.device), mask.to(self.device),
            self.normalizer, n_repeats=n_repeats,
        )

    # ------------------------------------------------------------- helpers --
    @staticmethod
    def _ensure_batch(tensor: torch.Tensor) -> torch.Tensor:
        return tensor if tensor.dim() >= 2 and tensor.size(0) == 1 else tensor.unsqueeze(0)

    @staticmethod
    def _metadata_names(dim: int) -> list[str]:
        from app.dataset.features import FEATURE_NAMES

        if dim == len(FEATURE_NAMES):
            return list(FEATURE_NAMES)
        return [f"feat_{i}" for i in range(dim)]
