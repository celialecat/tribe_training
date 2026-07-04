"""Neural network definitions: brain encoder and multitask head.

Public surface:
    - :class:`SuccessPredictor`   end-to-end model (encoder + multitask head)
    - :class:`BrainEncoder`       (T, V) -> latent
    - :class:`MultitaskPredictor` latent + metadata -> mean/log-var
    - :class:`MaskedMultitaskLoss` training objective
    - :class:`TargetNormalizer`   per-target standardisation
    - :func:`build_model`         construct a model from a Hydra/omegaconf mapping
"""

from __future__ import annotations

from typing import Any

from app.models.brain_encoder import BrainEncoder
from app.models.config import BrainEncoderConfig, ModelConfig, PredictorConfig
from app.models.losses import MaskedMultitaskLoss
from app.models.model import Prediction, SuccessPredictor, virality_score
from app.models.normalizer import TargetNormalizer
from app.models.predictor import MultitaskPredictor

__all__ = [
    "BrainEncoder",
    "BrainEncoderConfig",
    "MaskedMultitaskLoss",
    "ModelConfig",
    "MultitaskPredictor",
    "Prediction",
    "PredictorConfig",
    "SuccessPredictor",
    "TargetNormalizer",
    "build_model",
    "virality_score",
]


def build_model(model_cfg: Any) -> SuccessPredictor:
    """Build a :class:`SuccessPredictor` from a model config mapping.

    Accepts an OmegaConf ``DictConfig`` or any nested mapping with
    ``brain_encoder`` and ``predictor`` sections.
    """
    return SuccessPredictor(ModelConfig.from_mapping(model_cfg))
