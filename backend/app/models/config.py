"""Typed model configuration.

Frozen dataclasses that mirror ``configs/model/*.yaml``. Keeping a typed layer
between Hydra/OmegaConf and the ``nn.Module`` constructors means the modules
have no dependency on OmegaConf, are trivially unit-testable with plain Python,
and fail loudly on malformed configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.constants import BRAIN_LATENT_DIM, NUM_TARGETS
from app.dataset.features import FEATURE_DIM


@dataclass(frozen=True, slots=True)
class BrainEncoderConfig:
    """Configuration for :class:`app.models.brain_encoder.BrainEncoder`."""

    input_vertices: int = 20_484
    vertex_proj_dim: int = 256
    temporal_channels: tuple[int, ...] = (256, 384)
    temporal_kernel_sizes: tuple[int, ...] = (5, 3)
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 4
    ff_dim: int = 1536
    dropout: float = 0.1
    pool: str = "attention"  # attention | mean | cls
    latent_dim: int = BRAIN_LATENT_DIM

    def __post_init__(self) -> None:
        if len(self.temporal_channels) != len(self.temporal_kernel_sizes):
            raise ValueError("temporal_channels and temporal_kernel_sizes must be equal length")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.pool not in {"attention", "mean", "cls"}:
            raise ValueError(f"unknown pool: {self.pool!r}")

    @classmethod
    def from_mapping(cls, cfg: Any) -> BrainEncoderConfig:
        return cls(
            input_vertices=int(cfg.get("input_vertices", 20_484)),
            vertex_proj_dim=int(cfg.get("vertex_proj_dim", 256)),
            temporal_channels=tuple(int(c) for c in cfg.get("temporal_channels", (256, 384))),
            temporal_kernel_sizes=tuple(int(k) for k in cfg.get("temporal_kernel_sizes", (5, 3))),
            d_model=int(cfg.get("d_model", 384)),
            n_heads=int(cfg.get("n_heads", 6)),
            n_layers=int(cfg.get("n_layers", 4)),
            ff_dim=int(cfg.get("ff_dim", 1536)),
            dropout=float(cfg.get("dropout", 0.1)),
            pool=str(cfg.get("pool", "attention")),
            latent_dim=int(cfg.get("latent_dim", BRAIN_LATENT_DIM)),
        )


@dataclass(frozen=True, slots=True)
class PredictorConfig:
    """Configuration for :class:`app.models.predictor.MultitaskPredictor`."""

    latent_dim: int = BRAIN_LATENT_DIM
    metadata_dim: int = FEATURE_DIM
    hidden_dims: tuple[int, ...] = (512, 256)
    dropout: float = 0.2
    num_targets: int = NUM_TARGETS
    heteroscedastic: bool = True

    @classmethod
    def from_mapping(cls, cfg: Any) -> PredictorConfig:
        return cls(
            latent_dim=int(cfg.get("latent_dim", BRAIN_LATENT_DIM)),
            metadata_dim=int(cfg.get("metadata_dim", FEATURE_DIM)),
            hidden_dims=tuple(int(h) for h in cfg.get("hidden_dims", (512, 256))),
            dropout=float(cfg.get("dropout", 0.2)),
            num_targets=int(cfg.get("num_targets", NUM_TARGETS)),
            heteroscedastic=bool(cfg.get("heteroscedastic", True)),
        )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Top-level model config bundling encoder + predictor."""

    brain_encoder: BrainEncoderConfig = field(default_factory=BrainEncoderConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)

    @classmethod
    def from_mapping(cls, cfg: Any) -> ModelConfig:
        return cls(
            brain_encoder=BrainEncoderConfig.from_mapping(cfg["brain_encoder"]),
            predictor=PredictorConfig.from_mapping(cfg["predictor"]),
        )
