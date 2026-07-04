"""BrainEncoder — compress TRIBE brain activity ``(T, V)`` into a latent vector.

Design rationale
================
TRIBE emits a very wide, moderately long signal: ``T`` timesteps (fMRI TRs, up
to a few hundred) by ``V ≈ 20k`` cortical vertices. Feeding that directly to a
predictor is both statistically and computationally hopeless. The encoder is a
four-stage funnel, each stage chosen for a specific property of the signal:

1. **Spatial read-out** (``LayerNorm`` over vertices → ``Linear`` → ``GELU``).
   Per-vertex activity scales vary widely, so we normalise across the cortex
   first, then learn a low-rank spatial projection - effectively a *learned
   parcellation* that maps ~20k vertices to a few hundred spatial components.
   This is the bulk of the parameters and where cortical topography is read.

2. **Temporal convolutions** (depthwise-separable ``Conv1d`` stack with
   residuals). fMRI responses are smooth and autocorrelated over neighbouring
   TRs (the haemodynamic response function blurs ~5-6 s). Local convolutions
   model these short-range dynamics cheaply before the quadratic-cost attention.

3. **Transformer encoder over time**. Long-range temporal structure — a payoff
   that lands seconds after its setup, narrative arcs — needs global mixing.
   A standard pre-norm Transformer with sinusoidal positions and a key-padding
   mask handles variable-length, right-padded sequences.

4. **Pooling to a fixed latent**. Attention pooling (a learned query attends
   over time, respecting the mask) collapses the sequence to one vector, then a
   projection + ``LayerNorm`` yields the ``latent_dim`` (512) representation.
   Attention weights are exposed for explainability (which moments mattered).

The module also exposes attention maps so the explainability layer can attribute
predictions back to specific moments in the video.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from app.models.config import BrainEncoderConfig


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to the temporal sequence."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self.pe[: x.size(1)].unsqueeze(0)


class TemporalConvBlock(nn.Module):
    """Residual depthwise-separable 1D conv over time (channels-first)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size, padding=padding, groups=in_ch
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        residual = self.residual(x)
        y = self.pointwise(self.depthwise(x))
        y = self.dropout(self.act(self.norm(y)))
        return y + residual


class AttentionPool(nn.Module):
    """Single-query attention pooling over time with padding-mask support."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.scale = d_model**-0.5

    def forward(
        self, x: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D). Returns (pooled (B, D), weights (B, T)).
        scores = (self.query * self.scale) @ x.transpose(1, 2)  # (B, 1, T)
        scores = scores.squeeze(1)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # (B, T)
        pooled = torch.einsum("bt,btd->bd", weights, x)
        return pooled, weights


class BrainEncoder(nn.Module):
    """Encode ``(B, T, V)`` brain activity to ``(B, latent_dim)``."""

    def __init__(self, config: BrainEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or BrainEncoderConfig()
        cfg = self.config

        # Stage 1 — spatial read-out.
        self.vertex_norm = nn.LayerNorm(cfg.input_vertices)
        self.vertex_proj = nn.Linear(cfg.input_vertices, cfg.vertex_proj_dim)
        self.spatial_act = nn.GELU()

        # Stage 2 — temporal convolutions.
        conv_blocks: list[nn.Module] = []
        in_ch = cfg.vertex_proj_dim
        for out_ch, kernel in zip(cfg.temporal_channels, cfg.temporal_kernel_sizes, strict=True):
            conv_blocks.append(TemporalConvBlock(in_ch, out_ch, kernel, cfg.dropout))
            in_ch = out_ch
        self.temporal_convs = nn.ModuleList(conv_blocks)

        # Stage 3 — Transformer over time.
        self.input_proj = nn.Linear(in_ch, cfg.d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is incompatible with norm_first pre-norm layers;
        # disable it explicitly to avoid a spurious runtime warning.
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.n_layers, enable_nested_tensor=False
        )

        # Stage 4 — pooling + projection to latent.
        if cfg.pool == "attention":
            self.pool = AttentionPool(cfg.d_model)
        else:
            self.pool = None  # mean/cls handled inline
        if cfg.pool == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.to_latent = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
        )

    @property
    def latent_dim(self) -> int:
        return self.config.latent_dim

    def forward(
        self,
        brain: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of brain activity.

        Parameters
        ----------
        brain: ``(B, T, V)`` float tensor.
        pad_mask: ``(B, T)`` bool tensor, True where padded (ignored).
        return_attention: also return temporal attention weights ``(B, T)``.
        """
        x = self.vertex_norm(brain)
        x = self.spatial_act(self.vertex_proj(x))          # (B, T, vertex_proj_dim)

        x = x.transpose(1, 2)                              # (B, C, T)
        for block in self.temporal_convs:
            x = block(x)
        x = x.transpose(1, 2)                              # (B, T, C)

        x = self.input_proj(x)                             # (B, T, d_model)

        cls_prefix = 0
        if self.config.pool == "cls":
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
            if pad_mask is not None:
                pad_mask = torch.cat(
                    [torch.zeros(x.size(0), 1, dtype=torch.bool, device=x.device), pad_mask],
                    dim=1,
                )
            cls_prefix = 1

        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=pad_mask)

        pooled, weights = self._pool(x, pad_mask, cls_prefix)
        latent = self.to_latent(pooled)
        if return_attention:
            return latent, weights
        return latent

    def encode(
        self, brain: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Alias returning only the latent (used by embedding/explainability)."""
        out = self.forward(brain, pad_mask=pad_mask, return_attention=False)
        assert isinstance(out, torch.Tensor)
        return out

    def _pool(
        self, x: torch.Tensor, pad_mask: torch.Tensor | None, cls_prefix: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.pool == "attention":
            assert self.pool is not None
            return self.pool(x, pad_mask)
        if self.config.pool == "cls":
            pooled = x[:, 0]
            # No per-timestep attribution for CLS pooling; return zeros over the
            # original (non-CLS) timesteps so the attention shape stays (B, T).
            weights = torch.zeros(x.size(0), x.size(1) - cls_prefix, device=x.device)
            return pooled, weights
        # mean pooling over valid timesteps
        if pad_mask is not None:
            valid = (~pad_mask).float().unsqueeze(-1)      # (B, T, 1)
            pooled = (x * valid).sum(1) / valid.sum(1).clamp_min(1.0)
            weights = (~pad_mask).float()
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1.0)
        else:
            pooled = x.mean(1)
            weights = torch.full((x.size(0), x.size(1)), 1.0 / x.size(1), device=x.device)
        del cls_prefix
        return pooled, weights
