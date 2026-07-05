"""Tests for the brain encoder, predictor, losses, normalizer and full model."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
import torch

from app.core.constants import NUM_TARGETS, TARGET_ORDER
from app.dataset.features import FEATURE_DIM
from app.models import (
    BrainEncoder,
    MaskedMultitaskLoss,
    MultitaskPredictor,
    SuccessPredictor,
    TargetNormalizer,
    build_model,
)
from app.models.config import BrainEncoderConfig, ModelConfig, PredictorConfig

torch.manual_seed(0)

TARGET_NAMES = {t.value for t in TARGET_ORDER}

V = 64  # small vertex count for fast tests
SMALL_ENCODER = BrainEncoderConfig(
    input_vertices=V, vertex_proj_dim=32, temporal_channels=(32, 48),
    temporal_kernel_sizes=(5, 3), d_model=48, n_heads=4, n_layers=2, ff_dim=96, latent_dim=64,
)
SMALL_PREDICTOR = PredictorConfig(latent_dim=64, metadata_dim=FEATURE_DIM, hidden_dims=(64, 32))
SMALL_MODEL = ModelConfig(brain_encoder=SMALL_ENCODER, predictor=SMALL_PREDICTOR)


def _batch(b: int = 4, t: int = 20):
    brain = torch.randn(b, t, V)
    pad_mask = torch.zeros(b, t, dtype=torch.bool)
    pad_mask[0, t - 3 :] = True  # first sample partly padded
    metadata = torch.randn(b, FEATURE_DIM)
    return brain, pad_mask, metadata


# ------------------------------------------------------------------ encoder --
@pytest.mark.parametrize("pool", ["attention", "mean", "cls"])
def test_brain_encoder_output_shape(pool: str) -> None:
    cfg = BrainEncoderConfig(**{**asdict(SMALL_ENCODER), "pool": pool})
    enc = BrainEncoder(cfg)
    brain, pad_mask, _ = _batch()
    latent, attn = enc(brain, pad_mask=pad_mask, return_attention=True)
    assert latent.shape == (4, cfg.latent_dim)
    assert attn.shape == (4, brain.size(1))
    assert torch.isfinite(latent).all()


def test_encoder_config_validation() -> None:
    with pytest.raises(ValueError, match="divisible"):
        BrainEncoderConfig(d_model=48, n_heads=5)
    with pytest.raises(ValueError, match="equal length"):
        BrainEncoderConfig(temporal_channels=(32,), temporal_kernel_sizes=(3, 5))


# ---------------------------------------------------------------- predictor --
def test_predictor_shapes_and_homoscedastic_zeros() -> None:
    pred = MultitaskPredictor(PredictorConfig(latent_dim=64, heteroscedastic=False))
    mean, log_var = pred(torch.randn(3, 64), torch.randn(3, FEATURE_DIM))
    assert mean.shape == (3, NUM_TARGETS)
    assert torch.count_nonzero(log_var) == 0  # homoscedastic -> zeros


# -------------------------------------------------------------------- loss --
def test_masked_loss_ignores_masked_targets() -> None:
    loss_fn = MaskedMultitaskLoss(heteroscedastic=True)
    mean = torch.zeros(2, NUM_TARGETS, requires_grad=True)
    log_var = torch.zeros(2, NUM_TARGETS, requires_grad=True)
    target = torch.ones(2, NUM_TARGETS)
    mask = torch.zeros(2, NUM_TARGETS)
    mask[0, 0] = 1.0  # only one observed target
    loss, logs = loss_fn(mean, log_var, target, mask)
    loss.backward()
    assert torch.isfinite(loss)
    # Gradient only where observed.
    assert mean.grad[0, 0] != 0
    assert mean.grad[1, 1] == 0
    assert set(logs.keys()) == TARGET_NAMES


def test_loss_reduces_to_mse_when_homoscedastic() -> None:
    loss_fn = MaskedMultitaskLoss(heteroscedastic=False)
    mean = torch.zeros(1, NUM_TARGETS)
    mean[0, 0] = 2.0
    target = torch.zeros(1, NUM_TARGETS)
    mask = torch.zeros(1, NUM_TARGETS)
    mask[0, 0] = 1.0
    loss, _ = loss_fn(mean, torch.zeros_like(mean), target, mask)
    assert loss.item() == pytest.approx(0.5 * 4.0)  # 0.5 * (2-0)^2


# ------------------------------------------------------------- normalizer --
def test_target_normalizer_roundtrip_with_mask() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(5.0, 2.0, size=(100, NUM_TARGETS)).astype(np.float32)
    mask = np.ones_like(values)
    norm = TargetNormalizer.fit(values, mask)
    z = norm.normalize(torch.from_numpy(values))
    assert z.mean(0).abs().max() < 0.2
    back = norm.denormalize(z)
    torch.testing.assert_close(back, torch.from_numpy(values), rtol=1e-4, atol=1e-3)
    # Serialisation round-trip.
    norm2 = TargetNormalizer.from_dict(norm.to_dict())
    np.testing.assert_allclose(norm.mean, norm2.mean)


# ------------------------------------------------------------------ model --
def test_build_model_from_mapping_and_predict() -> None:
    mapping = {
        "brain_encoder": asdict(SMALL_ENCODER),
        "predictor": asdict(SMALL_PREDICTOR),
    }
    model = build_model(mapping)
    assert isinstance(model, SuccessPredictor)
    brain, pad_mask, metadata = _batch()
    norm = TargetNormalizer.identity()
    preds = model.predict(brain, metadata, norm, pad_mask=pad_mask, confidence_level=0.9)
    assert len(preds) == 4
    p = preds[0]
    assert set(p.values.keys()) == TARGET_NAMES
    # Confidence intervals bracket the point estimate.
    for target, (lo, hi) in p.intervals.items():
        assert lo <= p.values[target] + 1e-5
        assert hi >= p.values[target] - 1e-5
    # Bounded targets stay within [0, 1].
    for bounded in ("engagement", "retention", "virality"):
        assert 0.0 <= p.values[bounded] <= 1.0


def test_model_overfits_tiny_batch() -> None:
    """Sanity check the whole model can learn: loss must drop on a fixed batch."""
    model = SuccessPredictor(SMALL_MODEL)
    loss_fn = MaskedMultitaskLoss(heteroscedastic=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    brain, pad_mask, metadata = _batch(b=8, t=24)
    target = torch.randn(8, NUM_TARGETS)
    mask = torch.ones(8, NUM_TARGETS)

    model.train()
    first = last = None
    for step in range(60):
        opt.zero_grad()
        mean, log_var = model(brain, metadata, pad_mask=pad_mask)
        loss, _ = loss_fn(mean, log_var, target, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first * 0.5, f"loss did not drop enough: {first:.3f} -> {last:.3f}"
