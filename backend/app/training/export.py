"""Export a trained model to TorchScript and ONNX for portable inference.

A thin :class:`InferenceWrapper` exposes a fixed ``(brain, metadata, pad_mask)
-> (mean, log_var)`` signature with no Python-side branching, which is what both
``torch.jit.trace`` and ``torch.onnx.export`` need. Exports are best-effort:
failures are logged and returned as ``None`` rather than aborting training.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from app.core.logging import get_logger
from app.models.model import SuccessPredictor

logger = get_logger(__name__)


class InferenceWrapper(nn.Module):
    """Deterministic forward for tracing/export: returns (mean, log_var)."""

    def __init__(self, model: SuccessPredictor) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, brain: torch.Tensor, metadata: torch.Tensor, pad_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.model.encoder(brain, pad_mask=pad_mask)
        return self.model.predictor(latent, metadata)


def _example_inputs(
    model: SuccessPredictor, *, batch: int = 1, time_steps: int = 32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    v = model.config.brain_encoder.input_vertices
    f = model.config.predictor.metadata_dim
    brain = torch.randn(batch, time_steps, v)
    metadata = torch.randn(batch, f)
    pad_mask = torch.zeros(batch, time_steps, dtype=torch.bool)
    return brain, metadata, pad_mask


def export_torchscript(model: SuccessPredictor, path: str | Path) -> Path | None:
    """Trace the model to TorchScript. Returns the path or None on failure."""
    path = Path(path)
    wrapper = InferenceWrapper(model).eval()
    inputs = _example_inputs(model)
    try:
        with torch.no_grad():
            scripted = torch.jit.trace(wrapper, inputs, check_trace=False)
        scripted.save(str(path))
        logger.info("Exported TorchScript -> %s", path)
        return path
    except Exception as exc:
        logger.warning("TorchScript export failed: %s", exc)
        return None


def export_onnx(model: SuccessPredictor, path: str | Path, *, opset: int = 17) -> Path | None:
    """Export the model to ONNX with dynamic batch + time axes."""
    path = Path(path)
    wrapper = InferenceWrapper(model).eval()
    brain, metadata, pad_mask = _example_inputs(model)
    try:
        torch.onnx.export(
            wrapper,
            (brain, metadata, pad_mask),
            str(path),
            input_names=["brain", "metadata", "pad_mask"],
            output_names=["mean", "log_var"],
            opset_version=opset,
            dynamic_axes={
                "brain": {0: "batch", 1: "time"},
                "metadata": {0: "batch"},
                "pad_mask": {0: "batch", 1: "time"},
                "mean": {0: "batch"},
                "log_var": {0: "batch"},
            },
        )
        logger.info("Exported ONNX -> %s", path)
        return path
    except Exception as exc:
        logger.warning("ONNX export failed: %s", exc)
        return None
