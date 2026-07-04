"""Evaluator — run a trained model over a dataloader and build a report.

Collects de-normalised predictions, ground truth, masks and predicted sigmas,
then assembles regression metrics, residual diagnostics and calibration curves
per target. The report is JSON-serialisable for storage and the dashboard.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.core.constants import TARGET_ORDER
from app.core.logging import get_logger
from app.evaluation.calibration import calibration_curve
from app.evaluation.metrics import evaluate_regression, residual_summary
from app.models.model import SuccessPredictor
from app.models.normalizer import TargetNormalizer

logger = get_logger(__name__)


class Evaluator:
    """Evaluate a :class:`SuccessPredictor` on a dataset split."""

    def __init__(
        self, model: SuccessPredictor, normalizer: TargetNormalizer, *, device: str = "cpu"
    ):
        self.model = model.to(device).eval()
        self.normalizer = normalizer
        self.device = torch.device(device)

    @torch.no_grad()
    def collect(self, loader: DataLoader) -> dict[str, np.ndarray]:
        """Run inference, returning stacked ``y_true``, ``y_pred``, ``sigma``, ``mask``."""
        y_true, y_pred, sigma, mask = [], [], [], []
        for batch in loader:
            brain = batch["brain"].to(self.device)
            metadata = batch["features"].to(self.device)
            pad_mask = batch["pad_mask"].to(self.device)
            mean_z, log_var_z = self.model(brain, metadata, pad_mask)
            mean = self.normalizer.denormalize(mean_z)
            sig = self.normalizer.denormalize_std(torch.exp(0.5 * log_var_z))
            y_pred.append(mean.cpu().numpy())
            sigma.append(sig.cpu().numpy())
            y_true.append(batch["targets"].numpy())
            mask.append(batch["target_mask"].numpy())
        return {
            "y_true": np.concatenate(y_true, axis=0),
            "y_pred": np.concatenate(y_pred, axis=0),
            "sigma": np.concatenate(sigma, axis=0),
            "mask": np.concatenate(mask, axis=0),
        }

    def evaluate(self, loader: DataLoader) -> dict[str, Any]:
        """Return a full, JSON-serialisable evaluation report."""
        data = self.collect(loader)
        y_true, y_pred, sigma, mask = (
            data["y_true"], data["y_pred"], data["sigma"], data["mask"]
        )

        report: dict[str, Any] = {
            "n_samples": int(y_true.shape[0]),
            "metrics": evaluate_regression(y_true, y_pred, mask),
            "residuals": {},
            "calibration": {},
        }
        for i, target in enumerate(TARGET_ORDER):
            keep = mask[:, i] > 0
            yt, yp, sg = y_true[keep, i], y_pred[keep, i], sigma[keep, i]
            report["residuals"][target.value] = residual_summary(yt, yp)
            report["calibration"][target.value] = calibration_curve(yt, yp, sg).as_dict()

        logger.info(
            "Evaluated %d samples | macro R2=%.3f",
            report["n_samples"], report["metrics"].get("macro", {}).get("r2", float("nan")),
        )
        return report
