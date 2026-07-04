"""Regression metrics with masking.

All metrics operate on the observed entries only (via the target mask), so a
target that is unsupervised for some videos never contributes spurious error.
Everything is computed in the model's target units (log-space for views), which
is the meaningful space for these predictions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr

from app.core.constants import TARGET_ORDER

# Minimum observed samples before a correlation is meaningful.
_MIN_SAMPLES_FOR_CORR = 3


@dataclass(slots=True)
class TargetMetrics:
    """Metrics for a single target."""

    n: int
    mae: float
    rmse: float
    r2: float
    pearson: float
    spearman: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _safe(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def compute_target_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> TargetMetrics:
    """Compute metrics for one target over its observed entries."""
    n = int(y_true.shape[0])
    if n == 0:
        nan = float("nan")
        return TargetMetrics(0, nan, nan, nan, nan, nan)
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    r2 = _r2(y_true, y_pred)
    if n >= _MIN_SAMPLES_FOR_CORR and np.std(y_true) > 1e-9 and np.std(y_pred) > 1e-9:
        pearson = _safe(pearsonr(y_true, y_pred)[0])
        spearman = _safe(spearmanr(y_true, y_pred)[0])
    else:
        pearson = spearman = float("nan")
    return TargetMetrics(n=n, mae=mae, rmse=rmse, r2=_safe(r2), pearson=pearson, spearman=spearman)


def evaluate_regression(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray
) -> dict[str, dict[str, float]]:
    """Per-target + macro-averaged metrics for a batch of predictions.

    Shapes: ``y_true``, ``y_pred``, ``mask`` are ``(N, NUM_TARGETS)``.
    """
    results: dict[str, dict[str, float]] = {}
    per_target_values: list[TargetMetrics] = []
    for i, target in enumerate(TARGET_ORDER):
        keep = mask[:, i] > 0
        metrics = compute_target_metrics(y_true[keep, i], y_pred[keep, i])
        results[target.value] = metrics.as_dict()
        if metrics.n > 0:
            per_target_values.append(metrics)

    if per_target_values:
        results["macro"] = {
            "mae": float(np.nanmean([m.mae for m in per_target_values])),
            "rmse": float(np.nanmean([m.rmse for m in per_target_values])),
            "r2": float(np.nanmean([m.r2 for m in per_target_values])),
            "pearson": float(np.nanmean([m.pearson for m in per_target_values])),
            "spearman": float(np.nanmean([m.spearman for m in per_target_values])),
        }
    return results


def residual_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Summary statistics of residuals for diagnostic plots."""
    if y_true.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "skew": float("nan")}
    resid = y_pred - y_true
    mean = float(np.mean(resid))
    std = float(np.std(resid))
    skew = float(np.mean(((resid - mean) / std) ** 3)) if std > 1e-9 else 0.0
    return {"mean": mean, "std": std, "skew": skew}
