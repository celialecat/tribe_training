"""Metrics, calibration and residual analysis.

Public surface:
    - :func:`evaluate_regression`   per-target MAE/RMSE/R²/Pearson/Spearman
    - :func:`residual_summary`      residual diagnostics
    - :func:`calibration_curve`     uncertainty reliability curve
    - :class:`Evaluator`            end-to-end report over a dataloader
"""

from app.evaluation.calibration import CalibrationCurve, calibration_curve, coverage_at_level
from app.evaluation.evaluator import Evaluator
from app.evaluation.metrics import (
    TargetMetrics,
    compute_target_metrics,
    evaluate_regression,
    residual_summary,
)

__all__ = [
    "CalibrationCurve",
    "Evaluator",
    "TargetMetrics",
    "calibration_curve",
    "compute_target_metrics",
    "coverage_at_level",
    "evaluate_regression",
    "residual_summary",
]
