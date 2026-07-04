"""Uncertainty calibration for the heteroscedastic predictions.

A model that emits confidence intervals is only trustworthy if those intervals
are *calibrated*: a nominal 90% interval should contain the truth ~90% of the
time. We measure empirical coverage across nominal levels (a reliability curve)
and summarise miscalibration with a single expected-calibration-error-style area.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

DEFAULT_LEVELS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


@dataclass(slots=True)
class CalibrationCurve:
    """Nominal vs empirical coverage for a Gaussian predictive distribution."""

    levels: list[float]
    empirical: list[float]
    miscalibration_area: float

    def as_dict(self) -> dict[str, object]:
        return {
            "levels": self.levels,
            "empirical": self.empirical,
            "miscalibration_area": self.miscalibration_area,
        }


def coverage_at_level(
    y_true: np.ndarray, mean: np.ndarray, sigma: np.ndarray, level: float
) -> float:
    """Empirical fraction of truths inside the central ``level`` interval."""
    if y_true.size == 0:
        return float("nan")
    z = float(norm.ppf(0.5 + level / 2.0))
    sigma = np.maximum(sigma, 1e-9)
    inside = np.abs(y_true - mean) <= z * sigma
    return float(np.mean(inside))


def calibration_curve(
    y_true: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    levels: tuple[float, ...] = DEFAULT_LEVELS,
) -> CalibrationCurve:
    """Reliability curve + miscalibration area (∫|empirical - nominal|)."""
    empirical = [coverage_at_level(y_true, mean, sigma, lvl) for lvl in levels]
    valid = [(lvl, emp) for lvl, emp in zip(levels, empirical, strict=True) if np.isfinite(emp)]
    if len(valid) >= 2:
        xs = np.array([lvl for lvl, _ in valid])
        ys = np.array([abs(emp - lvl) for lvl, emp in valid])
        area = float(np.trapz(ys, xs))
    else:
        area = float("nan")
    return CalibrationCurve(
        levels=list(levels), empirical=empirical, miscalibration_area=area
    )
