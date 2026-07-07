"""Latent brain analysis package."""

from __future__ import annotations

from app.analysis.data import AnalysisData, TimeWindowSpec, load_analysis_data
from app.analysis.latent import CONDITIONAL_VARIANCE_JUSTIFICATION, list_methods, run_analysis

__all__ = [
    "CONDITIONAL_VARIANCE_JUSTIFICATION",
    "AnalysisData",
    "TimeWindowSpec",
    "list_methods",
    "load_analysis_data",
    "run_analysis",
]
