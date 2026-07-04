"""Attribution: SHAP, Integrated Gradients, attention, regions.

Public surface:
    - :class:`Explainer`                 per-sample + dataset attributions facade
    - :func:`integrated_gradients`       IG over brain + metadata
    - :func:`temporal_attention`         attention timeline aligned to video time
    - :func:`permutation_importance`     model-agnostic metadata importance
    - :func:`aggregate_region_importance` vertex -> region attribution
    - :func:`explain_metadata_shap`      SHAP values (optional dependency)
"""

from app.explainability.attention import AttentionTimeline, temporal_attention
from app.explainability.brain_regions import (
    aggregate_region_importance,
    default_region_labels,
)
from app.explainability.explainer import Explainer, SampleExplanation
from app.explainability.feature_importance import permutation_importance
from app.explainability.integrated_gradients import IGAttribution, integrated_gradients
from app.explainability.shap_explainer import explain_metadata_shap, shap_available

__all__ = [
    "AttentionTimeline",
    "Explainer",
    "IGAttribution",
    "SampleExplanation",
    "aggregate_region_importance",
    "default_region_labels",
    "explain_metadata_shap",
    "integrated_gradients",
    "permutation_importance",
    "shap_available",
    "temporal_attention",
]
