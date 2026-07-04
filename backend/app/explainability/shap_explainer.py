"""SHAP attributions for metadata features (optional dependency).

SHAP values fairly distribute a prediction among features using game theory.
The ``shap`` package is an optional extra (``pip install '.[explain]'``); when it
is absent, callers should fall back to
:func:`app.explainability.feature_importance.permutation_importance`, which is
always available. This module keeps the SHAP path isolated behind a guard so the
rest of the platform never hard-depends on it.
"""

from __future__ import annotations

import numpy as np
import torch

from app.dataset.features import FEATURE_NAMES
from app.models.model import SuccessPredictor


def shap_available() -> bool:
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False


def explain_metadata_shap(
    model: SuccessPredictor,
    brain: torch.Tensor,          # (1, T, V) — fixed context for the sample
    metadata_background: np.ndarray,  # (K, F) representative feature rows
    metadata_sample: np.ndarray,      # (F,) the row to explain
    *,
    target_index: int = 0,
    pad_mask: torch.Tensor | None = None,
    nsamples: int = 100,
) -> dict[str, float]:
    """Return ``{feature_name: shap_value}`` for one sample and target.

    Raises ``RuntimeError`` if ``shap`` is not installed — call
    :func:`shap_available` first or use the permutation fallback.
    """
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - exercised only without shap
        raise RuntimeError(
            "shap is not installed. Install with `pip install '.[explain]'` or use "
            "permutation_importance()."
        ) from exc

    model.eval()
    device = brain.device

    def predict(rows: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            meta = torch.as_tensor(rows, dtype=torch.float32, device=device)
            brain_rep = brain.expand(meta.size(0), -1, -1)
            mask_rep = pad_mask.expand(meta.size(0), -1) if pad_mask is not None else None
            mean, _ = model(brain_rep, meta, mask_rep)
            return mean[:, target_index].cpu().numpy()

    explainer = shap.KernelExplainer(predict, metadata_background)
    values = explainer.shap_values(metadata_sample[None, :], nsamples=nsamples)
    values = np.asarray(values).reshape(-1)
    names = FEATURE_NAMES if values.size == len(FEATURE_NAMES) else [
        f"feat_{i}" for i in range(values.size)
    ]
    return {name: float(values[i]) for i, name in enumerate(names)}
