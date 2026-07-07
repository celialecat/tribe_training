"""Latent brain analysis methods built on cached TRIBE tensors."""

from __future__ import annotations

import csv
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

from app.analysis.data import AnalysisData
from app.core.constants import FSAVERAGE5_VERTICES_PER_HEMI
from app.core.logging import get_logger
from app.explainability.brain_regions import aggregate_region_importance

logger = get_logger(__name__)


@dataclass(slots=True)
class AnalysisMethodSpec:
    name: str
    description: str


METHODS: list[AnalysisMethodSpec] = [
    AnalysisMethodSpec(
        name="pca",
        description="Unsupervised principal component analysis of brain features.",
    ),
    AnalysisMethodSpec(
        name="supervised_pca",
        description="Screen features by association with log-likes, then run PCA.",
    ),
    AnalysisMethodSpec(
        name="pls",
        description="Supervised latent components maximizing covariance with target(s).",
    ),
    AnalysisMethodSpec(
        name="cca",
        description="Canonical correlation between brain features and popularity metrics.",
    ),
    AnalysisMethodSpec(
        name="conditional_variance",
        description=(
            "Sliced inverse regression estimate of the sufficient dimension reduction subspace."
        ),
    ),
]

CONDITIONAL_VARIANCE_JUSTIFICATION = (
    "Conditional variance analysis is framed as sufficient dimension reduction: "
    "we seek a central subspace S such that Y ⟂ X | P_S X. PCA is unsupervised "
    "and therefore mathematically unsuitable because it ignores Y entirely. "
    "SIR is a rigorous SDR estimator that uses inverse regression slices to "
    "recover the central subspace. PLS is a supervised covariance-maximizing "
    "alternative, while SAVE captures second-moment / conditional-variance "
    "structure and mutual-information projections are another option; SIR is "
    "the primary implementation here because it is stable, lightweight, and "
    "well aligned with the sufficient-dimension-reduction formulation."
)


def list_methods() -> list[dict[str, str]]:
    return [{"name": spec.name, "description": spec.description} for spec in METHODS] + [
        {
            "name": "conditional_variance_justification",
            "description": CONDITIONAL_VARIANCE_JUSTIFICATION,
        }
    ]


def run_analysis(
    data: AnalysisData,
    *,
    method: str,
    n_components: int = 2,
    target: str = "log_likes",
    metrics: list[str] | None = None,
    screening_threshold: float = 0.2,
    top_k: int | None = None,
    n_slices: int = 10,
) -> dict[str, Any]:
    if data.error is not None:
        return {"method": method, "error": data.error}
    if data.n_samples < 3:
        return {
            "method": method,
            "error": {
                "code": "insufficient_samples",
                "message": "At least 3 cached videos are required for latent analysis.",
                "n_samples": data.n_samples,
            },
        }

    method = method.lower()
    target_y = _target_vector(data, target)
    max_components = _max_components(method, data)
    used = min(max(int(n_components), 1), max_components)
    result: dict[str, Any] = {
        "method": method,
        "n_samples": data.n_samples,
        "n_features": data.n_features,
        "n_components_requested": int(n_components),
        "n_components": used,
        "clamped": used != int(n_components),
        "time_window": data.time_window,
        "normalization": data.normalization,
        "video_ids": data.video_ids,
        "selected_regions": data.selected_regions,
    }

    if method == "pca":
        result.update(_run_pca(data, used, target_y))
    elif method == "supervised_pca":
        result.update(
            _run_supervised_pca(
                data,
                used,
                target_y,
                screening_threshold=screening_threshold,
                top_k=top_k,
            )
        )
    elif method == "pls":
        result.update(_run_pls(data, used, target_y))
    elif method == "cca":
        result.update(_run_cca(data, used, metrics=metrics))
    elif method == "conditional_variance":
        result.update(_run_sir(data, used, target_y, n_slices=n_slices))
    else:
        raise ValueError(f"unknown analysis method: {method}")

    return result


def export_analysis(result: dict[str, Any], *, fmt: str) -> tuple[Path, str, str]:
    fmt = fmt.lower()
    stem = f"analysis_{result.get('method', 'result')}"
    if fmt == "csv":
        path = Path(tempfile.mkstemp(prefix=stem, suffix=".csv")[1])
        _write_csv(path, result)
        return path, path.name, "text/csv"
    if fmt == "npz":
        path = Path(tempfile.mkstemp(prefix=stem, suffix=".npz")[1])
        _write_npz(path, result)
        return path, path.name, "application/zip"
    if fmt == "pt":
        path = Path(tempfile.mkstemp(prefix=stem, suffix=".pt")[1])
        torch.save(_to_torch(result), path)
        return path, path.name, "application/octet-stream"
    raise ValueError(f"unsupported export format: {fmt}")


def _run_pca(data: AnalysisData, n_components: int, y: np.ndarray) -> dict[str, Any]:
    model = PCA(n_components=n_components)
    scores = model.fit_transform(data.X)
    loadings = model.components_
    return {
        "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
        "cumulative_explained_variance": np.cumsum(model.explained_variance_ratio_).tolist(),
        "scree": [
            {
                "component": idx + 1,
                "variance": float(var),
                "ratio": float(ratio),
            }
            for idx, (var, ratio) in enumerate(
                zip(model.explained_variance_, model.explained_variance_ratio_, strict=True)
            )
        ],
        "projections": scores.tolist(),
        "loadings": loadings.tolist(),
        "components": _component_payload(loadings, scores, data),
        "variance_explained": float(np.sum(model.explained_variance_ratio_)),
    }


def _run_supervised_pca(
    data: AnalysisData,
    n_components: int,
    y: np.ndarray,
    *,
    screening_threshold: float,
    top_k: int | None,
) -> dict[str, Any]:
    scores = _screen_features(data, y, threshold=screening_threshold, top_k=top_k)
    selected = np.flatnonzero(scores > 0)
    if selected.size == 0:
        selected = np.arange(min(data.n_features, max(1, n_components)))
    x_selected = data.X[:, selected]
    pca = PCA(n_components=min(n_components, x_selected.shape[1]))
    projections = pca.fit_transform(x_selected)
    supervised_importance = scores[selected]
    loadings = pca.components_
    region_importance = aggregate_region_importance(
        _back_project_to_full(loadings[0], data, selected),
        labels=data.vertex_labels,
        names=data.region_names,
    )
    return {
        "screening_scores": scores.tolist(),
        "screened_feature_indices": selected.tolist(),
        "screened_feature_names": [data.feature_names[i] for i in selected.tolist()],
        "supervised_importance": supervised_importance.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "projections": projections.tolist(),
        "loadings": loadings.tolist(),
        "region_importance": region_importance,
        "components": _component_payload(loadings, projections, data, feature_indices=selected),
        "variance_explained_selected_features": float(np.sum(pca.explained_variance_ratio_)),
    }


def _run_pls(data: AnalysisData, n_components: int, y: np.ndarray) -> dict[str, Any]:
    model = PLSRegression(n_components=n_components, scale=False)
    x = data.X
    y_2d = y.reshape(-1, 1)
    model.fit(x, y_2d)
    x_scores = model.x_scores_
    y_scores = model.y_scores_
    covariances = [
        float(abs(np.cov(x_scores[:, idx], y_scores[:, idx], bias=True)[0, 1]))
        for idx in range(x_scores.shape[1])
    ]
    total_cov = sum(covariances) or 1.0
    cv_r2 = _cross_validated_r2(model, x, y_2d)
    vip = _vip(model, x, y_2d)
    return {
        "x_scores": x_scores.tolist(),
        "y_scores": y_scores.tolist(),
        "covariance_explained": [cov / total_cov for cov in covariances],
        "fit_r2": float(model.score(x, y_2d)),
        "cv_r2": cv_r2,
        "x_weights": model.x_weights_.tolist(),
        "y_weights": model.y_weights_.tolist(),
        "vip": vip.tolist(),
        "components": _component_payload(model.x_weights_.T, x_scores, data),
        "region_importance": aggregate_region_importance(
            _back_project_to_full(model.x_weights_[:, 0], data),
            labels=data.vertex_labels,
            names=data.region_names,
        ),
    }


def _run_cca(
    data: AnalysisData,
    n_components: int,
    *,
    metrics: list[str] | None,
) -> dict[str, Any]:
    metric_names = metrics or ["likes", "views", "comments", "engagement"]
    y_metrics = np.column_stack([_target_vector(data, metric) for metric in metric_names])
    model = CCA(n_components=n_components, max_iter=1000)
    x_scores, y_scores = model.fit_transform(data.X, y_metrics)
    canonical = [
        float(abs(np.corrcoef(x_scores[:, idx], y_scores[:, idx])[0, 1]))
        for idx in range(x_scores.shape[1])
    ]
    return {
        "metrics": metric_names,
        "canonical_correlations": canonical,
        "x_scores": x_scores.tolist(),
        "y_scores": y_scores.tolist(),
        "x_loadings": model.x_loadings_.tolist(),
        "y_loadings": model.y_loadings_.tolist(),
        "components": _component_payload(model.x_loadings_.T, x_scores, data),
    }


def _run_sir(
    data: AnalysisData,
    n_components: int,
    y: np.ndarray,
    *,
    n_slices: int,
) -> dict[str, Any]:
    x = data.X
    x_centered = x - x.mean(axis=0, keepdims=True)
    cov = np.cov(x_centered, rowvar=False)
    if np.ndim(cov) == 0:
        return {
            "method_justification": CONDITIONAL_VARIANCE_JUSTIFICATION,
            "error": {
                "code": "degenerate_rank",
                "message": "SIR requires more than one feature dimension.",
            },
        }
    eigvals, eigvecs = np.linalg.eigh(cov + np.eye(cov.shape[0]) * 1e-8)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    keep = eigvals > 1e-10
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]
    if eigvals.size == 0:
        return {
            "method_justification": CONDITIONAL_VARIANCE_JUSTIFICATION,
            "error": {
                "code": "degenerate_rank",
                "message": "SIR could not invert the feature covariance.",
            },
        }
    whitening = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
    z = x_centered @ whitening
    slices = _make_slices(y, n_slices)
    if len(slices) < 2:
        return {
            "method_justification": CONDITIONAL_VARIANCE_JUSTIFICATION,
            "error": {
                "code": "insufficient_slices",
                "message": "SIR needs at least two non-empty response slices.",
            },
        }
    m = np.zeros((z.shape[1], z.shape[1]), dtype=np.float64)
    for slc in slices:
        z_h = z[slc]
        p_h = z_h.shape[0] / z.shape[0]
        mean_h = z_h.mean(axis=0, keepdims=True)
        m += p_h * (mean_h.T @ mean_h)
    sir_vals, sir_vecs = np.linalg.eigh(m)
    order = np.argsort(sir_vals)[::-1]
    sir_vals = sir_vals[order]
    sir_vecs = sir_vecs[:, order]
    used = min(n_components, sir_vecs.shape[1])
    directions = whitening @ sir_vecs[:, :used]
    projections = x_centered @ directions
    fractions = sir_vals[:used] / (sir_vals.sum() or 1.0)
    component_corrs = _projection_correlations(projections, data)
    return {
        "method_justification": CONDITIONAL_VARIANCE_JUSTIFICATION,
        "eigenvalues": sir_vals[:used].tolist(),
        "between_slice_fraction": fractions.tolist(),
        "edr_directions": directions.tolist(),
        "projections": projections.tolist(),
        "components": _sir_components(directions, projections, data),
        "region_importance": [
            aggregate_region_importance(
                _back_project_to_full(directions[:, idx], data),
                labels=data.vertex_labels,
                names=data.region_names,
            )
            for idx in range(used)
        ],
        "component_correlations": component_corrs,
    }


def _component_payload(
    loadings: np.ndarray,
    projections: np.ndarray,
    data: AnalysisData,
    *,
    feature_indices: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for idx in range(loadings.shape[0]):
        vector = loadings[idx]
        if feature_indices is not None:
            full = _back_project_to_full(vector, data, feature_indices)
        else:
            full = _back_project_to_full(vector, data)
        payload.append(
            {
                "component": idx + 1,
                "correlations": _component_correlations(projections[:, idx], data),
                "brain_map": split_hemi(full),
                "region_importance": aggregate_region_importance(
                    full, labels=data.vertex_labels, names=data.region_names
                ),
            }
        )
    return payload


def _sir_components(
    directions: np.ndarray,
    projections: np.ndarray,
    data: AnalysisData,
) -> list[dict[str, Any]]:
    return [
        {
            "component": idx + 1,
            "correlations": _component_correlations(projections[:, idx], data),
            "brain_map": split_hemi(_back_project_to_full(directions[:, idx], data)),
        }
        for idx in range(directions.shape[1])
    ]


def split_hemi(vertex_vector: np.ndarray) -> dict[str, list[float]]:
    vector = np.asarray(vertex_vector, dtype=np.float32)
    left = vector[:FSAVERAGE5_VERTICES_PER_HEMI]
    right = vector[FSAVERAGE5_VERTICES_PER_HEMI : 2 * FSAVERAGE5_VERTICES_PER_HEMI]
    return {"left": left.tolist(), "right": right.tolist()}


def _back_project_to_full(
    vector: np.ndarray,
    data: AnalysisData,
    feature_indices: np.ndarray | None = None,
) -> np.ndarray:
    full = np.zeros(data.full_vertex_count, dtype=np.float32)
    vector = np.asarray(vector, dtype=np.float32)
    indices = feature_indices if feature_indices is not None else data.vertex_indices
    length = min(indices.shape[0], vector.shape[0])
    full[indices[:length]] = vector[:length]
    return full


def _component_correlations(component: np.ndarray, data: AnalysisData) -> dict[str, float]:
    return {
        "likes": _safe_corr(component, data.y["likes"]),
        "views": _safe_corr(component, data.y["views"]),
        "engagement": _safe_corr(component, data.y["engagement"]),
    }


def _projection_correlations(
    projections: np.ndarray,
    data: AnalysisData,
) -> list[dict[str, float]]:
    return [
        _component_correlations(projections[:, idx], data) for idx in range(projections.shape[1])
    ]


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    value = np.corrcoef(a, b)[0, 1]
    return float(0.0 if np.isnan(value) else value)


def _target_vector(data: AnalysisData, target: str) -> np.ndarray:
    if target not in data.y:
        raise KeyError(f"unknown target: {target}")
    return np.asarray(data.y[target], dtype=np.float32)


def _max_components(method: str, data: AnalysisData) -> int:
    if method == "cca":
        return max(1, min(data.n_samples - 1, data.n_features, 4))
    if method == "conditional_variance":
        return max(1, min(data.n_samples - 1, data.n_features, 10))
    return max(1, min(data.n_samples - 1, data.n_features))


def _screen_features(
    data: AnalysisData,
    y: np.ndarray,
    *,
    threshold: float,
    top_k: int | None,
) -> np.ndarray:
    scores = np.array([abs(_safe_corr(data.X[:, idx], y)) for idx in range(data.n_features)])
    if top_k is not None:
        keep = np.argsort(scores)[::-1][: max(int(top_k), 1)]
        mask = np.zeros_like(scores, dtype=bool)
        mask[keep] = True
        scores = scores * mask
    else:
        scores = np.where(scores >= threshold, scores, 0.0)
    return scores


def _cross_validated_r2(
    model: PLSRegression,
    x: np.ndarray,
    y: np.ndarray,
) -> float | None:
    if x.shape[0] < 6:
        return None
    folds = min(5, x.shape[0])
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    preds = cross_val_predict(
        PLSRegression(n_components=model.n_components, scale=False),
        x,
        y,
        cv=cv,
    )
    return float(r2_score(y, preds))


def _vip(model: PLSRegression, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p = x.shape[1]
    s = np.array(
        [np.sum((t[:, i] ** 2) * (q[:, i] ** 2)) for i in range(t.shape[1])],
        dtype=np.float64,
    )
    total = s.sum() or 1.0
    weight = (w**2) @ s
    return np.sqrt(p * weight / total)


def _make_slices(y: np.ndarray, n_slices: int) -> list[np.ndarray]:
    if y.size == 0:
        return []
    n_slices = max(2, min(int(n_slices), y.shape[0]))
    quantiles = np.quantile(y, np.linspace(0, 1, n_slices + 1))
    edges = np.unique(quantiles)
    if edges.size < 2:
        return [np.arange(y.shape[0])]
    slices: list[np.ndarray] = []
    for idx in range(edges.size - 1):
        left = edges[idx]
        right = edges[idx + 1]
        mask = (y >= left) & (y <= right) if right == edges[-1] else (y >= left) & (y < right)
        indices = np.flatnonzero(mask)
        if indices.size:
            slices.append(indices)
    return slices


def _write_csv(path: Path, result: dict[str, Any]) -> None:
    components = result.get("components") or []
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "component",
            "correlation_likes",
            "correlation_views",
            "correlation_engagement",
            "brain_map_left",
            "brain_map_right",
            "region_importance",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for component in components:
            correlations = component.get("correlations", {})
            brain_map = component.get("brain_map", {})
            writer.writerow(
                {
                    "component": component.get("component"),
                    "correlation_likes": correlations.get("likes"),
                    "correlation_views": correlations.get("views"),
                    "correlation_engagement": correlations.get("engagement"),
                    "brain_map_left": json.dumps(brain_map.get("left", [])),
                    "brain_map_right": json.dumps(brain_map.get("right", [])),
                    "region_importance": json.dumps(component.get("region_importance", {})),
                }
            )


def _write_npz(path: Path, result: dict[str, Any]) -> None:
    flattened: dict[str, Any] = {}
    _flatten(flattened, "result", result)
    np.savez_compressed(path, **flattened)


def _flatten(out: dict[str, Any], prefix: str, value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(out, f"{prefix}.{key}", item)
        return
    if isinstance(value, list):
        try:
            out[prefix] = np.asarray(value)
        except Exception:
            out[prefix] = np.asarray([json.dumps(value)])
        return
    if isinstance(value, (str, bytes)):
        out[prefix] = np.asarray([value])
        return
    if value is None:
        out[prefix] = np.asarray([None], dtype=object)
        return
    if isinstance(value, (int, float, bool, np.number)):
        out[prefix] = np.asarray([value])
        return
    out[prefix] = np.asarray([json.dumps(value, default=_json_default)])


def _to_torch(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_torch(item) for key, item in value.items()}
    if isinstance(value, list):
        try:
            return torch.as_tensor(value)
        except Exception:
            return [_to_torch(item) for item in value]
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value)
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"unsupported type: {type(value)!r}")
