"""Load brain tensors and aligned targets for latent analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.constants import FSAVERAGE5_VERTICES, FSAVERAGE5_VERTICES_PER_HEMI
from app.core.logging import get_logger
from app.dataset.targets import compute_targets
from app.db.models.embedding import Embedding
from app.db.models.enums import EmbeddingKind, VideoStatus
from app.db.models.video import Video
from app.explainability.brain_regions import default_region_labels

logger = get_logger(__name__)

ANALYSIS_METHOD_NOTE = (
    "Load cached TRIBE brain-activity tensors, reduce them to a per-video "
    "feature vector, and align each sample with popularity targets."
)


@dataclass(slots=True)
class TimeWindowSpec:
    """Temporal reduction rule applied to each cached tensor."""

    mode: str = "mean"
    index: int | None = None
    start: int | None = None
    end: int | None = None


@dataclass(slots=True)
class AnalysisData:
    """Feature matrix and aligned target arrays for latent analysis."""

    X: np.ndarray
    y: dict[str, np.ndarray]
    video_ids: list[int]
    feature_names: list[str]
    vertex_indices: np.ndarray
    vertex_labels: np.ndarray
    region_names: list[str]
    full_vertex_count: int
    time_window: dict[str, Any]
    normalization: str
    error: dict[str, Any] | None = None
    selected_regions: list[str] | None = None

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1]) if self.X.ndim == 2 else 0


def load_analysis_data(
    session: Session,
    *,
    settings: Settings | None = None,
    time_window: TimeWindowSpec | None = None,
    region_names: list[str] | None = None,
    normalization: str = "none",
) -> AnalysisData:
    """Build the analysis matrix from cached brain tensors and video metadata."""
    settings = settings or get_settings()
    time_window = time_window or TimeWindowSpec()
    producer = f"tribev2:{settings.tribe_model_id}"

    embeddings = list(
        session.scalars(
            select(Embedding).where(
                Embedding.kind == EmbeddingKind.brain_activity,
                Embedding.producer == producer,
            )
        )
    )

    rows: list[np.ndarray] = []
    video_ids: list[int] = []
    likes: list[float] = []
    views: list[float] = []
    comments: list[float] = []
    engagement: list[float] = []
    virality: list[float] = []
    log_likes: list[float] = []

    vertex_labels: np.ndarray | None = None
    region_names_all: list[str] | None = None
    selected_indices: np.ndarray | None = None
    target_regions = {name.lower() for name in region_names or []}

    for embedding in embeddings:
        video = session.get(Video, embedding.video_id)
        if video is None or video.status != VideoStatus.ready:
            continue
        if video.like_count is None or video.view_count is None:
            continue
        tensor = _load_tensor(settings, embedding.path)
        if tensor is None or tensor.ndim != 2:
            continue
        if tensor.shape[1] != FSAVERAGE5_VERTICES:
            logger.info(
                "Skipping video %s because tensor has %s vertices",
                video.id,
                tensor.shape[1],
            )
            continue
        if vertex_labels is None:
            vertex_labels, region_names_all = default_region_labels(tensor.shape[1])
            selected_indices = _selected_vertices(
                vertex_labels,
                region_names_all,
                target_regions,
            )
            if selected_indices.size == 0:
                return AnalysisData(
                    X=np.zeros((0, 0), dtype=np.float32),
                    y=_empty_targets(),
                    video_ids=[],
                    feature_names=[],
                    vertex_indices=np.zeros(0, dtype=np.int64),
                    vertex_labels=vertex_labels,
                    region_names=region_names_all,
                    full_vertex_count=FSAVERAGE5_VERTICES,
                    time_window=_time_window_dict(time_window),
                    normalization=normalization,
                    error={
                        "code": "empty_region_selection",
                        "message": "The requested region filter did not match any vertices.",
                        "regions": region_names,
                    },
                    selected_regions=[],
                )
        assert vertex_labels is not None
        assert region_names_all is not None
        assert selected_indices is not None
        reduced = _reduce_time(tensor, time_window)
        reduced = reduced[selected_indices]
        rows.append(reduced.astype(np.float32, copy=False))
        video_ids.append(video.id)
        sample_targets = compute_targets(video, approximate_view_curve=True)
        log_likes.append(float(sample_targets.values[-1]))
        likes.append(float(video.like_count))
        views.append(float(video.view_count))
        comments.append(float(video.comment_count or 0))
        engagement.append(float(sample_targets.values[2]))
        virality.append(float(sample_targets.values[4]))

    if vertex_labels is None or region_names_all is None or selected_indices is None:
        empty = np.zeros((0, 0), dtype=np.float32)
        return AnalysisData(
            X=empty,
            y=_empty_targets(),
            video_ids=[],
            feature_names=[],
            vertex_indices=np.zeros(0, dtype=np.int64),
            vertex_labels=np.zeros(0, dtype=np.int64),
            region_names=[],
            full_vertex_count=FSAVERAGE5_VERTICES,
            time_window=_time_window_dict(time_window),
            normalization=normalization,
            error={
                "code": "no_cached_data",
                "message": "No cached brain tensors matched the analysis filters.",
            },
        )

    if len(rows) < 3:
        return AnalysisData(
            X=np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 0), dtype=np.float32),
            y={
                "log_likes": np.asarray(log_likes, dtype=np.float32),
                "likes": np.asarray(likes, dtype=np.float32),
                "views": np.asarray(views, dtype=np.float32),
                "comments": np.asarray(comments, dtype=np.float32),
                "engagement": np.asarray(engagement, dtype=np.float32),
                "virality": np.asarray(virality, dtype=np.float32),
            },
            video_ids=video_ids,
            feature_names=[f"v{idx}" for idx in selected_indices.tolist()],
            vertex_indices=selected_indices,
            vertex_labels=vertex_labels,
            region_names=region_names_all,
            full_vertex_count=FSAVERAGE5_VERTICES,
            time_window=_time_window_dict(time_window),
            normalization=normalization,
            error={
                "code": "insufficient_samples",
                "message": "At least 3 cached videos are required for latent analysis.",
                "n_samples": len(rows),
                "min_required": 3,
            },
            selected_regions=_selected_region_names(
                region_names_all,
                vertex_labels,
                selected_indices,
            ),
        )

    x = np.stack(rows, axis=0)
    if normalization == "zscore":
        x = _zscore(x)

    feature_names = [f"v{idx}" for idx in selected_indices.tolist()]
    return AnalysisData(
        X=x,
        y={
            "log_likes": np.asarray(log_likes, dtype=np.float32),
            "likes": np.asarray(likes, dtype=np.float32),
            "views": np.asarray(views, dtype=np.float32),
            "comments": np.asarray(comments, dtype=np.float32),
            "engagement": np.asarray(engagement, dtype=np.float32),
            "virality": np.asarray(virality, dtype=np.float32),
        },
        video_ids=video_ids,
        feature_names=feature_names,
        vertex_indices=selected_indices,
        vertex_labels=vertex_labels,
        region_names=region_names_all,
        full_vertex_count=FSAVERAGE5_VERTICES,
        time_window=_time_window_dict(time_window),
        normalization=normalization,
        selected_regions=_selected_region_names(region_names_all, vertex_labels, selected_indices),
    )


def _empty_targets() -> dict[str, np.ndarray]:
    return {
        "log_likes": np.zeros(0, dtype=np.float32),
        "likes": np.zeros(0, dtype=np.float32),
        "views": np.zeros(0, dtype=np.float32),
        "comments": np.zeros(0, dtype=np.float32),
        "engagement": np.zeros(0, dtype=np.float32),
        "virality": np.zeros(0, dtype=np.float32),
    }


def _load_tensor(settings: Settings, stored_path: str) -> np.ndarray | None:
    path = Path(stored_path)
    if not path.is_absolute():
        path = settings.data_dir / path
    if not path.exists():
        return None
    with np.load(path) as data:
        return np.asarray(data["activity"], dtype=np.float32)


def _reduce_time(tensor: np.ndarray, window: TimeWindowSpec) -> np.ndarray:
    if tensor.ndim != 2:
        raise ValueError("cached tensor must have shape (T, V)")
    mode = window.mode.lower()
    if mode in {"mean", "average", "avg"}:
        return tensor.mean(axis=0)
    if mode in {"index", "timestep"}:
        index = 0 if window.index is None else int(window.index)
        index = max(min(index, tensor.shape[0] - 1), 0)
        return tensor[index]
    if mode == "window":
        start = 0 if window.start is None else max(int(window.start), 0)
        end = tensor.shape[0] if window.end is None else min(int(window.end), tensor.shape[0])
        if end <= start:
            end = min(start + 1, tensor.shape[0])
        return tensor[start:end].mean(axis=0)
    raise ValueError(f"unsupported time window mode: {window.mode}")


def _selected_vertices(
    labels: np.ndarray,
    region_names: list[str],
    selected_regions: set[str],
) -> np.ndarray:
    if not selected_regions:
        return np.arange(labels.shape[0], dtype=np.int64)
    indices: list[int] = []
    for region_id, name in enumerate(region_names):
        if name.lower() in selected_regions:
            indices.extend(np.flatnonzero(labels == region_id).tolist())
    return np.asarray(sorted(set(indices)), dtype=np.int64)


def _selected_region_names(
    all_region_names: list[str],
    labels: np.ndarray,
    selected_indices: np.ndarray,
) -> list[str]:
    selected = sorted({int(labels[index]) for index in selected_indices})
    return [all_region_names[index] for index in selected]


def _zscore(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (x - mean) / std


def _time_window_dict(window: TimeWindowSpec) -> dict[str, Any]:
    return {
        "mode": window.mode,
        "index": window.index,
        "start": window.start,
        "end": window.end,
    }


def full_vertex_count() -> int:
    return FSAVERAGE5_VERTICES


def split_hemispheres(vertex_vector: np.ndarray) -> dict[str, list[float]]:
    """Split a full cortical vector into left/right hemisphere lists."""
    vector = np.asarray(vertex_vector, dtype=np.float32)
    left = vector[:FSAVERAGE5_VERTICES_PER_HEMI]
    right = vector[FSAVERAGE5_VERTICES_PER_HEMI : 2 * FSAVERAGE5_VERTICES_PER_HEMI]
    return {"left": left.tolist(), "right": right.tolist()}
