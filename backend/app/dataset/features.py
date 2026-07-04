"""Metadata feature engineering.

Produces the fixed-width metadata vector concatenated with the brain latent
before the prediction head. Features are deliberately *causal* — computable at
publish time — so the model never peeks at post-publication signals (views,
likes) when predicting future success. Popularity signals are reserved for the
targets, not the features.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from app.db.models.video import Video

# Must equal configs/model/default.yaml -> predictor.metadata_dim.
FEATURE_DIM = 12

FEATURE_NAMES: tuple[str, ...] = (
    "log_subscribers",
    "log_duration",
    "upload_hour_sin",
    "upload_hour_cos",
    "upload_weekday_sin",
    "upload_weekday_cos",
    "title_len_norm",
    "title_word_count_norm",
    "description_len_norm",
    "has_transcript",
    "aspect_ratio",
    "fps_norm",
)


def _log1p10(value: float | int | None) -> float:
    return math.log10(1.0 + float(value)) if value else 0.0


def _cyclical(value: float, period: float) -> tuple[float, float]:
    angle = 2.0 * math.pi * (value % period) / period
    return math.sin(angle), math.cos(angle)


def build_feature_vector(video: Video) -> np.ndarray:
    """Build the causal metadata feature vector for a video.

    All features are publish-time observable. The output is a float32 vector of
    length :data:`FEATURE_DIM`, ordered as :data:`FEATURE_NAMES`.
    """
    upload: datetime | None = video.upload_date
    hour_sin, hour_cos = _cyclical(upload.hour, 24.0) if upload else (0.0, 0.0)
    wday_sin, wday_cos = _cyclical(upload.weekday(), 7.0) if upload else (0.0, 0.0)

    title = video.title or ""
    aspect = (video.width / video.height) if (video.width and video.height) else 0.0

    features = np.array(
        [
            _log1p10(video.subscriber_count),
            _log1p10(video.duration_seconds),
            hour_sin,
            hour_cos,
            wday_sin,
            wday_cos,
            min(len(title) / 100.0, 2.0),          # normalised title length
            min(len(title.split()) / 20.0, 2.0),   # normalised word count
            min(len(video.description or "") / 1000.0, 2.0),
            1.0 if video.transcript else 0.0,
            aspect,
            (video.fps or 0.0) / 60.0,
        ],
        dtype=np.float32,
    )
    assert features.shape == (FEATURE_DIM,), features.shape
    return features
