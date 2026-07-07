"""Prediction API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class PredictionUrlRequest(BaseModel):
    url: str
    confidence_level: float = 0.95
    horizon_days: int | None = None


class PredictionHorizonResponse(BaseModel):
    horizon_days: int


class PredictionResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    video_id: int
    youtube_id: str | None = None
    horizon_days: int
    horizon_label: str
    prediction: dict[str, Any]
    explanation: dict[str, Any]
