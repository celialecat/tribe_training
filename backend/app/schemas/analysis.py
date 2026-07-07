"""Analysis API schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TimeWindowPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: Literal["mean", "index", "window"] = "mean"
    index: int | None = None
    start: int | None = None
    end: int | None = None


class AnalysisRunRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    method: str
    n_components: int = Field(default=2, ge=1)
    target: str = "log_likes"
    metrics: list[str] | None = None
    region_names: list[str] | None = None
    time_window: TimeWindowPayload | None = None
    normalization: Literal["none", "zscore"] = "none"
    screening_threshold: float = 0.2
    top_k: int | None = None
    n_slices: int = 10


class AnalysisExportRequest(AnalysisRunRequest):
    fmt: Literal["csv", "npz", "pt"] = "csv"


class AnalysisMethodResponse(BaseModel):
    name: str
    description: str
