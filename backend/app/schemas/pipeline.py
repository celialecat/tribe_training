"""Pipeline API schemas."""

from __future__ import annotations

from pydantic import BaseModel


class PipelineRunRequest(BaseModel):
    mode: str
    n_videos: int
    horizon_days: int | None = None
