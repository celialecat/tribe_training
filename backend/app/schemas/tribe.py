"""TRIBE API schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TribeRunRequest(BaseModel):
    force: bool = False
    retry_failed: bool = False
    limit: int | None = None


class TribeStatusResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cached: int
    remaining: int
    failed: int
    processing_video: str | None = None
    eta_seconds: float | None = None
