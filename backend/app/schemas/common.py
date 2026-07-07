"""Shared API response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JobSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    type: str
    status: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: dict[str, Any] = Field(default_factory=dict)
    logs: list[str] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
