"""Training API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TrainingStartRequest(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)


class TrainingControlResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    type: str
    status: str
    progress: dict[str, Any]
    logs: list[str]
    result: dict[str, Any] | None = None
    error: str | None = None
