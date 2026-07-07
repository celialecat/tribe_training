"""Monitoring API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class SystemStatus(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cpu_percent: float
    memory_used: int
    memory_total: int
    disk_used: int
    disk_total: int
    gpu: dict[str, Any] | None = None
