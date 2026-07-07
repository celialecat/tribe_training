"""Tensor inspection API schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TensorInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    video_id: int
    shape: list[int]
    dtype: str
    num_timesteps: int | None = None
    num_features: int | None = None
    file_size_bytes: int
    path: str
