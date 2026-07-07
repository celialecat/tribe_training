"""Dataset API schemas."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict


class DatasetFilterPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    min_duration_seconds: float | None = None
    max_duration_seconds: float | None = None
    language: str | None = None
    categories: list[str] | None = None
    min_likes: int | None = None
    max_likes: int | None = None
    min_views: int | None = None
    max_views: int | None = None
    date_from: datetime | date | None = None
    date_to: datetime | date | None = None


class ModeChannelRequest(BaseModel):
    url: str


class DatasetPreviewRequest(BaseModel):
    mode: str
    filter: DatasetFilterPayload | None = None
    count: int | None = None


class DatasetBuildRequest(DatasetPreviewRequest):
    force: bool = False


class DatasetValidateRequest(BaseModel):
    filter: DatasetFilterPayload | None = None
    count: int | None = None


class VideoListQuery(BaseModel):
    limit: int = 50
    offset: int = 0
