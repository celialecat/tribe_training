"""Plain data structures for the dataset pipeline.

These decouple the download/extraction stages from the ORM: extractors return
these dataclasses, and the builder maps them onto :class:`Video` rows. This
keeps the media code free of any database dependency and easy to unit-test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(slots=True)
class VideoMetadata:
    """Metadata harvested from a source, before persistence."""

    youtube_id: str | None = None
    url: str | None = None
    language: str | None = None
    categories: list[str] | None = None
    title: str | None = None
    description: str | None = None
    channel: str | None = None
    channel_id: str | None = None
    subscriber_count: int | None = None
    view_count: int | None = None
    like_count: int | None = None
    comment_count: int | None = None
    duration_seconds: float | None = None
    upload_date: datetime | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None


@dataclass(slots=True)
class MediaArtifacts:
    """Filesystem locations of extracted media for one video."""

    root: Path
    video_path: Path | None = None
    audio_path: Path | None = None
    thumbnail_path: Path | None = None
    frames_dir: Path | None = None
    frame_paths: list[Path] = field(default_factory=list)
    transcript: str | None = None
