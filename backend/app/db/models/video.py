"""Video model — the central entity of the dataset.

Stores the YouTube/local source, all extracted metadata (the observed labels we
train against), and filesystem pointers to downloaded media artefacts. Large
media never lives in the database; only paths do.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.db.models.enums import BrainStatus, VideoSource, VideoStatus

if TYPE_CHECKING:
    from app.db.models.embedding import Embedding
    from app.db.models.prediction import Prediction
    from app.db.models.user import User


class Video(Base, TimestampMixin):
    __tablename__ = "videos"
    __table_args__ = (
        Index("ix_videos_channel_id", "channel_id"),
        Index("ix_videos_status", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), default=None)

    # ---- Source ----
    source: Mapped[VideoSource] = mapped_column(String(16), default=VideoSource.youtube)
    youtube_id: Mapped[str | None] = mapped_column(String(32), unique=True, index=True)
    url: Mapped[str | None] = mapped_column(String(512), default=None)

    # ---- Metadata (observed labels + features) ----
    title: Mapped[str | None] = mapped_column(String(1024), default=None)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    channel: Mapped[str | None] = mapped_column(String(256), default=None)
    channel_id: Mapped[str | None] = mapped_column(String(64), default=None)
    subscriber_count: Mapped[int | None] = mapped_column(BigInteger, default=None)
    view_count: Mapped[int | None] = mapped_column(BigInteger, default=None)
    like_count: Mapped[int | None] = mapped_column(BigInteger, default=None)
    comment_count: Mapped[int | None] = mapped_column(BigInteger, default=None)
    duration_seconds: Mapped[float | None] = mapped_column(Float, default=None)
    upload_date: Mapped[datetime | None] = mapped_column(default=None)
    fps: Mapped[float | None] = mapped_column(Float, default=None)
    width: Mapped[int | None] = mapped_column(Integer, default=None)
    height: Mapped[int | None] = mapped_column(Integer, default=None)

    # ---- Media artefact paths (relative to processed_dir) ----
    video_path: Mapped[str | None] = mapped_column(String(512), default=None)
    audio_path: Mapped[str | None] = mapped_column(String(512), default=None)
    thumbnail_path: Mapped[str | None] = mapped_column(String(512), default=None)
    frames_dir: Mapped[str | None] = mapped_column(String(512), default=None)
    transcript: Mapped[str | None] = mapped_column(Text, default=None)

    # ---- Pipeline state ----
    status: Mapped[VideoStatus] = mapped_column(String(16), default=VideoStatus.pending)
    brain_status: Mapped[BrainStatus] = mapped_column(String(16), default=BrainStatus.absent)
    error: Mapped[str | None] = mapped_column(Text, default=None)

    # ---- Relationships ----
    owner: Mapped[User | None] = relationship(back_populates="videos")
    embeddings: Mapped[list[Embedding]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )
    predictions: Mapped[list[Prediction]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        ref = self.youtube_id or self.id
        return f"<Video {ref} status={self.status} title={(self.title or '')[:40]!r}>"
