"""Video repository."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.db.models.enums import VideoStatus
from app.db.models.video import Video
from app.db.repositories.base import BaseRepository


class VideoRepository(BaseRepository[Video]):
    model = Video

    def get_by_youtube_id(self, youtube_id: str) -> Video | None:
        return self.session.scalar(select(Video).where(Video.youtube_id == youtube_id))

    def get_with_relations(self, video_id: int) -> Video | None:
        stmt = (
            select(Video)
            .where(Video.id == video_id)
            .options(selectinload(Video.embeddings), selectinload(Video.predictions))
        )
        return self.session.scalar(stmt)

    def get_or_create_youtube(self, youtube_id: str, url: str) -> tuple[Video, bool]:
        """Return (video, created). Idempotent ingestion entry point."""
        existing = self.get_by_youtube_id(youtube_id)
        if existing is not None:
            return existing, False
        video = Video(youtube_id=youtube_id, url=url)
        return self.add(video), True

    def list_ready_for_training(self) -> list[Video]:
        """Videos with media + observed labels sufficient to build a sample."""
        stmt = select(Video).where(
            Video.status == VideoStatus.ready,
            Video.view_count.is_not(None),
        )
        return list(self.session.scalars(stmt))

    def set_status(self, video: Video, status: VideoStatus, error: str | None = None) -> Video:
        video.status = status
        if error is not None:
            video.error = error
        self.session.flush()
        return video
