"""Brain-activity service: compute-or-load orchestration.

Bridges :class:`TribeExtractor` (expensive inference) and :class:`BrainCache`
(persistence). Callers ask for a video's brain activity and get it from cache
when possible, otherwise TRIBE is run once and the result is cached. This is the
single entry point used by dataset preparation and by online inference.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.models.enums import BrainStatus
from app.db.models.video import Video
from app.tribe.cache import BrainCache
from app.tribe.extractor import TribeExtractor
from app.tribe.types import BrainActivity

logger = get_logger(__name__)


class BrainActivityService:
    """Produce (and cache) TRIBE brain activity for videos."""

    def __init__(
        self,
        session: Session,
        *,
        extractor: TribeExtractor | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.extractor = extractor or TribeExtractor(settings=self.settings)
        self.cache = BrainCache(session, model_id=self.extractor.model_id, settings=self.settings)

    def get_or_compute(self, video: Video, *, force: bool = False) -> BrainActivity:
        """Return brain activity for a ready video, computing it if needed."""
        if not force:
            cached = self.cache.get(video)
            if cached is not None:
                logger.debug("Brain activity cache hit for video %s", video.id)
                return cached

        source = self._resolve_media_path(video)
        video.brain_status = BrainStatus.computing
        self.session.flush()
        try:
            activity = self.extractor.extract(
                source,
                audio_path=self._resolve_optional(video.audio_path),
                transcript=video.transcript,
            )
            self.cache.put(video, activity, source_path=source)
        except Exception as exc:
            video.brain_status = BrainStatus.failed
            video.error = f"TRIBE inference failed: {exc}"
            self.session.flush()
            raise
        return activity

    # ------------------------------------------------------------ helpers --
    def _resolve_media_path(self, video: Video) -> Path:
        if not video.video_path:
            raise ValueError(f"Video {video.id} has no downloaded media to run TRIBE on.")
        path = Path(video.video_path)
        return path if path.is_absolute() else self.settings.processed_dir / path

    def _resolve_optional(self, rel: str | None) -> Path | None:
        if not rel:
            return None
        path = Path(rel)
        return path if path.is_absolute() else self.settings.processed_dir / path
