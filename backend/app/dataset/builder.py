"""Dataset builder — orchestrates ingestion of a video into DB + filesystem.

Given a YouTube URL or a local file, it: downloads (or copies) the media,
probes/extracts audio + frames + transcript, writes artefacts under
``processed_dir/<key>/``, and upserts a :class:`Video` row with all metadata and
paths. Ingestion is idempotent and resumable — an already-``ready`` video is
returned untouched unless ``force`` is set.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.dataset.download import YouTubeDownloader, extract_youtube_id
from app.dataset.media import MediaExtractor, probe_video
from app.dataset.schemas import VideoMetadata
from app.dataset.transcript import fetch_youtube_transcript
from app.db.models.enums import VideoSource, VideoStatus
from app.db.models.video import Video
from app.db.repositories.video import VideoRepository

logger = get_logger(__name__)


class DatasetBuilder:
    """Ingest videos into the platform's dataset."""

    def __init__(
        self,
        session: Session,
        *,
        settings: Settings | None = None,
        downloader: YouTubeDownloader | None = None,
        media_extractor: MediaExtractor | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.settings.ensure_directories()
        self.repo = VideoRepository(session)
        self.downloader = downloader or YouTubeDownloader(self.settings.processed_dir)
        self.media = media_extractor or MediaExtractor()

    # ------------------------------------------------------------------ URLs --
    def ingest_url(self, url: str, *, force: bool = False) -> Video:
        """Ingest a YouTube URL end-to-end. Idempotent."""
        youtube_id = extract_youtube_id(url)
        if youtube_id is None:
            raise ValueError(f"Could not parse a YouTube id from URL: {url!r}")

        video, created = self.repo.get_or_create_youtube(youtube_id, url)
        if not created and video.status == VideoStatus.ready and not force:
            logger.info("Video %s already ingested; skipping.", youtube_id)
            return video

        video.source = VideoSource.youtube
        self.repo.set_status(video, VideoStatus.downloading)
        try:
            metadata, video_path, thumb_path = self.downloader.download(url)
            transcript = fetch_youtube_transcript(youtube_id)
            self._finalise(video, metadata, video_path, thumb_path, transcript)
        except Exception as exc:
            logger.exception("Ingestion failed for %s", youtube_id)
            self.repo.set_status(video, VideoStatus.failed, error=str(exc))
            raise
        return video

    # ---------------------------------------------------------------- Local --
    def ingest_local(self, path: str | Path, *, force: bool = False) -> Video:
        """Ingest a local video file."""
        src = Path(path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(src)

        key = f"local_{src.stem}"
        existing = self.repo.get_by_youtube_id(key)
        if existing and existing.status == VideoStatus.ready and not force:
            return existing
        video = existing or self.repo.add(
            Video(youtube_id=key, url=str(src), source=VideoSource.local, title=src.stem)
        )

        self.repo.set_status(video, VideoStatus.downloading)
        try:
            dest_dir = self.settings.processed_dir / key
            dest_dir.mkdir(parents=True, exist_ok=True)
            video_path = dest_dir / src.name
            if not video_path.exists():
                shutil.copy2(src, video_path)
            self._finalise(video, VideoMetadata(title=src.stem), video_path, None, None)
        except Exception as exc:
            logger.exception("Local ingestion failed for %s", src)
            self.repo.set_status(video, VideoStatus.failed, error=str(exc))
            raise
        return video

    # -------------------------------------------------------------- shared --
    def _finalise(
        self,
        video: Video,
        metadata: VideoMetadata,
        video_path: Path,
        thumb_path: Path | None,
        transcript: str | None,
    ) -> None:
        """Extract media, fill probe-derived gaps, persist all fields."""
        self._apply_metadata(video, metadata)
        self.repo.set_status(video, VideoStatus.extracting)

        artifacts = self.media.extract(video_path, video_path.parent)

        # Fill missing geometry/duration from ffprobe when metadata lacked them.
        probed = probe_video(video_path)
        video.duration_seconds = video.duration_seconds or probed.get("duration")
        video.fps = video.fps or probed.get("fps")
        video.width = video.width or (int(probed["width"]) if "width" in probed else None)
        video.height = video.height or (int(probed["height"]) if "height" in probed else None)

        root = self.settings.processed_dir
        video.video_path = _rel(video_path, root)
        video.audio_path = _rel(artifacts.audio_path, root)
        video.thumbnail_path = _rel(thumb_path, root)
        video.frames_dir = _rel(artifacts.frames_dir, root)
        video.transcript = transcript

        self.repo.set_status(video, VideoStatus.ready)
        logger.info("Ingested video %s (status=ready)", video.youtube_id)

    @staticmethod
    def _apply_metadata(video: Video, metadata: VideoMetadata) -> None:
        for field_name in (
            "title", "description", "channel", "channel_id", "subscriber_count",
            "view_count", "like_count", "comment_count", "duration_seconds",
            "upload_date", "fps", "width", "height",
        ):
            value = getattr(metadata, field_name)
            if value is not None:
                setattr(video, field_name, value)


def _rel(path: Path | None, root: Path) -> str | None:
    """Store artefact paths relative to processed_dir when possible."""
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
