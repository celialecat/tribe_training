"""Dataset builder — orchestrates ingestion of a video into DB + filesystem.

Given a YouTube URL, playlist, channel, or a local file, it: downloads (or
copies) the media + subtitles + thumbnail, probes/extracts audio + frames +
transcript, **validates the downloaded asset**, and only then upserts a ``ready``
:class:`Video` row. Videos that fail validation are marked ``rejected`` with the
failing checks recorded, and never enter the training set. Ingestion is
idempotent and resumable — an already-``ready`` video is returned untouched
unless ``force`` is set.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.dataset.download import DownloadResult, YouTubeDownloader, extract_youtube_id
from app.dataset.media import MediaExtractor, probe_video
from app.dataset.schemas import VideoMetadata
from app.dataset.transcript import fetch_youtube_transcript, parse_subtitle_file
from app.dataset.validation import ValidationResult, VideoValidator
from app.db.models.enums import VideoSource, VideoStatus
from app.db.models.video import Video
from app.db.repositories.video import VideoRepository

logger = get_logger(__name__)


class IngestionError(RuntimeError):
    """Raised when a video cannot be ingested (download failure or rejection)."""


@dataclass(slots=True)
class BatchIngestReport:
    """Summary of ingesting a batch/playlist/channel."""

    requested: int = 0
    ingested: list[int] = field(default_factory=list)   # ready video ids
    skipped: list[str] = field(default_factory=list)     # already-ready ids
    rejected: list[str] = field(default_factory=list)    # failed validation
    failed: list[str] = field(default_factory=list)      # download/pipeline error

    def summary(self) -> dict[str, int]:
        return {
            "requested": self.requested,
            "ingested": len(self.ingested),
            "skipped": len(self.skipped),
            "rejected": len(self.rejected),
            "failed": len(self.failed),
        }


class DatasetBuilder:
    """Ingest videos into the platform's dataset."""

    def __init__(
        self,
        session: Session,
        *,
        settings: Settings | None = None,
        downloader: YouTubeDownloader | None = None,
        media_extractor: MediaExtractor | None = None,
        validator: VideoValidator | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.settings.ensure_directories()
        self.repo = VideoRepository(session)
        self.downloader = downloader or YouTubeDownloader(self.settings.processed_dir)
        self.media = media_extractor or MediaExtractor()
        self.validator = validator or VideoValidator()

    # --------------------------------------------------------------- source --
    def ingest_source(
        self, url: str, *, force: bool = False, limit: int | None = None
    ) -> BatchIngestReport:
        """Expand a URL (video / playlist / channel) and ingest every video.

        Errors on individual videos are captured in the report rather than
        aborting the whole batch, so a channel with a few dead videos still
        ingests the rest.
        """
        video_urls = self.downloader.expand_source(url)
        if limit is not None:
            video_urls = video_urls[:limit]
        report = BatchIngestReport(requested=len(video_urls))
        for video_url in video_urls:
            vid = extract_youtube_id(video_url) or video_url
            existing = self.repo.get_by_youtube_id(vid) if vid else None
            already_ready = existing is not None and existing.status == VideoStatus.ready
            try:
                video = self.ingest_url(video_url, force=force)
                if already_ready and not force:
                    report.skipped.append(vid)
                else:
                    report.ingested.append(video.id)
            except IngestionError:
                report.rejected.append(vid)
            except Exception:
                report.failed.append(vid)
        logger.info("Batch ingest of %s: %s", url, report.summary())
        return report

    # ------------------------------------------------------------------ URLs --
    def ingest_url(self, url: str, *, force: bool = False) -> Video:
        """Ingest a single YouTube video end-to-end. Idempotent + validated."""
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
            result = self.downloader.download(url)
            transcript = fetch_youtube_transcript(youtube_id) or (
                parse_subtitle_file(result.subtitle_path) if result.subtitle_path else None
            )
            self._finalise(video, result, transcript)
        except IngestionError:
            # Validation rejection: _finalise already set status=rejected; do not
            # reclassify as a download failure. Propagate for the caller/report.
            raise
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
            result = DownloadResult(metadata=VideoMetadata(title=src.stem), video_path=video_path)
            self._finalise(video, result, None)
        except Exception as exc:
            logger.exception("Local ingestion failed for %s", src)
            self.repo.set_status(video, VideoStatus.failed, error=str(exc))
            raise
        return video

    # -------------------------------------------------------------- shared --
    def _finalise(
        self, video: Video, result: DownloadResult, transcript: str | None
    ) -> None:
        """Extract media, VALIDATE, fill probe gaps, persist all fields.

        A video is only marked ``ready`` after passing asset validation;
        otherwise it is marked ``rejected`` with the failing checks recorded and
        an :class:`IngestionError` is raised.
        """
        video_path = result.video_path
        self._apply_metadata(video, result.metadata)
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
        video.thumbnail_path = _rel(result.thumbnail_path, root)
        video.subtitle_path = _rel(result.subtitle_path, root)
        video.frames_dir = _rel(artifacts.frames_dir, root)
        video.transcript = transcript

        # ---- Validation gate: prove the asset is usable before admission. ----
        self.repo.set_status(video, VideoStatus.validating)
        validation: ValidationResult = self.validator.validate(video_path, result.metadata)
        video.validation = validation.as_dict()
        if not validation.is_valid:
            self.repo.set_status(video, VideoStatus.rejected, error=validation.reason())
            logger.warning("Rejected video %s: %s", video.youtube_id, validation.reason())
            raise IngestionError(f"{video.youtube_id} failed validation: {validation.reason()}")

        self.repo.set_status(video, VideoStatus.ready)
        logger.info("Ingested + validated video %s (status=ready)", video.youtube_id)

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
