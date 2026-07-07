"""Dataset orchestration for dashboards and background jobs."""

from __future__ import annotations

from collections import Counter
from contextlib import AbstractContextManager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import REPO_ROOT, Settings, get_settings
from app.core.logging import get_logger
from app.dataset.builder import DatasetBuilder, IngestionError
from app.dataset.report import build_dataset_report
from app.dataset.schemas import VideoMetadata
from app.db.models.enums import VideoStatus
from app.db.models.video import Video
from app.db.repositories.video import VideoRepository
from app.services.jobs import JobContext

logger = get_logger(__name__)

DATASET_MODES_PATH = REPO_ROOT / "configs" / "dataset_modes.yaml"
ASSUMED_720P_STORAGE_MB_PER_SECOND = 1.5
ASSUMED_720P_STORAGE_BYTES_PER_SECOND = int(ASSUMED_720P_STORAGE_MB_PER_SECOND * 1024 * 1024)


class DatasetFilter(BaseModel):
    """Optional filter controls applied before ingestion or preview."""

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

    def matches(self, item: VideoMetadata | Video) -> bool:
        """Return True when a metadata/video object satisfies the filter."""
        duration = getattr(item, "duration_seconds", None)
        if self.min_duration_seconds is not None and (
            duration is None or duration < self.min_duration_seconds
        ):
            return False
        if self.max_duration_seconds is not None and (
            duration is None or duration > self.max_duration_seconds
        ):
            return False

        language = getattr(item, "language", None)
        if self.language is not None and (
            language is None or str(language).lower() != self.language.lower()
        ):
            return False

        if self.categories:
            categories = [str(cat).lower() for cat in (getattr(item, "categories", None) or [])]
            if not categories or not any(cat.lower() in categories for cat in self.categories):
                return False

        likes = getattr(item, "like_count", None)
        if self.min_likes is not None and (likes is None or likes < self.min_likes):
            return False
        if self.max_likes is not None and (likes is None or likes > self.max_likes):
            return False

        views = getattr(item, "view_count", None)
        if self.min_views is not None and (views is None or views < self.min_views):
            return False
        if self.max_views is not None and (views is None or views > self.max_views):
            return False

        uploaded = getattr(item, "upload_date", None)
        if uploaded is not None:
            if isinstance(uploaded, datetime) and uploaded.tzinfo is None:
                uploaded = uploaded.replace(tzinfo=UTC)
            start = _coerce_dt(self.date_from)
            end = _coerce_dt(self.date_to)
            if start is not None and uploaded < start:
                return False
            if end is not None and uploaded > end:
                return False
        elif self.date_from is not None or self.date_to is not None:
            return False

        return True


class DatasetService:
    """High-level dataset operations reused by the API and jobs."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        modes_path: Path = DATASET_MODES_PATH,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_directories()
        self.modes_path = modes_path
        self.builder_factory = DatasetBuilder

    # --------------------------------------------------------------- modes --
    def load_modes(self) -> dict[str, dict[str, Any]]:
        if not self.modes_path.exists():
            return {}
        data = OmegaConf.load(self.modes_path)
        return cast(dict[str, dict[str, Any]], OmegaConf.to_container(data, resolve=True) or {})

    def get_mode(self, name: str) -> dict[str, Any]:
        modes = self.load_modes()
        if name not in modes:
            raise KeyError(name)
        return dict(modes[name])

    def add_channel(self, mode: str, url: str) -> dict[str, Any]:
        modes = self.load_modes()
        entry = dict(modes.get(mode, {"hint": "", "channels": []}))
        channels = list(entry.get("channels", []))
        if url not in channels:
            channels.append(url)
        entry["channels"] = channels
        modes[mode] = entry
        self._save_modes(modes)
        return entry

    def remove_channel(self, mode: str, url: str) -> dict[str, Any]:
        modes = self.load_modes()
        if mode not in modes:
            raise KeyError(mode)
        entry = dict(modes[mode])
        channels = [item for item in entry.get("channels", []) if item != url]
        entry["channels"] = channels
        modes[mode] = entry
        self._save_modes(modes)
        return entry

    # -------------------------------------------------------------- preview --
    def preview(
        self,
        mode: str,
        *,
        filter: DatasetFilter | None = None,
        count: int | None = None,
    ) -> dict[str, Any]:
        modes = self.load_modes()
        entry = modes.get(mode, {})
        filter = filter or DatasetFilter()
        metadata_rows = self._gather_metadata(entry.get("channels", []), filter=filter, count=count)
        if not metadata_rows:
            metadata_rows = self._db_preview(filter=filter, count=count)
        return self._preview_from_rows(metadata_rows, count=count)

    # --------------------------------------------------------------- build --
    def build(
        self,
        mode: str,
        *,
        filter: DatasetFilter | None = None,
        count: int | None = None,
        force: bool = False,
        job: JobContext | None = None,
    ) -> dict[str, Any]:
        filter = filter or DatasetFilter()
        modes = self.load_modes()
        entry = modes.get(mode)
        if entry is None:
            raise KeyError(mode)

        candidates = self._expand_sources(entry.get("channels", []))
        total = len(candidates)
        if job is not None:
            job.set_progress(stage="download", current=0, total=total, message=f"mode={mode}")

        stats: dict[str, int] = {
            "requested": 0,
            "ingested": 0,
            "skipped": 0,
            "rejected": 0,
            "failed": 0,
        }
        accepted = 0
        with self._session_scope() as session:
            builder = self.builder_factory(session, settings=self.settings)
            for index, url in enumerate(candidates, start=1):
                if count is not None and accepted >= count:
                    break
                if job is not None:
                    job.check_control()
                stats["requested"] += 1
                metadata = self._metadata_for_url(builder, url, job=job)
                if metadata is None:
                    stats["failed"] += 1
                    if job is not None:
                        job.set_progress(
                            current=index, total=total, message=f"metadata failed: {url}"
                        )
                    continue
                if not filter.matches(metadata):
                    stats["skipped"] += 1
                    if job is not None:
                        job.set_progress(current=index, total=total, message=f"filtered out: {url}")
                    continue
                try:
                    video = builder.ingest_url(url, force=force)
                except IngestionError:
                    stats["rejected"] += 1
                    if job is not None:
                        job.set_progress(current=index, total=total, message=f"rejected: {url}")
                    continue
                except Exception as exc:  # pragma: no cover - defensive
                    stats["failed"] += 1
                    logger.exception("Ingest failed for %s", url)
                    if job is not None:
                        job.log(f"ingest failed for {url}: {exc}")
                    continue
                accepted += 1
                stats["ingested"] += 1
                if job is not None:
                    job.set_progress(
                        current=index,
                        total=total,
                        message=f"ingested #{video.id} {video.youtube_id}",
                    )

        return {**stats, "mode": mode, "accepted": accepted}

    def validate(
        self,
        *,
        filter: DatasetFilter | None = None,
        count: int | None = None,
        job: JobContext | None = None,
    ) -> dict[str, Any]:
        filter = filter or DatasetFilter()
        stats: dict[str, int] = {
            "requested": 0,
            "valid": 0,
            "rejected": 0,
            "failed": 0,
        }
        with self._session_scope() as session:
            builder = self.builder_factory(session, settings=self.settings)
            videos = [
                video
                for video in session.scalars(select(Video).where(Video.status == VideoStatus.ready))
                if filter.matches(video)
            ]
            total = len(videos)
            for index, video in enumerate(videos, start=1):
                if count is not None and stats["requested"] >= count:
                    break
                if job is not None:
                    job.check_control()
                stats["requested"] += 1
                video_path = self._abs_path(video.video_path)
                if video_path is None:
                    stats["failed"] += 1
                    continue
                try:
                    result = builder.validator.validate(
                        video_path,
                        VideoMetadata(
                            duration_seconds=video.duration_seconds,
                            like_count=video.like_count,
                            view_count=video.view_count,
                            upload_date=video.upload_date,
                            language=None,
                        ),
                    )
                    video.validation = result.as_dict()
                    if result.is_valid:
                        stats["valid"] += 1
                    else:
                        stats["rejected"] += 1
                except Exception as exc:  # pragma: no cover - defensive
                    stats["failed"] += 1
                    logger.exception("Validation failed for video %s", video.id)
                    if job is not None:
                        job.log(f"validation failed for {video.id}: {exc}")
                if job is not None:
                    job.set_progress(
                        stage="validate", current=index, total=total, message=f"video={video.id}"
                    )
        return dict(stats)

    def build_report(self, session: Session) -> dict[str, Any]:
        return build_dataset_report(session).as_dict()

    def list_videos(self, session: Session, *, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        repo = VideoRepository(session)
        videos = repo.list(limit=limit, offset=offset)
        items = [
            {
                "id": video.id,
                "youtube_id": video.youtube_id,
                "title": video.title,
                "channel": video.channel,
                "status": video.status,
                "brain_status": video.brain_status,
                "duration_seconds": video.duration_seconds,
                "view_count": video.view_count,
                "like_count": video.like_count,
                "upload_date": video.upload_date,
                "validation": video.validation,
            }
            for video in videos
        ]
        return {"items": items, "total": repo.count(), "limit": limit, "offset": offset}

    # ------------------------------------------------------------- helpers --
    def _save_modes(self, modes: dict[str, Any]) -> None:
        self.modes_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(modes), self.modes_path)

    def _expand_sources(self, sources: list[str]) -> list[str]:
        candidates: list[str] = []
        from app.dataset.download import YouTubeDownloader

        source_downloader = YouTubeDownloader(self.settings.processed_dir)
        for source in sources:
            try:
                expanded = source_downloader.expand_source(source)
            except Exception as exc:
                logger.warning("Could not expand %s: %s", source, exc)
                continue
            candidates.extend(expanded)
        seen: set[str] = set()
        unique: list[str] = []
        for url in candidates:
            if url in seen:
                continue
            seen.add(url)
            unique.append(url)
        return unique

    def _gather_metadata(
        self, sources: list[str], *, filter: DatasetFilter, count: int | None
    ) -> list[VideoMetadata]:
        from app.dataset.download import YouTubeDownloader

        downloader = YouTubeDownloader(self.settings.processed_dir)
        rows: list[VideoMetadata] = []
        for source in sources:
            try:
                expanded = downloader.expand_source(source)
            except Exception as exc:
                logger.warning("Preview expansion failed for %s: %s", source, exc)
                continue
            for url in expanded:
                if count is not None and len(rows) >= count:
                    return rows
                try:
                    metadata = downloader.fetch_metadata(url)
                except Exception as exc:
                    logger.info("Preview metadata failed for %s: %s", url, exc)
                    continue
                if filter.matches(metadata):
                    rows.append(metadata)
        return rows

    def _metadata_for_url(
        self, builder: DatasetBuilder, url: str, *, job: JobContext | None
    ) -> VideoMetadata | None:
        try:
            return builder.downloader.fetch_metadata(url)
        except Exception as exc:
            if job is not None:
                job.log(f"metadata fetch failed for {url}: {exc}")
            logger.info("Metadata fetch failed for %s: %s", url, exc)
            return None

    def _db_preview(self, *, filter: DatasetFilter, count: int | None) -> list[VideoMetadata]:
        with self._session_scope() as session:
            rows: list[VideoMetadata] = []
            for video in session.scalars(select(Video)).all():
                if not filter.matches(video):
                    continue
                rows.append(
                    VideoMetadata(
                        youtube_id=video.youtube_id,
                        url=video.url,
                        language=None,
                        categories=None,
                        title=video.title,
                        description=video.description,
                        channel=video.channel,
                        channel_id=video.channel_id,
                        subscriber_count=video.subscriber_count,
                        view_count=video.view_count,
                        like_count=video.like_count,
                        comment_count=video.comment_count,
                        duration_seconds=video.duration_seconds,
                        upload_date=video.upload_date,
                        fps=video.fps,
                        width=video.width,
                        height=video.height,
                    )
                )
                if count is not None and len(rows) >= count:
                    break
            return rows

    def _preview_from_rows(self, rows: list[VideoMetadata], *, count: int | None) -> dict[str, Any]:
        if count is not None:
            rows = rows[:count]
        month_hist: Counter[str] = Counter()
        year_hist: Counter[str] = Counter()
        total_bytes = 0.0
        durations: list[float] = []
        for item in rows:
            if item.upload_date is not None:
                uploaded = item.upload_date
                if uploaded.tzinfo is None:
                    uploaded = uploaded.replace(tzinfo=UTC)
                month_hist[uploaded.strftime("%Y-%m")] += 1
                year_hist[uploaded.strftime("%Y")] += 1
            if item.duration_seconds is not None:
                durations.append(float(item.duration_seconds))
                total_bytes += float(item.duration_seconds) * ASSUMED_720P_STORAGE_BYTES_PER_SECOND

        if count is not None and rows and len(rows) < count and durations:
            total_bytes = (
                (sum(durations) / len(durations)) * count * ASSUMED_720P_STORAGE_BYTES_PER_SECOND
            )

        return {
            "estimated_count": len(rows),
            "publication_date_histogram": {
                "by_month": dict(sorted(month_hist.items())),
                "by_year": dict(sorted(year_hist.items())),
            },
            "estimated_storage_bytes": int(total_bytes),
        }

    def _session_scope(self) -> AbstractContextManager[Session]:
        from app.db.base import session_scope

        return session_scope()

    def _abs_path(self, path: str | None) -> str | None:
        if path is None:
            return None
        p = Path(path)
        return str(p if p.is_absolute() else self.settings.processed_dir / p)


def _coerce_dt(value: datetime | date | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value
