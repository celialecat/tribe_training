"""TRIBE orchestration for the dashboard backend."""

from __future__ import annotations

from time import monotonic
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.base import session_scope
from app.db.models.enums import BrainStatus, VideoStatus
from app.db.models.video import Video
from app.services.jobs import JobContext, JobStatus, JobType, get_job_manager
from app.tribe.extractor import TribeExtractor
from app.tribe.service import BrainActivityService

logger = get_logger(__name__)


class TribeService:
    """Iterate pending videos and run BrainActivityService for each one."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        extractor: TribeExtractor | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.extractor = extractor or TribeExtractor(settings=self.settings)

    def pending_video_ids(
        self,
        session: Session,
        *,
        force: bool = False,
        retry_failed: bool = False,
        limit: int | None = None,
    ) -> list[int]:
        stmt = select(Video.id).where(Video.status == VideoStatus.ready)
        if not force:
            allowed = [BrainStatus.absent]
            if retry_failed:
                allowed.append(BrainStatus.failed)
            stmt = stmt.where(Video.brain_status.in_(allowed))
        ids = list(session.scalars(stmt))
        return ids[:limit] if limit is not None else ids

    def run(
        self,
        *,
        job: JobContext | None = None,
        force: bool = False,
        retry_failed: bool = False,
        limit: int | None = None,
    ) -> dict[str, Any]:
        with session_scope() as session:
            video_ids = self.pending_video_ids(
                session, force=force, retry_failed=retry_failed, limit=limit
            )

        if job is not None:
            job.set_progress(stage="tribe", current=0, total=len(video_ids), message="pending")

        processed = failed = 0
        eta = None
        start = monotonic()
        for index, video_id in enumerate(video_ids, start=1):
            if job is not None:
                job.check_control()
            with session_scope() as session:
                video = session.get(Video, video_id)
                if video is None:
                    continue
                service = BrainActivityService(session, extractor=self.extractor)
                try:
                    activity = service.get_or_compute(video, force=force)
                    processed += 1
                    if job is not None:
                        elapsed = monotonic() - start
                        eta = self._eta(processed, len(video_ids), elapsed)
                        job.set_progress(
                            stage="tribe",
                            current=index,
                            total=len(video_ids),
                            elapsed=elapsed,
                            message=f"video={video.youtube_id}",
                        )
                        job.log(
                            "computed brain activity for "
                            f"{video.youtube_id} "
                            f"({activity.n_timesteps}x{activity.n_vertices})"
                        )
                except Exception as exc:  # pragma: no cover - defensive
                    failed += 1
                    logger.exception("TRIBE failed for video %s", video_id)
                    if job is not None:
                        job.log(f"TRIBE failed for {video.youtube_id}: {exc}")
                        job.set_progress(
                            stage="tribe",
                            current=index,
                            total=len(video_ids),
                            message=f"failed: {video.youtube_id}",
                        )

        return {
            "cached": processed,
            "remaining": max(0, len(video_ids) - processed - failed),
            "failed": failed,
            "eta_seconds": eta,
        }

    def status(self) -> dict[str, Any]:
        try:
            with session_scope() as session:
                cached = session.scalar(
                    select(func.count())
                    .select_from(Video)
                    .where(Video.brain_status == BrainStatus.cached)
                )
                remaining = len(self.pending_video_ids(session))
                failed = session.scalar(
                    select(func.count())
                    .select_from(Video)
                    .where(Video.brain_status == BrainStatus.failed)
                )
                processing = session.scalar(
                    select(Video.youtube_id)
                    .where(Video.brain_status == BrainStatus.computing)
                    .limit(1)
                )
        except SQLAlchemyError:
            cached = 0
            remaining = 0
            failed = 0
            processing = None
        eta = None
        jobs = get_job_manager()
        for job in jobs.list():
            if job.type == JobType.tribe.value and job.status in {
                JobStatus.running.value,
                JobStatus.paused.value,
                JobStatus.pending.value,
            }:
                eta = job.progress.get("eta_seconds")
                if processing is None:
                    processing = str(job.progress.get("message") or "")
                break
        return {
            "cached": int(cached or 0),
            "remaining": int(remaining),
            "failed": int(failed or 0),
            "processing_video": processing,
            "eta_seconds": eta,
        }

    @staticmethod
    def _eta(done: int, total: int, elapsed: float) -> float | None:
        if done <= 0 or total <= done:
            return None
        return max(0.0, elapsed / done * (total - done))
