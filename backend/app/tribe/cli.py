"""``ysp-compute-brain`` — run TRIBE v2 over the ready videos and cache tensors.

Iterates every ``ready`` video whose brain activity is not yet cached and runs
the official TRIBE v2 model once per video via :class:`BrainActivityService`,
persisting the ``(T, V)`` tensor to the cache. Videos already cached are skipped
(never recomputed). This stage requires a CUDA GPU and the installed ``tribev2``
weights — it is intended to run on the Vultr GPU instance, not the dev laptop.

Examples
--------
    ysp-compute-brain                 # compute for all ready, uncached videos
    ysp-compute-brain --limit 50
    ysp-compute-brain --retry-failed  # also retry videos whose TRIBE run failed
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import select

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.db.base import session_scope
from app.db.models.enums import BrainStatus, VideoStatus
from app.db.models.video import Video
from app.tribe.extractor import TribeExtractor, TribeNotInstalled
from app.tribe.service import BrainActivityService

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TRIBE v2 over ready videos and cache tensors."
    )
    parser.add_argument("--limit", type=int, default=None, help="Max videos to process.")
    parser.add_argument("--force", action="store_true", help="Recompute even if cached.")
    parser.add_argument(
        "--retry-failed", action="store_true", help="Also retry previously-failed videos."
    )
    return parser


def _pending_video_ids(session, *, force: bool, retry_failed: bool, limit: int | None) -> list[int]:
    stmt = select(Video.id).where(Video.status == VideoStatus.ready)
    if not force:
        allowed = [BrainStatus.absent]
        if retry_failed:
            allowed.append(BrainStatus.failed)
        stmt = stmt.where(Video.brain_status.in_(allowed))
    ids = list(session.scalars(stmt))
    return ids[:limit] if limit is not None else ids


def main(argv: list[str] | None = None) -> int:
    setup_logging(level=get_settings().log_level)
    args = _build_parser().parse_args(argv)

    with session_scope() as session:
        video_ids = _pending_video_ids(
            session, force=args.force, retry_failed=args.retry_failed, limit=args.limit
        )
    if not video_ids:
        logger.info("No videos require TRIBE inference. Nothing to do.")
        return 0

    logger.info("Computing TRIBE brain activity for %d video(s).", len(video_ids))
    # One extractor (loads the ~1B-param model once) reused across all videos.
    try:
        extractor = TribeExtractor()
        if not extractor.is_available():
            raise TribeNotInstalled()
    except TribeNotInstalled as exc:
        logger.error("%s", exc)
        return 2

    computed = failed = 0
    for video_id in video_ids:
        with session_scope() as session:
            video = session.get(Video, video_id)
            if video is None:
                continue
            service = BrainActivityService(session, extractor=extractor)
            try:
                activity = service.get_or_compute(video, force=args.force)
                computed += 1
                logger.info(
                    "[%d/%d] video %s -> brain %sx%s",
                    computed + failed, len(video_ids), video.youtube_id,
                    activity.n_timesteps, activity.n_vertices,
                )
            except Exception as exc:
                failed += 1
                logger.exception("TRIBE failed for video %s: %s", video.youtube_id, exc)

    logger.info("TRIBE stage complete: %d computed, %d failed.", computed, failed)
    return 1 if failed and not computed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
