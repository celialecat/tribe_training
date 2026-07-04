"""``ysp-build-dataset`` — ingest YouTube URLs / playlists / channels or files.

Every downloaded video is validated before admission; invalid videos are
rejected, not trained on. Use ``--report`` to print the dataset validation
summary.

Examples
--------
    ysp-build-dataset --url https://youtu.be/dQw4w9WgXcQ
    ysp-build-dataset --url "https://www.youtube.com/@channel/videos" --limit 100
    ysp-build-dataset --url "https://www.youtube.com/playlist?list=PL..."
    ysp-build-dataset --file ./clip.mp4
    ysp-build-dataset --urls-file urls.txt --force
    ysp-build-dataset --report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core.logging import get_logger, setup_logging
from app.dataset.builder import DatasetBuilder
from app.dataset.report import build_dataset_report
from app.db.base import session_scope

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest + validate videos into the YSP dataset.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="A YouTube video, playlist or channel URL.")
    src.add_argument("--file", help="A single local video file.")
    src.add_argument("--urls-file", help="Text file with one YouTube URL per line.")
    src.add_argument("--report", action="store_true", help="Print the dataset report and exit.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Max videos to ingest from a source."
    )
    parser.add_argument("--force", action="store_true", help="Re-ingest even if ready.")
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = _build_parser().parse_args(argv)

    if args.report:
        with session_scope() as session:
            print(build_dataset_report(session).render())
        return 0

    with session_scope() as session:
        builder = DatasetBuilder(session)
        if args.file:
            video = builder.ingest_local(args.file, force=args.force)
            logger.info("-> #%s %s [%s]", video.id, video.youtube_id, video.status)
            return 0

        sources = _collect_sources(args)
        totals = {"ingested": 0, "skipped": 0, "rejected": 0, "failed": 0, "requested": 0}
        for source in sources:
            report = builder.ingest_source(source, force=args.force, limit=args.limit)
            for key in totals:
                totals[key] += report.summary()[key]

    logger.info("Dataset build complete: %s", totals)
    print(build_dataset_report_str())
    return 1 if totals["failed"] and not totals["ingested"] else 0


def _collect_sources(args: argparse.Namespace) -> list[str]:
    if args.url:
        return [args.url]
    if args.urls_file:
        return [
            line.strip()
            for line in Path(args.urls_file).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    return []


def build_dataset_report_str() -> str:
    """Render the current dataset report as a string (fresh session)."""
    with session_scope() as session:
        return build_dataset_report(session).render()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
