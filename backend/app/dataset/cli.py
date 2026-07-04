"""``ysp-build-dataset`` — ingest YouTube URLs or local files into the dataset.

Examples
--------
    ysp-build-dataset --url https://youtu.be/dQw4w9WgXcQ
    ysp-build-dataset --file ./clip.mp4
    ysp-build-dataset --urls-file urls.txt --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core.logging import get_logger, setup_logging
from app.dataset.builder import DatasetBuilder
from app.db.base import session_scope

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest videos into the YSP dataset.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="A single YouTube URL.")
    src.add_argument("--file", help="A single local video file.")
    src.add_argument("--urls-file", help="Text file with one YouTube URL per line.")
    parser.add_argument("--force", action="store_true", help="Re-ingest even if ready.")
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = _build_parser().parse_args(argv)

    urls: list[str] = []
    if args.url:
        urls = [args.url]
    elif args.urls_file:
        urls = [
            line.strip()
            for line in Path(args.urls_file).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    ok, failed = 0, 0
    with session_scope() as session:
        builder = DatasetBuilder(session)
        if args.file:
            _ingest_one(builder, local=args.file, force=args.force)
            return 0
        for url in urls:
            try:
                _ingest_one(builder, url=url, force=args.force)
                ok += 1
            except Exception:
                failed += 1

    logger.info("Dataset build complete: %d ok, %d failed.", ok, failed)
    return 1 if failed and not ok else 0


def _ingest_one(
    builder: DatasetBuilder, *, url: str | None = None, local: str | None = None, force: bool
) -> None:
    if url:
        video = builder.ingest_url(url, force=force)
    else:
        video = builder.ingest_local(local, force=force)
    logger.info("-> #%s %s [%s]", video.id, video.youtube_id, video.status)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
