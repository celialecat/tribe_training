"""``ysp-pipeline`` — run the whole ML workflow as one resumable command.

Stages (each can be skipped):

    1. migrate   apply Alembic migrations to the (persistent) database
    2. download  ingest + validate videos from a source (yt-dlp)
    3. report    print the dataset validation report
    4. brain     run TRIBE v2 over ready videos, cache the (T, V) tensors
    5. train     train the Brain Encoder + Success Predictor end-to-end

Stages are executed as subprocesses of the project's console scripts so each
runs with its own (Hydra / argparse) entry point exactly as when invoked
manually. Any argument after ``--`` is forwarded verbatim to ``ysp-train`` as a
Hydra override, e.g.::

    ysp-pipeline --source "<channel-url>" --limit 100 -- training.epochs=200

This is the command the Vultr launcher runs on the GPU instance. It intentionally
does NOT provision infrastructure — that is the launcher's job (see
``app.deploy.vultr``); keeping the two separate means the same pipeline runs on
Vultr, in Docker, or on any GPU box.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import app
from app.core.logging import get_logger, setup_logging

logger = get_logger(__name__)

ALL_STAGES = ("migrate", "download", "report", "brain", "train")
# Directory containing alembic.ini (backend/), used as CWD for migrations.
BACKEND_DIR = Path(app.__file__).resolve().parent.parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end YSP pipeline (download -> TRIBE -> train)."
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--source", help="Video / playlist / channel URL to ingest.")
    src.add_argument("--urls-file", help="File with one URL per line to ingest.")
    parser.add_argument("--limit", type=int, default=None, help="Max videos to ingest.")
    parser.add_argument(
        "--skip", nargs="*", default=[], choices=ALL_STAGES, help="Stages to skip."
    )
    parser.add_argument(
        "--only", nargs="*", default=None, choices=ALL_STAGES, help="Run only these stages."
    )
    parser.add_argument("--force", action="store_true", help="Force re-ingest / recompute.")
    parser.add_argument(
        "train_overrides", nargs="*", help="Hydra overrides for ysp-train (after --)."
    )
    return parser


def _run(stage: str, cmd: list[str], *, cwd: Path | None = None) -> None:
    logger.info("=== stage '%s' === %s", stage, " ".join(cmd))
    start = time.monotonic()
    subprocess.run(cmd, check=True, cwd=cwd)
    logger.info("=== stage '%s' done in %.1fs ===", stage, time.monotonic() - start)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = _build_parser().parse_args(argv)
    py = [sys.executable, "-m"]

    stages = set(args.only) if args.only else set(ALL_STAGES)
    stages -= set(args.skip)

    if "download" in stages and not (args.source or args.urls_file):
        logger.warning("No --source/--urls-file given; skipping download stage.")
        stages.discard("download")

    try:
        if "migrate" in stages:
            _run("migrate", [sys.executable, "-m", "alembic", "upgrade", "head"], cwd=BACKEND_DIR)

        if "download" in stages:
            cmd = [*py, "app.dataset.cli"]
            cmd += ["--url", args.source] if args.source else ["--urls-file", args.urls_file]
            if args.limit is not None:
                cmd += ["--limit", str(args.limit)]
            if args.force:
                cmd += ["--force"]
            _run("download", cmd)

        if "report" in stages:
            _run("report", [*py, "app.dataset.cli", "--report"])

        if "brain" in stages:
            cmd = [*py, "app.tribe.cli"]
            if args.limit is not None:
                cmd += ["--limit", str(args.limit)]
            if args.force:
                cmd += ["--force"]
            _run("brain", cmd)

        if "train" in stages:
            _run("train", [*py, "app.training.cli", *args.train_overrides])
    except subprocess.CalledProcessError as exc:
        logger.error("Pipeline stage failed (exit %s): %s", exc.returncode, exc.cmd)
        return exc.returncode or 1

    logger.info("Pipeline complete.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
