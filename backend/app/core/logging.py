"""Centralised logging configuration.

We use :mod:`rich` for human-friendly console output in development and a plain,
structured format in production so that log aggregators can parse it. All
modules obtain loggers via :func:`get_logger` and never call ``print``.
"""

from __future__ import annotations

import logging
import sys
from functools import cache

from rich.logging import RichHandler

_CONFIGURED = False
_PLAIN_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", *, rich_console: bool = True) -> None:
    """Configure the root logger once. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)

    if rich_console and sys.stderr.isatty():
        handler: logging.Handler = RichHandler(
            rich_tracebacks=True, show_path=False, markup=False
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    else:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(_PLAIN_FORMAT))

    root.addHandler(handler)

    # Quieten noisy third-party loggers that would otherwise flood inference.
    for noisy in ("httpx", "urllib3", "yt_dlp", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


@cache
def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring logging has been configured."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
