"""Transcript extraction.

Prefers YouTube's own captions (fast, accurate) via youtube-transcript-api.
Returns None when captions are unavailable — the pipeline treats text as an
optional modality, and TRIBE can run on video+audio alone.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)

# Matches VTT/SRT timestamp cue lines and sequence numbers we strip out.
_CUE_TIMESTAMP = re.compile(r"^\d{1,2}:\d{2}:\d{2}[.,]\d{3}\s*-->")
_SEQ_NUMBER = re.compile(r"^\d+$")
_TAGS = re.compile(r"<[^>]+>")


def parse_subtitle_file(path: str | Path) -> str | None:
    """Parse a downloaded VTT/SRT subtitle file into plain, de-duplicated text.

    yt-dlp's auto-captions repeat lines across cues (rolling captions); we
    collapse consecutive duplicates so the transcript reads naturally.
    """
    path = Path(path)
    if not path.exists():
        return None
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = _TAGS.sub("", raw).strip()
        if not line or line == "WEBVTT" or line.startswith(("Kind:", "Language:")):
            continue
        if _CUE_TIMESTAMP.search(line) or _SEQ_NUMBER.match(line):
            continue
        if not lines or lines[-1] != line:  # drop consecutive duplicates
            lines.append(line)
    text = " ".join(lines).strip()
    return text or None


def fetch_youtube_transcript(
    youtube_id: str, *, languages: tuple[str, ...] = ("en",)
) -> str | None:
    """Fetch and concatenate the caption track for a YouTube id."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        logger.warning("youtube-transcript-api not installed; skipping transcript.")
        return None

    try:
        segments = YouTubeTranscriptApi.get_transcript(youtube_id, languages=list(languages))
    except Exception as exc:  # library raises many subclasses; treat all as "no transcript"
        logger.info("No transcript for %s: %s", youtube_id, type(exc).__name__)
        return None

    text = " ".join(seg["text"].strip() for seg in segments if seg.get("text"))
    return text or None
