"""Transcript extraction.

Prefers YouTube's own captions (fast, accurate) via youtube-transcript-api.
Returns None when captions are unavailable — the pipeline treats text as an
optional modality, and TRIBE can run on video+audio alone.
"""

from __future__ import annotations

from app.core.logging import get_logger

logger = get_logger(__name__)


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
