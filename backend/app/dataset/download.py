"""YouTube download + metadata extraction via yt-dlp.

The downloader is intentionally thin and dependency-injectable: the yt-dlp
handle is created lazily and can be swapped in tests. It downloads the best
MP4-compatible stream plus the thumbnail, and normalises yt-dlp's info dict into
a :class:`VideoMetadata`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.dataset.schemas import VideoMetadata

logger = get_logger(__name__)


@dataclass(slots=True)
class DownloadResult:
    """Filesystem locations produced by a single video download."""

    metadata: VideoMetadata
    video_path: Path
    thumbnail_path: Path | None = None
    subtitle_path: Path | None = None

# Accepts standard watch URLs, youtu.be short links, shorts and embeds.
_YOUTUBE_ID_RE = re.compile(
    r"(?:v=|/shorts/|youtu\.be/|/embed/|/v/)([0-9A-Za-z_-]{11})"
)


def extract_youtube_id(url: str) -> str | None:
    """Return the 11-char video id from a YouTube URL, or None."""
    if match := _YOUTUBE_ID_RE.search(url):
        return match.group(1)
    # Bare id passed directly.
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", url):
        return url
    return None


def _parse_upload_date(info: dict[str, Any]) -> datetime | None:
    # yt-dlp exposes `upload_date` as YYYYMMDD and `timestamp` as epoch seconds.
    if ts := info.get("timestamp"):
        return datetime.fromtimestamp(int(ts), tz=UTC)
    if raw := info.get("upload_date"):
        try:
            return datetime.strptime(str(raw), "%Y%m%d").replace(tzinfo=UTC)
        except ValueError:
            return None
    return None


def info_to_metadata(info: dict[str, Any]) -> VideoMetadata:
    """Normalise a yt-dlp info dict into :class:`VideoMetadata`."""
    return VideoMetadata(
        youtube_id=info.get("id"),
        url=info.get("webpage_url") or info.get("original_url"),
        title=info.get("title"),
        description=info.get("description"),
        channel=info.get("channel") or info.get("uploader"),
        channel_id=info.get("channel_id") or info.get("uploader_id"),
        subscriber_count=info.get("channel_follower_count"),
        view_count=info.get("view_count"),
        like_count=info.get("like_count"),
        comment_count=info.get("comment_count"),
        duration_seconds=float(info["duration"]) if info.get("duration") else None,
        upload_date=_parse_upload_date(info),
        fps=info.get("fps"),
        width=info.get("width"),
        height=info.get("height"),
    )


class YouTubeDownloader:
    """Downloads video + thumbnail and extracts metadata using yt-dlp."""

    def __init__(self, output_root: Path, *, ydl_factory: Any | None = None) -> None:
        self.output_root = Path(output_root)
        self._ydl_factory = ydl_factory  # for tests; None -> real yt_dlp

    def _ydl(self, options: dict[str, Any]):
        if self._ydl_factory is not None:
            return self._ydl_factory(options)
        import yt_dlp

        return yt_dlp.YoutubeDL(options)

    def fetch_metadata(self, url: str) -> VideoMetadata:
        """Extract metadata without downloading media."""
        with self._ydl({"quiet": True, "skip_download": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        return info_to_metadata(info)

    def expand_source(self, url: str) -> list[str]:
        """Expand a playlist/channel URL into individual video watch URLs.

        Uses yt-dlp's flat extraction (no per-video network calls), so a channel
        with thousands of videos is enumerated cheaply. A plain video URL is
        returned unchanged as a single-element list.
        """
        options = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": "in_playlist",
            "skip_download": True,
        }
        with self._ydl(options) as ydl:
            info = ydl.extract_info(url, download=False)

        entries = info.get("entries") if isinstance(info, dict) else None
        if not entries:
            return [url]  # a single video

        urls: list[str] = []
        for entry in entries:
            if not entry:
                continue
            # Nested playlists (e.g. a channel's tabs) recurse one level.
            if entry.get("entries"):
                urls.extend(self._entry_urls(entry["entries"]))
            else:
                urls.extend(self._entry_urls([entry]))
        # De-duplicate while preserving order.
        seen: set[str] = set()
        unique = [u for u in urls if not (u in seen or seen.add(u))]
        logger.info("Expanded %s into %d video URL(s)", url, len(unique))
        return unique

    @staticmethod
    def _entry_urls(entries: list[dict[str, Any]]) -> list[str]:
        urls: list[str] = []
        for entry in entries:
            if not entry:
                continue
            if entry.get("url") and entry.get("ie_key") == "Youtube":
                # Flat extraction yields either a bare video id or a full URL.
                ref = entry["url"]
                if ref.startswith(("http://", "https://")):
                    urls.append(ref)
                else:
                    urls.append(f"https://www.youtube.com/watch?v={ref}")
            elif entry.get("id"):
                urls.append(f"https://www.youtube.com/watch?v={entry['id']}")
            elif entry.get("webpage_url"):
                urls.append(entry["webpage_url"])
        return urls

    def download(
        self, url: str, *, subtitle_langs: tuple[str, ...] = ("en", "en-US")
    ) -> DownloadResult:
        """Download the video + thumbnail + subtitles (resumable, deduped).

        Files are written under ``output_root/<youtube_id>/``. Downloads are
        resumable (``continuedl``): a re-run picks up partial ``.part`` files
        rather than starting over. Both manual and auto-generated subtitles are
        fetched when available.
        """
        video_id = extract_youtube_id(url) or "local"
        dest = self.output_root / video_id
        dest.mkdir(parents=True, exist_ok=True)

        options = {
            "quiet": True,
            "no_warnings": True,
            "outtmpl": str(dest / "%(id)s.%(ext)s"),
            # Prefer an H.264/AAC MP4 <=720p: plenty for TRIBE's visual encoder
            # and dramatically cheaper to download and decode.
            "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]/best",
            "merge_output_format": "mp4",
            # Resumable downloads + skip already-complete files.
            "continuedl": True,
            "retries": 5,
            "fragment_retries": 5,
            # Thumbnails.
            "writethumbnail": True,
            # Subtitles: manual first, fall back to auto-generated captions.
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": list(subtitle_langs),
            "subtitlesformat": "vtt/srt/best",
            "postprocessors": [
                {"key": "FFmpegThumbnailsConvertor", "format": "jpg"},
            ],
        }
        logger.info("Downloading %s", url)
        with self._ydl(options) as ydl:
            info = ydl.extract_info(url, download=True)

        metadata = info_to_metadata(info)
        video_path = self._locate(dest, video_id, (".mp4", ".mkv", ".webm"))
        thumb_path = self._locate(dest, video_id, (".jpg", ".png", ".webp"))
        subtitle_path = self._locate(dest, video_id, (".vtt", ".srt"))
        if video_path is None:
            raise FileNotFoundError(f"yt-dlp reported success but no video file in {dest}")
        logger.info("Downloaded %s -> %s", video_id, video_path.name)
        return DownloadResult(
            metadata=metadata,
            video_path=video_path,
            thumbnail_path=thumb_path,
            subtitle_path=subtitle_path,
        )

    @staticmethod
    def _locate(directory: Path, stem: str, extensions: tuple[str, ...]) -> Path | None:
        for ext in extensions:
            candidate = directory / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        # Fall back to any file with a matching extension.
        for ext in extensions:
            matches = sorted(directory.glob(f"*{ext}"))
            if matches:
                return matches[0]
        return None
