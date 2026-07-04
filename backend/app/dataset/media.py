"""Media extraction: audio (ffmpeg), frames (OpenCV), probing (ffprobe).

TRIBE v2 consumes the video file directly, so frame/audio extraction here is for
(a) lightweight previews/thumbnails in the dashboard and (b) fallback feature
computation. Extraction is defensive: a missing ffmpeg or an unreadable stream
degrades gracefully rather than aborting ingestion.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from app.core.logging import get_logger
from app.dataset.schemas import MediaArtifacts

logger = get_logger(__name__)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe_video(video_path: Path) -> dict[str, float | int]:
    """Return {duration, fps, width, height} via ffprobe (best-effort)."""
    if shutil.which("ffprobe") is None:
        return {}
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
        "-of", "json", str(video_path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
        data = json.loads(out.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
        logger.warning("ffprobe failed for %s: %s", video_path.name, exc)
        return {}

    stream = (data.get("streams") or [{}])[0]
    result: dict[str, float | int] = {}
    if "width" in stream:
        result["width"] = int(stream["width"])
    if "height" in stream:
        result["height"] = int(stream["height"])
    if rate := stream.get("avg_frame_rate"):
        num, _, den = rate.partition("/")
        with_den = float(den) if den else 0.0
        result["fps"] = float(num) / with_den if with_den else 0.0
    if dur := data.get("format", {}).get("duration"):
        result["duration"] = float(dur)
    return result


def extract_audio(video_path: Path, out_path: Path, *, sample_rate: int = 16_000) -> Path | None:
    """Extract a mono WAV suitable for audio models. Returns None on failure."""
    if shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg not found; skipping audio extraction.")
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "wav", str(out_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=600, check=True)
    except subprocess.SubprocessError as exc:
        logger.warning("Audio extraction failed for %s: %s", video_path.name, exc)
        return None
    return out_path if out_path.exists() else None


def extract_frames(
    video_path: Path, out_dir: Path, *, num_frames: int = 16, max_side: int = 512
) -> list[Path]:
    """Sample ``num_frames`` evenly-spaced frames, resized to ``max_side``.

    Uses OpenCV for portability. Returns the written frame paths (possibly
    fewer than requested for very short clips).
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV unavailable; skipping frame extraction.")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("OpenCV could not open %s", video_path)
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        return []

    indices = _even_indices(total, num_frames)
    written: list[Path] = []
    for order, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = _resize_max_side(cv2, frame, max_side)
        path = out_dir / f"frame_{order:03d}.jpg"
        cv2.imwrite(str(path), frame)
        written.append(path)
    cap.release()
    logger.debug("Extracted %d/%d frames from %s", len(written), len(indices), video_path.name)
    return written


def _even_indices(total: int, count: int) -> list[int]:
    if count >= total:
        return list(range(total))
    step = total / count
    return [min(total - 1, int(i * step + step / 2)) for i in range(count)]


def _resize_max_side(cv2, frame, max_side: int):  # type: ignore[no-untyped-def]
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame
    scale = max_side / longest
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


class MediaExtractor:
    """Orchestrates audio + frame extraction into a video's artefact directory."""

    def __init__(self, *, num_frames: int = 16, audio_sr: int = 16_000) -> None:
        self.num_frames = num_frames
        self.audio_sr = audio_sr

    def extract(self, video_path: Path, root: Path) -> MediaArtifacts:
        root.mkdir(parents=True, exist_ok=True)
        artifacts = MediaArtifacts(root=root, video_path=video_path)

        artifacts.audio_path = extract_audio(
            video_path, root / "audio.wav", sample_rate=self.audio_sr
        )
        frames_dir = root / "frames"
        artifacts.frame_paths = extract_frames(
            video_path, frames_dir, num_frames=self.num_frames
        )
        artifacts.frames_dir = frames_dir if artifacts.frame_paths else None
        return artifacts
