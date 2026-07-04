"""Asset validation — prove a downloaded video is actually usable.

Every video must pass this gate before it is admitted to the dataset. A video
that yt-dlp "successfully" downloads can still be truncated, corrupted, audio-
less, or wrongly muxed; training on such files silently poisons the model. The
validator runs a battery of independent checks and returns a structured,
serialisable :class:`ValidationResult` recording each check's outcome.

Checks
------
- ``file_exists``       : the media file exists and is non-empty
- ``ffprobe``           : ffprobe parses the container
- ``video_stream``      : a decodable video stream is present
- ``audio_stream``      : a decodable audio stream is present
- ``codec_supported``   : video codec is in the supported allow-list
- ``duration_matches``  : container duration is within tolerance of metadata
- ``frame_count``       : reported/nominal frame count is plausible (> 0)
- ``frames_decodable``  : a sample of frames decodes without error (corruption)

The validator shells out to the official ffprobe/ffmpeg binaries and uses
OpenCV for frame sampling; all are treated as hard requirements on ingestion
hosts (missing tools are reported as failed checks, never silently skipped).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from app.core.logging import get_logger
from app.dataset.schemas import VideoMetadata

logger = get_logger(__name__)

# Video codecs we can reliably decode/feed downstream. Extendable via config.
SUPPORTED_VIDEO_CODECS: frozenset[str] = frozenset(
    {"h264", "hevc", "h265", "vp8", "vp9", "av1", "mpeg4"}
)
# Minimum acceptable media file size (guards against 0-byte / stub files).
MIN_FILE_BYTES = 1024
# Duration agreement tolerance: max(abs_seconds, rel_fraction * expected).
DURATION_ABS_TOLERANCE_S = 2.0
DURATION_REL_TOLERANCE = 0.05


@dataclass(slots=True)
class CheckResult:
    """Outcome of a single validation check."""

    name: str
    passed: bool
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(slots=True)
class ValidationResult:
    """Aggregate validation outcome for one video."""

    checks: list[CheckResult] = field(default_factory=list)
    probe: dict[str, object] = field(default_factory=dict)

    def add(self, name: str, passed: bool, detail: str = "") -> bool:
        self.checks.append(CheckResult(name=name, passed=passed, detail=detail))
        return passed

    @property
    def is_valid(self) -> bool:
        return all(check.passed for check in self.checks) and len(self.checks) > 0

    def failed_checks(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed]

    def reason(self) -> str:
        failed = [f"{c.name}: {c.detail}" for c in self.checks if not c.passed]
        return "; ".join(failed) if failed else "ok"

    def as_dict(self) -> dict[str, object]:
        return {
            "is_valid": self.is_valid,
            "failed": self.failed_checks(),
            "reason": self.reason(),
            "checks": [c.as_dict() for c in self.checks],
            "probe": self.probe,
        }


class VideoValidator:
    """Validate downloaded video assets before dataset admission."""

    def __init__(
        self,
        *,
        sample_frames: int = 8,
        supported_codecs: frozenset[str] = SUPPORTED_VIDEO_CODECS,
        duration_abs_tol_s: float = DURATION_ABS_TOLERANCE_S,
        duration_rel_tol: float = DURATION_REL_TOLERANCE,
        require_audio: bool = True,
    ) -> None:
        self.sample_frames = sample_frames
        self.supported_codecs = supported_codecs
        self.duration_abs_tol_s = duration_abs_tol_s
        self.duration_rel_tol = duration_rel_tol
        self.require_audio = require_audio

    def validate(
        self, video_path: str | Path, metadata: VideoMetadata | None = None
    ) -> ValidationResult:
        """Run all checks and return the aggregated result."""
        result = ValidationResult()
        path = Path(video_path)

        # 1. file exists and is non-empty
        if not result.add(
            "file_exists",
            path.exists() and path.stat().st_size >= MIN_FILE_BYTES,
            "" if path.exists() else f"missing: {path}",
        ):
            return result  # nothing else is meaningful without a file

        # 2. ffprobe parses the container
        probe = self._ffprobe(path)
        if not result.add("ffprobe", bool(probe), "ffprobe returned no data"):
            return result
        result.probe = self._summarise_probe(probe)

        streams = probe.get("streams", [])
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

        # 3. video stream present
        result.add("video_stream", bool(video_streams), "no video stream")
        # 4. audio stream present (optional per config)
        if self.require_audio:
            result.add("audio_stream", bool(audio_streams), "no audio stream")

        # 5. codec supported
        codec = video_streams[0].get("codec_name", "").lower() if video_streams else ""
        result.add(
            "codec_supported",
            codec in self.supported_codecs,
            f"codec {codec!r} not in {sorted(self.supported_codecs)}",
        )

        # 6. duration agrees with metadata
        self._check_duration(result, probe, metadata)

        # 7. frame count plausible
        nb_frames = self._frame_count(video_streams[0]) if video_streams else 0
        result.add("frame_count", nb_frames > 0, f"nb_frames={nb_frames}")

        # 8. sampled frames decode without error (real corruption detection)
        self._check_frames_decodable(result, path)

        if not result.is_valid:
            logger.warning("Validation failed for %s: %s", path.name, result.reason())
        return result

    # ------------------------------------------------------------- ffprobe --
    @staticmethod
    def _ffprobe(path: Path) -> dict:
        if shutil.which("ffprobe") is None:
            return {}
        cmd = [
            "ffprobe", "-v", "error", "-show_streams", "-show_format",
            "-of", "json", str(path),
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
            return json.loads(out.stdout)
        except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
            logger.debug("ffprobe failed for %s: %s", path.name, exc)
            return {}

    @staticmethod
    def _summarise_probe(probe: dict) -> dict[str, object]:
        streams = probe.get("streams", [])
        video = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
        fmt = probe.get("format", {})
        return {
            "duration": float(fmt["duration"]) if fmt.get("duration") else None,
            "video_codec": video.get("codec_name"),
            "audio_codec": audio.get("codec_name"),
            "width": video.get("width"),
            "height": video.get("height"),
            "n_streams": len(streams),
        }

    @staticmethod
    def _frame_count(video_stream: dict) -> int:
        nb = video_stream.get("nb_frames")
        if nb and str(nb).isdigit() and int(nb) > 0:
            return int(nb)
        # Fall back to duration * avg_frame_rate when nb_frames is absent.
        rate = video_stream.get("avg_frame_rate", "0/0")
        try:
            num, _, den = rate.partition("/")
            fps = float(num) / float(den) if den and float(den) else 0.0
            dur = float(video_stream.get("duration", 0.0) or 0.0)
            return int(fps * dur)
        except (ValueError, ZeroDivisionError):
            return 0

    def _check_duration(
        self, result: ValidationResult, probe: dict, metadata: VideoMetadata | None
    ) -> None:
        expected = metadata.duration_seconds if metadata else None
        actual = None
        fmt_dur = probe.get("format", {}).get("duration")
        if fmt_dur:
            actual = float(fmt_dur)
        if expected is None or actual is None:
            # Can't compare — pass but record why (metadata gap handled elsewhere).
            result.add("duration_matches", True, "no reference duration to compare")
            return
        tol = max(self.duration_abs_tol_s, self.duration_rel_tol * expected)
        ok = abs(actual - expected) <= tol
        result.add(
            "duration_matches", ok,
            f"expected {expected:.1f}s, got {actual:.1f}s (tol {tol:.1f}s)",
        )

    def _check_frames_decodable(self, result: ValidationResult, path: Path) -> None:
        """Decode a sample of frames with OpenCV to detect corruption."""
        try:
            import cv2
        except ImportError:
            result.add("frames_decodable", False, "OpenCV unavailable")
            return

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            result.add("frames_decodable", False, "OpenCV could not open file")
            return
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            cap.release()
            result.add("frames_decodable", False, "no decodable frames reported")
            return

        n = min(self.sample_frames, total)
        indices = [min(total - 1, int(i * total / n)) for i in range(n)]
        decoded = 0
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                decoded += 1
        cap.release()
        result.add(
            "frames_decodable",
            decoded == len(indices),
            f"decoded {decoded}/{len(indices)} sampled frames",
        )
