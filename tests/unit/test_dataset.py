"""Tests for the dataset pipeline: URL parsing, features, targets, ingestion."""

from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from app.core.constants import NUM_TARGETS, TARGET_ORDER, Target
from app.dataset.download import extract_youtube_id, info_to_metadata
from app.dataset.features import FEATURE_DIM, build_feature_vector
from app.dataset.targets import compute_targets
from app.db.models.video import Video


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/abcdefghijk", "abcdefghijk"),
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://example.com/not-a-video", None),
    ],
)
def test_extract_youtube_id(url: str, expected: str | None) -> None:
    assert extract_youtube_id(url) == expected


def test_info_to_metadata_maps_yt_dlp_fields() -> None:
    info = {
        "id": "abc", "title": "T", "view_count": 100, "like_count": 10,
        "duration": 61.0, "channel_follower_count": 5000, "upload_date": "20240115",
        "width": 1920, "height": 1080,
    }
    meta = info_to_metadata(info)
    assert meta.youtube_id == "abc"
    assert meta.subscriber_count == 5000
    assert meta.duration_seconds == 61.0
    assert meta.upload_date.year == 2024 and meta.upload_date.month == 1


def test_feature_vector_shape_and_causality() -> None:
    video = Video(
        title="A catchy title here", description="x" * 500,
        subscriber_count=10_000, duration_seconds=120.0,
        upload_date=datetime(2024, 1, 1, 15, 30, tzinfo=UTC),
        width=1920, height=1080, fps=30.0, transcript="hello world",
    )
    vec = build_feature_vector(video)
    assert vec.shape == (FEATURE_DIM,)
    assert vec.dtype == np.float32
    # Cyclical encodings are bounded; aspect ratio ~16:9.
    assert np.all(np.abs(vec[2:6]) <= 1.0)
    assert vec[10] == pytest.approx(1920 / 1080, rel=1e-4)


def test_targets_engagement_observed_views_masked_by_default() -> None:
    video = Video(
        view_count=1000, like_count=100, comment_count=50,
        upload_date=datetime.now(UTC) - timedelta(days=90),
    )
    sample = compute_targets(video)  # approximate_view_curve=False
    d = {t: (sample.values[i], sample.mask[i]) for i, t in enumerate(TARGET_ORDER)}
    # Engagement observed = 150/1000 = 0.15.
    assert d[Target.engagement][1] == 1.0
    assert d[Target.engagement][0] == pytest.approx(0.15)
    # 30d observable (>=30d old); 7d masked without approximation.
    assert d[Target.log_views_30d][1] == 1.0
    assert d[Target.log_views_7d][1] == 0.0
    # Retention never inferred.
    assert d[Target.retention][1] == 0.0


def test_targets_approximate_curve_fills_all_view_targets() -> None:
    video = Video(
        view_count=100_000, like_count=1000, comment_count=200,
        upload_date=datetime.now(UTC) - timedelta(days=10),
    )
    sample = compute_targets(video, approximate_view_curve=True, retention=0.42)
    mask = {t: sample.mask[i] for i, t in enumerate(TARGET_ORDER)}
    assert mask[Target.log_views_7d] == 1.0
    assert mask[Target.log_views_30d] == 1.0
    assert mask[Target.virality] == 1.0
    assert mask[Target.retention] == 1.0
    assert sample.values.shape == (NUM_TARGETS,)


FFMPEG = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")


def _write_real_mp4(path: Path, *, seconds: int = 3, with_audio: bool = True) -> None:
    """Generate a genuine, decodable H.264/AAC mp4 with ffmpeg."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=320x240:rate=30"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(path)]
    subprocess.run(cmd, capture_output=True, check=True)


class _FakeYDL:
    """Context-manager stub mimicking yt_dlp.YoutubeDL, producing a REAL video."""

    def __init__(self, options: dict, dest_root: Path, *, valid: bool = True) -> None:
        self.options = options
        self.dest_root = dest_root
        self.valid = valid

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url: str, download: bool = False) -> dict:
        vid = "test1234567"
        if download:
            out = self.dest_root / vid / f"{vid}.mp4"
            if self.valid:
                _write_real_mp4(out, seconds=3)
            else:  # non-empty but corrupt file (passes size check, fails ffprobe)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 4096)
        return {
            "id": vid, "webpage_url": url, "title": "Fake Video",
            "view_count": 4200, "like_count": 300, "comment_count": 20,
            "channel_follower_count": 12_345, "duration": 3.0,
            "upload_date": "20240301", "width": 320, "height": 240, "fps": 30,
        }


def _builder_with_fake_ydl(db_session, monkeypatch, *, valid: bool = True):
    from app.core.config import get_settings
    from app.dataset.builder import DatasetBuilder
    from app.dataset.download import YouTubeDownloader

    settings = get_settings()
    downloader = YouTubeDownloader(
        settings.processed_dir,
        ydl_factory=lambda opts: _FakeYDL(opts, settings.processed_dir, valid=valid),
    )
    monkeypatch.setattr("app.dataset.builder.fetch_youtube_transcript", lambda *a, **k: None)
    return DatasetBuilder(db_session, settings=settings, downloader=downloader)


@requires_ffmpeg
def test_dataset_builder_ingests_and_validates_real_video(db_session, monkeypatch) -> None:
    builder = _builder_with_fake_ydl(db_session, monkeypatch, valid=True)
    video = builder.ingest_url("https://youtu.be/test1234567")
    db_session.commit()

    assert video.status == "ready"
    assert video.view_count == 4200
    assert video.video_path is not None and video.video_path.endswith(".mp4")
    # Validation ran and passed, and is recorded on the row.
    assert video.validation is not None
    assert video.validation["is_valid"] is True
    assert "frames_decodable" in {c["name"] for c in video.validation["checks"]}
    # Idempotency: second ingest returns the same row.
    again = builder.ingest_url("https://youtu.be/test1234567")
    assert again.id == video.id


@requires_ffmpeg
def test_dataset_builder_rejects_corrupt_video(db_session, monkeypatch) -> None:
    from app.dataset.builder import IngestionError

    builder = _builder_with_fake_ydl(db_session, monkeypatch, valid=False)
    with pytest.raises(IngestionError):
        builder.ingest_url("https://youtu.be/test1234567")
    db_session.commit()

    video = builder.repo.get_by_youtube_id("test1234567")
    assert video.status == "rejected"
    assert video.validation is not None and video.validation["is_valid"] is False
    assert "ffprobe" in video.validation["failed"]


class _FakePlaylistYDL:
    """Fake yt-dlp returning a flat playlist of entries for expand_source."""

    def __init__(self, options: dict) -> None:
        self.options = options

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url: str, download: bool = False) -> dict:
        return {
            "_type": "playlist",
            "entries": [
                {"id": "aaaaaaaaaaa", "ie_key": "Youtube", "url": "aaaaaaaaaaa"},
                {"id": "bbbbbbbbbbb", "ie_key": "Youtube", "url": "bbbbbbbbbbb"},
                {"id": "aaaaaaaaaaa", "ie_key": "Youtube", "url": "aaaaaaaaaaa"},  # dup
            ],
        }


def test_expand_source_flattens_and_dedupes_playlist() -> None:
    from app.dataset.download import YouTubeDownloader

    dl = YouTubeDownloader(Path("/tmp"), ydl_factory=lambda opts: _FakePlaylistYDL(opts))
    urls = dl.expand_source("https://www.youtube.com/playlist?list=PLxxx")
    assert len(urls) == 2  # duplicate collapsed
    assert all("watch?v=" in u for u in urls)


def test_expand_source_single_video_returns_itself() -> None:
    from app.dataset.download import YouTubeDownloader

    class _SingleYDL(_FakePlaylistYDL):
        def extract_info(self, url: str, download: bool = False) -> dict:
            return {"id": "ccccccccccc"}  # no 'entries'

    dl = YouTubeDownloader(Path("/tmp"), ydl_factory=lambda opts: _SingleYDL(opts))
    url = "https://youtu.be/ccccccccccc"
    assert dl.expand_source(url) == [url]


def test_parse_subtitle_file_vtt(tmp_path: Path) -> None:
    from app.dataset.transcript import parse_subtitle_file

    vtt = tmp_path / "sub.en.vtt"
    vtt.write_text(
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:03.000\nHello world\n\n"
        "00:00:03.000 --> 00:00:05.000\nHello world\n\n"  # duplicate (rolling caption)
        "00:00:05.000 --> 00:00:07.000\n<c>Second line</c>\n"
    )
    text = parse_subtitle_file(vtt)
    assert text == "Hello world Second line"  # dedup + tag stripping


def test_dataset_report_counts(db_session) -> None:
    from app.dataset.report import build_dataset_report
    from app.db.models.enums import VideoStatus

    # A valid video with full metadata + subtitle.
    db_session.add(Video(
        youtube_id="rdy00000001", status=VideoStatus.ready, channel="C", channel_id="c1",
        view_count=100, upload_date=datetime(2024, 1, 1, tzinfo=UTC),
        transcript="hi", subtitle_path="x/s.vtt",
    ))
    # A ready video missing metadata + subtitles.
    db_session.add(Video(youtube_id="rdy00000002", status=VideoStatus.ready))
    # A rejected (corrupt) video.
    db_session.add(Video(
        youtube_id="rej00000001", status=VideoStatus.rejected,
        validation={"is_valid": False, "failed": ["ffprobe"]},
    ))
    # A failed download.
    db_session.add(Video(youtube_id="fail0000001", status=VideoStatus.failed))
    db_session.flush()

    report = build_dataset_report(db_session)
    assert report.total_videos == 4
    assert report.valid_videos == 2
    assert report.rejected_videos == 1
    assert report.corrupted_files == 1
    assert report.failed_downloads == 1
    assert report.duplicate_video_ids == 0
    assert report.missing_subtitles == 1  # only rdy00000002
    assert report.missing_metadata == 1
    assert report.rejection_reasons.get("ffprobe") == 1
    assert "Dataset Validation Report" in report.render()
