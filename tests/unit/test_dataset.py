"""Tests for the dataset pipeline: URL parsing, features, targets, ingestion."""

from __future__ import annotations

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


class _FakeYDL:
    """Context-manager stub mimicking yt_dlp.YoutubeDL for download tests."""

    def __init__(self, options: dict, dest_root: Path) -> None:
        self.options = options
        self.dest_root = dest_root

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url: str, download: bool = False) -> dict:
        vid = "test1234567"
        if download:
            d = self.dest_root / vid
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{vid}.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42fakevideo")
        return {
            "id": vid, "webpage_url": url, "title": "Fake Video",
            "view_count": 4200, "like_count": 300, "comment_count": 20,
            "channel_follower_count": 12_345, "duration": 42.0,
            "upload_date": "20240301", "width": 1280, "height": 720, "fps": 30,
        }


def test_dataset_builder_ingests_url(db_session, monkeypatch) -> None:
    from app.dataset.builder import DatasetBuilder
    from app.dataset.download import YouTubeDownloader

    settings = __import__("app.core.config", fromlist=["get_settings"]).get_settings()
    downloader = YouTubeDownloader(
        settings.processed_dir,
        ydl_factory=lambda opts: _FakeYDL(opts, settings.processed_dir),
    )
    # Avoid network transcript lookups.
    monkeypatch.setattr("app.dataset.builder.fetch_youtube_transcript", lambda *a, **k: None)

    builder = DatasetBuilder(db_session, settings=settings, downloader=downloader)
    video = builder.ingest_url("https://youtu.be/test1234567")
    db_session.commit()

    assert video.status == "ready"
    assert video.view_count == 4200
    assert video.title == "Fake Video"
    assert video.video_path is not None and video.video_path.endswith(".mp4")
    # Idempotency: second ingest returns the same row without re-downloading.
    again = builder.ingest_url("https://youtu.be/test1234567")
    assert again.id == video.id
