"""Tests for the TRIBE integration: extractor wiring, cache, service, dataset.

These exercise the real code paths with a *fake TRIBE model* injected at the
import boundary — we never mock the platform's own logic, only stand in for the
1B-parameter upstream weights that aren't present in CI.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from app.core.constants import NUM_TARGETS
from app.db.models.enums import BrainStatus, VideoStatus
from app.db.models.video import Video
from app.tribe.extractor import TribeExtractor, TribeNotInstalled
from app.tribe.types import BrainActivity

TR, V = 40, 200  # small brain geometry for fast tests


class _FakeEvents:
    pass


class _FakeTribeModel:
    """Stands in for tribev2.TribeModel with the documented API surface."""

    def __init__(self, n_t: int = TR, n_v: int = V) -> None:
        self.n_t, self.n_v = n_t, n_v

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        cache_folder: str,
        device: str = "auto",
        config_update: dict | None = None,
    ):
        return cls()

    def get_events_dataframe(self, **kwargs):
        assert "video_path" in kwargs
        return _FakeEvents()

    def predict(self, events):
        rng = np.random.default_rng(0)
        preds = rng.standard_normal((self.n_t, self.n_v)).astype(np.float32)
        return preds, None


@pytest.fixture
def fake_extractor(monkeypatch, tmp_path):
    monkeypatch.setattr(TribeExtractor, "_import_tribe", staticmethod(lambda: _FakeTribeModel))
    ext = TribeExtractor(model_id="facebook/tribev2", device="cpu")
    ext.cache_folder = tmp_path / "weights"
    return ext


def test_extractor_raises_clear_error_when_tribe_missing() -> None:
    ext = TribeExtractor(device="cpu")
    # tribev2 is not installed in the test environment.
    assert ext.is_available() is False
    with pytest.raises(TribeNotInstalled) as excinfo:
        ext.extract(__file__)  # any existing path; import fails before use
    assert "make install-tribe" in str(excinfo.value)


def test_extractor_returns_brain_activity(fake_extractor, tmp_path) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    activity = fake_extractor.extract(video, transcript="hello world")
    assert isinstance(activity, BrainActivity)
    assert activity.array.shape == (TR, V)
    assert activity.tr_seconds > 0
    assert activity.hemodynamic_offset_seconds == 5.0


def test_to_numpy_coerces_tensor_and_dataframe() -> None:
    import torch

    arr = TribeExtractor._to_numpy(torch.zeros(3, 4))
    assert arr.shape == (3, 4) and isinstance(arr, np.ndarray)


def _make_ready_video(session, channel_id: str, days_old: int, views: int) -> Video:
    v = Video(
        youtube_id=f"vid{channel_id}{days_old}{views}"[:11].ljust(11, "0"),
        source="youtube", status=VideoStatus.ready, channel_id=channel_id,
        title="t", subscriber_count=1000, duration_seconds=100.0,
        view_count=views, like_count=views // 10, comment_count=views // 50,
        upload_date=datetime.now(UTC) - timedelta(days=days_old),
        video_path="x/v.mp4",
    )
    session.add(v)
    session.flush()
    return v


def test_brain_cache_round_trip(db_session) -> None:
    from app.tribe.cache import BrainCache

    video = _make_ready_video(db_session, "chanA", 40, 10_000)
    src = db_session  # unused placeholder
    del src
    cache = BrainCache(db_session, model_id="facebook/tribev2")
    activity = BrainActivity(
        array=np.random.default_rng(1).standard_normal((TR, V)).astype(np.float32),
        tr_seconds=1.49, hemodynamic_offset_seconds=5.0, model_id="facebook/tribev2",
    )
    # Use a real file as the hash source.
    import tempfile
    from pathlib import Path

    f = Path(tempfile.mkstemp(suffix=".mp4")[1])
    f.write_bytes(b"video-bytes")
    cache.put(video, activity, source_path=f)
    assert cache.has(video)
    assert video.brain_status == BrainStatus.cached
    loaded = cache.get(video)
    assert loaded is not None
    np.testing.assert_allclose(loaded.array, activity.array, rtol=1e-6)


def test_brain_activity_service_computes_and_caches(db_session, monkeypatch, tmp_path) -> None:
    from app.core.config import get_settings
    from app.tribe.service import BrainActivityService

    settings = get_settings()
    # Real media file so the cache can hash it.
    vid_dir = settings.processed_dir / "x"
    vid_dir.mkdir(parents=True, exist_ok=True)
    (vid_dir / "v.mp4").write_bytes(b"the-video")

    video = _make_ready_video(db_session, "chanA", 40, 5000)
    monkeypatch.setattr(
        TribeExtractor, "_import_tribe", staticmethod(lambda: _FakeTribeModel)
    )
    ext = TribeExtractor(model_id="facebook/tribev2", device="cpu")
    service = BrainActivityService(db_session, extractor=ext, settings=settings)

    activity = service.get_or_compute(video)
    assert activity.array.shape == (TR, V)
    # Second call is served from cache (extractor would fail if re-run without file change).
    again = service.get_or_compute(video)
    np.testing.assert_allclose(again.array, activity.array)


def test_make_splits_is_deterministic_and_channel_grouped(db_session) -> None:
    from app.tribe.dataset import make_splits

    for i in range(6):
        _make_ready_video(db_session, channel_id=f"c{i % 2}", days_old=40, views=1000 + i)
    db_session.flush()

    a = make_splits(db_session, val_fraction=0.2, test_fraction=0.2, seed=7)
    b = make_splits(db_session, val_fraction=0.2, test_fraction=0.2, seed=7)
    assert a == b  # deterministic

    # All videos of a channel share a split.
    by_channel: dict[str, set] = {}
    for vid, split in a.items():
        video = db_session.get(Video, vid)
        by_channel.setdefault(video.channel_id, set()).add(split)
    assert all(len(splits) == 1 for splits in by_channel.values())


def test_brain_dataset_getitem_shapes(db_session) -> None:
    from app.core.config import get_settings
    from app.tribe.cache import BrainCache
    from app.tribe.dataset import BrainDataset, make_splits

    settings = get_settings()
    cache = BrainCache(db_session, model_id="facebook/tribev2")
    import tempfile
    from pathlib import Path

    for i in range(3):
        video = _make_ready_video(db_session, channel_id=f"c{i}", days_old=40, views=10_000 + i)
        f = Path(tempfile.mkstemp(suffix=".mp4")[1])
        f.write_bytes(f"vid{i}".encode())
        cache.put(
            video,
            BrainActivity(
                array=np.random.default_rng(i).standard_normal((30, V)).astype(np.float32),
                tr_seconds=1.49, hemodynamic_offset_seconds=5.0, model_id="facebook/tribev2",
            ),
            source_path=f,
        )
    db_session.flush()

    assignment = dict.fromkeys(make_splits(db_session, val_fraction=0, test_fraction=0), "train")
    ds = BrainDataset.from_database(
        db_session, split="train", assignment=assignment,
        model_id="facebook/tribev2", settings=settings, max_time_steps=64,
    )
    assert len(ds) == 3
    item = ds[0]
    assert item["brain"].shape == (64, V)        # padded up from 30
    assert item["pad_mask"].shape == (64,)
    assert bool(item["pad_mask"][40]) is True     # padding region flagged
    assert item["features"].shape == (12,)
    assert item["targets"].shape == (NUM_TARGETS,)
    assert item["target_mask"].shape == (NUM_TARGETS,)
