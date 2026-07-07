from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.db.models.enums import BrainStatus, VideoStatus
from app.db.models.video import Video
from app.services.dataset_service import DatasetService


def test_dataset_modes_edit_roundtrip(tmp_path: Path) -> None:
    modes_path = tmp_path / "dataset_modes.yaml"
    modes_path.write_text(
        "education:\n  hint: learning\n  channels:\n    - https://example.com/a\n",
        encoding="utf-8",
    )
    service = DatasetService(modes_path=modes_path)
    assert service.load_modes()["education"]["channels"] == ["https://example.com/a"]
    service.add_channel("education", "https://example.com/b")
    assert "https://example.com/b" in service.get_mode("education")["channels"]
    service.remove_channel("education", "https://example.com/a")
    assert service.get_mode("education")["channels"] == ["https://example.com/b"]


def test_dataset_preview_falls_back_to_db(db_session: Session, tmp_path: Path) -> None:
    video = Video(
        youtube_id="abc123",
        url="https://youtube.com/watch?v=abc123",
        title="Sample",
        channel="Channel",
        view_count=100,
        like_count=10,
        duration_seconds=12.0,
        upload_date=datetime(2024, 1, 5, tzinfo=UTC),
        status=VideoStatus.ready,
        brain_status=BrainStatus.cached,
    )
    db_session.add(video)
    db_session.commit()

    service = DatasetService(modes_path=tmp_path / "missing.yaml")
    preview = service.preview("missing", count=10)
    assert preview["estimated_count"] == 1
    assert preview["publication_date_histogram"]["by_year"] == {"2024": 1}
    assert preview["estimated_storage_bytes"] > 0
