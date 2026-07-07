from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.analysis.data import AnalysisData, load_analysis_data
from app.analysis.latent import CONDITIONAL_VARIANCE_JUSTIFICATION, run_analysis
from app.core.config import get_settings
from app.core.constants import FSAVERAGE5_VERTICES
from app.db.models.embedding import Embedding
from app.db.models.enums import BrainStatus, EmbeddingKind, VideoStatus
from app.db.models.video import Video
from app.explainability.brain_regions import default_region_labels


def test_load_analysis_data_returns_structured_error_for_too_few_samples(
    db_session: Session,
) -> None:
    settings = get_settings()
    settings.ensure_directories()
    _create_cached_videos(db_session, settings.data_dir, count=2)

    data = load_analysis_data(db_session)

    assert data.error is not None
    assert data.error["code"] == "insufficient_samples"
    assert data.error["n_samples"] == 2


def test_methods_cover_all_latent_modes() -> None:
    data = synthetic_analysis_data()

    pca = run_analysis(data, method="pca", n_components=3)
    assert pca["components"]
    assert len(pca["explained_variance_ratio"]) == 3
    assert np.asarray(pca["projections"]).shape == (30, 3)

    spca = run_analysis(data, method="supervised_pca", n_components=2, top_k=10)
    assert spca["screened_feature_indices"]
    assert spca["region_importance"]

    pls = run_analysis(data, method="pls", n_components=2)
    assert pls["fit_r2"] is not None
    assert pls["components"]

    cca = run_analysis(data, method="cca", n_components=2)
    assert cca["metrics"] == ["likes", "views", "comments", "engagement"]
    assert len(cca["canonical_correlations"]) == 2

    sir = run_analysis(data, method="conditional_variance", n_components=2)
    assert sir["method_justification"] == CONDITIONAL_VARIANCE_JUSTIFICATION
    assert len(sir["edr_directions"]) == data.n_features
    assert len(sir["between_slice_fraction"]) == 2


def test_components_are_clamped_to_rank() -> None:
    data = synthetic_analysis_data()
    result = run_analysis(data, method="pca", n_components=999)

    assert result["clamped"] is True
    assert result["n_components"] == 29


def test_too_few_samples_returns_structured_error() -> None:
    data = synthetic_analysis_data(n_samples=2)
    result = run_analysis(data, method="pca", n_components=2)

    assert result["error"]["code"] == "insufficient_samples"


def test_analysis_router_methods_and_run_and_export(
    tmp_settings: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api.routes import analysis as analysis_route
    from app.main import create_app

    app = create_app()
    client = TestClient(app)

    synthetic = synthetic_analysis_data()
    monkeypatch.setattr(analysis_route, "load_analysis_data", lambda *args, **kwargs: synthetic)

    methods = client.get("/api/analysis/methods")
    assert methods.status_code == 200
    payload = methods.json()
    assert any(item["name"] == "pca" for item in payload)
    assert any(item["name"] == "conditional_variance_justification" for item in payload)

    run_body = {
        "method": "pca",
        "n_components": 2,
        "target": "log_likes",
        "normalization": "none",
    }
    run_response = client.post("/api/analysis/run", json=run_body)
    assert run_response.status_code == 200
    assert run_response.json()["method"] == "pca"

    for fmt, content_type in [
        ("csv", "text/csv"),
        ("npz", "application/zip"),
        ("pt", "application/octet-stream"),
    ]:
        export_response = client.post(
            "/api/analysis/export",
            json={**run_body, "fmt": fmt},
        )
        assert export_response.status_code == 200
        assert export_response.headers["content-type"].startswith(content_type)


def synthetic_analysis_data(*, n_samples: int = 30, n_features: int = 50) -> AnalysisData:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    signal = x[:, 0] * 0.8 + x[:, 1] * 0.2 + rng.normal(scale=0.1, size=n_samples)
    log_likes = signal.astype(np.float32)
    likes = np.maximum(np.power(10.0, log_likes) - 1.0, 0.1)
    views = np.maximum(likes * 20.0 + rng.normal(scale=5.0, size=n_samples), 1.0)
    comments = np.maximum(likes * 0.1 + rng.normal(scale=0.5, size=n_samples), 0.0)
    engagement = np.clip((likes + comments) / np.maximum(views, 1.0), 0.0, 1.0)
    virality = np.clip(rng.uniform(0.0, 1.0, size=n_samples), 0.0, 1.0)
    labels, names = default_region_labels(FSAVERAGE5_VERTICES)
    return AnalysisData(
        X=x,
        y={
            "log_likes": log_likes.astype(np.float32),
            "likes": likes.astype(np.float32),
            "views": views.astype(np.float32),
            "comments": comments.astype(np.float32),
            "engagement": engagement.astype(np.float32),
            "virality": virality.astype(np.float32),
        },
        video_ids=list(range(1, n_samples + 1)),
        feature_names=[f"v{i}" for i in range(n_features)],
        vertex_indices=np.arange(n_features, dtype=np.int64),
        vertex_labels=labels,
        region_names=names,
        full_vertex_count=FSAVERAGE5_VERTICES,
        time_window={"mode": "mean", "index": None, "start": None, "end": None},
        normalization="none",
    )


def _create_cached_videos(session: Session, data_dir: Path, *, count: int) -> None:
    settings = get_settings()
    producer = f"tribev2:{settings.tribe_model_id}"
    for idx in range(count):
        video = Video(
            youtube_id=f"video{idx}",
            url=f"https://example.com/{idx}",
            title=f"Video {idx}",
            channel="Channel",
            view_count=100 + idx,
            like_count=10 + idx,
            comment_count=2 + idx,
            duration_seconds=30.0,
            upload_date=None,
            status=VideoStatus.ready,
            brain_status=BrainStatus.cached,
        )
        session.add(video)
        session.flush()
        path = data_dir / f"activity_{idx}.npz"
        tensor = np.ones((4, FSAVERAGE5_VERTICES), dtype=np.float32) * (idx + 1)
        np.savez_compressed(
            path,
            activity=tensor,
            tr_seconds=np.float32(1.49),
            offset=np.float32(0.0),
        )
        session.add(
            Embedding(
                video_id=video.id,
                kind=EmbeddingKind.brain_activity,
                producer=producer,
                path=str(path),
                shape=[4, FSAVERAGE5_VERTICES],
                dtype="float32",
                num_timesteps=4,
                num_features=FSAVERAGE5_VERTICES,
            )
        )
    session.commit()
