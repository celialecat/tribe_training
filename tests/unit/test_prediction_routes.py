from __future__ import annotations

from fastapi.testclient import TestClient


def test_prediction_horizon_route_returns_default_horizon(tmp_settings: object) -> None:
    from app.main import create_app

    app = create_app()
    client = TestClient(app)

    response = client.get("/api/prediction/horizon")

    assert response.status_code == 200
    assert response.json() == {"horizon_days": 30}
