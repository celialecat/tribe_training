from __future__ import annotations

from fastapi.testclient import TestClient


def test_app_boot_and_empty_get_endpoints(tmp_settings: object) -> None:
    from app.main import create_app

    app = create_app()
    client = TestClient(app)

    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/api/tribe/status").status_code == 200
    assert client.get("/api/tensors/").status_code == 200
    assert client.get("/api/datasets/modes").status_code == 200
