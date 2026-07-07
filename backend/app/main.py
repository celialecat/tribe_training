"""FastAPI application factory for the dashboard backend."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.types import Scope

from app.api.routes import register_routes
from app.core.config import REPO_ROOT, get_settings
from app.core.logging import setup_logging


class SPARootFiles(StaticFiles):
    async def get_response(self, path: str, scope: Scope) -> Any:  # type: ignore[override]
        response = await super().get_response(path, scope)
        if response.status_code == 404:
            return await super().get_response("index.html", scope)
        return response


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(level=settings.log_level)
    app = FastAPI(title="YouTube Success Predictor", version="0.1.0")

    origins = list(dict.fromkeys([*settings.cors_origins, "http://localhost:5173"]))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_routes(app)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    frontend_dist = REPO_ROOT / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", SPARootFiles(directory=str(frontend_dist), html=True), name="frontend")

    return app


app = create_app()
