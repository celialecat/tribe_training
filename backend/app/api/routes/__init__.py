"""HTTP route handlers grouped by resource."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import (
    analysis,
    datasets,
    evaluation,
    monitoring,
    pipeline,
    prediction,
    tensors,
    training,
    tribe,
)


def register_routes(app: FastAPI) -> None:
    app.include_router(analysis.router, prefix="/api/analysis")
    app.include_router(datasets.router, prefix="/api")
    app.include_router(tribe.router, prefix="/api")
    app.include_router(tensors.router, prefix="/api")
    app.include_router(training.router, prefix="/api")
    app.include_router(evaluation.router, prefix="/api")
    app.include_router(prediction.router, prefix="/api")
    app.include_router(monitoring.router, prefix="/api")
    app.include_router(pipeline.router, prefix="/api")
