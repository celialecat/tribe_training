"""Analysis endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.analysis.data import TimeWindowSpec, load_analysis_data
from app.analysis.latent import export_analysis, list_methods, run_analysis
from app.db.base import session_scope
from app.schemas.analysis import (
    AnalysisExportRequest,
    AnalysisMethodResponse,
    AnalysisRunRequest,
    TimeWindowPayload,
)

router = APIRouter(tags=["analysis"])


@router.get("/methods", response_model=list[AnalysisMethodResponse])
def methods() -> list[dict[str, str]]:
    return list_methods()


@router.post("/run")
def run(body: AnalysisRunRequest) -> dict[str, Any]:
    try:
        with session_scope() as session:
            data = load_analysis_data(
                session,
                time_window=_time_window(body.time_window),
                region_names=body.region_names,
                normalization=body.normalization,
            )
            result = run_analysis(
                data,
                method=body.method,
                n_components=body.n_components,
                target=body.target,
                metrics=body.metrics,
                screening_threshold=body.screening_threshold,
                top_k=body.top_k,
                n_slices=body.n_slices,
            )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@router.post("/export")
def export(body: AnalysisExportRequest) -> FileResponse:
    try:
        with session_scope() as session:
            data = load_analysis_data(
                session,
                time_window=_time_window(body.time_window),
                region_names=body.region_names,
                normalization=body.normalization,
            )
            result = run_analysis(
                data,
                method=body.method,
                n_components=body.n_components,
                target=body.target,
                metrics=body.metrics,
                screening_threshold=body.screening_threshold,
                top_k=body.top_k,
                n_slices=body.n_slices,
            )
            path, filename, media_type = export_analysis(result, fmt=body.fmt)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(path, filename=filename, media_type=media_type)


def _time_window(payload: TimeWindowPayload | None) -> TimeWindowSpec:
    if payload is None:
        return TimeWindowSpec()
    return TimeWindowSpec(
        mode=payload.mode,
        index=payload.index,
        start=payload.start,
        end=payload.end,
    )
