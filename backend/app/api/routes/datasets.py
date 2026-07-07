from __future__ import annotations

from fastapi import APIRouter, Query

from app.db.base import session_scope
from app.schemas.datasets import (
    DatasetBuildRequest,
    DatasetFilterPayload,
    DatasetPreviewRequest,
    DatasetValidateRequest,
    ModeChannelRequest,
)
from app.services.dataset_service import DatasetFilter, DatasetService
from app.services.jobs import JobType, get_job_manager

router = APIRouter(prefix="/datasets", tags=["datasets"])
service = DatasetService()
jobs = get_job_manager()


@router.get("/modes")
def list_modes() -> dict:
    return service.load_modes()


@router.post("/modes/{mode}/channels")
def add_channel(mode: str, body: ModeChannelRequest) -> dict:
    return service.add_channel(mode, body.url)


@router.delete("/modes/{mode}/channels")
def remove_channel(mode: str, body: ModeChannelRequest) -> dict:
    return service.remove_channel(mode, body.url)


@router.post("/preview")
def preview(body: DatasetPreviewRequest) -> dict:
    return service.preview(body.mode, filter=_filter(body.filter), count=body.count)


@router.post("/build")
def build(body: DatasetBuildRequest) -> dict:
    job_id = jobs.submit(
        JobType.download.value,
        lambda ctx: service.build(
            body.mode,
            filter=_filter(body.filter),
            count=body.count,
            force=body.force,
            job=ctx,
        ),
    )
    return {"job_id": job_id}


@router.post("/validate")
def validate(body: DatasetValidateRequest) -> dict:
    job_id = jobs.submit(
        JobType.validate.value,
        lambda ctx: service.validate(filter=_filter(body.filter), count=body.count, job=ctx),
    )
    return {"job_id": job_id}


@router.get("/report")
def report() -> dict:
    with session_scope() as session:
        return service.build_report(session)


@router.get("/videos")
def videos(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)) -> dict:
    with session_scope() as session:
        return service.list_videos(session, limit=limit, offset=offset)


def _filter(payload: DatasetFilterPayload | None) -> DatasetFilter:
    return DatasetFilter.model_validate(payload.model_dump() if payload is not None else {})
