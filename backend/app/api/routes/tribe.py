from __future__ import annotations

from fastapi import APIRouter

from app.schemas.tribe import TribeRunRequest
from app.services.jobs import JobType, get_job_manager
from app.services.tribe_service import TribeService

router = APIRouter(prefix="/tribe", tags=["tribe"])
service = TribeService()
jobs = get_job_manager()


@router.post("/run")
def run(body: TribeRunRequest) -> dict:
    job_id = jobs.submit(
        JobType.tribe.value,
        lambda ctx: service.run(
            job=ctx, force=body.force, retry_failed=body.retry_failed, limit=body.limit
        ),
    )
    return {"job_id": job_id}


@router.get("/status")
def status() -> dict:
    return service.status()
