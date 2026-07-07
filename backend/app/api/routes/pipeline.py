from __future__ import annotations

from fastapi import APIRouter

from app.schemas.pipeline import PipelineRunRequest
from app.services.jobs import JobType, get_job_manager
from app.services.pipeline_service import PipelineService

router = APIRouter(prefix="/pipeline", tags=["pipeline"])
service = PipelineService()
jobs = get_job_manager()


@router.post("/run")
def run(body: PipelineRunRequest) -> dict:
    job_id = jobs.submit(
        JobType.pipeline.value,
        lambda ctx: service.run(
            mode=body.mode, n_videos=body.n_videos, horizon_days=body.horizon_days, job=ctx
        ),
    )
    return {"job_id": job_id}


@router.get("/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        return {}
    return job.snapshot()
