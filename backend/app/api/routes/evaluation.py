from __future__ import annotations

from fastapi import APIRouter

from app.services.evaluation_service import EvaluationService
from app.services.jobs import JobType, get_job_manager

router = APIRouter(prefix="/evaluation", tags=["evaluation"])
service = EvaluationService()
jobs = get_job_manager()


@router.post("/run")
def run() -> dict:
    job_id = jobs.submit(JobType.evaluate.value, lambda ctx: service.run(job=ctx))
    return {"job_id": job_id}


@router.get("/latest")
def latest() -> dict:
    for job in jobs.list():
        if job.type == JobType.evaluate.value:
            return job.snapshot()
    return {}
