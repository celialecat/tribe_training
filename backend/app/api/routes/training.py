from __future__ import annotations

from fastapi import APIRouter

from app.db.base import session_scope
from app.db.repositories.prediction import ExperimentRepository
from app.schemas.training import TrainingStartRequest
from app.services.jobs import JobType, get_job_manager
from app.services.training_service import TrainingService

router = APIRouter(prefix="/training", tags=["training"])
service = TrainingService()
jobs = get_job_manager()


@router.post("/start")
def start(body: TrainingStartRequest) -> dict:
    job_id = jobs.submit(
        JobType.train.value, lambda ctx: service.run(overrides=body.config, job=ctx)
    )
    return {"job_id": job_id}


@router.post("/{job_id}/pause")
def pause(job_id: str) -> dict:
    return jobs.pause(job_id).snapshot()


@router.post("/{job_id}/resume")
def resume(job_id: str) -> dict:
    return jobs.resume(job_id).snapshot()


@router.post("/{job_id}/stop")
def stop(job_id: str) -> dict:
    return jobs.cancel(job_id).snapshot()


@router.get("/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        return {}
    return job.snapshot()


@router.get("/experiments")
def experiments(limit: int = 20) -> list[dict]:
    with session_scope() as session:
        items: list[dict] = []
        for exp in ExperimentRepository(session).recent(limit=limit):
            items.append(
                {
                    "id": exp.id,
                    "name": exp.name,
                    "status": exp.status,
                    "run_dir": exp.run_dir,
                    "config": exp.config,
                    "epoch": exp.epoch,
                    "total_epochs": exp.total_epochs,
                    "best_metric": exp.best_metric,
                    "metrics": exp.metrics,
                    "error": exp.error,
                    "created_at": exp.created_at,
                    "updated_at": exp.updated_at,
                }
            )
        return items
