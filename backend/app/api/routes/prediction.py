from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, UploadFile

from app.core.config import get_settings
from app.schemas.prediction import PredictionHorizonResponse, PredictionUrlRequest
from app.services.jobs import JobType, get_job_manager
from app.services.prediction_service import PredictionService

router = APIRouter(prefix="/prediction", tags=["prediction"])
service = PredictionService()
jobs = get_job_manager()


@router.post("/url")
def predict_url(body: PredictionUrlRequest) -> dict:
    job_id = jobs.submit(
        JobType.predict.value,
        lambda ctx: service.run(
            url=body.url,
            confidence_level=body.confidence_level,
            horizon_days=body.horizon_days,
            job=ctx,
        ),
    )
    return {"job_id": job_id}


@router.post("/upload")
async def predict_upload(file: UploadFile = File(...)) -> dict:
    settings = get_settings()
    settings.ensure_directories()
    upload_dir = settings.cache_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    with NamedTemporaryFile(delete=False, suffix=suffix, dir=str(upload_dir)) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name
    job_id = jobs.submit(
        JobType.predict.value, lambda ctx: service.run(file_path=temp_path, job=ctx)
    )
    return {"job_id": job_id}


@router.get("/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        return {}
    return job.snapshot()


@router.get("/horizon")
def horizon() -> PredictionHorizonResponse:
    return PredictionHorizonResponse(horizon_days=30)
