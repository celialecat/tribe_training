from __future__ import annotations

from fastapi import APIRouter

from app.db.base import session_scope
from app.services.tensor_service import TensorService

router = APIRouter(prefix="/tensors", tags=["tensors"])
service = TensorService()


@router.get("/")
def list_cached() -> dict:
    with session_scope() as session:
        return service.list(session)


@router.get("/{video_id}")
def get_cached(video_id: int) -> dict:
    with session_scope() as session:
        return service.get(session, video_id)
