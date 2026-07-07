from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.schemas.monitoring import SystemStatus
from app.services.monitoring_service import MonitoringService

router = APIRouter(prefix="/monitoring", tags=["monitoring"])
service = MonitoringService()


@router.get("/system")
def system() -> SystemStatus:
    return SystemStatus.model_validate(service.system())


@router.get("/jobs")
def jobs() -> dict:
    return {"items": service.jobs_snapshot()}


@router.get("/logs")
def logs(limit: int = 200) -> dict:
    return {"items": service.logs(limit=limit)}


@router.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(service.payload())
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        return
