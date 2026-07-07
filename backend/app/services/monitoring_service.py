"""System and job monitoring helpers for the API."""

from __future__ import annotations

import subprocess
from typing import Any

import psutil

from app.core.config import Settings, get_settings
from app.services.jobs import JobManager, get_job_manager


class MonitoringService:
    """Collect lightweight system and job status for the dashboard."""

    def __init__(self, *, settings: Settings | None = None, jobs: JobManager | None = None) -> None:
        self.settings = settings or get_settings()
        self.jobs = jobs or get_job_manager()

    def system(self) -> dict[str, Any]:
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(str(self.settings.data_dir))
        return {
            "cpu_percent": cpu,
            "memory_used": memory.used,
            "memory_total": memory.total,
            "disk_used": disk.used,
            "disk_total": disk.total,
            "gpu": self._gpu(),
        }

    def jobs_snapshot(self) -> list[dict[str, Any]]:
        return [job.snapshot() for job in self.jobs.list()]

    def logs(self, *, limit: int = 200) -> list[str]:
        lines: list[str] = []
        for job in self.jobs.list():
            lines.extend(list(job.logs))
        return lines[-limit:]

    def payload(self) -> dict[str, Any]:
        return {"system": self.system(), "active_jobs": self.jobs_snapshot()}

    def _gpu(self) -> dict[str, Any] | None:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None
        if not output:
            return None
        name, used, total = [part.strip() for part in output.split(",", 2)]
        return {"name": name, "memory_used": int(float(used)), "memory_total": int(float(total))}
