"""Threaded job orchestration for dashboard-backed long-running work."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from threading import Event, Lock, Thread
from time import monotonic, sleep
from typing import Any, ClassVar
from uuid import uuid4

from app.core.logging import get_logger

logger = get_logger(__name__)


class JobType(StrEnum):
    download = "download"
    validate = "validate"
    tribe = "tribe"
    train = "train"
    evaluate = "evaluate"
    pipeline = "pipeline"
    predict = "predict"


class JobStatus(StrEnum):
    pending = "pending"
    running = "running"
    paused = "paused"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class JobCancelled(RuntimeError):  # noqa: N818
    """Raised when a job is stopped cooperatively."""


@dataclass(slots=True)
class JobProgress:
    """Rolling progress helper with ETA estimation."""

    stage: str = "pending"
    current: int = 0
    total: int = 0
    message: str = ""
    eta_seconds: float | None = None
    window: int = 12
    _durations: deque[float] = field(default_factory=lambda: deque(maxlen=12), repr=False)
    _last_tick: float | None = field(default=None, repr=False)
    _last_current: int = field(default=0, repr=False)

    def update(
        self,
        *,
        stage: str | None = None,
        current: int | None = None,
        total: int | None = None,
        message: str | None = None,
        elapsed: float | None = None,
    ) -> dict[str, Any]:
        if stage is not None:
            self.stage = stage
        if total is not None:
            self.total = max(int(total), 0)
        if current is not None:
            current = max(int(current), 0)
            if elapsed is not None and current > self._last_current:
                delta = current - self._last_current
                if delta > 0:
                    self._durations.append(float(elapsed) / delta)
            self._last_current = current
            self.current = current
        if message is not None:
            self.message = message
        if self.total > 0 and self.current < self.total and self._durations:
            avg = sum(self._durations) / len(self._durations)
            self.eta_seconds = max(0.0, avg * (self.total - self.current))
        else:
            self.eta_seconds = None
        self._last_tick = monotonic()
        return self.as_dict()

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "current": self.current,
            "total": self.total,
            "message": self.message,
            "eta_seconds": self.eta_seconds,
        }


@dataclass(slots=True)
class Job:
    """Mutable state for a submitted job."""

    id: str = field(default_factory=lambda: uuid4().hex)
    type: str = "download"
    status: str = JobStatus.pending.value
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: dict[str, Any] = field(default_factory=lambda: JobProgress().as_dict())
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=500))
    result: dict[str, Any] | None = None
    error: str | None = None
    pause_event: Event = field(default_factory=Event, repr=False)
    stop_event: Event = field(default_factory=Event, repr=False)
    lock: Lock = field(default_factory=Lock, repr=False)
    thread: Thread | None = field(default=None, repr=False)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.id,
                "type": self.type,
                "status": self.status,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "progress": dict(self.progress),
                "logs": list(self.logs),
                "result": self.result,
                "error": self.error,
            }


class JobContext:
    """Control surface exposed to job functions."""

    def __init__(self, job: Job, manager: JobManager) -> None:
        self.job = job
        self.manager = manager
        self._progress = JobProgress(
            stage=str(job.progress.get("stage", "pending")),
            current=int(job.progress.get("current", 0)),
            total=int(job.progress.get("total", 0)),
            message=str(job.progress.get("message", "")),
            eta_seconds=job.progress.get("eta_seconds"),
        )

    def log(self, message: str) -> None:
        line = f"{datetime.now(UTC).isoformat()} | {message}"
        with self.job.lock:
            self.job.logs.append(line)
        logger.info("[%s] %s", self.job.id, message)

    def set_progress(self, **kw: Any) -> dict[str, Any]:
        progress = self._progress.update(
            stage=kw.get("stage"),
            current=kw.get("current"),
            total=kw.get("total"),
            message=kw.get("message"),
            elapsed=kw.get("elapsed"),
        )
        with self.job.lock:
            self.job.progress = progress
        return progress

    def check_control(self) -> None:
        while self.job.pause_event.is_set():
            with self.job.lock:
                self.job.status = JobStatus.paused.value
            if self.job.stop_event.is_set():
                raise JobCancelled(self.job.id)
            sleep(0.2)
        if self.job.stop_event.is_set():
            raise JobCancelled(self.job.id)
        with self.job.lock:
            if self.job.status == JobStatus.paused.value:
                self.job.status = JobStatus.running.value

    def is_cancelled(self) -> bool:
        return self.job.stop_event.is_set()


JobFn = Callable[[JobContext], dict[str, Any] | None]


class JobManager:
    """Singleton job registry backed by daemon worker threads."""

    exclusive_types: ClassVar[set[str]] = {
        JobType.tribe.value,
        JobType.train.value,
        JobType.pipeline.value,
    }

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()

    def submit(self, job_type: str, fn: JobFn) -> str:
        job_type = str(job_type)
        with self._lock:
            if job_type in self.exclusive_types and self._has_running_exclusive():
                raise RuntimeError("another exclusive job is already running")
            job = Job(type=job_type)
            self._jobs[job.id] = job

        def runner() -> None:
            context = JobContext(job, self)
            try:
                with job.lock:
                    job.status = JobStatus.running.value
                    job.started_at = datetime.now(UTC)
                result = fn(context)
                with job.lock:
                    if job.status not in {JobStatus.cancelled.value, JobStatus.failed.value}:
                        job.status = JobStatus.completed.value
                    if result is not None:
                        job.result = result
            except JobCancelled:
                with job.lock:
                    job.status = JobStatus.cancelled.value
                    job.error = "cancelled"
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Job %s failed", job.id)
                with job.lock:
                    job.status = JobStatus.failed.value
                    job.error = str(exc)
            finally:
                with job.lock:
                    job.finished_at = datetime.now(UTC)

        thread = Thread(target=runner, name=f"job-{job.id[:8]}", daemon=True)
        job.thread = thread
        thread.start()
        return job.id

    def pause(self, job_id: str) -> Job:
        job = self._require(job_id)
        job.pause_event.set()
        with job.lock:
            if job.status == JobStatus.running.value:
                job.status = JobStatus.paused.value
        return job

    def resume(self, job_id: str) -> Job:
        job = self._require(job_id)
        job.pause_event.clear()
        with job.lock:
            if job.status == JobStatus.paused.value:
                job.status = JobStatus.running.value
        return job

    def cancel(self, job_id: str) -> Job:
        job = self._require(job_id)
        job.stop_event.set()
        job.pause_event.clear()
        with job.lock:
            if job.status not in {JobStatus.completed.value, JobStatus.failed.value}:
                job.status = JobStatus.cancelled.value
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def _has_running_exclusive(self) -> bool:
        for job in self._jobs.values():
            if job.type in self.exclusive_types and job.status in {
                JobStatus.pending.value,
                JobStatus.running.value,
                JobStatus.paused.value,
            }:
                return True
        return False

    def _require(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job


_MANAGER: JobManager | None = None


def get_job_manager() -> JobManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = JobManager()
    return _MANAGER
