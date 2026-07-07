from __future__ import annotations

import time
from collections.abc import Callable
from threading import Event
from time import sleep

import pytest

from app.services.jobs import Job, JobContext, JobManager, JobStatus


def test_job_manager_lifecycle() -> None:
    manager = JobManager()
    started = Event()

    def worker(ctx: JobContext) -> dict[str, bool]:
        started.set()
        for _ in range(20):
            ctx.check_control()
            sleep(0.02)
        return {"ok": True}

    job_id = manager.submit("download", worker)
    assert started.wait(1.0)

    manager.pause(job_id)
    wait_for(lambda: _job_status(manager, job_id) == JobStatus.paused.value)
    manager.resume(job_id)
    manager.cancel(job_id)
    wait_for(lambda: _job_status(manager, job_id) == JobStatus.cancelled.value)
    assert _job(manager, job_id).result is None


def test_exclusive_job_guard() -> None:
    manager = JobManager()
    started = Event()
    release = Event()

    def worker(ctx: JobContext) -> dict[str, bool]:
        started.set()
        while not release.is_set():
            ctx.check_control()
            sleep(0.01)
        return {"done": True}

    job_id = manager.submit("tribe", worker)
    assert started.wait(1.0)
    with pytest.raises(RuntimeError):
        manager.submit("tribe", worker)
    release.set()
    wait_for(lambda: _job_status(manager, job_id) == JobStatus.completed.value)


def wait_for(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        sleep(0.02)
    pytest.fail("condition not met before timeout")


def _job(manager: JobManager, job_id: str) -> Job:
    job = manager.get(job_id)
    assert job is not None
    return job


def _job_status(manager: JobManager, job_id: str) -> str:
    return _job(manager, job_id).status
