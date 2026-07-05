"""Tests for core configuration, constants and device resolution."""

from __future__ import annotations

from app.core.constants import NUM_TARGETS, TARGET_ORDER, Target
from app.core.device import resolve_device


def test_target_order_is_canonical_and_unique() -> None:
    assert NUM_TARGETS == len(TARGET_ORDER) == 6
    assert len(set(TARGET_ORDER)) == NUM_TARGETS
    assert TARGET_ORDER[0] is Target.log_views_7d
    assert Target.log_likes in TARGET_ORDER


def test_settings_resolves_relative_paths(tmp_settings: object) -> None:
    # data_dir must be absolute and created by ensure_directories().
    assert tmp_settings.data_dir.is_absolute()  # type: ignore[attr-defined]
    assert tmp_settings.data_dir.exists()  # type: ignore[attr-defined]


def test_cors_origins_accepts_csv(monkeypatch) -> None:
    from app.core.config import Settings

    monkeypatch.setenv("YSP_CORS_ORIGINS", "http://a.com, http://b.com")
    settings = Settings()
    assert settings.cors_origins == ["http://a.com", "http://b.com"]


def test_resolve_device_falls_back_to_cpu_without_torch() -> None:
    # On a CPU-only host this must never raise and must return a real device.
    assert resolve_device("auto") in {"cpu", "cuda", "mps"}
    assert resolve_device("cuda") in {"cpu", "cuda"}
