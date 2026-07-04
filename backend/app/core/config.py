"""Application settings.

A single :class:`Settings` object is the source of truth for runtime
configuration that is *environment-specific* (paths, database URL, device,
network binding). It is loaded once and cached with :func:`get_settings`.

This is deliberately separate from the *experiment* configuration handled by
Hydra (``configs/``): Hydra owns model/training hyper-parameters that vary per
run and belong in version-controlled YAML, whereas ``Settings`` owns deployment
facts that come from the environment (``.env`` / real env vars). Keeping the two
apart means a training sweep never has to know the production database URL, and
the API never has to parse a model config to start.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Repository root = three levels up from this file (backend/app/core/config.py).
REPO_ROOT = Path(__file__).resolve().parents[3]


class Environment(StrEnum):
    """Deployment environment, controls defaults and safety checks."""

    development = "development"
    production = "production"
    test = "test"


class Settings(BaseSettings):
    """Environment-driven configuration.

    Field values are resolved (highest priority first) from: real environment
    variables, the ``.env`` file, then the defaults declared here. All variables
    use the ``YSP_`` prefix except TRIBE/HuggingFace ones, which keep their
    conventional names via explicit aliases.
    """

    model_config = SettingsConfigDict(
        env_prefix="YSP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Application ----
    env: Environment = Environment.development
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # NoDecode: keep pydantic-settings from JSON-decoding this env var so the
    # validator below can accept a plain comma-separated string from .env.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"]
    )

    # ---- Database ----
    database_url: str = f"sqlite:///{REPO_ROOT / 'data' / 'ysp.db'}"

    # ---- Storage ----
    data_dir: Path = REPO_ROOT / "data"
    raw_dir: Path = REPO_ROOT / "data" / "raw"
    processed_dir: Path = REPO_ROOT / "data" / "processed"
    cache_dir: Path = REPO_ROOT / "data" / "cache"
    models_dir: Path = REPO_ROOT / "models"

    # ---- Compute ----
    device: str = "cuda"
    num_workers: int = 4

    # ---- TRIBE v2 (non-YSP-prefixed env vars) ----
    tribe_model_id: str = Field(
        default="facebook/tribev2",
        validation_alias=AliasChoices("TRIBE_MODEL_ID", "YSP_TRIBE_MODEL_ID"),
    )
    tribe_cache_dir: Path = Field(
        default=REPO_ROOT / "data" / "cache" / "tribe",
        validation_alias=AliasChoices("TRIBE_CACHE_DIR", "YSP_TRIBE_CACHE_DIR"),
    )
    tribe_device: str = Field(
        default="cuda",
        validation_alias=AliasChoices("TRIBE_DEVICE", "YSP_TRIBE_DEVICE"),
    )
    hf_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("HUGGING_FACE_HUB_TOKEN", "HF_TOKEN"),
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Allow CORS origins to be given as a comma-separated string in .env."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator(
        "data_dir", "raw_dir", "processed_dir", "cache_dir", "models_dir", "tribe_cache_dir"
    )
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        """Resolve relative paths against the repo root for stable behaviour."""
        return value if value.is_absolute() else (REPO_ROOT / value).resolve()

    @property
    def is_production(self) -> bool:
        return self.env is Environment.production

    def ensure_directories(self) -> None:
        """Create all managed storage directories (idempotent)."""
        for path in (
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.cache_dir,
            self.models_dir,
            self.tribe_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""
    return Settings()
