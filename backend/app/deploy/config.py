"""Deployment configuration for the Vultr launcher.

Loaded from a YAML file (``deploy/vultr.yaml`` by default) and overlaid with
environment variables for anything secret. The Vultr API key is NEVER read from
the YAML — only from ``VULTR_API_KEY`` — so config files stay safe to commit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

DEFAULT_CONFIG_PATH = Path("deploy/vultr.yaml")


@dataclass(slots=True)
class DeployConfig:
    """All knobs for provisioning + running a remote training job."""

    # ---- Vultr infrastructure ----
    region: str = "ewr"                       # Vultr region code (e.g. ewr, lax)
    plan: str = ""                            # GPU plan id (see `ysp-vultr plans`)
    os_label: str = "Ubuntu 24.04 LTS x64"    # matched against GET /os
    block_size_gb: int = 100                  # persistent block storage size
    label: str = "ysp-train"                  # instance + block storage label
    ssh_key_path: str = "~/.ssh/id_ed25519.pub"
    ssh_user: str = "root"

    # ---- Source (Mac -> GitHub -> Vultr) ----
    repo_url: str = ""                        # git URL cloned on the instance
    branch: str = "main"

    # ---- Persistent storage layout on the instance ----
    mount_point: str = "/mnt/ysp-data"

    # ---- Pipeline arguments ----
    dataset_source: str = ""                  # video / playlist / channel URL
    dataset_limit: int | None = None
    train_overrides: list[str] = field(default_factory=list)  # Hydra overrides
    pipeline_skip: list[str] = field(default_factory=list)

    # ---- Lifecycle ----
    auto_halt: bool = True                    # halt (stop billing compute) when done
    auto_destroy: bool = False                # destroy instance (keeps block storage)

    # ---- Secret (env only) ----
    api_key: str = ""

    @classmethod
    def load(cls, path: str | Path | None = None) -> DeployConfig:
        """Load config from YAML (if present) + environment overrides."""
        cfg = cls()
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        if path.exists():
            data = OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}
            for key, value in data.items():
                if hasattr(cfg, key) and key != "api_key":
                    setattr(cfg, key, value)
        cfg.api_key = os.environ.get("VULTR_API_KEY", cfg.api_key)
        return cfg

    def require_api_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                "VULTR_API_KEY is not set. Export it before running: "
                "`export VULTR_API_KEY=...` (get it from the Vultr dashboard)."
            )
        return self.api_key

    def ssh_public_key(self) -> str:
        path = Path(self.ssh_key_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"SSH public key not found at {path}. Generate one with "
                "`ssh-keygen -t ed25519` or set ssh_key_path in deploy/vultr.yaml."
            )
        return path.read_text().strip()
