"""VultrLauncher — provision, bootstrap, run the pipeline, tear down.

Orchestration only: infrastructure via :class:`VultrClient`, remote commands via
``ssh``/``scp`` subprocesses. The heavy lifting on the instance lives in the
``scripts/vultr/*.sh`` shell scripts, which this launcher uploads and runs. This
separation means the same scripts work whether driven by this launcher, by
cloud-init, or by hand.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.logging import get_logger
from app.deploy.config import DeployConfig
from app.deploy.vultr_api import VultrClient

logger = get_logger(__name__)

# Repo-root-relative location of the instance-side scripts.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2].parent / "scripts" / "vultr"
_SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15"]


@dataclass(slots=True)
class Instance:
    """A provisioned instance we can talk to."""

    id: str
    ip: str
    block_id: str | None = None


class VultrLauncher:
    """Drive the full remote training lifecycle on Vultr."""

    def __init__(self, config: DeployConfig, *, client: VultrClient | None = None) -> None:
        self.cfg = config
        self.client = client or VultrClient(config.require_api_key())

    # --------------------------------------------------------- provisioning --
    def provision(self) -> Instance:
        """Create (or reuse) the GPU instance + block storage; wait until ready."""
        cfg = self.cfg
        if not cfg.plan:
            raise RuntimeError("No GPU plan set. Run `ysp-vultr plans` and set `plan` in config.")

        key_id = self.client.ensure_ssh_key(cfg.label, cfg.ssh_public_key())
        block = self.client.create_block(cfg.region, cfg.block_size_gb, cfg.label)

        existing = self.client.find_instance_by_label(cfg.label)
        if existing:
            logger.info("Reusing instance %s (%s)", existing["id"], cfg.label)
            if existing.get("power_status") != "running":
                self.client.start_instance(existing["id"])
            instance_id = existing["id"]
        else:
            os_id = self.client.find_os_id(cfg.os_label)
            logger.info("Creating %s GPU instance in %s ...", cfg.plan, cfg.region)
            created = self.client.create_instance(
                region=cfg.region, plan=cfg.plan, os_id=os_id,
                label=cfg.label, sshkey_ids=[key_id],
            )
            instance_id = created["id"]

        ip = self._wait_active(instance_id)
        self.client.attach_block(block["id"], instance_id)
        self._wait_ssh(ip)
        logger.info("Instance ready: %s (%s)", instance_id, ip)
        return Instance(id=instance_id, ip=ip, block_id=block["id"])

    def _wait_active(self, instance_id: str, *, timeout: float = 900.0) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            inst = self.client.get_instance(instance_id)
            ip = inst.get("main_ip", "0.0.0.0")
            running = inst.get("status") == "active" and inst.get("power_status") == "running"
            if running and ip not in ("", "0.0.0.0"):
                return ip
            time.sleep(10)
        raise TimeoutError(f"Instance {instance_id} did not become active in {timeout:.0f}s")

    def _wait_ssh(self, ip: str, *, timeout: float = 600.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = subprocess.run(
                ["ssh", *_SSH_OPTS, "-i", self._private_key(), f"{self.cfg.ssh_user}@{ip}", "true"],
                capture_output=True,
            )
            if result.returncode == 0:
                return
            time.sleep(10)
        raise TimeoutError(f"SSH to {ip} not reachable in {timeout:.0f}s")

    # ------------------------------------------------------------ bootstrap --
    def bootstrap(self, instance: Instance) -> None:
        """Upload scripts and run system bootstrap (mount, clone, install)."""
        self._upload_scripts(instance.ip)
        env = {
            "REPO_URL": self.cfg.repo_url,
            "BRANCH": self.cfg.branch,
            "MOUNT_POINT": self.cfg.mount_point,
            "BLOCK_LABEL": self.cfg.label,
        }
        self._ssh(instance.ip, "bash /root/ysp-scripts/bootstrap.sh", env=env)

    def run_pipeline(self, instance: Instance) -> None:
        """Run the end-to-end pipeline on the instance."""
        cfg = self.cfg
        env = {
            "MOUNT_POINT": cfg.mount_point,
            "DATASET_SOURCE": cfg.dataset_source,
            "DATASET_LIMIT": str(cfg.dataset_limit) if cfg.dataset_limit else "",
            "TRAIN_OVERRIDES": " ".join(cfg.train_overrides),
            "PIPELINE_SKIP": " ".join(cfg.pipeline_skip),
        }
        self._ssh(instance.ip, "bash /root/ysp-scripts/run_pipeline.sh", env=env)

    def teardown(self, instance: Instance) -> None:
        if self.cfg.auto_destroy:
            logger.info(
                "Destroying instance %s (block storage %s retained)",
                instance.id, instance.block_id,
            )
            self.client.destroy_instance(instance.id)
        elif self.cfg.auto_halt:
            logger.info("Halting instance %s (stops compute billing)", instance.id)
            self.client.halt_instance(instance.id)

    def launch(self) -> Instance:
        """Full lifecycle: provision -> bootstrap -> run -> teardown."""
        instance = self.provision()
        try:
            self.bootstrap(instance)
            self.run_pipeline(instance)
        finally:
            self.teardown(instance)
        return instance

    # ------------------------------------------------------------- ssh utils --
    def _private_key(self) -> str:
        return str(Path(self.cfg.ssh_key_path).expanduser()).removesuffix(".pub")

    def _ssh(self, ip: str, command: str, *, env: dict[str, str] | None = None) -> None:
        prefix = "".join(f"export {k}={_shquote(v)}; " for k, v in (env or {}).items())
        full = prefix + command
        logger.info("ssh %s: %s", ip, command)
        subprocess.run(
            ["ssh", *_SSH_OPTS, "-i", self._private_key(), f"{self.cfg.ssh_user}@{ip}", full],
            check=True,
        )

    def _upload_scripts(self, ip: str) -> None:
        self._ssh(ip, "mkdir -p /root/ysp-scripts")
        for script in sorted(_SCRIPTS_DIR.glob("*.sh")):
            subprocess.run(
                ["scp", *_SSH_OPTS, "-i", self._private_key(), str(script),
                 f"{self.cfg.ssh_user}@{ip}:/root/ysp-scripts/"],
                check=True,
            )


def _shquote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"
