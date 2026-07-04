"""``ysp-vultr`` — provision and drive remote training on Vultr from the Mac.

Subcommands
-----------
    plans      list GPU-capable Vultr plans (id, vcpu, ram, gpu, price)
    launch     provision -> bootstrap -> run pipeline -> teardown (full lifecycle)
    provision  create/start the instance + block storage only
    bootstrap  (re)run system bootstrap on an existing instance
    run        run the pipeline on an existing instance
    down       halt or destroy the instance (per config)
    status     show the instance's current state

The single-command milestone is simply::

    export VULTR_API_KEY=...   # asked for once
    ysp-vultr launch           # provision -> download -> TRIBE -> train -> ckpt -> halt
"""

from __future__ import annotations

import argparse
import sys

from app.core.logging import get_logger, setup_logging
from app.deploy.config import DeployConfig
from app.deploy.launcher import Instance, VultrLauncher
from app.deploy.vultr_api import VultrClient

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Provision + run remote training on Vultr.")
    parser.add_argument("--config", default=None, help="Path to deploy/vultr.yaml.")
    parser.add_argument(
        "command",
        choices=["plans", "launch", "provision", "bootstrap", "run", "down", "status"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = _build_parser().parse_args(argv)
    cfg = DeployConfig.load(args.config)

    if args.command == "plans":
        return _cmd_plans(cfg)

    launcher = VultrLauncher(cfg)
    if args.command == "launch":
        launcher.launch()
    elif args.command == "provision":
        inst = launcher.provision()
        logger.info("Provisioned: id=%s ip=%s", inst.id, inst.ip)
    elif args.command in {"bootstrap", "run", "down", "status"}:
        existing = launcher.client.find_instance_by_label(cfg.label)
        if existing is None:
            logger.error(
                "No instance labelled %r found. Run `ysp-vultr provision` first.",
                cfg.label,
            )
            return 1
        instance = Instance(id=existing["id"], ip=existing.get("main_ip", ""))
        if args.command == "bootstrap":
            launcher.bootstrap(instance)
        elif args.command == "run":
            launcher.run_pipeline(instance)
        elif args.command == "down":
            launcher.teardown(instance)
        elif args.command == "status":
            logger.info(
                "Instance %s: status=%s power=%s ip=%s",
                existing["id"], existing.get("status"),
                existing.get("power_status"), existing.get("main_ip"),
            )
    return 0


def _cmd_plans(cfg: DeployConfig) -> int:
    with VultrClient(cfg.require_api_key()) as client:
        plans = client.list_gpu_plans()
    if not plans:
        logger.info("No GPU plans returned.")
        return 0
    logger.info("Available GPU plans:")
    for p in sorted(plans, key=lambda x: x.get("monthly_cost", 0)):
        logger.info(
            "  %-28s vcpu=%s ram=%sMB gpu=%s vram=%sGB $%s/mo regions=%s",
            p.get("id"), p.get("vcpu_count"), p.get("ram"),
            p.get("gpu_type", "?"), p.get("gpu_vram_gb", "?"),
            p.get("monthly_cost", "?"), ",".join(p.get("locations", []))[:40],
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
