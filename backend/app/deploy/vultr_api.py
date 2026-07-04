"""Minimal Vultr API v2 client (built on httpx — no extra SDK dependency).

Covers exactly the operations the launcher needs: SSH keys, GPU plans, OS
lookup, block storage, and instance lifecycle. Every call raises on HTTP error
with the Vultr error body surfaced for debuggability.

Reference: https://www.vultr.com/api/
"""

from __future__ import annotations

from typing import Any

import httpx

from app.core.logging import get_logger

logger = get_logger(__name__)

BASE_URL = "https://api.vultr.com/v2"


class VultrError(RuntimeError):
    """A non-2xx response from the Vultr API."""


class VultrClient:
    """Thin, typed wrapper over the Vultr REST API."""

    def __init__(self, api_key: str, *, timeout: float = 60.0) -> None:
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> VultrClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------- request --
    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        resp = self._client.request(method, path, **kwargs)
        if resp.status_code >= 400:
            raise VultrError(f"{method} {path} -> {resp.status_code}: {resp.text}")
        return resp.json() if resp.content else {}

    def _paginated(self, path: str, key: str) -> list[dict[str, Any]]:
        """Collect all pages of a list endpoint."""
        items: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params = {"per_page": 500}
            if cursor:
                params["cursor"] = cursor
            data = self._request("GET", path, params=params)
            items.extend(data.get(key, []))
            cursor = data.get("meta", {}).get("links", {}).get("next") or ""
            if not cursor:
                break
        return items

    # ---------------------------------------------------------- discovery --
    def list_gpu_plans(self) -> list[dict[str, Any]]:
        """Return GPU-capable plans (Vultr 'vcg' cloud-GPU family)."""
        plans = self._paginated("/plans", "plans")
        gpu = [
            p for p in plans if p.get("type") == "vcg" or "gpu" in str(p.get("id", ""))
        ]
        return gpu or plans

    def find_os_id(self, label: str) -> int:
        """Resolve an OS id from its human label (e.g. 'Ubuntu 24.04 LTS x64')."""
        for os_entry in self._paginated("/os", "os"):
            if os_entry.get("name") == label:
                return int(os_entry["id"])
        # Looser match on the first token(s).
        for os_entry in self._paginated("/os", "os"):
            if label.split(" LTS")[0] in os_entry.get("name", ""):
                return int(os_entry["id"])
        raise VultrError(f"No OS matching {label!r} found.")

    # ------------------------------------------------------------ ssh keys --
    def ensure_ssh_key(self, name: str, public_key: str) -> str:
        """Upload the SSH key if absent; return its id (idempotent by content)."""
        for key in self._paginated("/ssh-keys", "ssh_keys"):
            if key.get("ssh_key", "").strip() == public_key.strip():
                return str(key["id"])
        data = self._request("POST", "/ssh-keys", json={"name": name, "ssh_key": public_key})
        return str(data["ssh_key"]["id"])

    # -------------------------------------------------------- block storage --
    def find_block_by_label(self, label: str) -> dict[str, Any] | None:
        for block in self._paginated("/blocks", "blocks"):
            if block.get("label") == label:
                return block
        return None

    def create_block(self, region: str, size_gb: int, label: str) -> dict[str, Any]:
        existing = self.find_block_by_label(label)
        if existing:
            logger.info("Reusing existing block storage %s (%s)", existing["id"], label)
            return existing
        data = self._request(
            "POST", "/blocks",
            json={
                "region": region, "size_gb": size_gb,
                "label": label, "block_type": "storage_opt",
            },
        )
        return data["block"]

    def attach_block(self, block_id: str, instance_id: str) -> None:
        self._request(
            "POST", f"/blocks/{block_id}/attach",
            json={"instance_id": instance_id, "live": True},
        )

    def get_block(self, block_id: str) -> dict[str, Any]:
        return self._request("GET", f"/blocks/{block_id}")["block"]

    # ------------------------------------------------------------ instances --
    def find_instance_by_label(self, label: str) -> dict[str, Any] | None:
        for inst in self._paginated("/instances", "instances"):
            if inst.get("label") == label:
                return inst
        return None

    def create_instance(
        self,
        *,
        region: str,
        plan: str,
        os_id: int,
        label: str,
        sshkey_ids: list[str],
        user_data_b64: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "region": region,
            "plan": plan,
            "os_id": os_id,
            "label": label,
            "hostname": label,
            "sshkey_id": sshkey_ids,
            "backups": "disabled",
        }
        if user_data_b64:
            payload["user_data"] = user_data_b64
        return self._request("POST", "/instances", json=payload)["instance"]

    def get_instance(self, instance_id: str) -> dict[str, Any]:
        return self._request("GET", f"/instances/{instance_id}")["instance"]

    def start_instance(self, instance_id: str) -> None:
        self._request("POST", f"/instances/{instance_id}/start")

    def halt_instance(self, instance_id: str) -> None:
        self._request("POST", f"/instances/{instance_id}/halt")

    def destroy_instance(self, instance_id: str) -> None:
        self._request("DELETE", f"/instances/{instance_id}")
