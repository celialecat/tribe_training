"""Device resolution helpers.

A tiny abstraction so that every component ("cuda", "cpu", "auto", "mps") agrees
on how a requested device string maps to a concrete, available torch device.
Kept import-light: torch is imported lazily so `app.core` stays usable without
the GPU stack installed.
"""

from __future__ import annotations

from app.core.logging import get_logger

logger = get_logger(__name__)


def resolve_device(requested: str = "auto") -> str:
    """Resolve a requested device string to one that is actually available.

    Falls back to CPU with a warning when CUDA/MPS is requested but absent, so
    that the same code path runs on a laptop and on a Vultr GPU node.
    """
    try:
        import torch
    except ImportError:  # torch not installed (e.g. docs build) -> CPU.
        logger.warning("torch is not installed; defaulting device to 'cpu'.")
        return "cpu"

    requested = requested.lower()
    cuda_ok = torch.cuda.is_available()
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    if requested == "auto":
        if cuda_ok:
            return "cuda"
        if mps_ok:
            return "mps"
        return "cpu"

    if requested.startswith("cuda") and not cuda_ok:
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    if requested == "mps" and not mps_ok:
        logger.warning("MPS requested but unavailable; falling back to CPU.")
        return "cpu"
    return requested


def describe_device(device: str) -> str:
    """Human-readable one-line description of the active device."""
    try:
        import torch
    except ImportError:
        return f"{device} (torch missing)"
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = 0 if ":" not in device else int(device.split(":")[1])
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        return f"{device} · {name} · {total:.1f} GiB"
    return device
