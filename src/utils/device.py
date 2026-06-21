"""Device selection helper.

Single source of truth for picking a torch device. Import `select_device()`
from every script that runs the encoder or any other torch model. Do not
re-implement the cascade inline.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def select_device(prefer: str | None = None) -> torch.device:
    """Return the best available torch device.

    Cascade: cuda > mps > cpu. The POC hardcoded mps which is fine on Apple
    Silicon but breaks on Windows and Linux.

    Args:
        prefer: Optional override. If given and the requested device is
            available, return it. If unavailable, log a warning and fall
            through to the cascade.

    Returns:
        torch.device suitable for model.to() / tensor.to() calls.
    """
    if prefer:
        device = torch.device(prefer)
        if _is_available(device):
            return device
        logger.warning("Requested device %s is not available; falling through to cascade.", prefer)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _is_available(device: torch.device) -> bool:
    if device.type == "cuda":
        return torch.cuda.is_available()
    if device.type == "mps":
        return torch.backends.mps.is_available()
    return device.type == "cpu"
