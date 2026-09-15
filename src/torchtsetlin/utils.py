"""Small shared utilities: device handling, seeding, input coercion, memory chunking."""

from __future__ import annotations

import random
from typing import Iterator, Optional, Union

import numpy as np
import torch
from torch import Tensor

DeviceLike = Union[str, torch.device, None]

__all__ = [
    "DeviceLike",
    "as_device",
    "seed_everything",
    "as_bool_tensor",
    "as_long_tensor",
    "as_float_tensor",
    "chunk_indices",
    "elements_budget_chunks",
    "format_int",
]


def as_device(device: DeviceLike) -> torch.device:
    """Normalise a device specification (``None`` -> CPU)."""
    if device is None:
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + all CUDA devices) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(t: Tensor, device: torch.device) -> Tensor:
    """Move ``t`` to ``device``.

    ``non_blocking=True`` is used only when the destination is a CUDA device. A
    *device-to-host* copy issued non-blocking into ordinary (pageable) memory returns before
    the transfer has completed, so the caller can read the destination while it still holds
    stale data — which silently corrupts, for example, an encoder that holds its thresholds on
    the CPU but is handed a CUDA tensor.
    """
    return t.to(device, non_blocking=(device.type == "cuda"))


def as_bool_tensor(x, device: Optional[torch.device] = None) -> Tensor:
    """Coerce ``x`` (tensor / ndarray / sequence) to a boolean tensor.

    Any non-zero value becomes ``True``. Floats are compared against 0.5 so that
    ``0.0``/``1.0`` style inputs work as expected.
    """
    if not isinstance(x, Tensor):
        x = torch.as_tensor(np.asarray(x))
    if x.dtype != torch.bool:
        if x.is_floating_point():
            x = x > 0.5
        else:
            x = x != 0
    if device is not None and x.device != device:
        x = _to_device(x, device)
    return x


def as_long_tensor(y, device: Optional[torch.device] = None) -> Tensor:
    if not isinstance(y, Tensor):
        y = torch.as_tensor(np.asarray(y))
    if y.dtype != torch.long:
        y = y.long()
    if device is not None and y.device != device:
        y = _to_device(y, device)
    return y


def as_float_tensor(y, device: Optional[torch.device] = None, dtype=torch.float32) -> Tensor:
    if not isinstance(y, Tensor):
        y = torch.as_tensor(np.asarray(y))
    if y.dtype != dtype:
        y = y.to(dtype)
    if device is not None and y.device != device:
        y = _to_device(y, device)
    return y


def chunk_indices(n: int, chunk: int) -> Iterator[slice]:
    """Yield ``slice`` objects that partition ``range(n)`` into chunks."""
    chunk = max(1, int(chunk))
    for start in range(0, n, chunk):
        yield slice(start, min(n, start + chunk))


def elements_budget_chunks(n: int, per_item_elements: int, budget: int) -> int:
    """Chunk size so that ``chunk * per_item_elements <= budget`` (at least 1)."""
    if per_item_elements <= 0:
        return max(1, n)
    return max(1, min(n, budget // per_item_elements))


def format_int(n: int) -> str:
    return f"{n:,}"
