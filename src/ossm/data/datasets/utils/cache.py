from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Mapping, Optional

import torch

__all__ = ["cache_key", "maybe_save_cache"]

_LOGGER = logging.getLogger(__name__)


def cache_key(*parts: str) -> str:
    return "_".join(parts).replace(os.sep, "-")


def maybe_save_cache(
    dirpath: str,
    key: str,
    obj: Mapping[str, Any] | Any,
    *,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Best-effort helper that persists ``obj`` via :func:`torch.save`.

    Only filesystem and serialization issues are suppressed and a debug log
    records failures when a logger is provided (defaults to the module logger).
    """

    target_logger = logger or _LOGGER
    path = os.path.join(dirpath, f"{key}.pt")
    try:
        os.makedirs(dirpath, exist_ok=True)
        torch.save(obj, path)
    except (OSError, RuntimeError, pickle.PickleError, pickle.PicklingError) as err:
        target_logger.debug("Skipping cache write for %s: %s", path, err)
