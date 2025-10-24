from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Optional

import torch

__all__ = ["cache_key", "maybe_save_cache"]

_LOGGER = logging.getLogger(__name__)


def cache_key(*parts: str) -> str:
    """Join ``parts`` into a filesystem-friendly cache key."""

    return "_".join(parts).replace(os.sep, "-")


def maybe_save_cache(
    dirpath: str | os.PathLike[str],
    key: str,
    obj: Any,
    *,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Persist ``obj`` to ``dirpath``/``key`` if possible.

    Any filesystem or serialization errors are swallowed after logging a debug message,
    keeping the cache best-effort while still surfacing diagnostics when desired.
    """

    log = logger if logger is not None else _LOGGER
    cache_dir = Path(dirpath)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.debug("Failed to create cache directory %s: %s", cache_dir, exc, exc_info=True)
        return

    path = cache_dir / f"{key}.pt"
    try:
        torch.save(obj, path)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError, pickle.PickleError) as exc:
        log.debug("Failed to save cache file %s: %s", path, exc, exc_info=True)
        try:
            path.unlink(missing_ok=True)
        except OSError as cleanup_exc:  # pragma: no cover - highly unlikely
            log.debug(
                "Failed to remove incomplete cache file %s: %s", path, cleanup_exc, exc_info=True
            )
