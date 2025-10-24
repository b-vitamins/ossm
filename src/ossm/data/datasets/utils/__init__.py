"""Dataset utility helpers split into focused submodules.

The submodules expose IO helpers, labeling transforms, and cache utilities.
Importing :mod:`ossm.data.datasets.utils` retains backward compatibility so
existing monkeypatch strategies continue to work.
"""

from .io import ensure_uea_layout, load_uea_numpy, load_uea_tensors, deduplicate_pairs
from .labeling import encode_labels, fit_label_encoder, transform_labels
from .cache import cache_key, maybe_save_cache

__all__ = [
    "ensure_uea_layout",
    "load_uea_numpy",
    "load_uea_tensors",
    "deduplicate_pairs",
    "encode_labels",
    "fit_label_encoder",
    "transform_labels",
    "cache_key",
    "maybe_save_cache",
]
