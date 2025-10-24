from __future__ import annotations

"""Compatibility re-export for dataset helper utilities.

Prefer importing from :mod:`ossm.data.datasets.io`, ``.labeling``, or ``.cache``
for new code. The symbols are preserved here to avoid breaking downstream
imports.
"""

from .cache import cache_key, maybe_save_cache
from .io import deduplicate_pairs, ensure_uea_layout, load_uea_numpy, load_uea_tensors
from .labeling import encode_labels, fit_label_encoder, transform_labels

__all__ = [
    "cache_key",
    "deduplicate_pairs",
    "encode_labels",
    "ensure_uea_layout",
    "fit_label_encoder",
    "load_uea_numpy",
    "load_uea_tensors",
    "maybe_save_cache",
    "transform_labels",
]
