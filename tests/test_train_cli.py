from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from train import _normalize_hydra_overrides


def test_normalises_space_separated_pairs():
    overrides = ["optimizer", "adamw", "training.topk=10"]
    assert _normalize_hydra_overrides(overrides) == [
        "optimizer=adamw",
        "training.topk=10",
    ]


def test_preserves_standard_overrides():
    overrides = ["training.topk=10", "+scheduler=none", "--flag"]
    assert _normalize_hydra_overrides(overrides) == overrides


def test_missing_value_raises_clear_error():
    with pytest.raises(ValueError):
        _normalize_hydra_overrides(["optimizer"])
    with pytest.raises(ValueError):
        _normalize_hydra_overrides(["optimizer", "training.topk=10"])
