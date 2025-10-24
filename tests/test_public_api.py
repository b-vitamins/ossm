"""Smoke tests for curated public exports."""

import importlib


def test_models_exports_available_from_package() -> None:
    ossm = importlib.import_module("ossm")
    models = importlib.import_module("ossm.models")

    assert isinstance(models.__all__, tuple)
    assert isinstance(ossm.__all__, tuple)

    assert ossm.__all__ == ("data", "metrics") + models.__all__

    for symbol in models.__all__:
        # Accessing through __import__ emulates ``from ossm import symbol``.
        module = __import__("ossm", fromlist=[symbol])
        assert getattr(module, symbol) is getattr(models, symbol)
