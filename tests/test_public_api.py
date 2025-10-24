"""Smoke tests for curated public exports."""

import ossm


def test_models_public_api_is_curated() -> None:
    expected_model_exports = (
        "Backbone",
        "Head",
        "SequenceBackboneOutput",
        "ClassificationHead",
        "RegressionHead",
        "DampedLinOSSBackbone",
        "DampedLinOSSBlock",
        "DampedLinOSSLayer",
        "Dlinoss4Rec",
        "ItemEmbeddingEncoder",
        "Mamba4Rec",
        "MambaLayer",
        "LinOSSBackbone",
        "LinOSSBlock",
        "LinOSSLayer",
        "LRUBackbone",
        "LRUBlock",
        "LRULayer",
        "NCDEVectorField",
        "NCDELayer",
        "NRDELayer",
        "NCDEBackbone",
        "AbstractRNNCell",
        "LinearRNNCell",
        "GRURNNCell",
        "LSTMRNNCell",
        "MLPRNNCell",
        "RNNBackbone",
        "RNNLayer",
        "S5Backbone",
        "S5Block",
        "S5Layer",
        "TiedSoftmaxHead",
    )

    assert ossm.models.__all__ == expected_model_exports

    for symbol in expected_model_exports:
        assert hasattr(ossm, symbol)
        assert getattr(ossm, symbol) is getattr(ossm.models, symbol)


def test_package_all_includes_models_exports() -> None:
    expected_model_exports = tuple(ossm.models.__all__)
    assert ossm.__all__ == ("data", "metrics") + expected_model_exports
