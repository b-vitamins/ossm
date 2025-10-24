"""Model backbones and task heads."""

from . import base as _base
from . import dlinoss as _dlinoss
from . import dlinossrec as _dlinossrec
from . import heads as _heads
from . import linoss as _linoss
from . import lru as _lru
from . import mambarec as _mambarec
from . import ncde as _ncde
from . import rnn as _rnn
from . import s5 as _s5

_MODELS_PUBLIC_API = (
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

__all__ = _MODELS_PUBLIC_API

_EXPORT_SOURCES = (
    (_base, ("Backbone", "Head", "SequenceBackboneOutput")),
    (_heads, ("ClassificationHead", "RegressionHead", "TiedSoftmaxHead")),
    (_dlinoss, ("DampedLinOSSBackbone", "DampedLinOSSBlock", "DampedLinOSSLayer")),
    (_dlinossrec, ("Dlinoss4Rec", "ItemEmbeddingEncoder")),
    (_mambarec, ("Mamba4Rec", "MambaLayer")),
    (_linoss, ("LinOSSBackbone", "LinOSSBlock", "LinOSSLayer")),
    (_lru, ("LRUBackbone", "LRUBlock", "LRULayer")),
    (_ncde, ("NCDEVectorField", "NCDELayer", "NRDELayer", "NCDEBackbone")),
    (
        _rnn,
        (
            "AbstractRNNCell",
            "GRURNNCell",
            "LinearRNNCell",
            "LSTMRNNCell",
            "MLPRNNCell",
            "RNNBackbone",
            "RNNLayer",
        ),
    ),
    (_s5, ("S5Backbone", "S5Block", "S5Layer")),
)

for _module, _names in _EXPORT_SOURCES:
    for _name in _names:
        globals()[_name] = getattr(_module, _name)

del _module, _names
