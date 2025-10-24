"""Model backbones and task heads."""

from .base import Backbone, Head, SequenceBackboneOutput
from .dlinoss import DampedLinOSSBackbone, DampedLinOSSBlock, DampedLinOSSLayer
from .dlinossrec import Dlinoss4Rec, ItemEmbeddingEncoder
from .heads import ClassificationHead, RegressionHead, TiedSoftmaxHead
from .linoss import LinOSSBackbone, LinOSSBlock, LinOSSLayer
from .lru import LRUBackbone, LRUBlock, LRULayer
from .mambarec import Mamba4Rec, MambaLayer
from .ncde import NCDEBackbone, NCDELayer, NCDEVectorField, NRDELayer
from .rnn import (
    AbstractRNNCell,
    GRURNNCell,
    LinearRNNCell,
    LSTMRNNCell,
    MLPRNNCell,
    RNNBackbone,
    RNNLayer,
)
from .s5 import S5Backbone, S5Block, S5Layer

__all__: tuple[str, ...] = (
    "Backbone",
    "BatchOnDevice",
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

MODELS_PUBLIC_EXPORTS: tuple[str, ...] = __all__
