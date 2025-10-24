"""Model backbones and task heads."""

from .base import Backbone, BatchOnDevice, Head, SequenceBackboneOutput
from .heads import ClassificationHead, RegressionHead, TiedSoftmaxHead
from .dlinoss import DampedLinOSSBackbone, DampedLinOSSBlock, DampedLinOSSLayer
from .dlinossrec import Dlinoss4Rec, ItemEmbeddingEncoder
from .mambarec import Mamba4Rec, MambaLayer
from .linoss import LinOSSBackbone, LinOSSBlock, LinOSSLayer
from .lru import LRUBackbone, LRUBlock, LRULayer
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
    "AbstractRNNCell",
    "Backbone",
    "BatchOnDevice",
    "ClassificationHead",
    "DampedLinOSSBackbone",
    "DampedLinOSSBlock",
    "DampedLinOSSLayer",
    "Dlinoss4Rec",
    "GRURNNCell",
    "Head",
    "ItemEmbeddingEncoder",
    "LinOSSBackbone",
    "LinOSSBlock",
    "LinOSSLayer",
    "LinearRNNCell",
    "LRUBackbone",
    "LRUBlock",
    "LRULayer",
    "LSTMRNNCell",
    "Mamba4Rec",
    "MambaLayer",
    "MLPRNNCell",
    "NCDEBackbone",
    "NCDELayer",
    "NCDEVectorField",
    "NRDELayer",
    "RegressionHead",
    "RNNBackbone",
    "RNNLayer",
    "S5Backbone",
    "S5Block",
    "S5Layer",
    "SequenceBackboneOutput",
    "TiedSoftmaxHead",
)
