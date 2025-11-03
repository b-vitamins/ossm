"""Model backbones and task heads."""

from .base import Backbone, BatchOnDevice, Head, ResidualSSMBlock, SequenceBackboneOutput
from .heads import ClassificationHead, RegressionHead, TiedSoftmaxHead
from .dlinoss import DampedLinOSSBackbone, DampedLinOSSBlock, DampedLinOSSLayer
from .dlinossrec import Dlinoss4Rec, ItemEmbeddingEncoder
from .mambarec import Mamba4Rec, MambaLayer
from .linoss import LinOSSBackbone, LinOSSBlock, LinOSSLayer
from .lru import LRUBackbone, LRUBlock, LRULayer
from .ncde import NCDEBackbone, NCDELayer, NCDEVectorField, NRDELayer
from .sdlinossrec import Sdlinoss4Rec
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
from .sdlinoss import (
    SelectiveDLinOSSBackbone,
    SelectiveDLinOSSBlock,
    SelectiveDLinOSSLayer,
    run_sdlinoss,
    has_fast_kernels,
)

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
    "Sdlinoss4Rec",
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
    "ResidualSSMBlock",
    "RNNBackbone",
    "RNNLayer",
    "S5Backbone",
    "S5Block",
    "S5Layer",
    "SequenceBackboneOutput",
    "SelectiveDLinOSSBackbone",
    "SelectiveDLinOSSBlock",
    "SelectiveDLinOSSLayer",
    "TiedSoftmaxHead",
    "run_sdlinoss",
    "has_fast_kernels",
)
