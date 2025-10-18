"""PyTorch implementation of RNN variants from LinOSS."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

from ._rnn_scan import try_run_linear_rnn_scan
from .base import Backbone, SequenceBackboneOutput

__all__ = [
    "AbstractRNNCell",
    "LinearRNNCell",
    "GRURNNCell",
    "LSTMRNNCell",
    "MLPRNNCell",
    "RNNLayer",
    "RNNBackbone",
]

State = Union[Tensor, Tuple[Tensor, Tensor]]


class AbstractRNNCell(nn.Module):
    """Common interface for RNN cells."""

    hidden_size: int

    def init_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> State:
        raise NotImplementedError

    def forward(self, state: State, inputs: Tensor) -> State:  # pragma: no cover - interface
        raise NotImplementedError

    def hidden_from_state(self, state: State) -> Tensor:
        if not isinstance(state, torch.Tensor):
            raise TypeError("Expected tensor hidden state for this cell")
        return state

    def scan(self, inputs: Tensor, state: State) -> Optional[Tuple[Tensor, State]]:
        return None


class LinearRNNCell(AbstractRNNCell):
    """RNN cell with a single linear transformation on concatenated input and state."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.hidden_size = hidden_dim
        self.linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def init_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def forward(self, state: Tensor, inputs: Tensor) -> Tensor:
        combined = torch.cat([state, inputs], dim=-1)
        return self.linear(combined)

    def scan(self, inputs: Tensor, state: State) -> Optional[Tuple[Tensor, Tensor]]:
        if not isinstance(state, torch.Tensor):
            raise TypeError("LinearRNNCell expects tensor state")
        weight = self.linear.weight
        bias = self.linear.bias if self.linear.bias is not None else torch.zeros(
            self.hidden_size, device=weight.device, dtype=weight.dtype
        )
        weight_hh = weight[:, : self.hidden_size]
        weight_xh = weight[:, self.hidden_size :]

        fused = try_run_linear_rnn_scan(weight_hh, weight_xh, bias, inputs, state)
        if fused is not None:
            return fused, fused[-1]

        hidden = state
        outputs = inputs.new_empty(inputs.size(0), inputs.size(1), self.hidden_size)
        weight_hh_t = weight_hh.transpose(0, 1).contiguous()
        weight_xh_t = weight_xh.transpose(0, 1).contiguous()
        bias_vec = bias.to(dtype=inputs.dtype, device=inputs.device)
        hidden = hidden.to(dtype=inputs.dtype)

        for t in range(inputs.size(0)):
            hidden = torch.addmm(bias_vec, hidden, weight_hh_t)
            hidden = hidden + inputs[t].matmul(weight_xh_t)
            outputs[t] = hidden

        return outputs, hidden


class GRURNNCell(AbstractRNNCell):
    """Wrapper around :class:`torch.nn.GRUCell`."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.hidden_size = hidden_dim
        self.cell = nn.GRUCell(input_dim, hidden_dim)

    def init_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def forward(self, state: Tensor, inputs: Tensor) -> Tensor:
        return self.cell(inputs, state)


class LSTMRNNCell(AbstractRNNCell):
    """Wrapper around :class:`torch.nn.LSTMCell`."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.hidden_size = hidden_dim
        self.cell = nn.LSTMCell(input_dim, hidden_dim)

    def init_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> State:
        h = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return h, c

    def forward(self, state: State, inputs: Tensor) -> State:
        if not isinstance(state, tuple):
            raise TypeError("LSTMRNNCell expects tuple state")
        return self.cell(inputs, state)

    def hidden_from_state(self, state: State) -> Tensor:
        if not isinstance(state, tuple):
            raise TypeError("LSTMRNNCell expects tuple state")
        return state[0]


class MLPRNNCell(AbstractRNNCell):
    """RNN cell backed by a feed-forward network on the concatenated state and input."""

    def __init__(self, input_dim: int, hidden_dim: int, depth: int, width: int) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if depth < 0 or width <= 0:
            raise ValueError("depth must be non-negative and width must be positive")
        self.hidden_size = hidden_dim
        layers = []
        in_features = input_dim + hidden_dim
        if depth == 0:
            layers.append(nn.Linear(in_features, hidden_dim))
        else:
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.ReLU())
            for _ in range(depth - 1):
                layers.append(nn.Linear(width, width))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(width, hidden_dim))
        self.network = nn.Sequential(*layers)

    def init_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def forward(self, state: Tensor, inputs: Tensor) -> Tensor:
        combined = torch.cat([state, inputs], dim=-1)
        return self.network(combined)


class RNNLayer(nn.Module):
    """Sequence processor wrapping an RNN cell."""

    def __init__(self, cell: AbstractRNNCell) -> None:
        super().__init__()
        self.cell = cell

    def forward(self, inputs: Tensor, state: Optional[State] = None) -> Tuple[Tensor, State]:
        if inputs.dim() != 3:
            raise ValueError("RNNLayer expects input of shape (batch, length, input_dim).")
        batch, length, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        if state is None:
            state = self.cell.init_state(batch, device=device, dtype=dtype)

        if length == 0:
            hidden = self.cell.hidden_from_state(state)
            empty = hidden.new_zeros(batch, 0, hidden.size(-1))
            return empty, state

        inputs_time_major = inputs.permute(1, 0, 2).contiguous()

        if isinstance(self.cell, LinearRNNCell):
            result = self.cell.scan(inputs_time_major, state)
            if result is not None:
                states_seq, final_state = result
                features = states_seq.permute(1, 0, 2).contiguous()
                return features, final_state

        return self._sequential_scan(inputs_time_major, state)

    def _sequential_scan(self, inputs: Tensor, state: State) -> Tuple[Tensor, State]:
        outputs = []
        current_state = state
        for t in range(inputs.size(0)):
            current_state = self.cell(current_state, inputs[t])
            outputs.append(self.cell.hidden_from_state(current_state))
        features = torch.stack(outputs, dim=1)
        return features, current_state


class RNNBackbone(Backbone):
    """Backbone that exposes RNN sequence features and pooled states."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        cell: str = "linear",
        mlp_depth: int = 1,
        mlp_width: int = 128,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell = self._build_cell(cell, input_dim, hidden_dim, mlp_depth, mlp_width)
        self.layer = RNNLayer(self.cell)

    def _build_cell(
        self,
        cell: str,
        input_dim: int,
        hidden_dim: int,
        mlp_depth: int,
        mlp_width: int,
    ) -> AbstractRNNCell:
        cell = cell.lower()
        if cell == "linear":
            return LinearRNNCell(input_dim, hidden_dim)
        if cell == "gru":
            return GRURNNCell(input_dim, hidden_dim)
        if cell == "lstm":
            return LSTMRNNCell(input_dim, hidden_dim)
        if cell == "mlp":
            return MLPRNNCell(input_dim, hidden_dim, mlp_depth, mlp_width)
        raise ValueError(f"Unsupported RNN cell '{cell}'.")

    def forward(self, inputs: Tensor) -> SequenceBackboneOutput:
        if inputs.dim() != 3:
            raise ValueError("RNNBackbone expects input of shape (batch, length, input_dim).")
        if inputs.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, received {inputs.size(-1)}."
            )
        features, final_state = self.layer(inputs)
        pooled = self.cell.hidden_from_state(final_state)
        return SequenceBackboneOutput(features=features, pooled=pooled)
