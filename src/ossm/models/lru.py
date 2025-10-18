"""PyTorch implementation of the Linear Recurrent Unit (LRU)."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ._lru_scan import try_run_lru_scan
from .base import Backbone, SequenceBackboneOutput
from .linoss import GatedLinearUnit

__all__ = ["LRULayer", "LRUBlock", "LRUBackbone"]


def _complex_from_pair(param: Tensor) -> Tensor:
    return torch.view_as_complex(param.contiguous())


def _sequential_scan(lambda_bar: Tensor, b_seq: Tensor) -> Tensor:
    length, batch, state = b_seq.shape
    outputs = b_seq.new_zeros(length, batch, state)
    state_vec = b_seq.new_zeros(batch, state)
    for idx in range(length):
        state_vec = lambda_bar * state_vec + b_seq[idx]
        outputs[idx] = state_vec
    return outputs


class LRULayer(nn.Module):
    """Single Linear Recurrent Unit layer."""

    def __init__(
        self,
        ssm_size: int,
        hidden_dim: int,
        *,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 6.28,
    ) -> None:
        super().__init__()
        if ssm_size <= 0:
            raise ValueError("ssm_size must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if r_min < 0 or r_max <= r_min:
            raise ValueError("Require 0 <= r_min < r_max")
        if max_phase <= 0:
            raise ValueError("max_phase must be positive")

        self.ssm_size = ssm_size
        self.hidden_dim = hidden_dim
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.max_phase = float(max_phase)

        self.nu_log = nn.Parameter(torch.empty(ssm_size))
        self.theta_log = nn.Parameter(torch.empty(ssm_size))
        self.B = nn.Parameter(torch.empty(ssm_size, hidden_dim, 2))
        self.C = nn.Parameter(torch.empty(hidden_dim, ssm_size, 2))
        self.D = nn.Parameter(torch.empty(hidden_dim))
        self.gamma_log = nn.Parameter(torch.empty(ssm_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            u1 = torch.rand(self.ssm_size)
            u2 = torch.rand(self.ssm_size)
            radius_sq = self.r_min**2 + u1 * (self.r_max**2 - self.r_min**2)
            radius_sq = torch.clamp(radius_sq, min=1e-6)
            nu = -0.5 * torch.log(radius_sq)
            theta = torch.clamp(self.max_phase * u2, min=1e-6)
            self.nu_log.copy_(torch.log(nu))
            self.theta_log.copy_(torch.log(theta))

            std_b = math.sqrt(1.0 / (2.0 * self.hidden_dim))
            std_c = math.sqrt(1.0 / self.ssm_size)
            self.B.copy_(torch.randn_like(self.B) * std_b)
            self.C.copy_(torch.randn_like(self.C) * std_c)
            self.D.copy_(torch.randn_like(self.D))

            lambda_complex = torch.exp(
                -torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log)
            )
            gamma = torch.sqrt(torch.clamp(1.0 - torch.abs(lambda_complex) ** 2, min=1e-9))
            self.gamma_log.copy_(torch.log(gamma.real))

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() != 3:
            raise ValueError("LRULayer expects input of shape (batch, length, hidden_dim).")
        batch, length, hidden_dim = inputs.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, received {hidden_dim}."
            )
        if length == 0:
            return inputs.new_zeros(batch, 0, hidden_dim)

        device = inputs.device
        real_dtype = inputs.dtype
        complex_dtype = (
            torch.complex64
            if real_dtype in {torch.float16, torch.bfloat16, torch.float32}
            else torch.complex128
        )

        nu = self.nu_log.to(device=device, dtype=real_dtype)
        theta = self.theta_log.to(device=device, dtype=real_dtype)
        lambda_diag = torch.exp(-torch.exp(nu) + 1j * torch.exp(theta)).to(complex_dtype)

        gamma = torch.exp(self.gamma_log.to(device=device, dtype=real_dtype)).to(real_dtype)
        b_complex = _complex_from_pair(self.B).to(device=device, dtype=complex_dtype)
        b_complex = b_complex * gamma.unsqueeze(-1).to(dtype=complex_dtype)
        c_complex = _complex_from_pair(self.C).to(device=device, dtype=complex_dtype)
        d_vec = self.D.to(device=device, dtype=real_dtype)

        inputs_complex = inputs.to(dtype=real_dtype).to(dtype=complex_dtype)
        bu = torch.einsum("blh,ph->blp", inputs_complex, b_complex)
        bu_seq = bu.permute(1, 0, 2).contiguous()

        states_seq = try_run_lru_scan(lambda_diag, bu_seq)
        if states_seq is None:
            states_seq = _sequential_scan(lambda_diag, bu_seq)

        states = states_seq.permute(1, 0, 2).contiguous()
        projected = torch.einsum("blp,hp->blh", states, c_complex).real
        du = inputs * d_vec
        return projected + du


class LRUBlock(nn.Module):
    """LRU processing block with normalization, recurrence, GLU, and dropout."""

    def __init__(
        self,
        ssm_size: int,
        hidden_dim: int,
        *,
        dropout: float = 0.1,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 6.28,
    ) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(hidden_dim, affine=False)
        self.layer = LRULayer(
            ssm_size,
            hidden_dim,
            r_min=r_min,
            r_max=r_max,
            max_phase=max_phase,
        )
        self.activation = nn.GELU()
        self.glu = GatedLinearUnit(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() != 3:
            raise ValueError("LRUBlock expects input of shape (batch, length, hidden_dim).")
        batch, length, hidden_dim = inputs.shape
        residual = inputs
        normed = self.norm(inputs.reshape(-1, hidden_dim)).reshape(batch, length, hidden_dim)
        outputs = self.layer(normed)
        outputs = self.dropout(self.activation(outputs))
        outputs = self.glu(outputs)
        outputs = self.dropout(outputs)
        return outputs + residual


class LRUBackbone(Backbone):
    """Stacked LRU backbone."""

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        ssm_size: int,
        hidden_dim: int,
        *,
        dropout: float = 0.1,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 6.28,
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                LRUBlock(
                    ssm_size,
                    hidden_dim,
                    dropout=dropout,
                    r_min=r_min,
                    r_max=r_max,
                    max_phase=max_phase,
                )
                for _ in range(num_blocks)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> SequenceBackboneOutput:
        if x.dim() != 3:
            raise ValueError("LRUBackbone expects input of shape (batch, length, input_dim).")
        features = self.encoder(x)
        for block in self.blocks:
            features = block(features)
        features = self.dropout(features)
        pooled = features.mean(dim=1)
        return SequenceBackboneOutput(features=features, pooled=pooled)

