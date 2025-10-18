"""PyTorch implementations of Neural CDE variants."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, cast

import torch
from torch import Tensor, nn

import torchcde
from torchdiffeq import odeint

from .base import Backbone, SequenceBackboneOutput

__all__ = [
    "NCDEVectorField",
    "NCDELayer",
    "NRDELayer",
    "NCDEBackbone",
]


def _build_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation '{name}'.")


class _ScaledMLP(nn.Module):
    """Feed-forward network with optional activation scaling."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_width: int,
        depth: int,
        activation: str = "relu",
        final_activation: Optional[str] = "tanh",
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        if depth < 0:
            raise ValueError("depth must be non-negative")

        layers = []
        current_dim = in_dim
        act = _build_activation(activation)
        for _ in range(depth):
            linear = nn.Linear(current_dim, hidden_width)
            layers.append(linear)
            layers.append(act.__class__())
            current_dim = hidden_width

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(current_dim, out_dim)
        self.final_activation = None if final_activation is None else _build_activation(final_activation)
        self.scale = float(scale)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.hidden:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                module.weight.data.div_(self.scale)
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1.0 / math.sqrt(fan_in)
                    nn.init.uniform_(module.bias, -bound, bound)
                    module.bias.data.div_(self.scale)

        nn.init.kaiming_uniform_(self.output.weight, a=math.sqrt(5))
        self.output.weight.data.div_(self.scale)
        if self.output.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.output.weight)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.output.bias, -bound, bound)
            self.output.bias.data.div_(self.scale)

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.hidden(inputs)
        out = self.output(out)
        if self.final_activation is not None:
            out = self.final_activation(out)
        return out


class NCDEVectorField(nn.Module):
    """Vector field used by Neural CDE variants."""

    def __init__(
        self,
        hidden_dim: int,
        input_dim: int,
        *,
        width: int,
        depth: int,
        activation: str = "relu",
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.mlp = _ScaledMLP(
            hidden_dim,
            hidden_dim * input_dim,
            hidden_width=width,
            depth=depth,
            activation=activation,
            final_activation="tanh",
            scale=scale,
        )

    def forward(self, t: Tensor, state: Tensor) -> Tensor:
        """Evaluate the vector field.

        Args:
            t: Time tensor (ignored, included for signature compatibility).
            state: Hidden state of shape ``(batch, hidden_dim)``.

        Returns:
            Tensor of shape ``(batch, hidden_dim, input_dim)``.
        """

        del t  # unused
        output = self.mlp(state)
        return output.view(state.shape[0], self.hidden_dim, self.input_dim)


def _canonicalize_times(times: Tensor, length: int) -> Tensor:
    if times.dim() == 2:
        if times.size(0) != 1:
            raise ValueError("Batch-specific time grids are not supported.")
        times = times.squeeze(0)
    if times.dim() != 1:
        raise ValueError("times must be a one-dimensional tensor")
    if times.numel() != length:
        raise ValueError("times and coefficients length mismatch")
    return times


def _resolve_cde_method(method: str) -> str:
    alias = method.lower()
    if alias == "heun2":
        return "rk4"
    return method


class NCDELayer(nn.Module):
    """Single Neural CDE layer leveraging torchcde for integration."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        vf_width: int,
        vf_depth: int,
        activation: str = "relu",
        scale: float = 1.0,
        solver: str = "heun2",
        step_size: float = 1.0,
        rtol: float = 1e-4,
        atol: float = 1e-5,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if vf_depth < 0:
            raise ValueError("vf_depth must be non-negative")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vector_field = NCDEVectorField(
            hidden_dim,
            input_dim,
            width=vf_width,
            depth=vf_depth,
            activation=activation,
            scale=scale,
        )
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.method = solver
        self.step_size = float(step_size)
        self.rtol = rtol
        self.atol = atol

    def forward(
        self,
        times: Tensor,
        coeffs: Tensor,
        initial: Tensor,
        *,
        evaluation_times: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if initial.dim() != 2 or initial.size(-1) != self.input_dim:
            raise ValueError("initial must have shape (batch, input_dim)")
        if coeffs.dim() != 3 or coeffs.size(-1) != 4 * self.input_dim:
            raise ValueError("coeffs must have shape (batch, intervals, 4 * input_dim)")
        _, intervals, _ = coeffs.shape
        times = _canonicalize_times(times, intervals + 1).to(device=initial.device)
        if evaluation_times is None:
            evaluation_times = times
        else:
            evaluation_times = evaluation_times.to(device=initial.device)
        if evaluation_times.dim() != 1:
            raise ValueError("evaluation_times must be one-dimensional")
        hidden0 = self.input_linear(initial.to(device=times.device))
        if intervals == 0:
            features = hidden0.unsqueeze(1).expand(-1, evaluation_times.numel(), -1).contiguous()
            return features, hidden0

        spline = torchcde.CubicSpline(
            coeffs.to(device=hidden0.device, dtype=hidden0.dtype),
            t=times.to(device=hidden0.device, dtype=hidden0.dtype),
        )
        solve_times = evaluation_times.to(device=hidden0.device, dtype=hidden0.dtype)
        method = _resolve_cde_method(self.method)
        options = {"step_size": self.step_size} if self.step_size > 0 else None
        solution = torchcde.cdeint(
            X=spline,
            z0=hidden0,
            func=self.vector_field,
            t=solve_times,
            method=method,
            options=options,
            rtol=self.rtol,
            atol=self.atol,
        )
        if isinstance(solution, tuple):
            solution = cast(Tensor, solution[0])
        features = solution.contiguous()
        final_state = features[:, -1]
        return features, final_state


class NRDELayer(nn.Module):
    """Neural RDE layer operating on log-signature inputs."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        logsig_dim: int,
        intervals: Sequence[float],
        *,
        vf_width: int,
        vf_depth: int,
        activation: str = "relu",
        scale: float = 1.0,
        solver: str = "heun2",
        step_size: float = 1.0,
        rtol: float = 1e-4,
        atol: float = 1e-5,
    ) -> None:
        super().__init__()
        if logsig_dim < 1:
            raise ValueError("logsig_dim must be at least 1")
        if vf_depth <= 0:
            raise ValueError("vf_depth must be positive")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.logsig_dim = logsig_dim - 1
        self.vector_field = _ScaledMLP(
            hidden_dim,
            vf_width,
            hidden_width=vf_width,
            depth=vf_depth - 1,
            activation=activation,
            final_activation="tanh",
            scale=scale,
        )
        self.mlp_linear = nn.Linear(vf_width, hidden_dim * self.logsig_dim)
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.register_buffer(
            "intervals",
            torch.as_tensor(intervals, dtype=torch.float32),
            persistent=False,
        )
        self.intervals: Tensor
        if self.intervals.ndim != 1 or self.intervals.numel() < 2:
            raise ValueError("intervals must be a one-dimensional tensor with at least two elements")
        self.method = solver
        self.step_size = float(step_size)
        self.rtol = rtol
        self.atol = atol

    def forward(
        self,
        times: Tensor,
        logsig: Tensor,
        initial: Tensor,
        *,
        evaluation_times: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if initial.dim() != 2 or initial.size(-1) != self.input_dim:
            raise ValueError("initial must have shape (batch, input_dim)")
        if logsig.dim() != 3 or logsig.size(-1) != self.logsig_dim + 1:
            raise ValueError("logsig must have shape (batch, intervals, logsig_dim)")

        batch, segments, _ = logsig.shape
        if segments != self.intervals.numel() - 1:
            raise ValueError("logsig segments must align with intervals")

        base_intervals = self.intervals.to(device=initial.device)
        times = _canonicalize_times(times, base_intervals.numel()).to(device=initial.device, dtype=base_intervals.dtype)
        if not torch.allclose(times, base_intervals):
            raise ValueError("times must match the registered intervals")

        hidden0 = self.input_linear(initial)
        if evaluation_times is None:
            evaluation_times = base_intervals.to(device=hidden0.device, dtype=hidden0.dtype)
        else:
            evaluation_times = evaluation_times.to(device=hidden0.device, dtype=hidden0.dtype)

        logsig = logsig.to(device=hidden0.device, dtype=hidden0.dtype)

        def func(t: Tensor, y: Tensor) -> Tensor:
            intervals = base_intervals.to(device=t.device, dtype=t.dtype)
            idx = torch.searchsorted(intervals, t.unsqueeze(0), right=False)
            idx = torch.clamp(idx, 1, intervals.numel() - 1)
            idx_int = int(idx.item())
            dt = intervals[idx_int] - intervals[idx_int - 1]
            coeff = logsig[:, idx_int - 1, 1:]
            vf = self.vector_field(y)
            vf = self.mlp_linear(vf).view(batch, self.hidden_dim, self.logsig_dim)
            return torch.einsum("bhd,bd->bh", vf, coeff) / dt

        options = {"step_size": self.step_size}
        solution = odeint(
            func,
            hidden0,
            evaluation_times,
            method=self.method,
            options=options,
            rtol=self.rtol,
            atol=self.atol,
        )
        if isinstance(solution, tuple):
            solution = cast(Tensor, solution[0])
        features = solution.permute(1, 0, 2).contiguous()
        final_state = features[:, -1]
        return features, final_state


@dataclass
class NCDEBatch:
    times: Tensor
    coeffs: Optional[Tensor] = None
    logsig: Optional[Tensor] = None
    initial: Optional[Tensor] = None
    evaluation_times: Optional[Tensor] = None


class NCDEBackbone(Backbone):
    """Backbone wrapping Neural CDE or Neural RDE layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        vf_width: int,
        vf_depth: int,
        activation: str = "relu",
        scale: float = 1.0,
        solver: str = "heun2",
        step_size: float = 1.0,
        rtol: float = 1e-4,
        atol: float = 1e-5,
        mode: str = "ncde",
        logsig_dim: Optional[int] = None,
        intervals: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        mode = mode.lower()
        self.mode = mode
        if mode == "ncde":
            self.layer = NCDELayer(
                input_dim,
                hidden_dim,
                vf_width=vf_width,
                vf_depth=vf_depth,
                activation=activation,
                scale=scale,
                solver=solver,
                step_size=step_size,
                rtol=rtol,
                atol=atol,
            )
        elif mode == "nrde":
            if logsig_dim is None or intervals is None:
                raise ValueError("logsig_dim and intervals are required for NRDE mode")
            self.layer = NRDELayer(
                input_dim,
                hidden_dim,
                logsig_dim,
                intervals,
                vf_width=vf_width,
                vf_depth=vf_depth,
                activation=activation,
                scale=scale,
                solver=solver,
                step_size=step_size,
                rtol=rtol,
                atol=atol,
            )
        else:
            raise ValueError("mode must be 'ncde' or 'nrde'")

    def forward(self, batch: NCDEBatch | Dict[str, Tensor]) -> SequenceBackboneOutput:
        if isinstance(batch, dict):
            batch = NCDEBatch(**batch)

        if batch.initial is None:
            raise ValueError("initial state must be provided")
        if batch.times is None:
            raise ValueError("times must be provided")

        if self.mode == "ncde":
            if batch.coeffs is None:
                raise ValueError("coeffs are required for NCDE mode")
            features, final = self.layer(
                batch.times,
                batch.coeffs,
                batch.initial,
                evaluation_times=batch.evaluation_times,
            )
        else:
            if batch.logsig is None:
                raise ValueError("logsig is required for NRDE mode")
            features, final = self.layer(
                batch.times,
                batch.logsig,
                batch.initial,
                evaluation_times=batch.evaluation_times,
            )

        return SequenceBackboneOutput(features=features, pooled=final)

