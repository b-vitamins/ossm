"""PyTorch implementation of the damped LinOSS backbone."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn, autocast

from ._dlinoss_scan import run_dlinoss_imex1
from .base import Backbone, ResidualSSMBlock, SequenceBackboneOutput
from .linoss import GatedLinearUnit

__all__ = [
    "DampedLinOSSLayer",
    "DampedLinOSSBlock",
    "DampedLinOSSBackbone",
]


@dataclass
class _InitializationConfig:
    initialization: str
    r_min: float
    r_max: float
    theta_min: float
    theta_max: float
    A_min: float
    A_max: float
    G_min: float
    G_max: float
    dt_std: float


def _safe_cat(samples: list[Tensor], target: int) -> Tensor:
    if not samples:
        return torch.empty(0)
    cat = torch.cat(samples)
    if cat.numel() >= target:
        return cat[:target]
    padding = target - cat.numel()
    return torch.cat((cat, cat.new_zeros(padding)))


class DampedLinOSSLayer(nn.Module):
    """Single Damped LinOSS state space layer."""

    def __init__(
        self,
        ssm_size: int,
        hidden_dim: int,
        *,
        variant: str = "imex1",
        initialization: str = "ring",
        r_min: float = 0.9,
        r_max: float = 1.0,
        theta_min: float = 0.0,
        theta_max: float = math.pi,
        A_min: float = 0.0,
        A_max: float = 1.0,
        G_min: float = 0.0,
        G_max: float = 1.0,
        dt_std: float = 0.5,
    ) -> None:
        super().__init__()
        variant = variant.lower()
        if variant != "imex1":
            raise ValueError("Only the 'imex1' damped LinOSS variant is supported.")
        self.variant = variant
        self.ssm_size = ssm_size
        self.hidden_dim = hidden_dim
        self.config = _InitializationConfig(
            initialization=initialization.lower(),
            r_min=float(r_min),
            r_max=float(r_max),
            theta_min=float(theta_min),
            theta_max=float(theta_max),
            A_min=float(A_min),
            A_max=float(A_max),
            G_min=float(G_min),
            G_max=float(G_max),
            dt_std=float(dt_std),
        )

        self.A_diag = nn.Parameter(torch.empty(ssm_size))
        self.G_diag = nn.Parameter(torch.empty(ssm_size))
        self.steps = nn.Parameter(torch.empty(ssm_size))
        self.B = nn.Parameter(torch.empty(ssm_size, hidden_dim, 2))
        self.C = nn.Parameter(torch.empty(hidden_dim, ssm_size, 2))
        self.D = nn.Parameter(torch.empty(hidden_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        b_std = 1.0 / math.sqrt(self.hidden_dim)
        c_std = 1.0 / math.sqrt(self.ssm_size)
        nn.init.uniform_(self.B, -b_std, b_std)
        nn.init.uniform_(self.C, -c_std, c_std)
        nn.init.normal_(self.D, mean=0.0, std=1.0)

        cfg = self.config
        if cfg.initialization == "uniform":
            a_vals, g_vals, dt_vals = self._uniform_init()
        elif cfg.initialization == "ring":
            a_vals, g_vals, dt_vals = self._ring_init()
        else:
            raise ValueError(f"Unknown initialization '{cfg.initialization}'.")

        with torch.no_grad():
            self.A_diag.copy_(a_vals.to(dtype=self.A_diag.dtype))
            self.G_diag.copy_(g_vals.to(dtype=self.G_diag.dtype))
            self.steps.copy_(dt_vals.to(dtype=self.steps.dtype))

    # ---------------------------------------------------------------------
    # Parameter initialization helpers
    # ---------------------------------------------------------------------
    def _uniform_init(self) -> Tuple[Tensor, Tensor, Tensor]:
        cfg = self.config
        device = self.A_diag.device
        dtype = self.A_diag.dtype
        batch = max(4 * self.ssm_size, 512)
        collected_a: list[Tensor] = []
        collected_g: list[Tensor] = []
        collected_dt: list[Tensor] = []

        while sum(t.numel() for t in collected_a) < self.ssm_size:
            A = torch.rand(batch, device=device, dtype=dtype) * (cfg.A_max - cfg.A_min) + cfg.A_min
            G = torch.rand(batch, device=device, dtype=dtype) * (cfg.G_max - cfg.G_min) + cfg.G_min
            dt = torch.randn(batch, device=device, dtype=dtype) * cfg.dt_std
            mask = self._is_valid(A, G, dt)
            if mask.any():
                collected_a.append(A[mask])
                collected_g.append(G[mask])
                collected_dt.append(dt[mask])

        A_vals = _safe_cat(collected_a, self.ssm_size)
        G_vals = _safe_cat(collected_g, self.ssm_size)
        dt_vals = _safe_cat(collected_dt, self.ssm_size)
        return A_vals, G_vals, dt_vals

    def _ring_init(self) -> Tuple[Tensor, Tensor, Tensor]:
        cfg = self.config
        device = self.A_diag.device
        dtype = self.A_diag.dtype

        dt_vals = torch.randn(self.ssm_size, device=device, dtype=dtype) * cfg.dt_std
        dt_sigmoid = torch.sigmoid(dt_vals)

        mags = torch.sqrt(
            torch.rand(self.ssm_size, device=device, dtype=dtype) * (cfg.r_max**2 - cfg.r_min**2) + cfg.r_min**2
        )
        args = torch.rand(self.ssm_size, device=device, dtype=dtype) * (cfg.theta_max - cfg.theta_min) + cfg.theta_min
        lam1 = torch.polar(mags, args)
        lam2 = lam1.conj()

        sums = (lam1 + lam2).real
        prods = (lam1 * lam2).real

        one = torch.ones_like(dt_sigmoid)
        g_vals = (one / torch.clamp(prods, min=1e-6) - one) / torch.clamp(dt_sigmoid, min=1e-6)
        g_vals = torch.clamp(g_vals, min=cfg.G_min)

        numerator = 2.0 + dt_sigmoid * g_vals - sums * (1.0 + dt_sigmoid * g_vals)
        denom = torch.clamp(dt_sigmoid * dt_sigmoid, min=1e-6)
        a_vals = numerator / denom

        return a_vals, g_vals, dt_vals

    def _is_valid(self, A_diag: Tensor, G_diag: Tensor, dt: Tensor) -> Tensor:
        step = torch.sigmoid(dt)
        return (G_diag >= 0) & (((G_diag - step * A_diag) ** 2 - 4.0 * A_diag) < 0)

    # ------------------------------------------------------------------
    # Core recurrence
    # ------------------------------------------------------------------
    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() != 3:
            raise ValueError("DampedLinOSSLayer expects input of shape (batch, length, hidden_dim).")
        batch, length, hidden_dim = inputs.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.hidden_dim}, received {hidden_dim}.")

        device = inputs.device
        dtype = inputs.dtype

        compute_dtype = dtype
        if dtype in (torch.float16, torch.bfloat16):
            compute_dtype = torch.float32

        scan_real_dtype = torch.float32
        scan_complex_dtype = torch.complex64
        if compute_dtype == torch.float64:
            scan_real_dtype = torch.float64
            scan_complex_dtype = torch.complex128

        a_diag, g_diag, step = self._project_parameters(device=device, dtype=compute_dtype)

        b_real = self.B[..., 0].to(device=device, dtype=compute_dtype)
        b_imag = self.B[..., 1].to(device=device, dtype=compute_dtype)
        c_real = self.C[..., 0].to(device=device, dtype=compute_dtype)
        c_imag = self.C[..., 1].to(device=device, dtype=compute_dtype)
        d_vec = self.D.to(device=device, dtype=compute_dtype)

        if compute_dtype == dtype:
            layer_inputs = inputs
        else:
            layer_inputs = inputs.to(dtype=compute_dtype)

        flat_inputs = layer_inputs.reshape(batch * length, hidden_dim)
        b_real_t = b_real.transpose(0, 1)
        b_imag_t = b_imag.transpose(0, 1)
        bu_real = flat_inputs @ b_real_t
        bu_imag = flat_inputs @ b_imag_t
        # Force complex64 to avoid ComplexHalf kernels
        bu = torch.complex(
            bu_real.to(dtype=scan_real_dtype),
            bu_imag.to(dtype=scan_real_dtype),
        ).to(dtype=scan_complex_dtype).reshape(batch, length, self.ssm_size)

        # Run the scan in fp32/complex64 regardless of surrounding dtype
        with autocast("cuda", enabled=False):
            outputs_complex = self._apply_damped_imex1(
                a_diag.to(dtype=scan_real_dtype),
                g_diag.to(dtype=scan_real_dtype),
                step.to(dtype=scan_real_dtype),
                bu.to(dtype=scan_complex_dtype),
            )

        states = outputs_complex.reshape(batch * length, self.ssm_size)
        states_real = states.real.to(dtype=compute_dtype)
        states_imag = states.imag.to(dtype=compute_dtype)

        c_real_t = c_real.transpose(0, 1).to(dtype=compute_dtype)
        c_imag_t = c_imag.transpose(0, 1).to(dtype=compute_dtype)
        projected_real = states_real @ c_real_t - states_imag @ c_imag_t
        projected = projected_real.reshape(batch, length, self.hidden_dim)
        du = layer_inputs * d_vec
        outputs = projected + du
        if compute_dtype == dtype:
            return outputs
        return outputs.to(dtype=dtype)

    def _project_parameters(self, *, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor, Tensor]:
        a_diag = self.A_diag.to(device=device, dtype=dtype)
        g_diag = self.G_diag.to(device=device, dtype=dtype)
        step = self.steps.to(device=device, dtype=dtype)

        step = torch.sigmoid(step)
        g_diag = torch.relu(g_diag)

        denom = torch.clamp(step * step, min=1e-6)
        s = step * g_diag
        base = torch.sqrt(torch.clamp(1.0 + s, min=1e-6))
        a_low = (2.0 + s - 2.0 * base) / denom
        a_high = (2.0 + s + 2.0 * base) / denom
        a_diag = a_low + torch.relu(a_diag - a_low) - torch.relu(a_diag - a_high)

        return a_diag, g_diag, step

    def _apply_damped_imex1(self, a_diag: Tensor, g_diag: Tensor, step: Tensor, bu: Tensor) -> Tensor:
        batch, length, _ = bu.shape
        if length == 0:
            return bu.new_zeros(batch, 0, self.ssm_size)

        bu_seq = bu.permute(1, 0, 2).contiguous()
        states = run_dlinoss_imex1(a_diag, g_diag, step, bu_seq)
        return states.permute(1, 0, 2).contiguous()


class DampedLinOSSBlock(ResidualSSMBlock):
    """Processing block composed of norm, D-LinOSS layer, and GLU."""

    def __init__(
        self,
        ssm_size: int,
        hidden_dim: int,
        *,
        variant: str = "imex1",
        initialization: str = "ring",
        r_min: float = 0.9,
        r_max: float = 1.0,
        theta_min: float = 0.0,
        theta_max: float = math.pi,
        A_min: float = 0.0,
        A_max: float = 1.0,
        G_min: float = 0.0,
        G_max: float = 1.0,
        dt_std: float = 0.5,
        dropout: float = 0.1,
    ) -> None:
        layer = DampedLinOSSLayer(
            ssm_size,
            hidden_dim,
            variant=variant,
            initialization=initialization,
            r_min=r_min,
            r_max=r_max,
            theta_min=theta_min,
            theta_max=theta_max,
            A_min=A_min,
            A_max=A_max,
            G_min=G_min,
            G_max=G_max,
            dt_std=dt_std,
        )
        super().__init__(
            hidden_dim,
            layer=layer,
            norm=nn.BatchNorm1d(hidden_dim, affine=False),
            activation=nn.GELU(),
            glu=GatedLinearUnit(hidden_dim, hidden_dim),
            dropout=dropout,
        )


class DampedLinOSSBackbone(Backbone):
    """Damped LinOSS backbone consisting of stacked blocks."""

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        ssm_size: int,
        hidden_dim: int,
        *,
        variant: str = "imex1",
        initialization: str = "ring",
        r_min: float = 0.9,
        r_max: float = 1.0,
        theta_min: float = 0.0,
        theta_max: float = math.pi,
        A_min: float = 0.0,
        A_max: float = 1.0,
        G_min: float = 0.0,
        G_max: float = 1.0,
        dt_std: float = 0.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                DampedLinOSSBlock(
                    ssm_size,
                    hidden_dim,
                    variant=variant,
                    initialization=initialization,
                    r_min=r_min,
                    r_max=r_max,
                    theta_min=theta_min,
                    theta_max=theta_max,
                    A_min=A_min,
                    A_max=A_max,
                    G_min=G_min,
                    G_max=G_max,
                    dt_std=dt_std,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: Tensor) -> SequenceBackboneOutput:
        if x.dim() != 3:
            raise ValueError("DampedLinOSSBackbone expects input of shape (batch, length, input_dim).")
        features = self.encoder(x)
        for block in self.blocks:
            features = block(features)
        pooled = features.mean(dim=1)
        return SequenceBackboneOutput(features=features, pooled=pooled)
