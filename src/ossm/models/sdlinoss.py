"""Selective D-LinOSS models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor, autocast, nn

from .base import Backbone, ResidualSSMBlock, SequenceBackboneOutput
from .linoss import GatedLinearUnit

__all__ = [
    "SelectiveDLinOSSLayer",
    "SelectiveDLinOSSBlock",
    "SelectiveDLinOSSBackbone",
    "run_sdlinoss",
]

_VALID_VARIANTS = ("imex1", "imex2", "im", "ex")


@torch.no_grad()
def _infer_real_dtype_from_complex(z: Tensor) -> torch.dtype:
    if z.dtype == torch.complex64:
        return torch.float32
    if z.dtype == torch.complex128:
        return torch.float64
    raise TypeError(f"Expected complex64/complex128, got {z.dtype}.")


def _expand_param(p: Tensor, L: int, B: int, M: int, *, device, dtype) -> Tensor:
    """Broadcast parameters to ``(L, B, M)``."""

    if p.dim() == 1 and p.shape[0] == M:
        return p.view(1, 1, M).to(device=device, dtype=dtype).expand(L, B, M)
    if p.dim() == 2 and p.shape == (L, M):
        return p.view(L, 1, M).to(device=device, dtype=dtype).expand(L, B, M)
    if p.dim() == 2 and p.shape == (B, M):
        return p.view(1, B, M).to(device=device, dtype=dtype).expand(L, B, M)
    if p.dim() == 3 and p.shape == (L, B, M):
        return p.to(device=device, dtype=dtype)
    raise ValueError(
        "Parameter must be (M,), (L,M), (B,M) or (L,B,M); "
        f"got {tuple(p.shape)}."
    )


def run_sdlinoss(
    variant: str,
    a_diag: Tensor,
    g_diag: Tensor,
    step: Tensor,
    bu_seq: Tensor,
) -> Tensor:
    """Pure PyTorch scan for the selective D-LinOSS recurrence."""

    variant = variant.lower()
    if variant not in _VALID_VARIANTS:
        raise ValueError(
            f"Unsupported variant '{variant}'. "
            f"Expected one of {', '.join(_VALID_VARIANTS)}."
        )

    if bu_seq.dim() != 3:
        raise ValueError("bu_seq must be (L,B,M) complex.")
    L, B, M = bu_seq.shape
    device = bu_seq.device
    c_real = _infer_real_dtype_from_complex(bu_seq)

    A = _expand_param(a_diag, L, B, M, device=device, dtype=c_real)
    G = _expand_param(g_diag, L, B, M, device=device, dtype=c_real)
    dt = _expand_param(step, L, B, M, device=device, dtype=c_real)
    dt = torch.clamp(dt, min=1e-6, max=1.0)

    z = torch.zeros(B, M, dtype=bu_seq.dtype, device=device)
    x = torch.zeros(B, M, dtype=bu_seq.dtype, device=device)

    xs: List[Tensor] = []
    if variant == "imex1":
        for t in range(L):
            S = 1.0 + dt[t] * G[t]
            z = (z + dt[t] * (-A[t] * x + bu_seq[t])) / torch.clamp(S, min=1e-6)
            x = x + dt[t] * z
            xs.append(x)
    elif variant == "imex2":
        for t in range(L):
            S = 1.0 + (dt[t] * dt[t]) * A[t]
            z = (z + dt[t] * (-A[t] * x - G[t] * z + bu_seq[t])) / torch.clamp(
                S, min=1e-6
            )
            x = x + dt[t] * z
            xs.append(x)
    elif variant == "im":
        for t in range(L):
            S = 1.0 + dt[t] * G[t] + (dt[t] * dt[t]) * A[t]
            z = (z + dt[t] * (-A[t] * x + bu_seq[t])) / torch.clamp(S, min=1e-6)
            x = x + dt[t] * z
            xs.append(x)
    else:  # "ex"
        for t in range(L):
            z = z + dt[t] * (-A[t] * x - G[t] * z + bu_seq[t])
            x = x + dt[t] * z
            xs.append(x)

    return torch.stack(xs, dim=0)


@dataclass
class _SelectiveConfig:
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
    selective_injection: bool
    per_step_dt: bool
    conv_kernel: int
class SelectiveDLinOSSLayer(nn.Module):
    """Selective D-LinOSS layer with spectral conditioning."""

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
        selective_injection: bool = True,
        per_step_dt: bool = False,
        conv_kernel: int = 4,
    ) -> None:
        super().__init__()
        variant = variant.lower()
        if variant not in _VALID_VARIANTS:
            raise ValueError(
                "SelectiveDLinOSSLayer received unsupported variant "
                f"'{variant}'. Expected one of {', '.join(_VALID_VARIANTS)}."
            )
        self.variant = variant
        self.ssm_size = ssm_size
        self.hidden_dim = hidden_dim
        self.cfg = _SelectiveConfig(
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
            selective_injection=bool(selective_injection),
            per_step_dt=bool(per_step_dt),
            conv_kernel=int(conv_kernel),
        )

        self.B = nn.Parameter(torch.empty(ssm_size, hidden_dim, 2))
        self.C = nn.Parameter(torch.empty(hidden_dim, ssm_size, 2))
        self.D = nn.Parameter(torch.empty(hidden_dim))

        k = self.cfg.conv_kernel
        pad = max(k - 1, 0)
        self.enc_linear = nn.Linear(hidden_dim, hidden_dim)
        self.enc_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=k,
            padding=pad,
            groups=hidden_dim,
        )
        self.enc_act = nn.SiLU()

        r0, th0 = self._init_ring_base()
        self.r_logit_base = nn.Parameter(torch.logit(r0))
        self.th_atanh_base = nn.Parameter(torch.atanh(th0 / math.pi))
        self.r_head = nn.Linear(hidden_dim, ssm_size, bias=False)
        self.th_head = nn.Linear(hidden_dim, ssm_size, bias=False)
        nn.init.zeros_(self.r_head.weight)
        nn.init.zeros_(self.th_head.weight)

        self.dt_base = nn.Parameter(torch.randn(ssm_size) * self.cfg.dt_std)
        self.dt_head = nn.Linear(hidden_dim, ssm_size, bias=False) if self.cfg.per_step_dt else None
        if self.dt_head is not None:
            nn.init.zeros_(self.dt_head.weight)

        self.inj_head = nn.Linear(hidden_dim, ssm_size, bias=False) if self.cfg.selective_injection else None
        if self.inj_head is not None:
            nn.init.zeros_(self.inj_head.weight)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        b_std = 1.0 / math.sqrt(self.hidden_dim)
        c_std = 1.0 / math.sqrt(self.ssm_size)
        nn.init.uniform_(self.B, -b_std, b_std)
        nn.init.uniform_(self.C, -c_std, c_std)
        nn.init.normal_(self.D, mean=0.0, std=1.0)

    def _init_ring_base(self) -> Tuple[Tensor, Tensor]:
        cfg = self.cfg
        device = torch.device("cpu")
        dtype = torch.float32
        mags = torch.sqrt(
            torch.rand(self.ssm_size, device=device, dtype=dtype)
            * (cfg.r_max**2 - cfg.r_min**2)
            + cfg.r_min**2
        )
        args = (
            torch.rand(self.ssm_size, device=device, dtype=dtype)
            * (cfg.theta_max - cfg.theta_min)
            + cfg.theta_min
        )
        return mags, args

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() != 3:
            raise ValueError(
                "SelectiveDLinOSSLayer expects input of shape (batch, length, hidden_dim)."
            )
        B, L, H = inputs.shape
        if H != self.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.hidden_dim}, received {H}.")

        device = inputs.device
        dtype = inputs.dtype
        compute_dtype = dtype if dtype not in (torch.float16, torch.bfloat16) else torch.float32

        scan_real_dtype = torch.float32 if compute_dtype != torch.float64 else torch.float64
        scan_complex_dtype = torch.complex64 if compute_dtype != torch.float64 else torch.complex128

        feats = self.enc_act(self.enc_linear(inputs))
        feats_c = feats.transpose(1, 2).contiguous()
        feats_c = self.enc_act(self.enc_conv(feats_c))
        feats = feats_c.transpose(1, 2).contiguous()
        if feats.size(1) != L:
            feats = feats[:, :L, :]

        delta_r = self.r_head(feats)
        delta_th = self.th_head(feats)
        base_r = self.r_logit_base.view(1, 1, -1).to(device=inputs.device, dtype=delta_r.dtype)
        base_th = self.th_atanh_base.view(1, 1, -1).to(device=inputs.device, dtype=delta_th.dtype)
        r = torch.sigmoid(base_r + delta_r).to(dtype=compute_dtype)
        theta = (math.pi * torch.tanh(base_th + delta_th)).to(dtype=compute_dtype)

        if self.dt_head is None:
            dt = torch.sigmoid(self.dt_base).view(1, 1, -1).expand(B, L, self.ssm_size)
        else:
            dt = torch.sigmoid(self.dt_base.view(1, 1, -1) + self.dt_head(feats))
        dt = dt.to(dtype=compute_dtype)

        r2 = torch.clamp(r * r, min=1e-8)
        cos_t = torch.cos(theta)
        dtc = torch.clamp(dt, min=1e-6)
        A = (r2 - 2.0 * r * cos_t + 1.0) / (dtc * dtc * r2)
        G = (1.0 - r2) / (dtc * r2)
        A = torch.clamp(A, min=0.0)
        G = torch.clamp(G, min=0.0)

        b_real = self.B[..., 0].to(device=device, dtype=compute_dtype)
        b_imag = self.B[..., 1].to(device=device, dtype=compute_dtype)
        flat_inputs = inputs.to(dtype=compute_dtype).reshape(B * L, H)
        bu_real = flat_inputs @ b_real.transpose(0, 1)
        bu_imag = flat_inputs @ b_imag.transpose(0, 1)
        bu = torch.complex(
            bu_real.to(dtype=scan_real_dtype),
            bu_imag.to(dtype=scan_real_dtype),
        ).to(dtype=scan_complex_dtype).reshape(B, L, self.ssm_size)

        if self.inj_head is not None:
            gate = torch.sigmoid(self.inj_head(feats)).to(dtype=scan_real_dtype)
            bu = bu * gate.to(dtype=bu.dtype)

        A_seq = A.to(dtype=scan_real_dtype).permute(1, 0, 2).contiguous()
        G_seq = G.to(dtype=scan_real_dtype).permute(1, 0, 2).contiguous()
        dt_seq = dtc.to(dtype=scan_real_dtype).permute(1, 0, 2).contiguous()
        bu_seq = bu.to(dtype=scan_complex_dtype).permute(1, 0, 2).contiguous()

        with autocast("cuda", enabled=False):
            x_seq = run_sdlinoss(self.variant, A_seq, G_seq, dt_seq, bu_seq)

        states = x_seq.permute(1, 0, 2).contiguous().reshape(B * L, self.ssm_size)
        states_real = states.real.to(dtype=compute_dtype)
        states_imag = states.imag.to(dtype=compute_dtype)

        c_real = self.C[..., 0].to(device=device, dtype=compute_dtype)
        c_imag = self.C[..., 1].to(device=device, dtype=compute_dtype)
        proj = states_real @ c_real.transpose(0, 1) - states_imag @ c_imag.transpose(0, 1)
        proj = proj.reshape(B, L, self.hidden_dim)

        out = proj + inputs.to(dtype=compute_dtype) * self.D.to(dtype=compute_dtype).view(1, 1, -1)
        return out.to(dtype=dtype) if compute_dtype != dtype else out


class SelectiveDLinOSSBlock(ResidualSSMBlock):
    """Residual block using :class:`SelectiveDLinOSSLayer`."""

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
        selective_injection: bool = True,
        per_step_dt: bool = False,
        conv_kernel: int = 4,
        dropout: float = 0.1,
    ) -> None:
        layer = SelectiveDLinOSSLayer(
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
            selective_injection=selective_injection,
            per_step_dt=per_step_dt,
            conv_kernel=conv_kernel,
        )
        super().__init__(
            hidden_dim,
            layer=layer,
            norm=nn.BatchNorm1d(hidden_dim, affine=False),
            activation=nn.GELU(),
            glu=GatedLinearUnit(hidden_dim, hidden_dim),
            dropout=dropout,
        )
        self.variant = variant


class SelectiveDLinOSSBackbone(Backbone):
    """Backbone composed of stacked :class:`SelectiveDLinOSSBlock`."""

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
        selective_injection: bool = True,
        per_step_dt: bool = False,
        conv_kernel: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.variant = variant
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                SelectiveDLinOSSBlock(
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
                    selective_injection=selective_injection,
                    per_step_dt=per_step_dt,
                    conv_kernel=conv_kernel,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: Tensor) -> SequenceBackboneOutput:
        if x.dim() != 3:
            raise ValueError(
                "SelectiveDLinOSSBackbone expects input of shape (batch, length, input_dim)."
            )
        features = self.encoder(x)
        for block in self.blocks:
            features = block(features)
        pooled = features.mean(dim=1)
        return SequenceBackboneOutput(features=features, pooled=pooled)

