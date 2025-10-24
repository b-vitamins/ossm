"""Structured State Space (S5) implementation."""

from __future__ import annotations

import math
from typing import Optional, Tuple, cast

import torch
from torch import Tensor, nn
from torch.autograd import Function

from ._s5_scan import try_run_s5_scan
from .base import Backbone, ResidualSSMBlock, SequenceBackboneOutput
from .linoss import GatedLinearUnit

__all__ = ["S5Layer", "S5Block", "S5Backbone"]


def _complex_from_pair(param: Tensor) -> Tensor:
    return torch.view_as_complex(param.contiguous())


def _make_hippo(size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    indices = torch.arange(size, device=device, dtype=dtype)
    p = torch.sqrt(1 + 2 * indices)
    a = torch.outer(p, p)
    a = torch.tril(a) - torch.diag(indices)
    return -a


def _make_nplr_hippo(size: int, *, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor, Tensor]:
    hippo = _make_hippo(size, device=device, dtype=dtype)
    p = torch.sqrt(torch.arange(size, device=device, dtype=dtype) + 0.5)
    b = torch.sqrt(2 * torch.arange(size, device=device, dtype=dtype) + 1.0)
    return hippo, p, b


def _make_dplr_hippo(size: int, *, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    a, p_vec, b_vec = _make_nplr_hippo(size, device=device, dtype=dtype)
    s = a + torch.outer(p_vec, p_vec)
    s_diag = torch.diagonal(s)
    lambda_real = torch.mean(s_diag) * torch.ones_like(s_diag)

    complex_dtype = torch.complex64 if dtype in {torch.float16, torch.bfloat16, torch.float32} else torch.complex128
    s_complex = (s * (-1j)).to(complex_dtype)
    lambda_imag, v = torch.linalg.eigh(s_complex)
    p_complex = p_vec.to(complex_dtype)
    b_complex = b_vec.to(complex_dtype)
    p_tilde = v.conj().T @ p_complex
    b_orig = b_vec
    b_tilde = v.conj().T @ b_complex
    return lambda_real + 1j * lambda_imag, p_tilde, b_tilde, v, b_orig


def _lecun_normal(shape: Tuple[int, ...], *, generator: torch.Generator, dtype: torch.dtype, device: torch.device) -> Tensor:
    fan_in = shape[-1]
    std = math.sqrt(1.0 / fan_in)
    return torch.randn(shape, generator=generator, device=device, dtype=dtype) * std


def _trunc_normal(shape: Tuple[int, ...], *, generator: torch.Generator, dtype: torch.dtype, device: torch.device) -> Tensor:
    tensor = torch.empty(shape, device=device, dtype=dtype)
    return nn.init.trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0, generator=generator)


def _init_vinvb(Vinv: Tensor, shape: Tuple[int, int], *, generator: torch.Generator, dtype: torch.dtype, device: torch.device) -> Tensor:
    base = _lecun_normal(shape, generator=generator, dtype=dtype, device=device)
    complex_dtype = torch.complex64 if dtype in {torch.float16, torch.bfloat16, torch.float32} else torch.complex128
    VinvB = Vinv.to(complex_dtype) @ base.to(complex_dtype)
    stacked = torch.stack((VinvB.real, VinvB.imag), dim=-1)
    return stacked


def _init_cv(
    init: str,
    V: Tensor,
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
    output_dim: int,
) -> Tensor:
    shape = (output_dim, V.size(0))
    if init == "trunc_standard_normal":
        base = _trunc_normal((*shape, 2), generator=generator, dtype=dtype, device=device)
    elif init == "lecun_normal":
        base = torch.stack(
            (
                _lecun_normal(shape, generator=generator, dtype=dtype, device=device),
                _lecun_normal(shape, generator=generator, dtype=dtype, device=device),
            ),
            dim=-1,
        )
    elif init == "complex_normal":
        std = math.sqrt(0.5)
        base = torch.randn((*shape, 2), generator=generator, dtype=dtype, device=device) * std
    else:
        raise ValueError(f"Unsupported C_init '{init}'.")

    complex_dtype = torch.complex64 if dtype in {torch.float16, torch.bfloat16, torch.float32} else torch.complex128
    complex_base = torch.view_as_complex(base.contiguous()).to(complex_dtype)
    projected = complex_base @ V.to(complex_dtype)
    return torch.view_as_real(projected.contiguous())


def _init_log_steps(num: int, *, generator: torch.Generator, dtype: torch.dtype, device: torch.device, dt_min: float, dt_max: float) -> Tensor:
    uniform = torch.rand((num, 1), generator=generator, device=device, dtype=dtype)
    log_dt_min = math.log(dt_min)
    log_dt_max = math.log(dt_max)
    return uniform * (log_dt_max - log_dt_min) + log_dt_min


def _sequential_scan(lambda_bar: Tensor, b_seq: Tensor) -> Tensor:
    length, batch, state = b_seq.shape
    outputs = b_seq.new_zeros(length, batch, state)
    state_vec = b_seq.new_zeros(batch, state)
    for idx in range(length):
        state_vec = lambda_bar * state_vec + b_seq[idx]
        outputs[idx] = state_vec
    return outputs


class _S5ScanFn(Function):
    """Autograd-aware wrapper around the fast S5 scan."""

    @staticmethod
    def forward(ctx, lambda_bar: Tensor, b_seq: Tensor) -> Tensor:  # type: ignore[override]
        if b_seq.size(0) == 0:
            ctx.save_for_backward(lambda_bar, b_seq)
            return b_seq

        ext_result = try_run_s5_scan(lambda_bar, b_seq)
        states: Tensor
        if ext_result is None:
            states = _sequential_scan(lambda_bar, b_seq)
        else:
            states = cast(Tensor, ext_result)

        ctx.save_for_backward(lambda_bar, states)
        return states

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor]:  # type: ignore[override]
        lambda_bar, states = ctx.saved_tensors
        length = grad_output.size(0)
        if length == 0:
            return torch.zeros_like(lambda_bar), grad_output

        grad_output = grad_output.contiguous()
        batch = grad_output.size(1)
        state = grad_output.size(2)
        device = grad_output.device

        grad_lambda = torch.zeros_like(lambda_bar)
        grad_b = grad_output.new_zeros(grad_output.shape)
        grad_next = grad_output.new_zeros((batch, state))
        lambda_conj = lambda_bar.conj().view(1, state)

        for step in range(length - 1, -1, -1):
            grad_state = grad_output[step] + grad_next
            grad_b[step] = grad_state
            if step > 0:
                prev_state = states[step - 1]
            else:
                prev_state = torch.zeros((batch, state), device=device, dtype=grad_output.dtype)

            grad_lambda = grad_lambda + (grad_state * prev_state.conj()).sum(dim=0)
            grad_next = grad_state * lambda_conj

        return grad_lambda, grad_b


class S5Layer(nn.Module):
    """Single S5 state space layer."""

    def __init__(
        self,
        ssm_size: int,
        hidden_dim: int,
        *,
        blocks: int = 1,
        C_init: str = "lecun_normal",
        conj_sym: bool = True,
        clip_eigs: bool = False,
        discretization: str = "zoh",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        step_rescale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if discretization not in {"zoh", "bilinear"}:
            raise ValueError(f"Unsupported discretization '{discretization}'.")
        if ssm_size % blocks != 0:
            raise ValueError("ssm_size must be divisible by blocks.")
        if conj_sym and ssm_size % 2 != 0:
            raise ValueError("ssm_size must be even when conj_sym is enabled.")

        device = torch.device("cpu") if device is None else device
        dtype = torch.get_default_dtype() if dtype is None else dtype
        generator = generator if generator is not None else torch.Generator(device=device)

        block_size = ssm_size // blocks
        lambda_init, _, _, V, _ = _make_dplr_hippo(block_size, device=device, dtype=dtype)

        if conj_sym:
            block_size //= 2

        lambda_init = lambda_init[:block_size]
        V = V[:, :block_size]
        Vinv = V.conj().T

        lambda_init = lambda_init.repeat(blocks)
        V = torch.block_diag(*([V] * blocks))
        Vinv = torch.block_diag(*([Vinv] * blocks))

        self.hidden_dim = hidden_dim
        self.state_size = lambda_init.shape[0]
        self.conj_sym = conj_sym
        self.clip_eigs = clip_eigs
        self.discretization = discretization
        self.step_rescale = step_rescale

        self.lambda_real = nn.Parameter(lambda_init.real.to(device=device, dtype=dtype))
        self.lambda_imag = nn.Parameter(lambda_init.imag.to(device=device, dtype=dtype))
        self.B = nn.Parameter(
            _init_vinvb(Vinv, (Vinv.size(1), hidden_dim), generator=generator, dtype=dtype, device=device)
        )

        self.C = nn.Parameter(
            _init_cv(
                C_init,
                V,
                generator=generator,
                dtype=dtype,
                device=device,
                output_dim=hidden_dim,
            )
        )

        self.D = nn.Parameter(torch.randn(hidden_dim, generator=generator, device=device, dtype=dtype))
        self.log_step = nn.Parameter(
            _init_log_steps(
                self.state_size,
                generator=generator,
                dtype=dtype,
                device=device,
                dt_min=dt_min,
                dt_max=dt_max,
            )
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() != 3:
            raise ValueError("S5Layer expects input of shape (batch, length, hidden_dim).")
        batch, length, hidden = inputs.shape
        if hidden != self.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.hidden_dim}, received {hidden}.")
        if length == 0:
            return inputs.new_zeros(batch, 0, hidden)

        device = inputs.device
        real_dtype = inputs.dtype
        complex_dtype = torch.complex64 if real_dtype in {torch.float16, torch.bfloat16, torch.float32} else torch.complex128

        lambda_real = self.lambda_real.to(device=device, dtype=real_dtype)
        lambda_imag = self.lambda_imag.to(device=device, dtype=real_dtype)
        lambda_complex = torch.complex(lambda_real, lambda_imag).to(dtype=complex_dtype)
        if self.clip_eigs:
            lambda_complex = torch.complex(torch.clamp(lambda_real, max=-1e-4), lambda_imag).to(dtype=complex_dtype)

        step = self.step_rescale * torch.exp(self.log_step.squeeze(-1)).to(device=device, dtype=real_dtype)
        step_complex = step.to(device=device, dtype=complex_dtype)

        b_complex = _complex_from_pair(self.B.to(device=device, dtype=real_dtype)).to(dtype=complex_dtype)
        c_complex = _complex_from_pair(self.C.to(device=device, dtype=real_dtype)).to(dtype=complex_dtype)

        if self.discretization == "zoh":
            lambda_bar = torch.exp(lambda_complex * step_complex)
            denom = torch.where(lambda_complex == 0, torch.ones_like(lambda_complex), lambda_complex)
            b_bar = ((lambda_bar - 1) / denom).unsqueeze(-1) * b_complex
        else:
            identity = torch.ones_like(lambda_complex)
            bl = 1.0 / (identity - 0.5 * step_complex * lambda_complex)
            lambda_bar = bl * (identity + 0.5 * step_complex * lambda_complex)
            b_bar = (bl * step_complex).unsqueeze(-1) * b_complex

        inputs_complex = inputs.to(dtype=real_dtype).to(dtype=complex_dtype)
        bu = torch.einsum("blh,ph->blp", inputs_complex, b_bar)

        bu_seq = bu.permute(1, 0, 2).contiguous()
        states_seq = cast(Tensor, _S5ScanFn.apply(lambda_bar, bu_seq))
        states = states_seq.permute(1, 0, 2)

        projected = torch.einsum("blp,hp->blh", states, c_complex)
        if self.conj_sym:
            projected = 2.0 * projected.real
        else:
            projected = projected.real

        du = inputs * self.D.to(device=device, dtype=real_dtype)
        return projected + du


class S5Block(ResidualSSMBlock):
    """S5 processing block with normalization, S5 layer and GLU."""

    def __init__(
        self,
        ssm_size: int,
        hidden_dim: int,
        *,
        blocks: int = 1,
        C_init: str = "lecun_normal",
        conj_sym: bool = True,
        clip_eigs: bool = False,
        discretization: str = "zoh",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        step_rescale: float = 1.0,
        dropout: float = 0.05,
    ) -> None:
        layer = S5Layer(
            ssm_size,
            hidden_dim,
            blocks=blocks,
            C_init=C_init,
            conj_sym=conj_sym,
            clip_eigs=clip_eigs,
            discretization=discretization,
            dt_min=dt_min,
            dt_max=dt_max,
            step_rescale=step_rescale,
        )
        super().__init__(
            hidden_dim,
            layer=layer,
            norm=nn.BatchNorm1d(hidden_dim, affine=False),
            activation=nn.GELU(),
            glu=GatedLinearUnit(hidden_dim, hidden_dim),
            dropout=dropout,
        )


class S5Backbone(Backbone):
    """Full S5 backbone built from stacked S5 blocks."""

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        ssm_size: int,
        ssm_blocks: int,
        hidden_dim: int,
        *,
        C_init: str = "lecun_normal",
        conj_sym: bool = True,
        clip_eigs: bool = False,
        discretization: str = "zoh",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        step_rescale: float = 1.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                S5Block(
                    ssm_size,
                    hidden_dim,
                    blocks=ssm_blocks,
                    C_init=C_init,
                    conj_sym=conj_sym,
                    clip_eigs=clip_eigs,
                    discretization=discretization,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    step_rescale=step_rescale,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: Tensor) -> SequenceBackboneOutput:
        if x.dim() != 3:
            raise ValueError("S5Backbone expects input of shape (batch, length, input_dim).")
        features = self.encoder(x)
        for block in self.blocks:
            features = block(features)
        pooled = features.mean(dim=1)
        return SequenceBackboneOutput(features=features, pooled=pooled)
