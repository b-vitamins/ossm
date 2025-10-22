"""PyTorch implementation of the LinOSS backbone."""

from __future__ import annotations

import math
from typing import Tuple, cast

import torch
from torch import Tensor, nn

from torch.autograd import Function

from ._linoss_scan import try_run_scan
from .base import Backbone, SequenceBackboneOutput


def _doubling_scan(a_matrix: Tensor, b_seq: Tensor) -> Tensor:
    """Binary-doubling parallel scan specialized for 2x2 diagonal blocks."""

    length = b_seq.shape[0]
    if length <= 1:
        return b_seq

    # Flatten the 2x2 matrices so we can apply elementwise fused formulas.
    a_flat = a_matrix.reshape(a_matrix.shape[0], 4)
    prefix_a = a_flat.unsqueeze(0).expand(length, -1, -1).clone()
    prefix_b = b_seq.clone()

    offset = 1
    while offset < length:
        right_a = prefix_a[offset:]
        left_a = prefix_a[:-offset]
        right_b = prefix_b[offset:]
        left_b = prefix_b[:-offset]

        ra11, ra12, ra21, ra22 = right_a.unbind(-1)
        la11, la12, la21, la22 = left_a.unbind(-1)

        c11 = ra11 * la11 + ra12 * la21
        c12 = ra11 * la12 + ra12 * la22
        c21 = ra21 * la11 + ra22 * la21
        c22 = ra21 * la12 + ra22 * la22
        combined_a = torch.stack((c11, c12, c21, c22), dim=-1)

        lb1, lb2 = left_b.unbind(-1)
        new_b1 = ra11.unsqueeze(1) * lb1 + ra12.unsqueeze(1) * lb2
        new_b2 = ra21.unsqueeze(1) * lb1 + ra22.unsqueeze(1) * lb2
        transformed_b_left = torch.stack((new_b1, new_b2), dim=-1)

        prefix_a[offset:] = combined_a
        prefix_b[offset:] = transformed_b_left + right_b
        offset <<= 1

    return prefix_b


class _LinossScanFn(Function):
    """Autograd-aware wrapper around the optimized LinOSS scan."""

    @staticmethod
    def forward(ctx, a_matrix: Tensor, b_seq: Tensor) -> Tensor:  # type: ignore[override]
        length = b_seq.size(0)
        if length <= 1:
            ctx.save_for_backward(a_matrix, b_seq)
            return b_seq

        b_seq = b_seq.contiguous()
        complex_dtype = b_seq.dtype
        m11 = a_matrix[:, 0, 0].to(dtype=complex_dtype).contiguous()
        m12 = a_matrix[:, 0, 1].to(dtype=complex_dtype).contiguous()
        m21 = a_matrix[:, 1, 0].to(dtype=complex_dtype).contiguous()
        m22 = a_matrix[:, 1, 1].to(dtype=complex_dtype).contiguous()
        ext_result = try_run_scan(m11, m12, m21, m22, b_seq)
        states: Tensor
        if ext_result is None:
            states = _doubling_scan(a_matrix, b_seq)
        else:
            states = cast(Tensor, ext_result)

        ctx.save_for_backward(a_matrix, states)
        return states

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor]:  # type: ignore[override]
        a_matrix, states = ctx.saved_tensors
        length = grad_output.size(0)
        if length == 0:
            return torch.zeros_like(a_matrix), grad_output

        grad_output = grad_output.contiguous()
        m11 = a_matrix[:, 0, 0]
        m12 = a_matrix[:, 0, 1]
        m21 = a_matrix[:, 1, 0]
        m22 = a_matrix[:, 1, 1]

        batch = grad_output.size(1)
        ssm = grad_output.size(2)
        device = grad_output.device
        dtype = grad_output.dtype

        grad_b = grad_output.new_zeros(grad_output.shape)
        grad_next = grad_output.new_zeros((batch, ssm, 2))

        grad_m11 = torch.zeros_like(m11)
        grad_m12 = torch.zeros_like(m12)
        grad_m21 = torch.zeros_like(m21)
        grad_m22 = torch.zeros_like(m22)

        m11_conj = m11.conj().view(1, ssm)
        m12_conj = m12.conj().view(1, ssm)
        m21_conj = m21.conj().view(1, ssm)
        m22_conj = m22.conj().view(1, ssm)

        for step in range(length - 1, -1, -1):
            grad_state = grad_output[step] + grad_next
            grad_b[step] = grad_state

            if step > 0:
                prev_state = states[step - 1]
            else:
                prev_state = torch.zeros((batch, ssm, 2), device=device, dtype=dtype)

            prev0 = prev_state[..., 0].conj()
            prev1 = prev_state[..., 1].conj()
            grad0 = grad_state[..., 0]
            grad1 = grad_state[..., 1]

            if torch.is_complex(m11):
                grad_m11 = grad_m11 + (grad0 * prev0).sum(dim=0)
                grad_m12 = grad_m12 + (grad0 * prev1).sum(dim=0)
                grad_m21 = grad_m21 + (grad1 * prev0).sum(dim=0)
                grad_m22 = grad_m22 + (grad1 * prev1).sum(dim=0)
            else:
                # Keep gradients real when inputs are real to avoid complex-to-real mismatch
                grad_m11 = grad_m11 + (grad0 * prev0).real.sum(dim=0)
                grad_m12 = grad_m12 + (grad0 * prev1).real.sum(dim=0)
                grad_m21 = grad_m21 + (grad1 * prev0).real.sum(dim=0)
                grad_m22 = grad_m22 + (grad1 * prev1).real.sum(dim=0)

            grad_prev0 = grad0 * m11_conj + grad1 * m21_conj
            grad_prev1 = grad0 * m12_conj + grad1 * m22_conj
            grad_next = torch.stack((grad_prev0, grad_prev1), dim=-1)

        grad_a = torch.stack(
            (
                torch.stack((grad_m11, grad_m12), dim=-1),
                torch.stack((grad_m21, grad_m22), dim=-1),
            ),
            dim=-2,
        )
        return grad_a, grad_b


def _run_associative_scan(a_matrix: Tensor, b_seq: Tensor) -> Tensor:
    """Evaluate the LinOSS recurrence via an associative parallel scan."""

    return cast(Tensor, _LinossScanFn.apply(a_matrix, b_seq))


class GatedLinearUnit(nn.Module):
    """Feature-wise gated linear unit."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gate = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x) * torch.sigmoid(self.gate(x))


class LinOSSLayer(nn.Module):
    """Single LinOSS state space layer."""

    def __init__(self, ssm_size: int, hidden_dim: int, discretization: str) -> None:
        super().__init__()
        if discretization not in {"IM", "IMEX"}:
            raise ValueError(f"Unsupported discretization '{discretization}'.")
        self.ssm_size = ssm_size
        self.hidden_dim = hidden_dim
        self.discretization = discretization

        self.A_diag = nn.Parameter(torch.empty(ssm_size))
        self.B = nn.Parameter(torch.empty(ssm_size, hidden_dim, 2))
        self.C = nn.Parameter(torch.empty(hidden_dim, ssm_size, 2))
        self.D = nn.Parameter(torch.empty(hidden_dim))
        self.steps = nn.Parameter(torch.empty(ssm_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        b_std = 1.0 / math.sqrt(self.hidden_dim)
        c_std = 1.0 / math.sqrt(self.ssm_size)
        nn.init.uniform_(self.A_diag, 0.0, 1.0)
        nn.init.uniform_(self.steps, 0.0, 1.0)
        nn.init.uniform_(self.B, -b_std, b_std)
        nn.init.uniform_(self.C, -c_std, c_std)
        nn.init.normal_(self.D, mean=0.0, std=1.0)

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the LinOSS layer."""

        if inputs.dim() != 3:
            raise ValueError("LinOSSLayer expects input of shape (batch, length, hidden_dim).")
        batch, length, hidden_dim = inputs.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, received {hidden_dim}."
            )

        device = inputs.device
        dtype = inputs.dtype

        a_diag = torch.relu(self.A_diag).to(device=device, dtype=dtype)
        step = torch.sigmoid(self.steps).to(device=device, dtype=dtype)

        b_real = self.B[..., 0].to(device=device, dtype=dtype)
        b_imag = self.B[..., 1].to(device=device, dtype=dtype)
        c_real = self.C[..., 0].to(device=device, dtype=dtype)
        c_imag = self.C[..., 1].to(device=device, dtype=dtype)
        d_vec = self.D.to(device=device, dtype=dtype)

        flat_inputs = inputs.to(dtype=dtype).reshape(batch * length, hidden_dim)
        b_real_t = b_real.transpose(0, 1)
        b_imag_t = b_imag.transpose(0, 1)
        bu_real = flat_inputs @ b_real_t
        bu_imag = flat_inputs @ b_imag_t
        bu = torch.complex(bu_real, bu_imag).reshape(batch, length, self.ssm_size)

        if self.discretization == "IM":
            outputs_complex = self._apply_im(a_diag, step, bu)
        else:
            outputs_complex = self._apply_imex(a_diag, step, bu)

        states = outputs_complex.reshape(batch * length, self.ssm_size)
        states_real = states.real
        states_imag = states.imag
        c_real_t = c_real.transpose(0, 1)
        c_imag_t = c_imag.transpose(0, 1)
        projected_real = states_real @ c_real_t - states_imag @ c_imag_t
        projected = projected_real.reshape(batch, length, self.hidden_dim)
        du = inputs * d_vec
        return projected + du

    def _apply_im(self, a_diag: Tensor, step: Tensor, bu: Tensor) -> Tensor:
        schur = 1.0 / (1.0 + (step**2) * a_diag)
        m11 = 1.0 - (step**2) * a_diag * schur
        m12 = -step * a_diag * schur
        m21 = step * schur
        m22 = schur

        return self._run_recurrence(m11, m12, m21, m22, step, bu)

    def _apply_imex(self, a_diag: Tensor, step: Tensor, bu: Tensor) -> Tensor:
        m11 = torch.ones_like(a_diag)
        m12 = -step * a_diag
        m21 = step
        m22 = 1.0 - (step**2) * a_diag
        return self._run_recurrence(m11, m12, m21, m22, step, bu, imex=True)

    def _run_recurrence(
        self,
        m11: Tensor,
        m12: Tensor,
        m21: Tensor,
        m22: Tensor,
        step: Tensor,
        bu: Tensor,
        *,
        imex: bool = False,
    ) -> Tensor:
        batch, length, _ = bu.shape
        if length == 0:
            return bu.new_zeros(batch, 0, self.ssm_size)

        device = bu.device
        dtype = bu.dtype
        matrix_dtype = dtype

        m11 = m11.to(device=device, dtype=matrix_dtype)
        m12 = m12.to(device=device, dtype=matrix_dtype)
        m21 = m21.to(device=device, dtype=matrix_dtype)
        m22 = m22.to(device=device, dtype=matrix_dtype)
        step = step.to(device=device, dtype=matrix_dtype)

        a_matrix = torch.stack(
            (
                torch.stack((m11, m12), dim=-1),
                torch.stack((m21, m22), dim=-1),
            ),
            dim=-2,
        ).to(device=device, dtype=matrix_dtype)

        bu_seq = bu.permute(1, 0, 2).contiguous()
        step_broadcast = step.view(1, 1, -1)
        if imex:
            f1 = bu_seq * step_broadcast
            f2 = bu_seq * (step_broadcast * step_broadcast)
        else:
            f1 = bu_seq * (m11 * step).view(1, 1, -1)
            f2 = bu_seq * (m21 * step).view(1, 1, -1)

        b_elems = torch.empty(
            bu_seq.size(0),
            bu_seq.size(1),
            bu_seq.size(2),
            2,
            dtype=bu_seq.dtype,
            device=bu_seq.device,
        )
        b_elems[..., 0] = f1
        b_elems[..., 1] = f2

        states = _run_associative_scan(a_matrix, b_elems)
        states = states.permute(1, 0, 2, 3).contiguous()
        return states[..., 1]


class LinOSSBlock(nn.Module):
    """LinOSS processing block with normalization, LinOSS layer and GLU."""

    def __init__(
        self,
        ssm_size: int,
        hidden_dim: int,
        discretization: str,
        *,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(hidden_dim, affine=False)
        self.layer = LinOSSLayer(ssm_size, hidden_dim, discretization)
        self.activation = nn.GELU()
        self.glu = GatedLinearUnit(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() != 3:
            raise ValueError("LinOSSBlock expects input of shape (batch, length, hidden_dim).")
        batch, length, hidden_dim = inputs.shape
        residual = inputs
        normed = self.norm(inputs.reshape(-1, hidden_dim)).reshape(batch, length, hidden_dim)
        outputs = self.layer(normed)
        outputs = self.dropout(self.activation(outputs))
        outputs = self.glu(outputs)
        outputs = self.dropout(outputs)
        return outputs + residual


class LinOSSBackbone(Backbone):
    """Full LinOSS backbone consisting of stacked LinOSS blocks."""

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        ssm_size: int,
        hidden_dim: int,
        discretization: str,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [LinOSSBlock(ssm_size, hidden_dim, discretization) for _ in range(num_blocks)]
        )

    def forward(self, x: Tensor) -> SequenceBackboneOutput:
        if x.dim() != 3:
            raise ValueError("LinOSSBackbone expects input of shape (batch, length, input_dim).")
        features = self.encoder(x)
        for block in self.blocks:
            features = block(features)
        pooled = features.mean(dim=1)
        return SequenceBackboneOutput(features=features, pooled=pooled)
