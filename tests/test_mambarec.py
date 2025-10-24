"""Tests for the Mamba selective scan implementation."""

from __future__ import annotations

from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.testing import assert_close

from ossm.models.mambarec import _MambaMixer, _selective_scan_discretized


def _naive_selective_scan(
    inputs: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B_t: torch.Tensor,
    C_t: torch.Tensor,
) -> torch.Tensor:
    batch, channels, seqlen = inputs.shape
    state_size = A.size(1)
    state = torch.zeros(batch, channels, state_size, device=inputs.device, dtype=torch.float32)
    outputs: list[torch.Tensor] = []

    u = inputs.to(torch.float32)
    dt = dt.to(torch.float32)
    A_matrix = A.to(torch.float32)
    B_proj = B_t.to(torch.float32)
    C_proj = C_t.to(torch.float32)

    for timestep in range(seqlen):
        dt_t = dt[:, :, timestep].unsqueeze(-1)
        u_t = u[:, :, timestep].unsqueeze(-1)
        B_step = B_proj[:, timestep, :].unsqueeze(1)
        C_step = C_proj[:, timestep, :]

        dA = dt_t * A_matrix
        A_bar = torch.exp(dA)
        phi = torch.expm1(dA) / A_matrix

        state = (A_bar * state) + (phi * B_step) * u_t
        outputs.append(torch.einsum("bcn,bn->bc", state, C_step))

    return torch.stack(outputs, dim=-1).to(dtype=inputs.dtype)


def test_selective_scan_discretized_matches_naive() -> None:
    torch.manual_seed(0)
    batch, channels, seqlen, state = 2, 3, 5, 4

    inputs = torch.randn(batch, channels, seqlen, requires_grad=True)
    dt = torch.rand(batch, channels, seqlen, dtype=inputs.dtype).abs()
    A = -torch.rand(channels, state, dtype=inputs.dtype)
    B_t = torch.randn(batch, seqlen, state, dtype=inputs.dtype)
    C_t = torch.randn(batch, seqlen, state, dtype=inputs.dtype)

    actual = _selective_scan_discretized(inputs, dt, A, B_t, C_t)
    expected = _naive_selective_scan(inputs, dt, A, B_t, C_t)

    assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    grad_actual = torch.autograd.grad(actual.sum(), inputs, retain_graph=True)[0]
    grad_expected = torch.autograd.grad(expected.sum(), inputs)[0]
    assert_close(grad_actual, grad_expected, atol=1e-5, rtol=1e-5)


def test_mamba_mixer_fallback_matches_fused() -> None:
    torch.manual_seed(0)
    mixer = _MambaMixer(
        d_model=8,
        d_state=4,
        d_conv=1,
        expand=1,
        dt_rank=2,
    )
    mixer.eval()

    inputs = torch.randn(2, 6, 8, requires_grad=True)

    with patch("ossm.models.mambarec._try_selective_scan", return_value=None):
        fallback_out = mixer(inputs)

    fallback_grad = torch.autograd.grad(fallback_out.sum(), inputs, retain_graph=True)[0]

    inputs_fused = inputs.detach().clone().requires_grad_(True)

    def fake_selective_scan(
        *,
        inputs: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        gate: torch.Tensor | None,
    ) -> torch.Tensor:
        outputs = _selective_scan_discretized(inputs, dt, A, B, C)
        if gate is None:
            return outputs
        return outputs * F.silu(gate)

    with patch("ossm.models.mambarec._try_selective_scan", side_effect=fake_selective_scan):
        fused_out = mixer(inputs_fused)

    fused_grad = torch.autograd.grad(fused_out.sum(), inputs_fused)[0]

    assert_close(fallback_out, fused_out, atol=1e-5, rtol=1e-5)
    assert_close(fallback_grad, fused_grad, atol=1e-5, rtol=1e-5)
