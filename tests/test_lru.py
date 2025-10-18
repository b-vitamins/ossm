"""Tests for the Linear Recurrent Unit implementation."""

from __future__ import annotations

import pytest
import torch

from ossm.models import LRUBackbone, LRULayer


def _sequential_scan(lambda_diag: torch.Tensor, bu: torch.Tensor) -> torch.Tensor:
    length, batch, state = bu.shape
    outputs = bu.new_zeros(length, batch, state)
    state_vec = bu.new_zeros(batch, state)
    for idx in range(length):
        state_vec = lambda_diag * state_vec + bu[idx]
        outputs[idx] = state_vec
    return outputs


def _reference_lru(layer: LRULayer, inputs: torch.Tensor) -> torch.Tensor:
    if inputs.dim() != 3:
        raise ValueError("inputs must be (batch, length, hidden_dim)")
    batch, length, hidden = inputs.shape
    if hidden != layer.hidden_dim:
        raise ValueError("hidden dimension mismatch")
    if length == 0:
        return inputs.new_zeros(batch, 0, hidden)

    device = inputs.device
    real_dtype = inputs.dtype
    complex_dtype = (
        torch.complex64 if real_dtype in {torch.float16, torch.bfloat16, torch.float32} else torch.complex128
    )

    nu = layer.nu_log.detach().to(device=device, dtype=real_dtype)
    theta = layer.theta_log.detach().to(device=device, dtype=real_dtype)
    lambda_diag = torch.exp(-torch.exp(nu) + 1j * torch.exp(theta)).to(dtype=complex_dtype)

    gamma = torch.exp(layer.gamma_log.detach().to(device=device, dtype=real_dtype)).to(dtype=real_dtype)
    b_complex = torch.view_as_complex(layer.B.detach().to(device=device, dtype=real_dtype)).to(dtype=complex_dtype)
    b_complex = b_complex * gamma.unsqueeze(-1).to(dtype=complex_dtype)
    c_complex = torch.view_as_complex(layer.C.detach().to(device=device, dtype=real_dtype)).to(dtype=complex_dtype)
    d_vec = layer.D.detach().to(device=device, dtype=real_dtype)

    inputs_complex = inputs.to(dtype=real_dtype).to(dtype=complex_dtype)
    bu = torch.einsum("blh,ph->blp", inputs_complex, b_complex)
    bu_seq = bu.permute(1, 0, 2).contiguous()
    states_seq = _sequential_scan(lambda_diag, bu_seq)
    states = states_seq.permute(1, 0, 2)

    projected = torch.einsum("blp,hp->blh", states, c_complex).real
    du = inputs * d_vec
    return projected + du


def test_lru_layer_matches_reference() -> None:
    torch.manual_seed(0)
    layer = LRULayer(ssm_size=8, hidden_dim=6)
    inputs = torch.randn(3, 11, 6)

    outputs = layer(inputs)
    reference = _reference_lru(layer, inputs)

    torch.testing.assert_close(outputs, reference, rtol=1e-5, atol=1e-6)


def test_lru_layer_zero_length() -> None:
    torch.manual_seed(1)
    layer = LRULayer(ssm_size=4, hidden_dim=3)
    inputs = torch.randn(2, 0, 3)

    outputs = layer(inputs)
    assert outputs.shape == (2, 0, 3)
    reference = _reference_lru(layer, inputs)
    torch.testing.assert_close(outputs, reference)


def test_lru_backbone_shapes() -> None:
    torch.manual_seed(2)
    backbone = LRUBackbone(
        num_blocks=2,
        input_dim=5,
        ssm_size=8,
        hidden_dim=6,
        dropout=0.0,
    )
    inputs = torch.randn(3, 13, 5)

    output = backbone(inputs)
    assert output.features.shape == (3, 13, 6)
    assert output.pooled.shape == (3, 6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_lru_layer_cuda_matches_cpu() -> None:
    torch.manual_seed(3)
    layer_cpu = LRULayer(ssm_size=6, hidden_dim=5)
    inputs_cpu = torch.randn(4, 19, 5)

    reference = layer_cpu(inputs_cpu)

    layer_cuda = LRULayer(ssm_size=6, hidden_dim=5).cuda()
    layer_cuda.load_state_dict(layer_cpu.state_dict())
    inputs_cuda = inputs_cpu.cuda()

    outputs_cuda = layer_cuda(inputs_cuda).cpu()
    torch.testing.assert_close(outputs_cuda, reference, rtol=1e-5, atol=1e-6)

