"""Tests for the PyTorch RNN implementation."""

from __future__ import annotations

import pytest
import torch

from ossm.models import GRURNNCell, LinearRNNCell, MLPRNNCell, RNNBackbone, RNNLayer


@pytest.mark.parametrize("cell_cls", [LinearRNNCell, GRURNNCell, MLPRNNCell])
def test_rnn_layer_matches_manual(cell_cls) -> None:
    torch.manual_seed(0)
    batch, length, input_dim, hidden_dim = 3, 5, 4, 6
    inputs = torch.randn(batch, length, input_dim)

    if cell_cls is MLPRNNCell:
        cell = cell_cls(input_dim, hidden_dim, depth=2, width=8)
    else:
        cell = cell_cls(input_dim, hidden_dim)
    layer = RNNLayer(cell)

    features, final_state = layer(inputs)

    # Manual sequential reference
    state = cell.init_state(batch, device=inputs.device, dtype=inputs.dtype)
    outputs = []
    current_state = state
    for t in range(length):
        current_state = cell(current_state, inputs[:, t])
        hidden = cell.hidden_from_state(current_state)
        outputs.append(hidden)
    reference_features = torch.stack(outputs, dim=1)

    assert torch.allclose(features, reference_features, atol=1e-6, rtol=1e-6)
    ref_hidden = cell.hidden_from_state(current_state)
    assert torch.allclose(cell.hidden_from_state(final_state), ref_hidden, atol=1e-6, rtol=1e-6)


def test_rnn_layer_zero_length() -> None:
    cell = LinearRNNCell(4, 3)
    layer = RNNLayer(cell)
    inputs = torch.randn(2, 0, 4)
    features, state = layer(inputs)
    assert features.shape == (2, 0, 3)
    assert torch.all(features == 0)
    hidden = cell.hidden_from_state(state)
    assert hidden.shape == (2, 3)
    assert torch.all(hidden == 0)


def test_rnn_backbone_shapes() -> None:
    backbone = RNNBackbone(input_dim=4, hidden_dim=5, cell="linear")
    x = torch.randn(2, 7, 4)
    output = backbone(x)
    assert output.features.shape == (2, 7, 5)
    assert output.pooled.shape == (2, 5)


def test_rnn_backbone_lstm() -> None:
    backbone = RNNBackbone(input_dim=3, hidden_dim=4, cell="lstm")
    x = torch.randn(2, 6, 3)
    output = backbone(x)
    assert output.features.shape == (2, 6, 4)
    assert output.pooled.shape == (2, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_linear_rnn_cuda_cpu_parity() -> None:
    torch.manual_seed(42)
    batch, length, input_dim, hidden_dim = 4, 9, 3, 5
    inputs = torch.randn(batch, length, input_dim)

    cell_cpu = LinearRNNCell(input_dim, hidden_dim)
    cell_gpu = LinearRNNCell(input_dim, hidden_dim).to("cuda")
    cell_gpu.load_state_dict(cell_cpu.state_dict())

    layer_cpu = RNNLayer(cell_cpu)
    layer_gpu = RNNLayer(cell_gpu).cuda()

    features_cpu, final_cpu = layer_cpu(inputs)
    features_gpu, final_gpu = layer_gpu(inputs.cuda())

    assert torch.allclose(features_gpu.cpu(), features_cpu, atol=1e-5, rtol=1e-5)
    assert torch.allclose(final_gpu.cpu(), final_cpu, atol=1e-5, rtol=1e-5)
