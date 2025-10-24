"""Tests for Neural CDE and RDE layers."""

from __future__ import annotations

import importlib

import pytest
import torch

from ossm.models import NCDEBackbone, NCDELayer, NRDELayer


def _natural_coeffs(path: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    if path.dim() != 3:
        raise ValueError("path must have shape (batch, length, channels)")
    torchcde = pytest.importorskip("torchcde")
    return torchcde.natural_cubic_coeffs(path, t=times)


def _make_zero_ncde_layer(input_dim: int, hidden_dim: int) -> NCDELayer:
    layer = NCDELayer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        vf_width=8,
        vf_depth=1,
    )
    with torch.no_grad():
        layer.input_linear.weight.zero_()
        layer.input_linear.bias.zero_()
        for param in layer.vector_field.parameters():
            param.zero_()
    return layer


def test_ncde_layer_constant_solution() -> None:
    torch.manual_seed(0)
    input_dim = hidden_dim = 3
    layer = _make_zero_ncde_layer(input_dim, hidden_dim)
    batch, length = 4, 6
    path = torch.randn(batch, length, input_dim)
    times = torch.linspace(0.0, 1.0, steps=length)
    coeffs = _natural_coeffs(path, times)
    initial = torch.randn(batch, input_dim)

    features, final = layer(times, coeffs, initial)

    expected = torch.zeros(batch, hidden_dim)
    torch.testing.assert_close(final, expected)
    torch.testing.assert_close(features, expected.unsqueeze(1).expand(-1, length, -1))


def test_ncde_layer_accepts_batched_time_grid() -> None:
    torch.manual_seed(4)
    input_dim = hidden_dim = 3
    layer = _make_zero_ncde_layer(input_dim, hidden_dim)
    batch, length = 5, 6
    path = torch.randn(batch, length, input_dim)
    times = torch.linspace(0.0, 1.0, steps=length)
    coeffs = _natural_coeffs(path, times)
    tiled_times = times.unsqueeze(0).expand(batch, -1)
    initial = torch.randn(batch, input_dim)
    mask = torch.ones(batch, length, dtype=torch.bool)

    features, final = layer(tiled_times, coeffs, initial, mask=mask)

    expected = torch.zeros(batch, hidden_dim)
    torch.testing.assert_close(final, expected)
    torch.testing.assert_close(features, expected.unsqueeze(1).expand(-1, length, -1))


def test_ncde_backbone_sequence_output() -> None:
    torch.manual_seed(1)
    input_dim, hidden_dim = 4, 5
    batch, length = 3, 7
    path = torch.randn(batch, length, input_dim)
    times = torch.linspace(0.0, 1.0, steps=length)
    coeffs = _natural_coeffs(path, times)
    initial = path[:, 0]

    backbone = NCDEBackbone(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        vf_width=16,
        vf_depth=2,
    )
    output = backbone({"times": times, "coeffs": coeffs, "initial": initial})
    assert output.features.shape == (batch, length, hidden_dim)
    assert output.pooled.shape == (batch, hidden_dim)


def test_nrde_layer_constant_solution() -> None:
    torch.manual_seed(2)
    input_dim = hidden_dim = 3
    logsig_dim = 5
    intervals = torch.linspace(0.0, 1.0, steps=4)
    layer = NRDELayer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        logsig_dim=logsig_dim,
        intervals=intervals,
        vf_width=12,
        vf_depth=2,
    )
    with torch.no_grad():
        layer.input_linear.weight.zero_()
        layer.input_linear.bias.zero_()
        layer.mlp_linear.weight.zero_()
        layer.mlp_linear.bias.zero_()
        for param in layer.vector_field.parameters():
            param.zero_()

    batch = 2
    logsig = torch.zeros(batch, intervals.numel() - 1, logsig_dim)
    initial = torch.randn(batch, input_dim)

    features, final = layer(intervals, logsig, initial)
    expected = torch.zeros(batch, hidden_dim)
    torch.testing.assert_close(final, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(features, expected.unsqueeze(1).expand(-1, intervals.numel(), -1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ncde_layer_cuda_matches_cpu() -> None:
    torch.manual_seed(3)
    layer_cpu = NCDELayer(input_dim=3, hidden_dim=4, vf_width=8, vf_depth=1)
    layer_cuda = NCDELayer(input_dim=3, hidden_dim=4, vf_width=8, vf_depth=1).cuda()
    layer_cuda.load_state_dict(layer_cpu.state_dict())

    batch, length = 2, 5
    path = torch.randn(batch, length, 3)
    times = torch.linspace(0.0, 1.0, steps=length)
    coeffs = _natural_coeffs(path, times)
    initial = torch.randn(batch, 3)

    features_cpu, final_cpu = layer_cpu(times, coeffs, initial)
    coeffs_cuda = coeffs.cuda()
    features_gpu, final_gpu = layer_cuda(times.cuda(), coeffs_cuda, initial.cuda())

    torch.testing.assert_close(features_gpu.cpu(), features_cpu, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(final_gpu.cpu(), final_cpu, atol=1e-5, rtol=1e-5)


def test_ncde_layer_missing_torchcde(monkeypatch) -> None:
    from ossm.models import ncde as ncde_mod

    original_import_module = importlib.import_module

    def fake_import(name: str, *args, **kwargs):
        if name == "torchcde":
            raise ModuleNotFoundError("No module named 'torchcde'")
        return original_import_module(name, *args, **kwargs)

    monkeypatch.setattr(ncde_mod.importlib, "import_module", fake_import)

    layer = NCDELayer(input_dim=2, hidden_dim=3, vf_width=4, vf_depth=1)
    times = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    coeffs = torch.zeros(1, times.numel() - 1, 4 * layer.input_dim)
    initial = torch.zeros(1, layer.input_dim)

    with pytest.raises(RuntimeError, match="torchcde is required for Neural CDE support"):
        layer(times, coeffs, initial)

