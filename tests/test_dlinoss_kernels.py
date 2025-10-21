from __future__ import annotations

import os
from unittest import mock

import pytest
import torch

from ossm.models.dlinoss import DampedLinOSSLayer
from ossm.models import _dlinoss_scan


def _run_layer(layer: DampedLinOSSLayer, inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
    layer.zero_grad(set_to_none=True)
    output = layer(inputs)
    loss = output.real.pow(2).sum()
    loss.backward()
    grads = [
        p.grad.detach().clone() if p.grad is not None else None
        for p in layer.parameters()
        if p.requires_grad
    ]
    return output.detach(), grads


@pytest.mark.skipif(not _dlinoss_scan.has_kernels(), reason="D-LinOSS kernels unavailable")
def test_dlinoss_kernel_matches_fallback() -> None:
    torch.manual_seed(0)
    layer = DampedLinOSSLayer(ssm_size=8, hidden_dim=16).double()
    layer.eval()

    base_inputs = torch.randn(3, 12, 16, dtype=torch.double)

    with mock.patch.dict(os.environ, {"OSSM_DLINOSS_DISABLE_KERNEL": "1"}):
        inputs_fallback = base_inputs.clone()
        ref_output, ref_grads = _run_layer(layer, inputs_fallback)

    inputs_kernel = base_inputs.clone()
    kernel_output, kernel_grads = _run_layer(layer, inputs_kernel)

    assert torch.allclose(kernel_output, ref_output, atol=1e-6, rtol=1e-6)

    for grad_kernel, grad_ref in zip(kernel_grads, ref_grads):
        if grad_ref is None:
            assert grad_kernel is None
            continue
        assert grad_kernel is not None
        assert torch.allclose(grad_kernel, grad_ref, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not _dlinoss_scan.has_kernels(), reason="D-LinOSS kernels unavailable")
@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_dlinoss_kernel_torch_compile() -> None:
    torch.manual_seed(0)
    layer = DampedLinOSSLayer(ssm_size=4, hidden_dim=8)
    layer.eval()

    inputs = torch.randn(2, 6, 8)

    compiled_layer = torch.compile(layer, mode="reduce-overhead")

    eager_out = layer(inputs)
    compiled_out = compiled_layer(inputs)

    assert torch.allclose(compiled_out, eager_out, atol=1e-5, rtol=1e-5)
