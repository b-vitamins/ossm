import os
from unittest import mock

import pytest
import torch

pytestmark = pytest.mark.filterwarnings(
    "ignore:Torchinductor does not support code generation for complex operators:UserWarning"
)

from ossm.models import _dlinoss_scan
from ossm.models.dlinoss import DampedLinOSSLayer

VARIANTS = ("imex1", "imex2", "im", "ex")
DTYPE_CASES = (
    (torch.float32, 1e-4, 5e-4),
    (torch.float64, 1e-7, 1e-7),
)


def _run_layer(
    layer: DampedLinOSSLayer, inputs: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
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


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dtype, atol, rtol", DTYPE_CASES)
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_dlinoss_kernel_matches_fallback(
    variant: str, dtype: torch.dtype, atol: float, rtol: float, device: str
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    torch.manual_seed(0)
    if device == "cuda":
        torch.cuda.manual_seed_all(0)

    layer = DampedLinOSSLayer(ssm_size=8, hidden_dim=16, variant=variant)
    layer = layer.to(device=device, dtype=dtype)
    layer.eval()

    base_inputs = torch.randn(3, 12, 16, dtype=dtype, device=device)

    with mock.patch.dict(os.environ, {"OSSM_DLINOSS_DISABLE_KERNEL": "1"}):
        fallback_out, fallback_grads = _run_layer(layer, base_inputs.clone())

    if not _dlinoss_scan.has_kernels(variant):
        pytest.skip(f"D-LinOSS {variant} kernels unavailable")

    kernel_out, kernel_grads = _run_layer(layer, base_inputs.clone())

    torch.testing.assert_close(kernel_out, fallback_out, atol=atol, rtol=rtol)

    for grad_kernel, grad_ref in zip(kernel_grads, fallback_grads):
        if grad_ref is None:
            assert grad_kernel is None
            continue
        assert grad_kernel is not None
        torch.testing.assert_close(grad_kernel, grad_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_dlinoss_kernel_torch_compile(variant: str) -> None:
    if not _dlinoss_scan.has_kernels(variant):
        pytest.skip(f"D-LinOSS {variant} kernels unavailable")

    torch.manual_seed(0)
    layer = DampedLinOSSLayer(ssm_size=4, hidden_dim=8, variant=variant).eval()
    inputs = torch.randn(2, 6, 8)

    compiled_layer = torch.compile(layer, mode="reduce-overhead")

    with mock.patch.dict(os.environ, {"OSSM_DLINOSS_DISABLE_KERNEL": "1"}):
        fallback_out, _ = _run_layer(layer, inputs.clone())

    eager_out, _ = _run_layer(layer, inputs.clone())
    compiled_out, _ = _run_layer(compiled_layer, inputs.clone())

    torch.testing.assert_close(eager_out, fallback_out, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(compiled_out, eager_out, atol=1e-6, rtol=1e-6)
