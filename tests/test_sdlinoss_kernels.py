from __future__ import annotations

import os
from unittest import mock

import pytest
import torch

from ossm.models import _sdlinoss_scan
from ossm.models.sdlinoss import SelectiveDLinOSSLayer

pytestmark = pytest.mark.filterwarnings(
    "ignore:Torchinductor does not support code generation for complex operators:UserWarning"
)

_VARIANTS = ("imex1", "imex2", "im", "ex")
_DTYPE_CASES = (
    (torch.float32, 5e-4, 5e-4),
    (torch.float64, 1e-6, 1e-6),
)
_COMPLEX_CASES = (
    (torch.complex64, 5e-4, 5e-4),
    (torch.complex128, 1e-7, 1e-7),
)


def _make_inputs(
    *,
    length: int,
    batch: int,
    ssm: int,
    device: torch.device,
    complex_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    real_dtype = torch.float32 if complex_dtype == torch.complex64 else torch.float64
    A = torch.randn(length, batch, ssm, device=device, dtype=real_dtype)
    G = torch.randn(length, batch, ssm, device=device, dtype=real_dtype)
    # Keep the step within (1e-3, 1] to stay in the stable region exercised by the
    # production code paths.
    step = torch.rand(length, batch, ssm, device=device, dtype=real_dtype) * 0.85 + 0.05
    real = torch.randn(length, batch, ssm, device=device, dtype=real_dtype)
    imag = torch.randn(length, batch, ssm, device=device, dtype=real_dtype)
    bu = torch.complex(real, imag).to(dtype=complex_dtype)
    return A, G, step, bu


def _run_layer(layer: SelectiveDLinOSSLayer, inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
    layer.zero_grad(set_to_none=True)
    output = layer(inputs)
    loss = output.pow(2).sum()
    loss.backward()
    grads = [
        p.grad.detach().clone() if p.grad is not None else None
        for p in layer.parameters()
        if p.requires_grad
    ]
    return output.detach(), grads


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("dtype, atol, rtol", _DTYPE_CASES)
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_sdlinoss_kernel_matches_fallback(
    variant: str, dtype: torch.dtype, atol: float, rtol: float, device: str
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    torch.manual_seed(0)
    if device == "cuda":
        torch.cuda.manual_seed_all(0)

    layer = SelectiveDLinOSSLayer(
        ssm_size=8,
        hidden_dim=16,
        variant=variant,
        selective_injection=True,
        per_step_dt=True,
    )
    layer = layer.to(device=device, dtype=dtype)
    layer.eval()

    base_inputs = torch.randn(3, 12, 16, dtype=dtype, device=device)

    with mock.patch.dict(os.environ, {"OSSM_SDLINOSS_DISABLE_KERNEL": "1"}):
        fallback_out, fallback_grads = _run_layer(layer, base_inputs.clone())

    if not _sdlinoss_scan.has_kernels(variant):
        pytest.skip(f"Selective D-LinOSS {variant} kernels unavailable")

    kernel_out, kernel_grads = _run_layer(layer, base_inputs.clone())

    torch.testing.assert_close(kernel_out, fallback_out, atol=atol, rtol=rtol)

    for grad_kernel, grad_ref in zip(kernel_grads, fallback_grads):
        if grad_ref is None:
            assert grad_kernel is None
            continue
        assert grad_kernel is not None
        torch.testing.assert_close(grad_kernel, grad_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("complex_dtype, atol, rtol", _COMPLEX_CASES)
def test_sdlinoss_native_bindings_match_reference(
    variant: str,
    device: str,
    complex_dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    kernels = getattr(_sdlinoss_scan, "_kernels", None)
    if kernels is None:
        pytest.skip("Selective D-LinOSS extension unavailable")

    forward_name = f"sdlinoss_{variant}_forward"
    backward_name = f"sdlinoss_{variant}_backward"
    if not hasattr(kernels, forward_name) or not hasattr(kernels, backward_name):
        pytest.skip(f"Selective D-LinOSS {variant} bindings missing")

    device_obj = torch.device(device)
    torch.manual_seed(1234)
    if device == "cuda":
        torch.cuda.manual_seed_all(1234)

    length, batch, ssm = 4, 3, 5
    A_base, G_base, step_base, bu_base = _make_inputs(
        length=length,
        batch=batch,
        ssm=ssm,
        device=device_obj,
        complex_dtype=complex_dtype,
    )

    A_ref = A_base.clone().requires_grad_(True)
    G_ref = G_base.clone().requires_grad_(True)
    step_ref = step_base.clone().requires_grad_(True)
    bu_ref = bu_base.clone().requires_grad_(True)

    output_ref = _sdlinoss_scan._fallback_sdlinoss(variant, A_ref, G_ref, step_ref, bu_ref)
    loss_ref = output_ref.abs().square().sum()
    grad_ref = torch.autograd.grad(loss_ref, (A_ref, G_ref, step_ref, bu_ref))

    A_ker = A_base.clone().requires_grad_(True)
    G_ker = G_base.clone().requires_grad_(True)
    step_ker = step_base.clone().requires_grad_(True)
    bu_ker = bu_base.clone().requires_grad_(True)

    forward_fn = getattr(kernels, forward_name)
    backward_fn = getattr(kernels, backward_name)

    states_ker = forward_fn(A_ker, G_ker, step_ker, bu_ker)
    assert states_ker.shape == (length, batch, ssm, 2)

    loss_ker = (states_ker[..., 1].abs().square()).sum()
    grad_states = torch.autograd.grad(loss_ker, states_ker, retain_graph=True)[0]
    grad_output = grad_states[..., 1].detach().contiguous()
    grad_kernel = torch.autograd.grad(loss_ker, (A_ker, G_ker, step_ker, bu_ker))

    with torch.no_grad():
        ref_states = _sdlinoss_scan._reference_sdlinoss_states(
            variant,
            A_base,
            G_base,
            step_base,
            bu_base,
        )

    torch.testing.assert_close(states_ker, ref_states, atol=atol, rtol=rtol)
    for grad_actual, grad_expected in zip(grad_kernel[:-1], grad_ref[:-1]):
        torch.testing.assert_close(grad_actual.detach(), grad_expected.detach(), atol=atol, rtol=rtol)
    torch.testing.assert_close(grad_kernel[-1].detach(), grad_ref[-1].detach(), atol=atol, rtol=rtol)

    # Exercise the explicit backward binding with a manually supplied grad_output
    # tensor to ensure it aligns with autograd's result.
    backward_grads = backward_fn(
        A_base.contiguous(),
        G_base.contiguous(),
        step_base.contiguous(),
        bu_base.contiguous(),
        states_ker.detach(),
        grad_output,
    )
    torch.testing.assert_close(backward_grads[0], grad_kernel[0], atol=atol, rtol=rtol)
    torch.testing.assert_close(backward_grads[1], grad_kernel[1], atol=atol, rtol=rtol)
    torch.testing.assert_close(backward_grads[2], grad_kernel[2], atol=atol, rtol=rtol)
    torch.testing.assert_close(backward_grads[3], grad_kernel[3], atol=atol, rtol=rtol)


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_sdlinoss_kernel_torch_compile(variant: str) -> None:
    if not _sdlinoss_scan.has_kernels(variant):
        pytest.skip(f"Selective D-LinOSS {variant} kernels unavailable")

    torch.manual_seed(0)
    layer = SelectiveDLinOSSLayer(
        ssm_size=4,
        hidden_dim=8,
        variant=variant,
        selective_injection=True,
        per_step_dt=True,
    ).eval()
    inputs = torch.randn(2, 6, 8)

    compiled_layer = torch.compile(layer, mode="reduce-overhead")

    with mock.patch.dict(os.environ, {"OSSM_SDLINOSS_DISABLE_KERNEL": "1"}):
        fallback_out, _ = _run_layer(layer, inputs.clone())

    eager_out, _ = _run_layer(layer, inputs.clone())
    compiled_out, _ = _run_layer(compiled_layer, inputs.clone())

    torch.testing.assert_close(eager_out, fallback_out, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(compiled_out, eager_out, atol=1e-6, rtol=1e-6)
