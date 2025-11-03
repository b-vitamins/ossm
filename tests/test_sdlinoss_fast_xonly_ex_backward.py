from __future__ import annotations

import pytest
import torch

from ossm.models._sdlinoss_scan_fast import has_fast_kernels


@pytest.mark.cuda
@pytest.mark.parametrize("length", [33, 128, 1024])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("ssm", [17, 64])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape_tag", ["M", "LM", "BM", "LBM"])
def test_fast_ex_backward_xonly_matches_states(
    length: int,
    batch: int,
    ssm: int,
    dtype: torch.dtype,
    shape_tag: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fast selective D-LinOSS kernels")

    if not has_fast_kernels("ex"):
        pytest.skip("Fast EX kernels are unavailable in this build")

    from ossm.models import _sdlinoss_scan_fast as fast

    kernels = fast._kernels
    assert kernels is not None

    device = torch.device("cuda")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64

    A_full = torch.randn(length, batch, ssm, dtype=real_dtype, device=device)
    G_full = torch.randn(length, batch, ssm, dtype=real_dtype, device=device)
    step_full = torch.rand(length, batch, ssm, dtype=real_dtype, device=device) * 0.9 + 0.05
    bu = torch.randn(length, batch, ssm, dtype=dtype, device=device)

    def make_param(tensor: torch.Tensor) -> torch.Tensor:
        if shape_tag == "M":
            return tensor[0, 0].unsqueeze(0).unsqueeze(0).contiguous()
        if shape_tag == "LM":
            return tensor[:, 0].unsqueeze(1).contiguous()
        if shape_tag == "BM":
            return tensor[0].unsqueeze(0).contiguous()
        if shape_tag == "LBM":
            return tensor.contiguous()
        raise AssertionError(f"Unexpected shape tag: {shape_tag}")

    A_param = make_param(A_full)
    G_param = make_param(G_full)
    step_param = make_param(step_full)

    x_only = kernels.sdlinoss_fast_ex_forward_xonly(A_param, G_param, step_param, bu)
    states = kernels.sdlinoss_fast_ex_forward(A_param, G_param, step_param, bu)
    grad_out = torch.randn(length, batch, ssm, dtype=dtype, device=device)

    gA_x, gG_x, gStep_x, gBu_x = kernels.sdlinoss_fast_ex_backward_xonly(
        A_param, G_param, step_param, bu, x_only, grad_out
    )
    gA, gG, gStep, gBu = kernels.sdlinoss_fast_ex_backward(
        A_param, G_param, step_param, bu, states, grad_out
    )

    assert gA_x.shape == gA.shape
    assert gG_x.shape == gG.shape
    assert gStep_x.shape == gStep.shape
    assert gBu_x.shape == gBu.shape

    if dtype == torch.complex64:
        rtol, atol = 1e-4, 2e-5
    else:
        rtol, atol = 1e-6, 1e-7

    torch.testing.assert_close(gA_x, gA, rtol=rtol, atol=atol)
    torch.testing.assert_close(gG_x, gG, rtol=rtol, atol=atol)
    torch.testing.assert_close(gStep_x, gStep, rtol=rtol, atol=atol)
    torch.testing.assert_close(gBu_x, gBu, rtol=rtol, atol=atol)


@pytest.mark.cuda
def test_fast_ex_autograd_invokes_xonly_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fast selective D-LinOSS kernels")

    if not has_fast_kernels("ex"):
        pytest.skip("Fast EX kernels are unavailable in this build")

    from ossm.models import _sdlinoss_scan_fast as fast

    kernels = fast._kernels
    assert kernels is not None

    device = torch.device("cuda")
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    length, batch, ssm = 16, 2, 8
    dtype = torch.complex64
    real_dtype = torch.float32

    A = torch.randn(length, batch, ssm, dtype=real_dtype, device=device, requires_grad=True)
    G = torch.randn(length, batch, ssm, dtype=real_dtype, device=device, requires_grad=True)
    step = torch.rand(length, batch, ssm, dtype=real_dtype, device=device, requires_grad=True) * 0.9 + 0.05
    bu = torch.randn(length, batch, ssm, dtype=dtype, device=device, requires_grad=True)

    call_tracker = {"forward": 0, "backward": 0}

    orig_forward = kernels.sdlinoss_fast_ex_forward_xonly
    orig_backward = kernels.sdlinoss_fast_ex_backward_xonly

    def forward_wrapper(*args, **kwargs):
        call_tracker["forward"] += 1
        result = orig_forward(*args, **kwargs)
        assert result.shape == (length, batch, ssm)
        return result

    def backward_wrapper(*args, **kwargs):
        call_tracker["backward"] += 1
        x_arg = args[4]
        assert isinstance(x_arg, torch.Tensor)
        assert x_arg.shape == (length, batch, ssm)
        return orig_backward(*args, **kwargs)

    monkeypatch.setattr(kernels, "sdlinoss_fast_ex_forward_xonly", forward_wrapper)
    monkeypatch.setattr(kernels, "sdlinoss_fast_ex_backward_xonly", backward_wrapper)
    monkeypatch.setattr(fast, "X_ONLY", True)

    output = fast.SdlinossExFastFn.apply(A, G, step, bu)
    loss = (output.abs() ** 2).sum()
    loss.backward()

    assert call_tracker["forward"] == 1
    assert call_tracker["backward"] == 1

    for tensor in (A, G, step, bu):
        tensor.grad = None

