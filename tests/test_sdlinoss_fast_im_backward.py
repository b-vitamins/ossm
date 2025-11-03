from __future__ import annotations

import pytest
import torch

from ossm.models._sdlinoss_scan import _SdlinossImFn
from ossm.models._sdlinoss_scan_fast import SdlinossImFastFn, has_fast_kernels


@pytest.mark.cuda
@pytest.mark.parametrize("length", [33, 128, 1024])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("ssm", [17, 64])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape_tag", ["M", "LM", "BM", "LBM"])
def test_fast_im_backward_matches_reference(length: int,
                                            batch: int,
                                            ssm: int,
                                            dtype: torch.dtype,
                                            shape_tag: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fast selective D-LinOSS kernels")

    if not has_fast_kernels("im"):
        pytest.skip("Fast IM kernels are unavailable in this build")

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

    A_fast = A_param.clone().detach().contiguous().requires_grad_(True)
    G_fast = G_param.clone().detach().contiguous().requires_grad_(True)
    step_fast = step_param.clone().detach().contiguous().requires_grad_(True)
    bu_fast = bu.clone().detach().contiguous().requires_grad_(True)

    out_fast = SdlinossImFastFn.apply(A_fast, G_fast, step_fast, bu_fast)
    loss_fast = (out_fast.real.square() + out_fast.imag.square()).sum()
    loss_fast.backward()

    A_ref = A_param.clone().detach().contiguous().requires_grad_(True)
    G_ref = G_param.clone().detach().contiguous().requires_grad_(True)
    step_ref = step_param.clone().detach().contiguous().requires_grad_(True)
    bu_ref = bu.clone().detach().contiguous().requires_grad_(True)

    out_ref = _SdlinossImFn.apply(A_ref, G_ref, step_ref, bu_ref)
    loss_ref = (out_ref.real.square() + out_ref.imag.square()).sum()
    loss_ref.backward()

    if dtype == torch.complex64:
        rtol, atol = 1e-4, 2e-5
    else:
        rtol, atol = 1e-6, 1e-7

    torch.testing.assert_close(A_fast.grad, A_ref.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(G_fast.grad, G_ref.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(step_fast.grad, step_ref.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(bu_fast.grad, bu_ref.grad, rtol=rtol, atol=atol)
