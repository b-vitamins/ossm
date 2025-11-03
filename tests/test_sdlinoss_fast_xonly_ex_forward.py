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
def test_fast_ex_xonly_matches_states(length: int,
                                      batch: int,
                                      ssm: int,
                                      dtype: torch.dtype,
                                      shape_tag: str) -> None:
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

    assert x_only.shape == (length, batch, ssm)
    assert states.shape == (length, batch, ssm, 2)

    if dtype == torch.complex64:
        rtol, atol = 1e-5, 5e-6
    else:
        rtol, atol = 1e-7, 1e-8

    torch.testing.assert_close(x_only, states[..., 1], rtol=rtol, atol=atol)
