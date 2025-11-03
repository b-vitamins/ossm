from __future__ import annotations

import importlib
import os

import pytest
import torch


@pytest.mark.cuda
@pytest.mark.parametrize("variant", ["ex", "imex1", "imex2", "im"])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", [(33, 1, 17), (128, 3, 64), (1024, 2, 32)])
def test_fast_parity_all(variant: str, dtype: torch.dtype, shape: tuple[int, int, int]) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fast selective D-LinOSS kernels")

    import ossm.models._sdlinoss_scan_fast as fast_mod
    import ossm.models._sdlinoss_scan as scan_mod

    original_env = os.environ.get("OSSM_SDLINOSS_FAST")
    os.environ["OSSM_SDLINOSS_FAST"] = "1"

    try:
        fast_mod = importlib.reload(fast_mod)
        scan_mod = importlib.reload(scan_mod)

        if not fast_mod.has_fast_kernels(variant):
            pytest.skip(f"Fast {variant} kernels are unavailable in this build")

        length, batch, ssm = shape
        device = torch.device("cuda")
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64

        A = torch.randn(ssm, dtype=real_dtype, device=device)
        G = torch.randn(ssm, dtype=real_dtype, device=device)
        step = torch.rand(ssm, dtype=real_dtype, device=device) * 0.5 + 0.1
        bu_real = torch.randn(length, batch, ssm, dtype=real_dtype, device=device)
        bu_imag = torch.randn(length, batch, ssm, dtype=real_dtype, device=device)
        bu = torch.complex(bu_real, bu_imag).to(dtype)

        prev_fast_use = scan_mod._FAST_USE
        try:
            scan_mod._FAST_USE = False
            y_legacy = scan_mod.run_sdlinoss(variant, A, G, step, bu)
        finally:
            scan_mod._FAST_USE = prev_fast_use

        scan_mod._FAST_USE = True
        try:
            y_fast = scan_mod.run_sdlinoss(variant, A, G, step, bu)
        finally:
            scan_mod._FAST_USE = prev_fast_use

        if dtype == torch.complex64:
            rtol, atol = 1e-5, 5e-6
        else:
            rtol, atol = 1e-7, 1e-8

        torch.testing.assert_close(y_fast, y_legacy, rtol=rtol, atol=atol)
    finally:
        if original_env is None:
            os.environ.pop("OSSM_SDLINOSS_FAST", None)
        else:
            os.environ["OSSM_SDLINOSS_FAST"] = original_env

        fast_mod = importlib.reload(fast_mod)
        importlib.reload(scan_mod)
