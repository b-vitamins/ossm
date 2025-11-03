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


@pytest.mark.cuda
@pytest.mark.parametrize("variant", ["ex", "imex1", "imex2", "im"])
def test_fast_gradients_match_reference_with_broadcast(variant: str) -> None:
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

        device = torch.device("cuda")
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)

        length, batch, ssm = 5, 3, 2
        real_dtype = torch.float64

        A_base = torch.randn(length, 1, ssm, dtype=real_dtype, device=device)
        G_base = torch.randn(1, batch, 1, dtype=real_dtype, device=device)
        step_base = torch.rand(length, batch, 1, dtype=real_dtype, device=device) * 0.4 + 0.2
        bu_real = torch.randn(length, batch, ssm, dtype=real_dtype, device=device)
        bu_imag = torch.randn(length, batch, ssm, dtype=real_dtype, device=device)
        bu_base = torch.complex(bu_real, bu_imag)

        def make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            return (
                A_base.clone().requires_grad_(True),
                G_base.clone().requires_grad_(True),
                step_base.clone().requires_grad_(True),
                bu_base.clone().requires_grad_(True),
            )

        original_fast_flag = scan_mod._FAST_USE
        try:
            scan_mod._FAST_USE = False
            A_ref, G_ref, step_ref, bu_ref = make_inputs()
            out_ref = scan_mod.run_sdlinoss(variant, A_ref, G_ref, step_ref, bu_ref)
            loss_ref = (out_ref.abs() ** 2).sum()
            loss_ref.backward()
            grads_ref = (A_ref.grad, G_ref.grad, step_ref.grad, bu_ref.grad)
        finally:
            scan_mod._FAST_USE = original_fast_flag

        scan_mod._FAST_USE = True
        try:
            A_fast, G_fast, step_fast, bu_fast = make_inputs()
            out_fast = scan_mod.run_sdlinoss(variant, A_fast, G_fast, step_fast, bu_fast)
            loss_fast = (out_fast.abs() ** 2).sum()
            loss_fast.backward()
            grads_fast = (A_fast.grad, G_fast.grad, step_fast.grad, bu_fast.grad)
        finally:
            scan_mod._FAST_USE = original_fast_flag

        for grad_fast, grad_ref in zip(grads_fast, grads_ref):
            torch.testing.assert_close(grad_fast, grad_ref, rtol=1e-5, atol=1e-6)
    finally:
        if original_env is None:
            os.environ.pop("OSSM_SDLINOSS_FAST", None)
        else:
            os.environ["OSSM_SDLINOSS_FAST"] = original_env

        fast_mod = importlib.reload(fast_mod)
        importlib.reload(scan_mod)


@pytest.mark.cuda
@pytest.mark.parametrize("variant", ["ex", "imex1", "imex2", "im"])
def test_fast_gradients_respect_step_clamp(variant: str) -> None:
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

        device = torch.device("cuda")
        real_dtype = torch.float64
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)

        length, batch, ssm = 4, 2, 1
        dt_min = torch.tensor(1e-6, dtype=real_dtype, device=device)
        dt_max = torch.tensor(1.0, dtype=real_dtype, device=device)
        step_values = torch.stack(
            [dt_min, dt_min + 1e-4, dt_max - 1e-4, dt_max]
        ).view(length, 1, 1)
        step_base = step_values.expand(-1, batch, ssm).clone()

        A_base = torch.randn(length, batch, ssm, dtype=real_dtype, device=device)
        G_base = torch.randn(length, batch, ssm, dtype=real_dtype, device=device)
        bu_real = torch.randn(length, batch, ssm, dtype=real_dtype, device=device)
        bu_imag = torch.randn(length, batch, ssm, dtype=real_dtype, device=device)
        bu_base = torch.complex(bu_real, bu_imag)

        def make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            return (
                A_base.clone().requires_grad_(True),
                G_base.clone().requires_grad_(True),
                step_base.clone().requires_grad_(True),
                bu_base.clone().requires_grad_(True),
            )

        original_fast_flag = scan_mod._FAST_USE
        try:
            scan_mod._FAST_USE = False
            A_ref, G_ref, step_ref, bu_ref = make_inputs()
            out_ref = scan_mod.run_sdlinoss(variant, A_ref, G_ref, step_ref, bu_ref)
            loss_ref = (out_ref.abs() ** 2).sum()
            loss_ref.backward()
            grad_step_ref = step_ref.grad.clone()
        finally:
            scan_mod._FAST_USE = original_fast_flag

        scan_mod._FAST_USE = True
        try:
            A_fast, G_fast, step_fast, bu_fast = make_inputs()
            out_fast = scan_mod.run_sdlinoss(variant, A_fast, G_fast, step_fast, bu_fast)
            loss_fast = (out_fast.abs() ** 2).sum()
            loss_fast.backward()
            grad_step_fast = step_fast.grad.clone()
        finally:
            scan_mod._FAST_USE = original_fast_flag

        torch.testing.assert_close(grad_step_fast, grad_step_ref, rtol=1e-6, atol=1e-7)
        assert torch.allclose(grad_step_fast[0], torch.zeros_like(grad_step_fast[0]))
        assert torch.allclose(grad_step_fast[-1], torch.zeros_like(grad_step_fast[-1]))
    finally:
        if original_env is None:
            os.environ.pop("OSSM_SDLINOSS_FAST", None)
        else:
            os.environ["OSSM_SDLINOSS_FAST"] = original_env

        fast_mod = importlib.reload(fast_mod)
        importlib.reload(scan_mod)
