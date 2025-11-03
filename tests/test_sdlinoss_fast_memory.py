from __future__ import annotations

import os
from unittest import mock

import pytest
import torch

import ossm.models._sdlinoss_scan as _sdlinoss_scan_module
import ossm.models._sdlinoss_scan_fast as _sdlinoss_scan_fast_module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_xonly_reduces_forward_memory() -> None:
    if not _sdlinoss_scan_fast_module.has_fast_kernels("ex"):
        pytest.skip("Selective D-LinOSS fast kernels unavailable")

    length, batch, state = 4096, 4, 128
    complex_dtype = torch.complex64
    real_dtype = torch.float32
    generator = torch.Generator(device="cuda").manual_seed(0)

    bu = torch.randn(length, batch, state, dtype=complex_dtype, device="cuda", generator=generator)
    A = torch.randn(state, dtype=real_dtype, device="cuda", generator=generator)
    G = torch.randn(state, dtype=real_dtype, device="cuda", generator=generator)
    step = torch.rand(state, dtype=real_dtype, device="cuda", generator=generator) * 0.5 + 0.1

    def _run() -> torch.Tensor:
        with torch.no_grad():
            return _sdlinoss_scan_module.run_sdlinoss("ex", A, G, step, bu)

    def _measure(x_only: bool) -> int:
        env_updates = {
            "OSSM_SDLINOSS_FAST": "1",
            "OSSM_SDLINOSS_FAST_X_ONLY": "1" if x_only else "0",
        }
        with mock.patch.dict(os.environ, env_updates, clear=False):
            with mock.patch.object(_sdlinoss_scan_module, "_FAST_USE", True):
                with mock.patch.object(_sdlinoss_scan_fast_module, "USE_FAST", True):
                    with mock.patch.object(_sdlinoss_scan_fast_module, "X_ONLY", x_only):
                        torch.cuda.reset_peak_memory_stats()
                        _run()
                        torch.cuda.synchronize()
                        return int(torch.cuda.max_memory_allocated())

    mem_full = _measure(False)
    mem_x = _measure(True)

    assert mem_full > 0
    assert mem_x < 0.7 * mem_full, f"expected ~2x less, got ratio {mem_x / mem_full:.2f}"
