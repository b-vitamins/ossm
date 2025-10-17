from __future__ import annotations
import importlib
import importlib.util


def _require_torchcde():
    if importlib.util.find_spec("torchcde") is None:
        raise RuntimeError(
            "torchcde is required for ToCubicSplineCoeffs. Install via `pip install torchcde`."
        )
    return importlib.import_module("torchcde")


class ToCubicSplineCoeffs:
    def __call__(self, sample):
        torchcde = _require_torchcde()
        x = sample["values"]
        t = sample["times"]
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x, t)
        sample["coeffs"] = coeffs
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
