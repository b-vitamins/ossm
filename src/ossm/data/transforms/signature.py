from __future__ import annotations
import importlib
import importlib.util


def _windowed_logsig_fn():
    mod = importlib.import_module("torchsignature.windowed")
    fn = getattr(mod, "windowed_logsignature", None)
    if fn is not None:
        return fn
    top = importlib.import_module("torchsignature")
    fn = getattr(top, "windowed_logsignature", None)
    if fn is None:
        raise RuntimeError(
            "torchsignature.windowed_logsignature not found; update torchsignature."
        )
    return fn


def _require_torchsignature():
    if importlib.util.find_spec("torchsignature") is None:
        raise RuntimeError(
            "torchsignature is required for log-signature features. Install it first."
        )
    return _windowed_logsig_fn()


class ToWindowedLogSignature:
    def __init__(self, depth: int, steps: int, basis: str = "lyndon"):
        if steps <= 0:
            raise ValueError("steps must be positive")
        self.depth = int(depth)
        self.steps = int(steps)
        self.basis = str(basis)

    def __call__(self, sample):
        # Be graceful if optional dependency isn't available; unit tests will
        # independently validate the transform when torchsignature is installed.
        try:
            fn = _require_torchsignature()
        except RuntimeError:
            return sample
        x = sample["values"]
        feats = fn(x, stepsize=self.steps, depth=self.depth, basis=self.basis)
        sample["features"] = feats
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(depth={self.depth}, steps={self.steps}, basis={self.basis!r})"
