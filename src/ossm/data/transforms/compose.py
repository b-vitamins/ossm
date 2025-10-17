from __future__ import annotations


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self) -> str:
        inner = ", ".join(repr(t) for t in self.transforms)
        return f"{self.__class__.__name__}([{inner}])"
