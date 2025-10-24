from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Generic, NotRequired, TypeVar, TypedDict

from torch import Tensor

__all__ = ["Compose", "TimeSeriesSample"]


class TimeSeriesSample(TypedDict, total=False):
    """Dictionary-style sample passed between transforms and dataset helpers."""

    values: NotRequired[Tensor]
    label: NotRequired[Tensor]
    times: NotRequired[Tensor]
    coeffs: NotRequired[Tensor]
    features: NotRequired[Tensor]
    grid: NotRequired[Tensor]
    initial: NotRequired[Tensor]
    logsig: NotRequired[Tensor]
    mask: NotRequired[Tensor]
    source_index: NotRequired[Tensor]
    source_split: NotRequired[Tensor]


SampleT = TypeVar("SampleT", bound=TimeSeriesSample)


class Compose(Generic[SampleT]):
    """Chain together a sequence of transforms operating on the same sample type."""

    def __init__(self, transforms: Iterable[Callable[[SampleT], SampleT]]) -> None:
        self.transforms = tuple(transforms)

    def __call__(self, sample: SampleT) -> SampleT:
        result = sample
        for transform in self.transforms:
            result = transform(result)
        return result

    def __repr__(self) -> str:
        inner = ", ".join(repr(t) for t in self.transforms)
        return f"{self.__class__.__name__}([{inner}])"
