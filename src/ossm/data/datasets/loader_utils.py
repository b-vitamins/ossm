from __future__ import annotations
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def infinite(loader: Iterable[T]) -> Iterator[T]:
    while True:
        for batch in loader:
            yield batch
