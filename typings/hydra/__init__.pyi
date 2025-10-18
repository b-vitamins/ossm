from typing import Any, Callable, Optional, Protocol, TypeVar

T = TypeVar("T", bound=Callable[..., Any])

class _MainDecorator(Protocol[T]):
    def __call__(self, fn: T, /) -> Callable[[], Any]: ...


def main(
    config_path: Optional[str] = ..., *,
    config_name: Optional[str] = ..., version_base: Optional[str] = ...,
) -> _MainDecorator[T]: ...
