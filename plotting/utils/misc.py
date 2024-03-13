from typing import TypeVar

T = TypeVar("T")


def F(li: T) -> T:
    if type(li) == list:
        return [1.0 - x for x in li]  # type: ignore[return-value]
    else:
        return 1.0 - li  # type: ignore[return-value]
