"""Structural contract for `roctx`, which ships untyped with ROCm (not on PyPI)."""

from typing import Protocol


class Roctx(Protocol):
    """The range/mark surface the tracer emits, visible under `rocprofv3 --marker-trace`."""

    def mark(self, message: str) -> None: ...
    def rangeStart(self, message: str) -> int: ...
    def rangeStop(self, range_id: int) -> None: ...
