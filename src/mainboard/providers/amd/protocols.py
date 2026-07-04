"""Structural contract for the untyped AMD ROCTx marker surface.

`roctx` ships with a ROCm install (not on PyPI) and carries no type stubs, so this Protocol
pins the range and mark API the tracer emits, visible under `rocprofv3 --marker-trace`.
"""

from typing import Protocol


class Roctx(Protocol):
    """The `roctx` range/mark surface mainboard emits to the ROCm marker trace."""

    def rangeStart(self, message: str) -> int: ...  # noqa: N802 — mirrors the ROCTx symbol
    def rangeStop(self, range_id: int) -> None: ...  # noqa: N802 — mirrors the ROCTx symbol
    def mark(self, message: str) -> None: ...
