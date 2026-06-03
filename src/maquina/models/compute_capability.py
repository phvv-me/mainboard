from __future__ import annotations

from typing import NamedTuple


class ComputeCapability(NamedTuple):
    """CUDA compute capability as a comparable (major, minor) pair.

    Comparison operators work correctly across two-digit minor versions:
    ``ComputeCapability(9, 0) > ComputeCapability(8, 10)`` is True.
    """

    major: int
    minor: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"

    def __repr__(self) -> str:
        return f"ComputeCapability({self.major}, {self.minor})"
