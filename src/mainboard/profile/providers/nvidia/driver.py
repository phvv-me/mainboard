"""The context query needed to bound a synchronized native activity window."""

from typing import Protocol, SupportsInt


class CudaDriver(Protocol):
    """The CUDA driver reports the context bound to the calling host thread."""

    def cuCtxGetCurrent(self) -> tuple[int, SupportsInt]: ...
