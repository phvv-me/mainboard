from __future__ import annotations

from pydantic import Field

from ...enums import Vendor
from ...gpu import GPU


class AMDGPU(GPU):
    """AMD GPU provider stub for future ROCm SMI / HIP support."""

    vendor: Vendor = Field(default=Vendor.AMD)
    backend: str = "rocm"

    @classmethod
    def is_available(cls) -> bool:
        """Whether AMD accelerators are available through the future provider."""
        return False

    @classmethod
    def all(cls) -> tuple[AMDGPU, ...]:
        """Return detected AMD GPUs; empty until the provider is implemented."""
        return ()
