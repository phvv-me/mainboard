from __future__ import annotations

from pydantic import Field

from ...enums import Vendor
from ...gpu import GPU


class QualcommGPU(GPU):
    """Qualcomm GPU provider stub for future Adreno telemetry support."""

    vendor: Vendor = Field(default=Vendor.QUALCOMM)
    backend: str = "adreno"

    @classmethod
    def is_available(cls) -> bool:
        """Whether Qualcomm GPUs are available through the future provider."""
        return False

    @classmethod
    def all(cls) -> tuple[QualcommGPU, ...]:
        """Return detected Qualcomm GPUs; empty until the provider is implemented."""
        return ()
