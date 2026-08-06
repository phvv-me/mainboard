from __future__ import annotations

from pydantic import Field

from ...enums import Vendor
from ...npu import NPU


class QualcommNPU(NPU):
    """Qualcomm NPU provider stub for future QNN support."""

    vendor: Vendor = Field(default=Vendor.QUALCOMM)
    backend: str = "qnn"

    @classmethod
    def is_available(cls) -> bool:
        """Whether Qualcomm NPUs are available through the future provider."""
        return False

    @classmethod
    def all(cls) -> tuple[QualcommNPU, ...]:
        """Return detected Qualcomm NPUs; empty until the provider is implemented."""
        return ()
