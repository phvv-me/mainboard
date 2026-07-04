from pydantic import Field

from ...enums import Vendor
from ...gpu import GPU


class IntelGPU(GPU):
    """Intel GPU provider stub for future Level Zero / oneAPI support."""

    vendor: Vendor = Field(default=Vendor.INTEL)
    backend: str = "level-zero"

    @classmethod
    def is_available(cls) -> bool:
        """Whether Intel GPUs are available through the future provider."""
        return False

    @classmethod
    def all(cls) -> tuple[IntelGPU, ...]:
        """Return detected Intel GPUs; empty until the provider is implemented."""
        return ()
