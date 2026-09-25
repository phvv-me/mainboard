from typing import ClassVar

from ...enums import Vendor
from ...units.gpu import GPU
from .silicon import AppleSilicon


class AppleGPU(AppleSilicon, GPU):
    """Apple Silicon integrated GPU backed by unified memory."""

    engine: ClassVar[str] = "GPU"
    vendor: Vendor = Vendor.APPLE
    backend: str = "metal"

    @classmethod
    def all(cls) -> tuple[AppleGPU, ...]:
        """The local Apple Silicon GPU when present."""
        return (cls(),) if cls.is_available() else ()
