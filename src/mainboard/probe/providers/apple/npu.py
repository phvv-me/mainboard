from typing import ClassVar

from ...enums import Vendor
from ...units.npu import NPU
from .silicon import AppleSilicon


class AppleNPU(AppleSilicon, NPU):
    """Apple Neural Engine backed by unified memory."""

    engine: ClassVar[str] = "Neural Engine"
    vendor: Vendor = Vendor.APPLE
    backend: str = "coreml"

    @classmethod
    def all(cls) -> tuple[AppleNPU, ...]:
        """The local Apple Neural Engine when present."""
        return (cls(),) if cls.is_available() else ()
