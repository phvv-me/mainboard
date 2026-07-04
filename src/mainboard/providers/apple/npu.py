import platform
from functools import cached_property

from pydantic import Field

from ...enums import Vendor
from ...models.clock import Clock
from ...models.memory import Memory
from ...npu import NPU
from . import profile


class AppleNPU(NPU):
    """Apple Neural Engine backed by unified memory."""

    vendor: Vendor = Field(default=Vendor.APPLE)
    backend: str = "coreml"

    @classmethod
    def is_available(cls) -> bool:
        """Whether this host is an Apple Silicon machine."""
        return platform.system() == "Darwin" and platform.machine() == "arm64"

    @classmethod
    def all(cls) -> tuple[AppleNPU, ...]:
        """Return the local Apple Neural Engine when present."""
        return (cls(),) if cls.is_available() else ()

    @cached_property
    def name(self) -> str:
        """Apple Neural Engine model name."""
        return f"{self.architecture} Neural Engine"

    @cached_property
    def architecture(self) -> str:
        """Apple SoC family backing the Neural Engine."""
        return str(profile.hardware_record().get("chip_type") or "Apple Silicon")

    @property
    def memory(self) -> Memory:
        """Unified memory visible to CPU, GPU, and Neural Engine."""
        return Memory.system(scope="unified", unified=True)

    @property
    def clock_readings(self) -> tuple[Clock, ...]:
        """Apple Neural Engine clocks are not exposed through public APIs."""
        return (Clock(domain="npu", source="coreml", supported=False),)
