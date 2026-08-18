import platform
from functools import cached_property

from ...enums import Vendor
from ...facts.memory import Memory
from ...shell import sysctl
from ...units.npu import NPU


class AppleNPU(NPU):
    """Apple Neural Engine backed by unified memory."""

    vendor: Vendor = Vendor.APPLE
    backend: str = "coreml"

    @cached_property
    def architecture(self) -> str:
        """Apple SoC family backing the Neural Engine, e.g. `Apple M4 Pro`."""
        return sysctl("machdep.cpu.brand_string") or "Apple Silicon"

    @cached_property
    def label(self) -> str:
        """Apple Neural Engine model name."""
        return f"{self.architecture} Neural Engine"

    @property
    def memory(self) -> Memory:
        """Unified memory visible to CPU, GPU, and Neural Engine."""
        return Memory.system(scope="unified", unified=True)

    @classmethod
    def all(cls) -> tuple["AppleNPU", ...]:
        """Return the local Apple Neural Engine when present."""
        return (cls(),) if cls.is_available() else ()

    @classmethod
    def is_available(cls) -> bool:
        """Whether this host is an Apple Silicon Mac."""
        return platform.system() == "Darwin" and platform.machine() == "arm64"
