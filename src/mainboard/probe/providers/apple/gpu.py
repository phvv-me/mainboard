import platform
from functools import cached_property

from ...enums import Vendor
from ...facts.memory import Memory
from ...shell import sysctl
from ...units.gpu import GPU


class AppleGPU(GPU):
    """Apple Silicon integrated GPU backed by unified memory.

    Identity comes from `sysctl machdep.cpu.brand_string`, which macOS populates
    with the SoC name (e.g. `Apple M4 Pro`) even for the integrated GPU cores,
    since Apple Silicon has no separate GPU model string on the command line.
    """

    vendor: Vendor = Vendor.APPLE
    backend: str = "metal"

    @cached_property
    def architecture(self) -> str:
        """Apple SoC family backing this GPU, e.g. `Apple M4 Pro`."""
        return sysctl("machdep.cpu.brand_string") or "Apple Silicon"

    @cached_property
    def label(self) -> str:
        """Apple GPU model name."""
        return f"{self.architecture} GPU"

    @property
    def memory(self) -> Memory:
        """Unified memory visible to CPU, GPU, and Neural Engine."""
        return Memory.system(scope="unified", unified=True)

    @classmethod
    def all(cls) -> tuple["AppleGPU", ...]:
        """Return the local Apple Silicon GPU when present."""
        return (cls(index=0),) if cls.is_available() else ()

    @classmethod
    def is_available(cls) -> bool:
        """Whether this host is an Apple Silicon Mac."""
        return platform.system() == "Darwin" and platform.machine() == "arm64"
