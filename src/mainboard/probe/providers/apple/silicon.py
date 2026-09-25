import platform
from functools import cached_property
from typing import ClassVar

from ...facts.memory import Memory
from ...shell import sysctl


class AppleSilicon:
    """An Apple Silicon engine, named after its SoC and backed by unified memory.

    Apple Silicon has no separate GPU or Neural Engine model string on the command line, so every
    engine takes its identity from `sysctl machdep.cpu.brand_string`, the SoC name.

    engine: the suffix naming the engine on that SoC, e.g. `GPU`.
    """

    engine: ClassVar[str]

    @cached_property
    def architecture(self) -> str:
        """The SoC family backing this engine, e.g. `Apple M4 Pro`."""
        return sysctl("machdep.cpu.brand_string") or "Apple Silicon"

    @cached_property
    def label(self) -> str:
        """The SoC name followed by the engine's, e.g. `Apple M4 Pro GPU`."""
        return f"{self.architecture} {self.engine}"

    @property
    def memory(self) -> Memory:
        """Unified memory visible to CPU, GPU, and Neural Engine."""
        return Memory.system(scope="unified", unified=True)

    @classmethod
    def is_available(cls) -> bool:
        """Whether this host is an Apple Silicon Mac."""
        return platform.system() == "Darwin" and platform.machine() == "arm64"
