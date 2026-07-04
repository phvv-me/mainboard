import re
import shutil
from collections import defaultdict
from dataclasses import dataclass

from plumbum import ProcessExecutionError

from .. import shell
from ..enums import ToolCategory
from .base import FrozenModel
from .detected_tool import DetectedTool


@dataclass(frozen=True, slots=True)
class ToolProbe:
    """Immutable recipe for discovering one build tool.

    A probe is pure data: adding a tool to the host inventory means appending one
    `ToolProbe` to `TOOL_PROBES`, with no change to the discovery logic.

    name: display name reported on the `DetectedTool`.
    category: toolchain group the tool belongs to.
    binaries: candidate executable names, tried in order until one is on PATH.
    version_args: arguments that make the binary print its version.
    pattern: regex whose first group captures the version from the command output.
    """

    name: str
    category: ToolCategory
    binaries: tuple[str, ...]
    version_args: tuple[str, ...] = ("--version",)
    pattern: re.Pattern[str] = re.compile(r"(\d+\.\d+(?:\.\d+)?)")

    def detect(self) -> DetectedTool:
        """Resolve the first available binary and parse its version."""
        for binary in self.binaries:
            if path := shutil.which(binary):
                return DetectedTool(
                    name=self.name,
                    category=self.category,
                    path=path,
                    version=self._version(path),
                    available=True,
                )
        return DetectedTool(name=self.name, category=self.category, available=False)

    def _version(self, path: str) -> str | None:
        """Capture the binary's version output and extract the version string."""
        try:
            output = shell.run(path, *self.version_args)
        except ProcessExecutionError:
            return None
        if match := self.pattern.search(output):
            return match.group(1)
        return None


_GNU_VERSION = re.compile(r"\)\s+(\d+\.\d+(?:\.\d+)?)")
_CUDA_VERSION = re.compile(r"release\s+(\d+\.\d+)")

TOOL_PROBES: tuple[ToolProbe, ...] = (
    ToolProbe("gcc", ToolCategory.C_COMPILER, ("gcc",), pattern=_GNU_VERSION),
    ToolProbe("clang", ToolCategory.C_COMPILER, ("clang",)),
    ToolProbe("g++", ToolCategory.CXX_COMPILER, ("g++",), pattern=_GNU_VERSION),
    ToolProbe("clang++", ToolCategory.CXX_COMPILER, ("clang++",)),
    ToolProbe(
        "nvcc",
        ToolCategory.CUDA_COMPILER,
        ("nvcc", "cuda-nvcc"),
        pattern=_CUDA_VERSION,
    ),
    ToolProbe("cmake", ToolCategory.BUILD_SYSTEM, ("cmake",)),
    ToolProbe("ninja", ToolCategory.BUILD_SYSTEM, ("ninja",)),
    ToolProbe("make", ToolCategory.BUILD_SYSTEM, ("make", "gmake")),
    ToolProbe("meson", ToolCategory.BUILD_SYSTEM, ("meson",)),
)


class Toolchain(FrozenModel):
    """Build tools discovered on the host, grouped by category.

    Each registered `ToolProbe` is run once at probe time; only tools found on PATH
    are kept, so the model reports the host's real build capability.

    tools: every available tool, ordered as registered in `TOOL_PROBES`.
    """

    tools: tuple[DetectedTool, ...] = ()

    @classmethod
    def probe(cls, probes: tuple[ToolProbe, ...] = TOOL_PROBES) -> Toolchain:
        """Run every probe and keep the tools that are present on PATH."""
        detected = (probe.detect() for probe in probes)
        return cls(tools=tuple(tool for tool in detected if tool.available))

    @property
    def by_category(self) -> dict[ToolCategory, tuple[DetectedTool, ...]]:
        """Available tools bucketed by their toolchain category."""
        buckets: dict[ToolCategory, list[DetectedTool]] = defaultdict(list)
        for tool in self.tools:
            buckets[tool.category].append(tool)
        return {category: tuple(found) for category, found in buckets.items()}
