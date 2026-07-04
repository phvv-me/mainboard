import re
from functools import cached_property
from pathlib import Path

from .. import shell
from ..enums import CompilerKind
from .base import FrozenModel

_NVCC_BINARIES = frozenset({"nvcc", "cuda-nvcc"})


class CompilerInfo(FrozenModel):
    """Detected compiler identity.

    path: absolute path to the compiler binary.
    """

    path: Path

    @cached_property
    def _version_output(self) -> str:
        """Raw lowercase `--version` output."""
        return shell.run(str(self.path), "--version").lower()

    @cached_property
    def kind(self) -> CompilerKind:
        """Normalized kind derived from binary name and version output."""
        binary = self.path.stem
        v = self._version_output
        if binary in _NVCC_BINARIES or "nvcc: nvidia" in v:
            return CompilerKind.NVCC
        if "grace" in v:
            return CompilerKind.CLANG_GRACE
        if "gcc" in v or "g++" in v:
            return CompilerKind.GCC
        if "clang" in v:
            return CompilerKind.CLANG
        return CompilerKind.UNKNOWN

    @cached_property
    def version(self) -> str | None:
        """Release string; `X.Y` for nvcc, first `--version` line otherwise."""
        if m := re.search(r"release\s+(\d+\.\d+)", self._version_output):
            return m.group(1)
        first = self._version_output.split("\n")[0].strip()
        return first or None
