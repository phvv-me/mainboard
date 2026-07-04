import shutil
import sys
from functools import cached_property
from pathlib import Path

from .. import shell
from .base import FrozenModel
from .compiler_info import CompilerInfo


class SystemCompilers(FrozenModel):
    """Detected compilers for the current machine.

    arch: host CPU architecture string, used to prefer Grace Clang on aarch64.
    cpu: CPU model name, used to select arch-specific compiler flags.
    cuda_arch: CUDA architecture string, e.g. `"90"` for sm_90.
    """

    arch: str
    cpu: str
    cuda_arch: str

    @cached_property
    def _flags(self) -> tuple[str, ...]:
        """Host compiler flags tuned for this CPU architecture."""
        if self.arch == "aarch64":
            if "neoverse-v2" in self.cpu.lower():
                return ("-mcpu=neoverse-v2",)
            return ("-mcpu=native",)
        if self.arch in {"x86_64", "amd64"}:
            return ("-march=native", "-mtune=native")
        return ()

    @cached_property
    def cxx(self) -> CompilerInfo:
        """Host C++ compiler; raises `FileNotFoundError` if none found."""
        if (
            self.arch == "aarch64"
            and (p := shutil.which("clang++"))
            and "grace" in shell.run(p, "--version").lower()
        ):
            return CompilerInfo(path=Path(p))
        if p := shutil.which("g++") or shutil.which("clang++"):
            return CompilerInfo(path=Path(p))
        raise FileNotFoundError("No C++ compiler found.")

    @cached_property
    def nvcc(self) -> CompilerInfo:
        """nvcc compiler; raises `FileNotFoundError` if CUDA toolkit is absent."""
        _dirs = [
            Path(sys.prefix) / "bin" / "nvcc",
            Path("/usr/local/cuda/bin/nvcc"),
            *sorted(Path("/usr/local").glob("cuda-*/bin/nvcc")),
        ]
        if p := (
            shutil.which("nvcc")
            or shutil.which("cuda-nvcc")
            or next((c for c in _dirs if c.exists()), None)
        ):
            return CompilerInfo(path=Path(p))
        raise FileNotFoundError("nvcc not found.")

    @property
    def cxx_flags_release_init(self) -> str | None:
        """Release C++ flags for CMake initialization."""
        return " ".join(self._flags) or None

    @property
    def cuda_flags_release_init(self) -> str | None:
        """Release CUDA flags for CMake initialization."""
        return " ".join(f"-Xcompiler={flag}" for flag in self._flags) or None
