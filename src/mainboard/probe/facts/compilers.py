import os
import shutil
import sys
from functools import cached_property
from pathlib import Path

from patos import FrozenModel

from ..enums import CompilerKind
from ..shell import run

_CUDA_ROOT = Path("/usr/local")
_NVCC_BINARIES = ("nvcc", "cuda-nvcc")
# Version-banner markers scanned in order, so a new family costs one pair rather than an
# edit to the classifier. `grace` comes before `clang` because NVIDIA's Grace toolchain
# banner also says clang, and both GNU spellings map to the same family.
_KIND_BY_MARKER = (
    ("nvcc: nvidia", CompilerKind.NVCC),
    ("grace", CompilerKind.CLANG_GRACE),
    ("gcc", CompilerKind.GCC),
    ("g++", CompilerKind.GCC),
    ("clang", CompilerKind.CLANG),
)


class Compiler(FrozenModel):
    """One discovered compiler binary.

    path: absolute path to the compiler executable.
    """

    path: Path

    @cached_property
    def kind(self) -> CompilerKind:
        """Compiler family, read from the binary name and then its `--version` banner."""
        if self.path.stem in _NVCC_BINARIES:
            return CompilerKind.NVCC
        banner = run(str(self.path), "--version").lower()
        return next(
            (kind for marker, kind in _KIND_BY_MARKER if marker in banner), CompilerKind.UNKNOWN
        )


class Compilers(FrozenModel):
    """Host C++ and CUDA compilers with the release flags a native build configures with.

    This is what a CMake or nanobind build reads to pin its toolchain, so each compiler is
    resolved lazily and raises `FileNotFoundError` when it is genuinely absent, letting the
    caller learn which half of the toolchain is missing instead of configuring a build
    against an empty path.

    arch: host CPU architecture, `aarch64` selecting Grace Clang when it is installed.
    cpu: host CPU model name, choosing the `-mcpu` target on aarch64.
    cuda_arch: CUDA compute capability without the dot, e.g. `89` for sm_89.
    """

    arch: str
    cpu: str
    cuda_arch: str

    @property
    def cuda_flags_release_init(self) -> str | None:
        """`CMAKE_CUDA_FLAGS_RELEASE_INIT`, the host flags forwarded through nvcc."""
        return " ".join(f"-Xcompiler={flag}" for flag in self.release_flags) or None

    @cached_property
    def cxx(self) -> Compiler:
        """Host C++ compiler, Grace Clang on aarch64 when installed, otherwise g++ then clang++."""
        grace = shutil.which("clang++")
        if self.arch == "aarch64" and grace and "grace" in run(grace, "--version").lower():
            return Compiler(path=Path(grace))
        if found := shutil.which("g++") or grace:
            return Compiler(path=Path(found))
        raise FileNotFoundError("No C++ compiler found on PATH.")

    @property
    def cxx_flags_release_init(self) -> str | None:
        """`CMAKE_CXX_FLAGS_RELEASE_INIT` tuned for this CPU, `None` when it is untuned."""
        return " ".join(self.release_flags) or None

    @cached_property
    def nvcc(self) -> Compiler:
        """CUDA compiler from PATH, declared toolkit homes, then conventional install roots."""
        if found := shutil.which("nvcc") or shutil.which("cuda-nvcc"):
            return Compiler(path=Path(found))
        declared = (
            Path(home) / "bin" / name
            for variable in ("CUDA_PATH", "CUDA_HOME")
            if (home := os.environ.get(variable))
            for name in ("nvcc", "nvcc.exe")
        )
        toolkits = (
            *declared,
            Path(sys.prefix) / "bin" / "nvcc",
            Path(sys.prefix) / "Library" / "bin" / "nvcc.exe",
            _CUDA_ROOT / "cuda" / "bin" / "nvcc",
            *sorted(_CUDA_ROOT.glob("cuda-*/bin/nvcc")),
        )
        if path := next((t for t in toolkits if t.exists()), None):
            return Compiler(path=path)
        raise FileNotFoundError(
            "No `nvcc` found on PATH, under CUDA_PATH/CUDA_HOME, or in the toolkit roots."
        )

    @cached_property
    def release_flags(self) -> tuple[str, ...]:
        """Host release flags tuned for this CPU, empty when the architecture is unrecognized."""
        if self.arch == "aarch64":
            core = "neoverse-v2" if "neoverse-v2" in self.cpu.lower() else "native"
            return (f"-mcpu={core}",)
        if self.arch in {"x86_64", "amd64"}:
            return ("-march=native", "-mtune=native")
        return ()
