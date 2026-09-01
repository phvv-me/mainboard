from collections.abc import Sequence
from pathlib import Path

import pytest

from mainboard.probe import Compiler, CompilerKind, Compilers
from mainboard.probe.facts import compilers as compilers_mod

_GRACE_BANNER = "clang version 17.0.6 for Grace"
_NVCC_BANNER = "nvcc: NVIDIA (R) Cuda compiler driver"


def only_on_path(monkeypatch: pytest.MonkeyPatch, *present: str) -> None:
    """Pretend `present` are the only compiler binaries on PATH, each under `/usr/bin`."""
    monkeypatch.setattr(
        compilers_mod.shutil, "which", lambda name: f"/usr/bin/{name}" if name in present else None
    )


def answer_version(monkeypatch: pytest.MonkeyPatch, banner: str) -> None:
    """Answer every `--version` probe with one banner instead of running a binary."""
    monkeypatch.setattr(compilers_mod, "run", lambda *command: banner)


@pytest.fixture
def bare_toolchain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A host with nothing on PATH and empty toolkit roots, away from any real CUDA install.

    Returns the throwaway directory holding the interpreter prefix and the `/usr/local` stand-in,
    so a test can install a compiler into either one.
    """
    only_on_path(monkeypatch)
    answer_version(monkeypatch, _NVCC_BANNER)
    monkeypatch.setattr(compilers_mod.sys, "prefix", str(tmp_path / "prefix"))
    monkeypatch.setattr(compilers_mod, "_CUDA_ROOT", tmp_path / "usr-local")
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    return tmp_path


@pytest.mark.parametrize(
    ("path", "banner", "expected"),
    [
        ("/usr/local/cuda/bin/nvcc", "unreadable banner", CompilerKind.NVCC),
        ("/opt/nvhpc/bin/cuda-nvcc", "unreadable banner", CompilerKind.NVCC),
        ("/opt/nvhpc/bin/nvcc-13", _NVCC_BANNER, CompilerKind.NVCC),
        ("/opt/nvidia/bin/clang++", _GRACE_BANNER, CompilerKind.CLANG_GRACE),
        ("/usr/bin/cc", "cc (GCC) 13.3.0", CompilerKind.GCC),
        ("/usr/bin/g++", "g++ (Ubuntu 13.3.0-6) 13.3.0", CompilerKind.GCC),
        ("/usr/bin/clang++", "Ubuntu clang version 17.0.6", CompilerKind.CLANG),
        ("/usr/bin/tcc", "tiny c compiler 0.9.27", CompilerKind.UNKNOWN),
    ],
)
def test_the_compiler_family_comes_from_the_binary_name_then_its_version_banner(
    path: str, banner: str, expected: CompilerKind, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Either nvcc spelling settles the family outright, since its banner is not always readable.
    Everything else is scanned for a marker in order, and Grace is checked before clang because
    NVIDIA's Grace toolchain banner also says clang, while both GNU spellings mean one family."""
    answer_version(monkeypatch, banner)
    assert Compiler(path=Path(path)).kind is expected


@pytest.mark.parametrize(
    ("arch", "present", "banner", "expected"),
    [
        pytest.param(
            "aarch64", ("clang++", "g++"), _GRACE_BANNER, "clang++", id="aarch64-grace-clang"
        ),
        pytest.param(
            "aarch64", ("clang++", "g++"), "clang version 17.0.6", "g++", id="aarch64-plain-clang"
        ),
        pytest.param("aarch64", ("g++",), "g++ (Ubuntu) 13.3.0", "g++", id="aarch64-no-clang"),
        pytest.param("x86_64", ("clang++", "g++"), "g++ (Ubuntu) 13.3.0", "g++", id="x86-both"),
        pytest.param(
            "x86_64", ("clang++",), "clang version 17.0.6", "clang++", id="x86-clang-only"
        ),
    ],
)
def test_the_host_compiler_prefers_grace_clang_then_gpp_then_clang(
    arch: str, present: Sequence[str], banner: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The C++ pick follows the platform's own preference order.

    Grace Clang only wins on aarch64 and only when its banner says so, so an ordinary clang++
    on the same machine loses to g++, and clang++ is taken only where g++ is absent.
    """
    only_on_path(monkeypatch, *present)
    answer_version(monkeypatch, banner)
    assert Compilers(arch=arch, cpu="Neoverse-V2", cuda_arch="90").cxx.path == Path(
        f"/usr/bin/{expected}"
    )


@pytest.mark.parametrize(
    ("on_path", "relative", "expected"),
    [
        pytest.param("nvcc", None, "/usr/bin/nvcc", id="path-nvcc"),
        pytest.param("cuda-nvcc", None, "/usr/bin/cuda-nvcc", id="path-cuda-nvcc"),
        pytest.param(None, "prefix/bin/nvcc", None, id="interpreter-prefix"),
        pytest.param(
            None, "prefix/Library/bin/nvcc.exe", None, id="windows-interpreter-prefix"
        ),
        pytest.param(None, "usr-local/cuda/bin/nvcc", None, id="toolkit-root"),
        pytest.param(None, "usr-local/cuda-13.0/bin/nvcc", None, id="versioned-toolkit-root"),
    ],
)
def test_nvcc_is_taken_from_path_first_and_from_the_toolkit_roots_after(
    on_path: str | None,
    relative: str | None,
    expected: str | None,
    bare_toolchain: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """nvcc is found on PATH first and in the conventional roots after.

    A CUDA toolkit often installs outside PATH, so either spelling on PATH is used first and
    the interpreter prefix then the conventional install roots are searched after it.
    """
    if on_path:
        only_on_path(monkeypatch, on_path)
    installed = bare_toolchain / relative if relative else None
    if installed:
        installed.parent.mkdir(parents=True)
        installed.touch()

    nvcc = Compilers(arch="x86_64", cpu="Xeon", cuda_arch="89").nvcc
    assert nvcc.path == (Path(expected) if expected else installed)
    assert nvcc.kind is CompilerKind.NVCC


@pytest.mark.parametrize("variable", ["CUDA_PATH", "CUDA_HOME"])
def test_nvcc_uses_the_declared_cross_platform_toolkit_home(
    variable: str, bare_toolchain: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NVIDIA's and the ecosystem's toolkit-home variables both lead to their native binary."""
    toolkit = bare_toolchain / variable.lower()
    binary = toolkit / "bin" / "nvcc.exe"
    binary.parent.mkdir(parents=True)
    binary.touch()
    monkeypatch.setenv(variable, str(toolkit))

    assert Compilers(arch="amd64", cpu="Ryzen", cuda_arch="120").nvcc.path == binary


@pytest.mark.parametrize(
    ("half", "message"),
    [("cxx", r"No C\+\+ compiler"), ("nvcc", "No `nvcc` found")],
)
def test_a_missing_half_of_the_toolchain_is_named_rather_than_left_a_blank_path(
    half: str, message: str, bare_toolchain: Path
) -> None:
    """A missing compiler refuses by name at first touch.

    A build configured against an empty path fails much later and far less clearly, so each
    compiler resolves lazily and says which half of the toolchain the host does not have.
    """
    compilers = Compilers(arch="x86_64", cpu="Xeon", cuda_arch="89")
    with pytest.raises(FileNotFoundError, match=message):
        getattr(compilers, half)


@pytest.mark.parametrize(
    ("arch", "cpu", "expected_cxx", "expected_cuda"),
    [
        ("aarch64", "Neoverse-V2", "-mcpu=neoverse-v2", "-Xcompiler=-mcpu=neoverse-v2"),
        ("aarch64", "Cortex-A78", "-mcpu=native", "-Xcompiler=-mcpu=native"),
        (
            "x86_64",
            "Xeon",
            "-march=native -mtune=native",
            "-Xcompiler=-march=native -Xcompiler=-mtune=native",
        ),
        (
            "amd64",
            "EPYC",
            "-march=native -mtune=native",
            "-Xcompiler=-march=native -Xcompiler=-mtune=native",
        ),
        ("riscv64", "SiFive", None, None),
    ],
)
def test_release_flags_track_the_cpu_architecture_and_forward_through_nvcc(
    arch: str, cpu: str, expected_cxx: str | None, expected_cuda: str | None
) -> None:
    """Release flags tune to the exact part where one is known.

    Grace gets its own `-mcpu` target while other aarch64 parts fall back to native, x86
    tunes both march and mtune, and an architecture with no tuning to offer leaves the flags
    unset.
    """
    compilers = Compilers(arch=arch, cpu=cpu, cuda_arch="90")
    assert compilers.cxx_flags_release_init == expected_cxx
    assert compilers.cuda_flags_release_init == expected_cuda
