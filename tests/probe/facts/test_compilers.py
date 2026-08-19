from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mainboard.probe import Compiler, CompilerKind, Compilers
from mainboard.probe.facts import compilers as compilers_mod

if TYPE_CHECKING:
    from collections.abc import Sequence


def only_on_path(monkeypatch: pytest.MonkeyPatch, *present: str) -> None:
    """Pretend `present` are the only compiler binaries on PATH, each under `/usr/bin`."""
    monkeypatch.setattr(
        compilers_mod.shutil, "which", lambda name: f"/usr/bin/{name}" if name in present else None
    )


def answer_version(monkeypatch: pytest.MonkeyPatch, banner: str) -> None:
    """Answer every `--version` probe with one banner instead of running a binary."""
    monkeypatch.setattr(compilers_mod, "run", lambda *command: banner)


def isolate_toolkits(monkeypatch: pytest.MonkeyPatch, prefix: Path, root: Path) -> None:
    """Point the nvcc fallback search at throwaway directories, away from a real CUDA install."""
    monkeypatch.setattr(compilers_mod.sys, "prefix", str(prefix))
    monkeypatch.setattr(compilers_mod, "CUDA_ROOT", root)


@pytest.mark.parametrize(
    ("path", "banner", "expected"),
    [
        ("/usr/local/cuda/bin/nvcc", "unreadable banner", CompilerKind.NVCC),
        ("/opt/nvhpc/bin/nvcc-13", "nvcc: NVIDIA (R) Cuda compiler driver", CompilerKind.NVCC),
        ("/opt/nvidia/bin/clang++", "clang version 17.0.6 for Grace", CompilerKind.CLANG_GRACE),
        ("/usr/bin/cc", "cc (GCC) 13.3.0", CompilerKind.GCC),
        ("/usr/bin/g++", "g++ (Ubuntu 13.3.0-6) 13.3.0", CompilerKind.GCC),
        ("/usr/bin/clang++", "Ubuntu clang version 17.0.6", CompilerKind.CLANG),
        ("/usr/bin/tcc", "tiny c compiler 0.9.27", CompilerKind.UNKNOWN),
    ],
)
def test_kind_comes_from_the_binary_name_then_the_banner(
    path: str, banner: str, expected: CompilerKind, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A known nvcc name settles the family outright, everything else reads its banner."""
    answer_version(monkeypatch, banner)
    assert Compiler(path=Path(path)).kind is expected


def test_cxx_prefers_grace_clang_on_aarch64(monkeypatch: pytest.MonkeyPatch) -> None:
    """On aarch64 a Grace-flavored clang++ wins over the g++ that is also installed."""
    only_on_path(monkeypatch, "clang++", "g++")
    answer_version(monkeypatch, "clang version 17.0.6 for Grace")

    cxx = Compilers(arch="aarch64", cpu="Neoverse-V2", cuda_arch="90").cxx
    assert cxx.path == Path("/usr/bin/clang++")
    assert cxx.kind is CompilerKind.CLANG_GRACE


@pytest.mark.parametrize(
    ("arch", "present", "banner", "expected"),
    [
        ("aarch64", ("clang++", "g++"), "Ubuntu clang version 17.0.6", "/usr/bin/g++"),
        ("aarch64", ("g++",), "g++ (Ubuntu 13.3.0-6) 13.3.0", "/usr/bin/g++"),
        ("x86_64", ("clang++", "g++"), "g++ (Ubuntu 13.3.0-6) 13.3.0", "/usr/bin/g++"),
        ("x86_64", ("clang++",), "Ubuntu clang version 17.0.6", "/usr/bin/clang++"),
    ],
)
def test_cxx_falls_back_through_gpp_then_clang(
    arch: str,
    present: Sequence[str],
    banner: str,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a Grace clang the host compiler is g++ when installed and clang++ otherwise."""
    only_on_path(monkeypatch, *present)
    answer_version(monkeypatch, banner)
    assert Compilers(arch=arch, cpu="Xeon", cuda_arch="90").cxx.path == Path(expected)


def test_cxx_raises_without_a_host_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty PATH names the missing half of the toolchain rather than yielding a blank path."""
    only_on_path(monkeypatch)
    compilers = Compilers(arch="x86_64", cpu="Xeon", cuda_arch="90")
    with pytest.raises(FileNotFoundError, match=r"No C\+\+ compiler"):
        _ = compilers.cxx


@pytest.mark.parametrize("binary", ["nvcc", "cuda-nvcc"])
def test_nvcc_is_taken_from_path_first(binary: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Either spelling of the CUDA compiler on PATH is used before any toolkit root."""
    only_on_path(monkeypatch, binary)
    answer_version(monkeypatch, "nvcc: NVIDIA (R) Cuda compiler driver")

    nvcc = Compilers(arch="x86_64", cpu="Xeon", cuda_arch="89").nvcc
    assert nvcc.path == Path(f"/usr/bin/{binary}")
    assert nvcc.kind is CompilerKind.NVCC


@pytest.mark.parametrize("relative", ["bin/nvcc", "cuda/bin/nvcc", "cuda-13.0/bin/nvcc"])
def test_nvcc_falls_back_to_the_toolkit_roots(
    relative: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With nothing on PATH, the interpreter prefix and the toolkit roots are searched."""
    prefix, root = tmp_path / "prefix", tmp_path / "usr-local"
    installed = (prefix if relative.startswith("bin/") else root) / relative
    installed.parent.mkdir(parents=True)
    installed.touch()
    only_on_path(monkeypatch)
    isolate_toolkits(monkeypatch, prefix, root)
    answer_version(monkeypatch, "nvcc: NVIDIA (R) Cuda compiler driver")

    assert Compilers(arch="x86_64", cpu="Xeon", cuda_arch="89").nvcc.path == installed


def test_nvcc_raises_without_a_cuda_toolkit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host with no CUDA toolkit anywhere reports that rather than guessing a path."""
    only_on_path(monkeypatch)
    isolate_toolkits(monkeypatch, tmp_path / "prefix", tmp_path / "usr-local")
    compilers = Compilers(arch="x86_64", cpu="Xeon", cuda_arch="89")
    with pytest.raises(FileNotFoundError, match="No `nvcc` found"):
        _ = compilers.nvcc


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
def test_release_flags_track_the_cpu_architecture(
    arch: str, cpu: str, expected_cxx: str | None, expected_cuda: str | None
) -> None:
    """Host flags are tuned per architecture and forwarded through nvcc, or left unset."""
    compilers = Compilers(arch=arch, cpu=cpu, cuda_arch="90")
    assert compilers.cxx_flags_release_init == expected_cxx
    assert compilers.cuda_flags_release_init == expected_cuda
