from __future__ import annotations

from pathlib import Path

import pytest

from mainboard.enums import CompilerKind
from mainboard.models.system_compilers import SystemCompilers


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
        ("riscv64", "SiFive", None, None),
    ],
)
def test_release_flags_track_architecture(
    arch: str, cpu: str, expected_cxx: str | None, expected_cuda: str | None
) -> None:
    """Host and CUDA release flags are tuned per CPU architecture family."""
    compilers = SystemCompilers(arch=arch, cpu=cpu, cuda_arch="90")
    assert compilers.cxx_flags_release_init == expected_cxx
    assert compilers.cuda_flags_release_init == expected_cuda


def test_cxx_prefers_grace_clang_on_aarch64(monkeypatch: pytest.MonkeyPatch) -> None:
    """On aarch64 a Grace-flavored clang++ is chosen over plain g++."""
    monkeypatch.setattr(
        "mainboard.models.system_compilers.shutil.which",
        lambda name: f"/opt/{name}" if name == "clang++" else None,
    )
    monkeypatch.setattr(
        "mainboard.models.system_compilers.cached_run", lambda *cmd: "clang version for grace"
    )
    monkeypatch.setattr(
        "mainboard.models.compiler_info.cached_run", lambda *cmd: "clang version for grace"
    )
    compilers = SystemCompilers(arch="aarch64", cpu="Neoverse-V2", cuda_arch="90")
    assert compilers.cxx.path == Path("/opt/clang++")
    assert compilers.cxx.kind == CompilerKind.CLANG_GRACE


def test_cxx_falls_back_to_gpp(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no Grace clang is present, g++ is selected first."""
    monkeypatch.setattr(
        "mainboard.models.system_compilers.shutil.which",
        lambda name: f"/usr/bin/{name}" if name == "g++" else None,
    )
    monkeypatch.setattr(
        "mainboard.models.compiler_info.cached_run", lambda *cmd: "g++ (GCC) 13.2.0"
    )
    compilers = SystemCompilers(arch="x86_64", cpu="Xeon", cuda_arch="90")
    assert compilers.cxx.path == Path("/usr/bin/g++")


def test_cxx_raises_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """No host compiler on PATH raises a clear error."""
    monkeypatch.setattr("mainboard.models.system_compilers.shutil.which", lambda name: None)
    compilers = SystemCompilers(arch="x86_64", cpu="Xeon", cuda_arch="90")
    with pytest.raises(FileNotFoundError, match="C\\+\\+ compiler"):
        _ = compilers.cxx


def test_nvcc_found_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """nvcc is resolved from PATH when present."""
    monkeypatch.setattr(
        "mainboard.models.system_compilers.shutil.which",
        lambda name: "/usr/local/cuda/bin/nvcc" if name == "nvcc" else None,
    )
    monkeypatch.setattr(
        "mainboard.models.compiler_info.cached_run",
        lambda *cmd: "nvcc: NVIDIA (R) Cuda compiler\nrelease 12.4, V12.4.131",
    )
    compilers = SystemCompilers(arch="x86_64", cpu="Xeon", cuda_arch="124")
    assert compilers.nvcc.path == Path("/usr/local/cuda/bin/nvcc")
    assert compilers.nvcc.kind == CompilerKind.NVCC


def test_nvcc_raises_when_toolkit_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing CUDA toolkit raises rather than silently succeeding."""
    monkeypatch.setattr("mainboard.models.system_compilers.shutil.which", lambda name: None)
    monkeypatch.setattr(Path, "exists", lambda self: False)
    monkeypatch.setattr(Path, "glob", lambda self, pattern: iter(()))
    compilers = SystemCompilers(arch="x86_64", cpu="Xeon", cuda_arch="90")
    with pytest.raises(FileNotFoundError, match="nvcc not found"):
        _ = compilers.nvcc
