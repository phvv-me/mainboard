from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from plumbum import ProcessExecutionError

from mainboard.enums import ToolCategory
from mainboard.models import toolchain as toolchain_mod
from mainboard.models.toolchain import TOOL_PROBES, Toolchain, ToolProbe

GCC_OUTPUT = "gcc (Ubuntu 13.2.0-23) 13.2.0\nCopyright (C) 2023"
CLANG_OUTPUT = "clang version 18.1.3\nTarget: x86_64-pc-linux-gnu"
NVCC_OUTPUT = "nvcc: NVIDIA (R) Cuda compiler\nrelease 12.4, V12.4.131"
CMAKE_OUTPUT = "cmake version 3.29.2\n\nCMake suite maintained and supported."


def fake_which(present: dict[str, str]) -> object:
    """A `shutil.which` stand-in returning paths only for the named binaries."""
    return lambda name: present.get(name)


def fake_run(output: str) -> object:
    """A `shell.run` stand-in returning fixed version output for any command."""
    return lambda *command: output


@pytest.mark.parametrize(
    ("probe_name", "output", "expected"),
    [
        ("gcc", GCC_OUTPUT, "13.2.0"),
        ("g++", GCC_OUTPUT, "13.2.0"),
        ("clang", CLANG_OUTPUT, "18.1.3"),
        ("nvcc", NVCC_OUTPUT, "12.4"),
        ("cmake", CMAKE_OUTPUT, "3.29.2"),
    ],
)
def test_probe_parses_version_per_tool(
    probe_name: str,
    output: str,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each registered probe extracts the version from its tool's real output."""
    probe = next(p for p in TOOL_PROBES if p.name == probe_name)
    binary = probe.binaries[0]
    monkeypatch.setattr(toolchain_mod.shutil, "which", fake_which({binary: f"/usr/bin/{binary}"}))
    monkeypatch.setattr(toolchain_mod.shell, "run", fake_run(output))

    tool = probe.detect()

    assert tool.available is True
    assert tool.path == f"/usr/bin/{binary}"
    assert tool.version == expected


def test_probe_tries_binary_candidates_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """A probe with several candidates resolves the first one present on PATH."""
    probe = ToolProbe("make", ToolCategory.BUILD_SYSTEM, ("make", "gmake"))
    monkeypatch.setattr(toolchain_mod.shutil, "which", fake_which({"gmake": "/usr/bin/gmake"}))
    monkeypatch.setattr(toolchain_mod.shell, "run", fake_run("GNU Make 4.4.1"))

    tool = probe.detect()

    assert tool.path == "/usr/bin/gmake"
    assert tool.version == "4.4.1"


def test_probe_marks_missing_tool_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A probe whose binaries are all absent yields an unavailable tool."""
    probe = next(p for p in TOOL_PROBES if p.name == "ninja")
    monkeypatch.setattr(toolchain_mod.shutil, "which", fake_which({}))

    tool = probe.detect()

    assert tool.available is False
    assert tool.path is None
    assert tool.version is None


def test_probe_version_none_when_command_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A binary that errors on `--version` is still available but versionless."""
    probe = next(p for p in TOOL_PROBES if p.name == "cmake")

    def boom(*command: str) -> str:
        raise ProcessExecutionError(list(command), 1, "", "broken")

    monkeypatch.setattr(toolchain_mod.shutil, "which", fake_which({"cmake": "/usr/bin/cmake"}))
    monkeypatch.setattr(toolchain_mod.shell, "run", boom)

    tool = probe.detect()

    assert tool.available is True
    assert tool.version is None


def test_probe_version_none_when_unparseable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Output without a version number leaves the version unset."""
    probe = next(p for p in TOOL_PROBES if p.name == "cmake")
    monkeypatch.setattr(toolchain_mod.shutil, "which", fake_which({"cmake": "/usr/bin/cmake"}))
    monkeypatch.setattr(toolchain_mod.shell, "run", fake_run("no digits here"))

    tool = probe.detect()

    assert tool.version is None


def test_probe_keeps_only_available_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Toolchain.probe` drops tools absent from PATH and keeps the present ones."""
    monkeypatch.setattr(toolchain_mod.shutil, "which", fake_which({"cmake": "/usr/bin/cmake"}))
    monkeypatch.setattr(toolchain_mod.shell, "run", fake_run(CMAKE_OUTPUT))

    chain = Toolchain.probe()

    assert [tool.name for tool in chain.tools] == ["cmake"]


def test_by_category_groups_available_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """`by_category` buckets discovered tools under their toolchain category."""
    present = {"gcc": "/usr/bin/gcc", "g++": "/usr/bin/g++", "cmake": "/usr/bin/cmake"}
    monkeypatch.setattr(toolchain_mod.shutil, "which", fake_which(present))
    monkeypatch.setattr(toolchain_mod.shell, "run", fake_run("gcc (X) 9.4.0"))

    grouped = Toolchain.probe().by_category

    assert {tool.name for tool in grouped[ToolCategory.C_COMPILER]} == {"gcc"}
    assert {tool.name for tool in grouped[ToolCategory.CXX_COMPILER]} == {"g++"}
    assert {tool.name for tool in grouped[ToolCategory.BUILD_SYSTEM]} == {"cmake"}


def test_registry_extends_with_one_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adding a tool is one `ToolProbe`; the discoverer needs no other change."""
    extra = ToolProbe("bazel", ToolCategory.BUILD_SYSTEM, ("bazel",))
    monkeypatch.setattr(toolchain_mod.shutil, "which", fake_which({"bazel": "/usr/bin/bazel"}))
    monkeypatch.setattr(toolchain_mod.shell, "run", fake_run("bazel 7.1.1"))

    chain = Toolchain.probe(probes=(extra,))

    assert chain.tools[0].name == "bazel"
    assert chain.tools[0].version == "7.1.1"


@given(
    major=st.integers(min_value=0, max_value=99),
    minor=st.integers(min_value=0, max_value=99),
    patch=st.integers(min_value=0, max_value=999),
    prefix=st.text(st.characters(blacklist_categories=("Nd",)), max_size=20),
)
def test_default_pattern_extracts_dotted_version(
    major: int, minor: int, patch: int, prefix: str
) -> None:
    """The default regex captures the first `X.Y` or `X.Y.Z` token in any prefix."""
    probe = ToolProbe("any", ToolCategory.BUILD_SYSTEM, ("any",))
    text = f"{prefix} {major}.{minor}.{patch} trailing"
    match = probe.pattern.search(text)
    assert match is not None
    assert match.group(1) == f"{major}.{minor}.{patch}"
