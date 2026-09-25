import sys
from typing import TYPE_CHECKING

import pytest
from plumbum import local

from mainboard import MissionError
from mainboard.engines.compile.backend import Process, Tool

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_subprocess import FakeProcess

_PYTHON = sys.executable


class _PythonTool(Tool):
    """A minimal `Tool` naming a cross-platform executable for the base class."""

    name = _PYTHON


class _Unavailable(_PythonTool):
    """A tool whose guard says this workspace has no business running it."""

    def available(self) -> bool:
        return False


def test_flags_convert_keyword_options_to_cli_args() -> None:
    assert Tool.flags(resolve=True, feature="serving", skip=False, empty="", extra=None) == [
        "--resolve",
        "--feature",
        "serving",
    ]


def test_a_tool_names_the_binary_it_runs_and_pins_nothing_by_default() -> None:
    """A tool's binary name is required only where it is used.

    A backend running through another tool names no binary, so the name is demanded at the
    one boundary that needs it rather than of every subclass.
    """
    tool = _PythonTool()
    assert str(tool.command) == _PYTHON
    assert tool.scope() == ()
    assert tool.cwd() is None
    assert tool.available() is True

    with pytest.raises(MissionError, match="names no command"):
        _ = Tool().command


def test_within_cwd_runs_in_the_declared_directory(tmp_path: Path) -> None:
    class _ScopedTool(_PythonTool):
        def cwd(self) -> Path:
            return tmp_path

    seen: list[str] = []
    _ScopedTool().within_cwd(lambda command: seen.append(str(local.cwd)), "hi")
    assert seen == [str(tmp_path)]


def test_a_failed_run_raises_or_preserves_its_code_depending_on_who_asked(
    fp: FakeProcess,
) -> None:
    """A run raises on failure and a passthrough relays the exit.

    Raising keeps a failed install from being reported as green, while a transparent
    passthrough has to exit with whatever the wrapped command exited.
    """
    fp.register([_PYTHON, "hi"], returncode=0)
    assert _PythonTool()("hi") is None

    fp.register([_PYTHON, "hi"], returncode=1)
    with pytest.raises(MissionError, match="failed"):
        _PythonTool()("hi")

    fp.register([_PYTHON, "hi"], returncode=9)
    assert _PythonTool().exit_code("hi") == 9


def test_an_unavailable_tool_runs_nothing_and_reports_success(fp: FakeProcess) -> None:
    _Unavailable()("hi")
    assert _Unavailable().exit_code("hi") == 0
    assert not fp.calls


def test_a_deferred_tool_builds_the_same_command_without_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(Process, "detached", lambda command: seen.append(str(command)))

    assert _PythonTool().defer("hi") is None
    assert seen == [f"{_PYTHON} hi"]

    _Unavailable().defer("ignored")
    assert seen == [f"{_PYTHON} hi"]


def test_on_windows_a_manager_runs_by_its_pathext_spelling_and_a_script_under_cmd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """conda ships `npm` as a POSIX script beside `npm.cmd`; only the launcher is a program."""
    from plumbum import local

    from mainboard.engines.compile.backend import tool as tool_module

    (tmp_path / "npm").write_text("#!/bin/sh\n")
    (tmp_path / "npm.cmd").write_text("@echo off\n")
    for program in ("node.exe", "cmd.exe"):
        (tmp_path / program).write_bytes(b"MZ")
        (tmp_path / program).chmod(0o755)
    monkeypatch.setattr(tool_module.platform, "system", lambda: "Windows")
    with local.env(PATH=str(tmp_path), PATHEXT=".EXE;.CMD;.BAT"):
        npm = tool_module.windows_launcher("npm")
        node = tool_module.windows_launcher("node")
        with pytest.raises(MissionError, match="yarn is not on PATH"):
            tool_module.windows_launcher("yarn")
        # A spelling that already names its program, by extension or by path, is taken as is.
        named = tool_module.windows_launcher("node.exe")
        located = tool_module.windows_launcher(str(tmp_path / "npm.cmd"))
    assert npm.formulate()[-3:] == ["/d", "/c", str(tmp_path / "npm.cmd")]
    assert npm.formulate()[0].lower().endswith("cmd.exe")
    assert node.formulate() == [str(tmp_path / "node.exe")]
    assert named.formulate() == [str(tmp_path / "node.exe")]
    assert located.formulate() == npm.formulate()


class _Node(Tool):
    """A manager conda ships both as a POSIX script and as the program Windows runs."""

    name = "node"


def test_on_windows_a_tools_own_command_is_the_launcher_and_never_the_posix_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "node").write_text("#!/bin/sh\n")
    (tmp_path / "node.exe").write_bytes(b"MZ")
    (tmp_path / "node.exe").chmod(0o755)
    monkeypatch.setattr("platform.system", lambda: "Windows")
    with local.env(PATH=str(tmp_path), PATHEXT=".EXE;.CMD;.BAT"):
        command = _Node().command
    assert command.formulate() == [str(tmp_path / "node.exe")]
