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
