from typing import TYPE_CHECKING

import pytest
from plumbum import local

from mainboard import MissionError
from mainboard.engines.compile.backend import Tool

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_subprocess import FakeProcess

_ECHO = str(local.which("echo"))


class _EchoTool(Tool):
    """A minimal `Tool` naming the real `echo` binary, for exercising the base class."""

    name = "echo"


def test_flags_convert_keyword_options_to_cli_args() -> None:
    assert Tool.flags(resolve=True, feature="serving", skip=False, empty="", extra=None) == [
        "--resolve",
        "--feature",
        "serving",
    ]


def test_command_raises_without_a_declared_name() -> None:
    with pytest.raises(MissionError, match="names no command"):
        _ = Tool().command


def test_command_resolves_the_declared_binary() -> None:
    assert str(_EchoTool().command) == _ECHO


def test_scope_and_cwd_default_to_nothing() -> None:
    tool = _EchoTool()
    assert tool.scope() == ()
    assert tool.cwd() is None
    assert tool.available() is True


def test_within_cwd_runs_in_the_declared_directory(tmp_path: Path) -> None:
    class _ScopedTool(_EchoTool):
        def cwd(self) -> Path:
            return tmp_path

    seen: list[str] = []
    _ScopedTool().within_cwd(lambda command: seen.append(str(local.cwd)), "hi")
    assert seen == [str(tmp_path)]


def test_call_is_a_noop_when_unavailable(fp: FakeProcess) -> None:
    class _Unavailable(_EchoTool):
        def available(self) -> bool:
            return False

    _Unavailable()("hi")
    assert not fp.calls


def test_call_raises_on_failure(fp: FakeProcess) -> None:
    fp.register([_ECHO, "hi"], returncode=1)
    with pytest.raises(MissionError, match="`echo hi` failed"):
        _EchoTool()("hi")


def test_call_succeeds_silently(fp: FakeProcess) -> None:
    fp.register([_ECHO, "hi"], returncode=0)
    assert _EchoTool()("hi") is None


def test_exit_code_is_zero_when_unavailable() -> None:
    class _Unavailable(_EchoTool):
        def available(self) -> bool:
            return False

    assert _Unavailable().exit_code("hi") == 0


def test_exit_code_preserves_the_real_code(fp: FakeProcess) -> None:
    fp.register([_ECHO, "hi"], returncode=9)
    assert _EchoTool().exit_code("hi") == 9
