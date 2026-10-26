import os
from collections.abc import Callable
from pathlib import Path

import pytest

from mainboard.apple import Apple
from mainboard.cli import build
from mainboard.proc import Processes

from .fakes import ProcessLog


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    """A result bundle whose name holds shell metacharacters."""
    path = tmp_path / "test ' $(literal).xcresult"
    path.mkdir()
    return path


@pytest.fixture
def bundles(tmp_path: Path) -> Path:
    """A folder holding an older and a newer result bundle."""
    for name, stamp in (("old.xcresult", 10), ("new.xcresult", 20)):
        (tmp_path / name).mkdir()
        os.utime(tmp_path / name, (stamp, stamp))
    return tmp_path


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """An Xcode project directory."""
    path = tmp_path / "App.xcodeproj"
    path.mkdir()
    return path


@pytest.fixture
def processes(monkeypatch: pytest.MonkeyPatch) -> ProcessLog:
    """`Processes.timeout` replaced by a recording double that runs nothing."""
    log = ProcessLog()
    monkeypatch.setattr(Processes, "timeout", log.timeout)
    return log


@pytest.fixture
def cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Callable[..., tuple[int, str]]:
    """Run `mb apple ARGV` against the given `Apple`; answers the exit code and what it printed."""

    def invoke(apple: Apple, *argv: str) -> tuple[int, str]:
        monkeypatch.setattr("mainboard.verbs.apps.apple.Apple", lambda: apple)
        with pytest.raises(SystemExit) as stopped:
            build()(["apple", *argv])
        return int(stopped.value.code or 0), capsys.readouterr().out

    return invoke
