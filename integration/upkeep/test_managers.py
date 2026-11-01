"""`ask` puts one read-only question to a package manager and reads silence as no answer."""

import subprocess
import sys
from pathlib import Path

import pytest

from mainboard.upkeep.managers import ask

from ..fakes import failing

_PRINTS = "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"
_PIXI_HOME = "import os; print(os.environ['PIXI_HOME'])"


def test_ask_answers_both_streams_of_a_child_whatever_it_exits_with() -> None:
    assert ask([sys.executable, "-c", _PRINTS]) == "out\nerr\n"


def test_ask_sets_the_pixi_home_the_dotfiles_lay_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert ask([sys.executable, "-c", _PIXI_HOME]) == f"{tmp_path / '.pixi'}\n"


@pytest.mark.parametrize(
    "fault",
    [
        pytest.param(FileNotFoundError("absent"), id="a-program-that-cannot-start"),
        pytest.param(subprocess.TimeoutExpired("slow", 20.0), id="a-command-past-its-timeout"),
    ],
)
def test_ask_answers_nothing_for_a_child_that_cannot_answer(
    fault: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subprocess, "run", failing(fault))

    assert ask(["slow"]) is None
