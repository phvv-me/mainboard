import os
import sys
import time
from typing import TYPE_CHECKING

from mainboard.lint.process import MISSING, TIMED_OUT, Invocation

from .conftest import Repository

if TYPE_CHECKING:
    from pathlib import Path


def _invocation(cwd: Path, *argv: str, timeout: float = 30.0) -> Invocation:
    return Invocation(step="probe", owner=".", cwd=cwd, argv=argv, env="default", timeout=timeout)


def test_a_command_runs_in_its_owner_and_reports_its_own_exit_and_words(tmp_path: Path) -> None:
    said = _invocation(
        tmp_path, sys.executable, "-c", "import os, sys; print(os.getcwd()); sys.exit(3)"
    ).run(os.environ)

    assert said.code == 3
    assert said.failed
    assert os.path.samefile(said.output.strip(), tmp_path)


def test_a_tool_missing_from_the_environment_fails_by_name_without_starting(
    tmp_path: Path,
) -> None:
    said = _invocation(tmp_path, "no-such-linter", "--check").run({"PATH": str(tmp_path)})

    assert said.code == MISSING
    assert "no-such-linter is not on the default environment's PATH" in said.output


def test_a_hung_command_is_killed_with_its_children_at_its_deadline(
    repository: Repository,
) -> None:
    started = time.monotonic()
    said = _invocation(
        repository.root, sys.executable, str(repository.root / "tool.py"), "hang", timeout=1.0
    ).run(os.environ)

    assert said.code == TIMED_OUT
    assert "probe exceeded its 1s deadline" in said.output
    assert time.monotonic() - started < 30
