import os
import shlex
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from ..builders import committed
from .engine.builders import written

_VERBS = ("ls-files", "check-attr")


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """A repository holding `README`, with an untracked `a/x.py` beside it."""
    return written(committed(tmp_path / "repository"), {"a/x.py": ""})


@pytest.fixture
def git_verbs(monkeypatch: pytest.MonkeyPatch, repository: Path) -> Callable[[], list[str]]:
    """The verbs (`ls-files`, `check-attr`) of the git commands run from here on, in order.

    A `git` ahead on PATH, beside the repository, logs its arguments and runs the real one; the
    repository is built before it counts.
    """
    log, shim = repository.parent / "git.log", repository.parent / "shim" / "git"
    shim.parent.mkdir()
    shim.write_text(
        f'#!/bin/sh\necho "$@" >> {shlex.quote(str(log))}\n'
        f'exec {shlex.quote(str(shutil.which("git")))} "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    log.touch()
    monkeypatch.setenv("PATH", f"{shim.parent}{os.pathsep}{os.environ['PATH']}")
    return lambda: [word for word in log.read_text().split() if word in _VERBS]
