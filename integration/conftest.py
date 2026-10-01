"""Run the real `mb` in a throwaway workspace, the way a person or an agent runs it.

Every test here starts the installed console script as a separate process, so what is checked is
what a terminal gets: argument parsing, startup, output and exit status, with nothing mocked.
"""

import os
import shutil
import subprocess
import sysconfig
from dataclasses import dataclass
from pathlib import Path

import pytest

# The console script installed with the interpreter running the tests: beside it on POSIX, in
# its `Scripts` folder on Windows.
MB = shutil.which("mb", path=sysconfig.get_path("scripts"))


@dataclass(frozen=True)
class Ran:
    """One finished `mb` process."""

    code: int
    out: str
    err: str

    @property
    def said(self) -> str:
        return self.out + self.err


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A fresh workspace holding only a manifest."""
    (tmp_path / "mb.toml").write_text('[workspace]\nname = "it"\n', encoding="utf-8")
    return tmp_path


@pytest.fixture
def mb(workspace: Path):
    """Run `mb` with `args` inside the workspace, refusing any traceback in what it printed."""
    if MB is None:
        pytest.skip("the mb console script is not installed beside this interpreter")

    def run(
        *args: str,
        cwd: Path | None = None,
        timeout: float = 120,
        env: dict[str, str] | None = None,
    ) -> Ran:
        done = subprocess.run(
            [MB, *args],
            cwd=cwd or workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env={**os.environ, "NO_COLOR": "1", **(env or {})},
            check=False,
        )
        ran = Ran(done.returncode, done.stdout, done.stderr)
        assert "Traceback (most recent call last)" not in ran.said, ran.said
        return ran

    return run
