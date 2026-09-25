import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import pytest

from mainboard.center.carrier import _BOOTSTRAP as BOOTSTRAP
from mainboard.center.carrier import MARKER
from mainboard.dispatch.transport import SshTransport

from ..git.conftest import Forge, Workspace, isolated_git, template

__all__ = ["LocalTransport", "isolated_git", "template"]


@pytest.fixture
def tree(template: Path, tmp_path: Path) -> Workspace:
    """A private copy of the git fixture tree: an owned root and library, two foreign
    references, and real bare remotes under the test's own directory."""
    forge = Forge(tmp_path / "forge")
    shutil.copytree(template, forge.root, symlinks=True)
    return Workspace(forge, forge.root / "work")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private home directory, so no test reads or writes the real one."""
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HOME", str(path))
    monkeypatch.setenv("USERPROFILE", str(path))
    return path


class LocalTransport(SshTransport):
    """An ssh whose far side is a Python here: the carrier's bootstrap run with its own home.

    Every byte the carrier would send travels through a real child process, so the loader and
    the destination agent run exactly as they would on a machine with nothing of ours on it.
    A function named in `canned` answers that JSON instead, for the calls that would reach this
    machine's own tools (a census, a gh login).

    home: the destination's home directory, what the child sees as HOME.
    canned: fixed answers by agent function name.
    """

    home: Path
    canned: dict[str, str] = {}

    def feed(
        self, command: tuple[str, ...], host: str, *, operation: str, chunks: Iterable[bytes]
    ) -> str:
        sent = b"".join(chunks)
        if operation in self.canned:
            return f"{MARKER}{self.canned[operation]}\n"
        done = subprocess.run(
            [sys.executable, "-c", BOOTSTRAP],
            input=sent,
            capture_output=True,
            check=False,
            env={**os.environ, "HOME": str(self.home), "USERPROFILE": str(self.home)},
        )
        if done.returncode:
            raise RuntimeError(done.stderr.decode())
        return done.stdout.decode()
