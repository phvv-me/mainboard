"""Run the real `mb` in a throwaway workspace, the way a person or an agent runs it.

Every test here starts the installed console script as a separate process, so what is checked is
what a terminal gets: argument parsing, startup, output and exit status, with nothing mocked.
"""

import os
import shutil
import subprocess
import sysconfig
from pathlib import Path

import pytest
from patos import FrozenModel

# The console script installed beside the interpreter running the tests.
MB = shutil.which("mb", path=sysconfig.get_path("scripts"))


class Ran(FrozenModel):
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
        ran = Ran(code=done.returncode, out=done.stdout, err=done.stderr)
        assert "Traceback (most recent call last)" not in ran.said, ran.said
        return ran

    return run


def git(where: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git in `where`, local-path submodules allowed and an author named."""
    settings = ("protocol.file.allow=always", "user.name=it", "user.email=it@example.invalid")
    flags = [word for setting in settings for word in ("-c", setting)]
    return subprocess.run(
        ["git", "-C", str(where), *flags, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def committed(where: Path) -> Path:
    """A repository at `where` holding one commit."""
    where.mkdir(parents=True)
    git(where, "init", "--quiet", "--initial-branch=main")
    (where / "README").write_text("one\n", encoding="utf-8", newline="\n")
    git(where, "add", "README")
    git(where, "commit", "--quiet", "-m", "one")
    return where


@pytest.fixture
def tracked(workspace: Path, tmp_path_factory) -> Path:
    """The workspace as a repository with an author, its manifest committed and pushed to a bare
    `origin`, the remote that makes it the workspace's own."""
    remote = tmp_path_factory.mktemp("remotes") / "root.git"
    git(workspace, "init", "--quiet", "--bare", "--initial-branch=main", remote.as_posix())
    git(workspace, "init", "--quiet", "--initial-branch=main")
    git(workspace, "config", "user.name", "it")
    git(workspace, "config", "user.email", "it@example.invalid")
    git(workspace, "remote", "add", "origin", remote.as_posix())
    git(workspace, "add", "-A")
    git(workspace, "commit", "--quiet", "-m", "manifest")
    git(workspace, "push", "--quiet", "-u", "origin", "main")
    return workspace
