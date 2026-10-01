"""The tree verbs on a repository git itself cannot walk.

A nested submodule checkout whose `.git` names a directory that is gone aborts a bare `git
status` in every repository above it (research/llm's googletest, 2026-09-30). `mb git status` and
`mb git check` never recurse, so they survive it, and they are the ones that name the path.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


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
def broken(workspace: Path, tmp_path_factory) -> Path:
    """The workspace as a repository whose foreign submodule holds an unreadable nested one."""
    remotes = tmp_path_factory.mktemp("remotes")
    nested = committed(remotes / "other" / "nested")
    library = committed(remotes / "other" / "library")
    git(library, "submodule", "add", "--quiet", nested.as_posix(), "deps/nested")
    git(library, "commit", "--quiet", "-m", "nest")
    git(workspace, "init", "--quiet", "--initial-branch=main")
    git(workspace, "config", "user.name", "it")
    git(workspace, "config", "user.email", "it@example.invalid")
    git(workspace, "remote", "add", "origin", (remotes / "me" / "root.git").as_posix())
    git(workspace, "submodule", "add", "--quiet", library.as_posix(), "vendor/library")
    git(workspace / "vendor" / "library", "submodule", "update", "--init", "--quiet")
    git(workspace, "add", "-A")
    git(workspace, "commit", "--quiet", "-m", "vendor")
    assert git(workspace, "status", "--porcelain").returncode == 0
    pointer = workspace / "vendor" / "library" / "deps" / "nested" / ".git"
    # Replaced rather than rewritten: git hides the file on Windows, which refuses a write.
    pointer.unlink()
    pointer.write_text("gitdir: ../../../../.git/modules/gone\n", encoding="utf-8", newline="\n")
    assert git(workspace, "status", "--porcelain").returncode != 0
    return workspace


def test_status_survives_an_unreadable_checkout_and_names_it(mb, broken: Path) -> None:
    ran = mb("git", "status", "--json")
    assert ran.code == 0, ran.said
    (root,) = json.loads(ran.out)
    assert root["repo"] == "." and "deps/nested" in root["broken"]


def test_check_survives_an_unreadable_checkout_and_names_it(mb, broken: Path) -> None:
    ran = mb("git", "check", "--json", timeout=300)
    found = [row for row in json.loads(ran.out) if row["check"] == "submodule"]
    assert [row["repo"] for row in found] == ["."]
    detail = found[0]["detail"]
    assert "deps/nested" in detail and "git submodule update --init" in detail
