import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path

import psutil
import pytest

import mainboard


def _pushed_elsewhere(repo: Path, *, clone: Path) -> None:
    """A new `clone` of `repo`'s origin pushes `shared.txt` holding `theirs`; `repo` fetches it."""
    remote = git(repo, "remote", "get-url", "origin").stdout.strip()
    git(repo, "clone", "--quiet", remote, clone.as_posix())
    (clone / "shared.txt").write_text("theirs\n", encoding="utf-8", newline="\n")
    git(clone, "add", "-A")
    git(clone, "commit", "--quiet", "-m", "theirs")
    git(clone, "push", "--quiet")
    git(repo, "fetch", "--quiet")


def committed(where: Path) -> Path:
    """A repository at `where` holding one commit."""
    where.mkdir(parents=True)
    git(where, "init", "--quiet", "--initial-branch=main")
    (where / "README").write_text("one\n", encoding="utf-8", newline="\n")
    git(where, "add", "README")
    git(where, "commit", "--quiet", "-m", "one")
    return where


def committed_links(repo: Path, *names: str) -> None:
    """Commit `names` in `repo` as symbolic links to a `target.txt` holding `kept`.

    Skips the test on a machine that cannot make symbolic links.
    """
    (repo / "target.txt").write_text("kept\n", encoding="utf-8", newline="\n")
    for name in names:
        try:
            (repo / name).symlink_to("target.txt")
        except OSError:
            pytest.skip("this machine cannot make symbolic links")
    git(repo, "add", "-A")
    git(repo, "commit", "--quiet", "-m", "links")


def diverged(repo: Path, name: str, *, clone: Path, commit_edit: bool) -> None:
    """Leave `repo` behind its origin, which another `clone` moved, with its own edit of `name`.

    commit_edit: commit the edit, else leave it in the working tree.
    """
    _pushed_elsewhere(repo, clone=clone)
    (repo / name).write_text("mine\n", encoding="utf-8", newline="\n")
    if commit_edit:
        git(repo, "add", "-A")
        git(repo, "commit", "--quiet", "-m", "mine")


def git(where: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git in `where`, local-path submodules allowed and an author named."""
    settings = ["protocol.file.allow=always", "user.name=it", "user.email=it@example.invalid"]
    flags = [word for setting in settings for word in ("-c", setting)]
    return subprocess.run(
        ["git", "-C", str(where), *flags, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def nesting_library(remotes: Path) -> Path:
    """A repository under `remotes` whose one commit adds a submodule at `deps/nested`."""
    nested = committed(remotes / "other" / "nested")
    library = committed(remotes / "other" / "library")
    git(library, "submodule", "add", "--quiet", nested.as_posix(), "deps/nested")
    git(library, "commit", "--quiet", "-m", "nest")
    return library


def replaced_by_files(repo: Path, contents: Mapping[str, str]) -> None:
    """Replace each link of `repo` that `contents` names with a plain file holding its text."""
    for name, text in contents.items():
        (repo / name).unlink()
        (repo / name).write_text(text, encoding="utf-8", newline="")


def run_with_libraries(where: Path, libraries: Mapping[str, str], script: str) -> str:
    """What `script` prints in a fresh interpreter that imports the stand-in `libraries` first.

    libraries: each library file's path relative to a folder under `where`, to its source. A test
        process cannot host them, since an optional-library flag or registry would keep what a
        stand-in let it build.
    """
    root = where / "libraries"
    for relative, source in libraries.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(source, encoding="utf-8")
    path = os.pathsep.join([str(root), str(Path(mainboard.__file__).parents[1])])
    return subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": path},
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    ).stdout


def survivors(
    members: Sequence[psutil.Process | int], *, within: float = 5.0
) -> list[psutil.Process]:
    """The `members`, processes or pids, still running `within` seconds on.

    A process that ended and was not collected yet does not count.
    """
    known: list[psutil.Process] = []
    for member in members:
        with suppress(psutil.NoSuchProcess):
            known.append(member if isinstance(member, psutil.Process) else psutil.Process(member))
    _, alive = psutil.wait_procs(known, timeout=within)
    found: list[psutil.Process] = []
    for member in alive:
        with suppress(psutil.NoSuchProcess):
            if member.status() != psutil.STATUS_ZOMBIE:
                found.append(member)
    return found


def unreadable(workspace: Path) -> None:
    """Point the vendored nested checkout's `.git` at a directory that is gone.

    A bare `git status` then aborts in `workspace` and in every repository above the checkout.
    """
    pointer = workspace / "vendor" / "library" / "deps" / "nested" / ".git"
    pointer.unlink()
    pointer.write_text("gitdir: ../../../../.git/modules/gone\n", encoding="utf-8", newline="\n")
    assert git(workspace, "status", "--porcelain").returncode != 0


def vendored(workspace: Path, *, library: Path, origin: Path) -> None:
    """Make `workspace` a repository with an author and an `origin` that vendors `library`.

    The library sits at `vendor/library` with its own submodules checked out, all committed.
    """
    git(workspace, "init", "--quiet", "--initial-branch=main")
    git(workspace, "config", "user.name", "it")
    git(workspace, "config", "user.email", "it@example.invalid")
    git(workspace, "remote", "add", "origin", origin.as_posix())
    git(workspace, "submodule", "add", "--quiet", library.as_posix(), "vendor/library")
    git(workspace / "vendor" / "library", "submodule", "update", "--init", "--quiet")
    git(workspace, "add", "-A")
    git(workspace, "commit", "--quiet", "-m", "vendor")
    assert git(workspace, "status", "--porcelain").returncode == 0
