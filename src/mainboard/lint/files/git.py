from pathlib import Path
from subprocess import CompletedProcess

from ...core.project import Project
from ...git.process import Git


def git(repository: Path, *arguments: str, stdin: str = "") -> CompletedProcess[str]:
    """One git command run in `repository`, a path's bytes kept whole so no locale reads it.

    repository: any directory inside the work tree the command addresses.
    stdin: what the command reads, for the `--stdin` queries.
    """
    return Git(repository).run(*arguments, stdin=stdin, errors="surrogateescape")


def topmost(directory: Path) -> Path | None:
    """The top working tree of the superprojects above `directory`'s repository.

    It is None outside git. It is the one workspace a hook in any nested repository answers to,
    even when a submodule declares a manifest of its own.
    """
    top = git(directory, "rev-parse", "--show-toplevel")
    if top.returncode:
        return None
    found = Path(top.stdout.strip())
    while True:
        above = git(found, "rev-parse", "--show-superproject-working-tree")
        if not (parent := above.stdout.strip()):
            return found
        found = Path(parent)


def nested_projects(root: Path) -> list[Path]:
    """The projects nested in the workspace at `root`, each a directory with its own manifest.

    These are what a monorepo keeps beside its members, each a workspace when cloned alone.

    A tracked manifest counts (submodules' included; one deleted from the work tree is gone), and
    so does a repository nested untracked and unignored (`apps/usagiunagi`, kept as its own
    repository), with its own nested projects.
    """
    project = Project()
    tracked = git(
        root,
        "ls-files",
        "-z",
        "--cached",
        "--recurse-submodules",
        "--",
        *(f":(glob)**/{manifest}" for manifest in project.manifests),
    )
    found = {
        (root / name).parent
        for name in tracked.stdout.split("\0")
        if name and (root / name).is_file()
    }
    # Untracked files one by one, except a nested repository, which git names as `path/`.
    loose = git(root, "ls-files", "-z", "--others", "--exclude-standard")
    for name in loose.stdout.split("\0"):
        home = root / name
        if name.endswith("/") and (home / ".git").exists() and project.manifest(home).is_file():
            found |= {home, *nested_projects(home)}
    return sorted(found - {root})
