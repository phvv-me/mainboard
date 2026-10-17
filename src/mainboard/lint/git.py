import subprocess
from pathlib import Path


def git(repository: Path, *arguments: str, stdin: str = "") -> subprocess.CompletedProcess[bytes]:
    """One git command run in `repository`, bytes in and out so no locale reads a path.

    repository: any directory inside the work tree the command addresses.
    stdin: what the command reads, for the `--stdin` queries.
    """
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        input=stdin.encode(),
        capture_output=True,
        check=False,
    )


def printed(output: bytes) -> str:
    """What git printed, as text: paths are UTF-8 on every platform, whatever the console says."""
    return output.decode("utf-8", errors="surrogateescape")


def topmost(directory: Path) -> Path | None:
    """The working tree at the top of the superprojects `directory`'s repository sits in, None
    outside git: the one workspace a hook in any nested repository answers to, even when a
    submodule declares a manifest of its own."""
    top = git(directory, "rev-parse", "--show-toplevel")
    if top.returncode:
        return None
    found = Path(printed(top.stdout).strip())
    while True:
        above = git(found, "rev-parse", "--show-superproject-working-tree")
        if not (parent := printed(above.stdout).strip()):
            return found
        found = Path(parent)
