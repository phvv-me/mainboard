import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
