import sys
from typing import NamedTuple


class CommandResult(NamedTuple):
    """One completed backend command with output retained for failure reporting."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0

    def replay(self) -> None:
        """Write retained output to the caller's streams, flushed."""
        sys.stdout.write(self.stdout)
        sys.stdout.flush()
        sys.stderr.write(self.stderr)
        sys.stderr.flush()
