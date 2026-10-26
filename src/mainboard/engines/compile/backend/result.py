from typing import NamedTuple

from ....log import logger


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
        logger.result(self.stdout, end="")
        if said := self.stderr.rstrip("\n"):
            logger.info(said)
