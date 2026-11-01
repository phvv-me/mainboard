"""What the output names: who reads it and how a scan ended."""

from collections.abc import Mapping
from enum import StrEnum, auto

from ..core.project import Project

# What the agent harnesses set in the processes they start, beside the converging `AI_AGENT`.
_VENDORS = (
    "CLAUDECODE",
    "CODEX_SANDBOX",
    "CODEX_CI",
    "CODEX_THREAD_ID",
    "OPENCODE",
    "GEMINI_CLI",
    "CURSOR_AGENT",
    "CLINE_ACTIVE",
)


class Ending(StrEnum):
    """How a scan ended: it read every row, it was stopped early, or it timed out."""

    COMPLETE = auto()
    STOPPED = auto()
    TIMED_OUT = auto()


class Audience(StrEnum):
    """Who reads what the tool prints."""

    AGENT = auto()
    HUMAN = auto()

    @classmethod
    def of(cls, environ: Mapping[str, str], *, tty: bool) -> Audience:
        """The audience, the first rule that matches deciding.

        1. `MB_AGENT`: `1` is an agent and `0` a person.
        2. A harness's marker (`marked`).
        3. Standard output that is not a terminal, which is a pipe or a file.
        4. Otherwise a person.

        environ: the process environment.
        tty: whether standard output is a terminal.
        """
        # Any other `MB_AGENT` value falls through to detection.
        if forced := {"1": cls.AGENT, "0": cls.HUMAN}.get(
            Project().variable("AGENT").read(environ)
        ):
            return forced
        return cls.AGENT if cls.marked(environ) or not tty else cls.HUMAN

    @staticmethod
    def marked(environ: Mapping[str, str]) -> bool:
        """Whether an agent harness started the process: a non-empty `AI_AGENT` or vendor name."""
        return any(environ.get(name) for name in ("AI_AGENT", *_VENDORS))
