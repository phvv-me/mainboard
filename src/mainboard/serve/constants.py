"""Every child `mb serve` declares, running or not, and why each one that does not run is off.

A task that builds a child fills in its command and clears its `off`, and nothing else changes:
the YAML, the unit, `serve status` and the doctor all read this tuple. An off child is left out
of the generated YAML altogether, so nothing can start it through the supervisor's socket.
"""

from itertools import chain
from typing import Self

from patos import FrozenModel
from pydantic import model_validator

from ..core.constants.names import CODEX_FLAGS
from ..core.project import Project
from .enums import Kind

# The program a command names to mean this tool itself, resolved to the installed one.
TOOL = Project().package

# Safety rule 1 of ROADMAP-v2 keeps both remote-control children off until N1 is applied,
# v2-policy has landed and v2-review has passed before and after it. The owner set N1 aside.
_REMOTE = (
    "remote control stays off: N1 is not applied (set aside by the owner, 2026-10-28), and "
    "safety rule 1 also waits for v2-policy and two v2-review passes"
)

# The subcommand both remote-control children run.
_REMOTE_CONTROL = "remote-control"


class Service(FrozenModel):
    """One child of the supervisor: a resident process, or a pass run on a period.

    command: what it runs, its program first; the tool's own name (`TOOL`) means the installed
        tool. Empty while no task has built it.
    environment: what it needs beyond the allowlist every child gets, never a secret; a child
        that needs one is given the path of the 0600 file holding it.
    every: how often a pass runs, a Go duration (`60s`, `20m`); empty for a resident child.
    name: its name in the YAML, its log's file name and its row in `serve status`.
    off: why it does not run; empty when it does.
    stop: seconds it has to stop on SIGTERM before the supervisor kills it.
    timeout: seconds one pass may run before it is killed, since the supervisor bounds none and
        a pass that hangs would skip every later one.
    """

    command: tuple[str, ...] = ()
    environment: tuple[tuple[str, str], ...] = ()
    every: str = ""
    name: str
    off: str = ""
    stop: int = 5
    timeout: int = 0

    @model_validator(mode="after")
    def _runnable(self) -> Self:
        """A child that runs has a command, and a pass that runs has a bound."""
        if self.off:
            return self
        if not self.command:
            raise ValueError(
                f"{self.name} runs but has no command; give it one, or say why it is off"
            )
        if self.every and self.timeout <= 0:
            raise ValueError(f"the {self.name} pass runs every {self.every} with no timeout")
        return self

    @property
    def kind(self) -> Kind:
        return Kind.PERIODIC if self.every else Kind.RESIDENT


SERVICES = (
    # The telemetry sink (`serve/events/sink.py`), given time to flush what it holds.
    Service(name="sink", command=(TOOL, "serve", "process", "sink"), stop=10),
    Service(name="claude", command=("claude", _REMOTE_CONTROL), off=_REMOTE),
    Service(
        name="codex",
        command=("codex", _REMOTE_CONTROL, *chain.from_iterable(CODEX_FLAGS)),
        off=_REMOTE,
    ),
    Service(name="save", every="60s", timeout=50, off="v2-jj builds the save pass"),
    Service(
        name="collect", command=(TOOL, "serve", "process", "collect"), every="60s", timeout=50
    ),
    Service(
        name="settle",
        command=(TOOL, "job", "list", "--json"),
        every="20m",
        timeout=900,
        off=(
            "today's settler unit settles until CVPR (Nov 16, ROADMAP-v2); the pass moves here "
            "when that unit retires"
        ),
    ),
    # The collect pass's backstop (`serve/events/drain.py`), ending clean while the sink is down.
    Service(name="drain", command=(TOOL, "serve", "process", "drain"), every="5m", timeout=240),
    Service(
        name="reap",
        command=(TOOL, "serve", "process", "reap"),
        every="10m",
        timeout=300,
        off="waits for the canaries",
    ),
    Service(
        name="tidy",
        every="24h",
        timeout=1800,
        off=(
            "no task owns the tidy pass yet: flushing and merging the lakes, store gc, the "
            "30-day expiry and rotating pc.log"
        ),
    ),
)
