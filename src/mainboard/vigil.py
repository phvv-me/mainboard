# What a `wait` says while it blocks, and the two things it decides on its own.
#
# A wait used to be silent for an hour, so whoever waited polled beside it. Now it says each
# test cell as it lands and one heartbeat per look on stderr, keeping stdout for the verdict.
#
# A job whose pytest session ended but whose process has not, past a grace period of silence, is
# settled on what the session said (a library job read `running` for half an hour after `1 known
# in 27s`, 2026-09-19). A job silent for the stall threshold while its host's cards sit idle is
# stalled, so the wait ends with a distinct exit status instead of burning its whole timeout. A
# card nobody can read cheaply does not veto the call: on a cluster the silence alone decides.

from math import inf
from time import monotonic
from typing import TYPE_CHECKING

from patos import FrozenModel

from .pulse import Pulse

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from .dispatch.state import RunRecord
    from .pulse import Pulses

# How often a wait looks at its jobs' output, far rarer than its scheduler polls since a look
# reads each whole log over the network.
LOOK_SECONDS = 30.0
# How long a running job may print nothing on an idle card before a wait calls it stalled.
STALL_SECONDS = 1200.0
# How long a process may outlive its finished pytest session before it settles on the session.
LINGER_SECONDS = 120.0
# The busiest a card may be and still count as idle.
IDLE_PCT = 5


class Linger(FrozenModel):
    """A job whose pytest session ended while its process lives on.

    session: the session's exit status, what the job settles on.
    """

    handle: str
    target: str
    session: int


class Look(FrozenModel):
    """What one look decided.

    lingering: the jobs to settle now on the sessions they already finished.
    stalled: why the wait should stop calling a job alive, empty while nothing stalled.
    """

    lingering: tuple[Linger, ...] = ()
    stalled: str = ""


class Vigil:
    """One wait's looks at the jobs it blocks on, and what it says about each."""

    def __init__(
        self,
        pulses: Pulses,
        *,
        stall: float = STALL_SECONDS,
        say: Callable[[str], None],
        clock: Callable[[], float] = monotonic,
    ) -> None:
        """stall: seconds of silence on an idle card that call a job stalled, 0 never.

        say: where each line goes, stderr at a command line.
        clock: monotonic seconds, spacing the looks.
        """
        self.pulses = pulses
        self.stall = stall
        self.say = say
        self.clock = clock
        self.said: dict[str, int] = {}
        self.last = -inf

    def look(self, records: Sequence[RunRecord]) -> Look:
        """Look at the live `records` if a look is due: say what landed, beat once, and decide."""
        if self.clock() - self.last < LOOK_SECONDS:
            return Look()
        self.last = self.clock()
        pulses = list(self.pulses.taken(records).values())
        if not pulses:
            return Look()
        for pulse in pulses:
            self.landed(pulse)
        self.say(heartbeat(pulses))
        lingering = tuple(
            Linger(handle=pulse.handle, target=pulse.target, session=session)
            for pulse in pulses
            if (session := pulse.progress.session) is not None
            and (pulse.quiet_s or 0) >= LINGER_SECONDS
        )
        stalled = [pulse for pulse in pulses if self.stuck(pulse)]
        return Look(lingering=lingering, stalled="; ".join(map(silence, stalled)))

    def landed(self, pulse: Pulse) -> None:
        """Say every cell of `pulse` that landed since the last look."""
        cells = pulse.progress.cells
        for cell, outcome in cells[self.said.get(pulse.handle, 0) :]:
            self.say(f"{pulse.handle} {outcome} {cell}")
        self.said[pulse.handle] = len(cells)

    def stuck(self, pulse: Pulse) -> bool:
        """Whether `pulse` has been silent past the threshold with no busy card to excuse it."""
        silent = pulse.quiet_s is not None and self.stall > 0 and pulse.quiet_s >= self.stall
        idle = pulse.gpu_pct is None or pulse.gpu_pct <= IDLE_PCT
        return silent and idle and pulse.progress.session is None


def silence(pulse: Pulse) -> str:
    """Why `pulse` reads as stalled, in one clause."""
    card = f" and its busiest card at {pulse.gpu_pct}%" if pulse.gpu_pct is not None else ""
    return f"{pulse.handle} on {pulse.target} printed nothing for {pulse.quiet_s}s{card}"


def heartbeat(pulses: Iterable[Pulse]) -> str:
    """One line over at least one job's pulse: cells, failures, freshest output, busiest card."""
    seen = list(pulses)
    totals = [pulse.progress.total for pulse in seen]
    total = "?" if None in totals else str(sum(total or 0 for total in totals))
    done = sum(pulse.progress.done for pulse in seen)
    failed = sum(pulse.progress.failed for pulse in seen)
    quiet = [pulse.quiet_s for pulse in seen if pulse.quiet_s is not None]
    cards = [pulse.gpu_pct for pulse in seen if pulse.gpu_pct is not None]
    parts = [f"{len(seen)} running", f"{done}/{total} cells", f"{failed} failed"]
    if quiet:
        parts.append(f"output {min(quiet)}s ago")
    if cards:
        parts.append(f"gpu {max(cards)}%")
    return "heartbeat: " + ", ".join(parts)
