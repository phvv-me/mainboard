# The one-word verdict vocabulary every scheduler backend reports, as a typed `Lifecycle`.
# Routing every move through this table turns a stale re-read into a raised `IllegalTransition`
# instead of a silent lie.

from patos import Lifecycle

QUEUED = "queued"
RUNNING = "running"
OK = "ok"
FAILED = "failed"
VANISHED = "vanished"
UNKNOWN = "unknown"
TIMEOUT = "timeout"

# Declared edges: queued -> running/vanished, running -> one terminal. Every terminal maps to
# the empty set, so a further move (a stale `running` after `ok`) raises rather than mutates.
VERDICTS: dict[str, set[str]] = {
    QUEUED: {RUNNING, VANISHED},
    RUNNING: {OK, FAILED, VANISHED, TIMEOUT},
    OK: set(),
    FAILED: set(),
    VANISHED: set(),
    UNKNOWN: set(),
    TIMEOUT: set(),
}


# The verdicts no declared move can leave. A job that reached one is settled for good, so a
# durable sweep trusts it straight from the cache instead of asking a queue that may already
# have forgotten the job.
TERMINAL = frozenset(verdict for verdict, moves in VERDICTS.items() if not moves)


def tracker(initial: str = QUEUED) -> Lifecycle[str]:
    """A fresh `Lifecycle` over the verdict table, started at `initial`."""
    return Lifecycle(VERDICTS, initial)
