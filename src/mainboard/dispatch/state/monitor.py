# The durable single-pass monitor's report types. One sweep resolves every tracked job across
# all hosts once (robustly, so a dead host never crashes it), classifies each by its verdict, and
# harvests the ones newly terminal since the last sweep. This module holds the value objects such
# a sweep builds; `mainboard.monitor.Monitor` runs the pass and `mainboard monitor` prints it.

from patos import FrozenModel
from pydantic import Field


class Finished(FrozenModel):
    """A job that reached `ok` since the last sweep, with where its results were pulled.

    handle: the scheduler's job handle.
    target: the host alias it ran on.
    pulled_path: the local path its recorded results were rsynced into, or None when the run
        had no fetch path or the pull failed.
    """

    handle: str
    target: str
    pulled_path: str | None = None


class Failed(FrozenModel):
    """A job that ended badly (`failed` or `vanished`) since the last sweep, with the cause.

    A failed run still carries its results path, because the work it did before it died is
    what a partial sweep is worth: 399 immutable receipt fragments out of 500 planned trials
    are the ordinary end of a metered rental, and they come home whatever the exit code said.

    handle: the scheduler's job handle.
    target: the host alias it ran on.
    reason: a short, network-free cause (a signal exit, a plain non-zero code, or that it is
        gone).
    pulled_path: the local path whatever it managed to write was rsynced into, or None when the
        run had no fetch path or the pull failed.
    """

    handle: str
    target: str
    reason: str
    pulled_path: str | None = None


class DownHost(FrozenModel):
    """A host that could not be probed this sweep, so its jobs stay unresolved.

    host: the host alias.
    reason: why it could not be reached (`daemon down` for a dead pueue, else ssh fault text).
    """

    host: str
    reason: str


class MonitorReport(FrozenModel):
    """One durable sweep's outcome.

    `changed` is a plain property, not a model field, so a caller building a report payload
    folds it in explicitly; it is true exactly when this sweep harvested a job newly terminal
    since the last one, the cheap flag a cron branches on to skip a no-op tick.

    running: how many tracked jobs are still in flight.
    finished: jobs newly `ok` this sweep, each with its pulled results path.
    failed: jobs newly `failed`/`vanished` this sweep, each with a reason and whatever partial
        results still came home.
    unreachable_hosts: hosts that could not be probed, each with why.
    """

    running: int = 0
    finished: list[Finished] = Field(default_factory=list)
    failed: list[Failed] = Field(default_factory=list)
    unreachable_hosts: list[DownHost] = Field(default_factory=list)

    @property
    def changed(self) -> bool:
        """Whether this sweep harvested any newly terminal job."""
        return bool(self.finished or self.failed)
