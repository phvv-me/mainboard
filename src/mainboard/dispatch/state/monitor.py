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
    """A computational or settlement failure, with the cause.

    A failed run still carries its results path, because the work it did before it died is
    what a partial sweep is worth: 399 immutable receipt fragments out of 500 planned trials
    are the ordinary end of a metered rental, and they come home whatever the exit code said.

    handle: the scheduler's job handle.
    target: the host alias it ran on.
    reason: the execution failure or the pending transfer/release. A settlement failure does
        not change the recorded computational verdict or advance its reported cursor.
    pulled_path: the local path whatever it managed to write was rsynced into, or None when the
        run had no fetch path or the pull failed.
    """

    handle: str
    target: str
    reason: str
    pulled_path: str | None = None


class Held(FrozenModel):
    """A dispatch a target's quota would not take yet, kept at this workstation until it will.

    A queue that is full has not rejected the job, so nothing about it is a verdict: the request
    is waiting, and every sweep asks again until there is room. It is counted as in flight for
    exactly that reason.

    handle: the local id the held request is recorded under, ours rather than a scheduler's.
    target: the host alias whose quota is full.
    reason: what that target said when it refused.
    """

    handle: str
    target: str
    reason: str


class Resumed(FrozenModel):
    """A dispatch a quota had been holding that this sweep finally got through.

    handle: the handle the target gave it once there was room.
    target: the host alias that took it.
    name: the run's label, which is what says which job of which batch just went.
    """

    handle: str
    target: str
    name: str = ""


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

    running: how many tracked jobs are still in flight, a dispatch a target's quota is holding
        included, since a held request is work this workspace still owes an outcome for.
        None when another monitor owns settlement and this pass did not inspect the fleet.
    resumed: dispatches a quota had been holding that this sweep got through, each with the
        handle the target finally gave it.
    held: dispatches still waiting on a quota after this sweep asked again, each with why.
    finished: jobs newly `ok` this sweep, each with its pulled results path.
    failed: jobs newly `failed`/`vanished` this sweep, each with a reason and whatever partial
        results still came home.
    unreachable_hosts: hosts that could not be probed, each with why.
    """

    running: int | None = 0
    resumed: list[Resumed] = Field(default_factory=list)
    held: list[Held] = Field(default_factory=list)
    finished: list[Finished] = Field(default_factory=list)
    failed: list[Failed] = Field(default_factory=list)
    unreachable_hosts: list[DownHost] = Field(default_factory=list)

    @property
    def changed(self) -> bool:
        """Whether this sweep harvested any newly terminal job."""
        return bool(self.finished or self.failed)
