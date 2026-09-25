# The value objects a durable monitor sweep builds: one pass resolves every tracked job on every
# host (a dead host never crashes it) and harvests those newly terminal since the last sweep.
# `mainboard.monitor.Monitor` runs the pass and `mainboard monitor` prints it.

from patos import FrozenModel
from pydantic import Field


class Finished(FrozenModel):
    """A job that reached `ok` since the last sweep.

    target: the host alias it ran on.
    pulled_path: where its results were pulled locally, None with no fetch path or a failed pull.
    """

    handle: str
    target: str
    pulled_path: str | None = None


class Failed(FrozenModel):
    """A computational or settlement failure, with the cause.

    A failed run still carries its results path: 399 immutable receipt fragments out of 500
    planned trials are the ordinary end of a metered rental, and they come home whatever the
    exit code said.

    target: the host alias it ran on.
    reason: the execution failure or the pending transfer/release. A settlement failure neither
        changes the recorded computational verdict nor advances its reported cursor.
    pulled_path: where its partial results were pulled locally, None as on `Finished`.
    """

    handle: str
    target: str
    reason: str
    pulled_path: str | None = None


class Held(FrozenModel):
    """A dispatch a target's quota would not take yet, kept at this workstation until it will.

    A full queue has not rejected the job, so it is no verdict: every sweep asks again, and it
    counts as in flight meanwhile.

    handle: the local id the held request is recorded under, ours rather than a scheduler's.
    reason: what the target said when it refused.
    """

    handle: str
    target: str
    reason: str


class Resumed(FrozenModel):
    """A dispatch a quota had been holding that this sweep finally got through.

    handle: the handle the target gave it once there was room.
    name: the run's label, which says which job of which batch just went.
    """

    handle: str
    target: str
    name: str = ""


class DownHost(FrozenModel):
    """A host that could not be probed this sweep, so its jobs stay unresolved.

    reason: `daemon down` for a dead pueue, else the ssh fault text.
    """

    host: str
    reason: str


class MonitorReport(FrozenModel):
    """One durable sweep's outcome.

    running: tracked jobs still in flight, held dispatches included since this workspace still
        owes them an outcome; None when another monitor owns settlement and this pass did not
        inspect the fleet.
    failed: jobs newly `failed`/`vanished` this sweep.
    """

    running: int | None = 0
    resumed: list[Resumed] = Field(default_factory=list)
    held: list[Held] = Field(default_factory=list)
    finished: list[Finished] = Field(default_factory=list)
    failed: list[Failed] = Field(default_factory=list)
    unreachable_hosts: list[DownHost] = Field(default_factory=list)

    @property
    def changed(self) -> bool:
        """Whether this sweep harvested a newly terminal job, the flag a cron skips a tick on.

        A property, not a field, so a caller building a report payload folds it in explicitly.
        """
        return bool(self.finished or self.failed)
