# The value objects a durable monitor sweep builds: one pass resolves every tracked job on every
# host (a dead host never crashes it) and harvests those newly terminal since the last sweep.
# `mainboard.monitor.Monitor` runs the pass and `mb job list` prints it.
#
# It also holds M34's bridge, which M24 deletes together with the file lock for a PostgreSQL
# advisory lock: the holder of the settlement lock writes who it is beside the lock file, so a
# listing can name it and say that what it shows is cached.

import os
import socket
from contextlib import suppress
from datetime import datetime
from pathlib import Path

from filelock import FileLock
from patos import FrozenModel
from pydantic import Field

from ..shared import now, since

# Written beside `settlement.lock`: its holder's record, and when a pass last ran to its end.
_OWNER = "settlement.owner.json"
_SUCCESS = "settlement.success"


class SettlementOwner(FrozenModel):
    """The process holding the settlement lock, as it last wrote itself down.

    Diagnostic only: the lock decides who owns settlement, never this record or its age.

    phase: what the holder is doing now.
    heartbeat: when the phase last changed, which says nothing of a worker blocked inside it.
    deadline: when a settling pass stops itself, empty for any other holder.
    """

    pid: int
    host: str
    started_at: str
    phase: str
    heartbeat: str
    deadline: str = ""


class SettlementBusy(FrozenModel):
    """A held settlement lock as an observer finds it; every state read meanwhile is cached.

    owner: the holder, None when it wrote no record (an older release, or a holder between
        taking the lock and writing itself down).
    age_s: how long the owner has held the lock.
    last_success: when a pass last ran to its end, empty when none ever did.
    """

    owner: SettlementOwner | None = None
    observed_at: str
    age_s: float | None = None
    last_success: str = ""

    @classmethod
    def read(cls, lock_file: Path) -> SettlementBusy:
        """The holder of `lock_file` as its record describes it, which may be missing or stale."""
        observed = now()
        owner: SettlementOwner | None = None
        age: float | None = None
        with suppress(OSError, ValueError):
            owner = SettlementOwner.model_validate_json(
                lock_file.with_name(_OWNER).read_text(encoding="utf-8")
            )
            age = (
                datetime.fromisoformat(observed) - datetime.fromisoformat(owner.started_at)
            ).total_seconds()
        return cls(
            owner=owner,
            observed_at=observed,
            age_s=age,
            last_success=last_settlement(lock_file.parent),
        )

    @property
    def note(self) -> str:
        """The line a listing or a wait prints, so cached states never read as current."""
        held = (
            f"pid {self.owner.pid} on {self.owner.host}, started {self.owner.started_at} "
            f"({since(self.owner.started_at)} ago), phase {self.owner.phase}, "
            f"heartbeat {self.owner.heartbeat}, deadline {self.owner.deadline or 'none'}"
            if self.owner
            else "holder unknown (an older release, or a pass still starting)"
        )
        return (
            f"settlement busy: {held}; states are cached; "
            f"last successful pass {self.last_success or 'unknown'}"
        )


class SettlementLock(FileLock):
    """The settlement lock made visible: its holder records who it is beside the lock file.

    The record is replaced atomically while the lock is held and removed on release. It only
    describes a holder, and never decides, steals or times out the lock.
    """

    owner: SettlementOwner

    def _acquire(self) -> None:
        super()._acquire()
        if self.is_locked:
            stamp = now()
            self.owner = SettlementOwner(
                pid=os.getpid(),
                host=socket.gethostname(),
                started_at=stamp,
                phase="launching, cancelling or settling",
                heartbeat=stamp,
            )
            self.phase(self.owner.phase)

    def _release(self) -> None:
        try:
            Path(self.lock_file).with_name(_OWNER).unlink(missing_ok=True)
        finally:
            super()._release()

    def phase(self, phase: str, *, deadline: str = "") -> None:
        """Record what the holder is doing now.

        deadline: when this pass stops itself, kept from an earlier call when empty.
        """
        self.owner = self.owner.model_copy(
            update={
                "phase": phase,
                "heartbeat": now(),
                "deadline": deadline or self.owner.deadline,
            }
        )
        _replace(Path(self.lock_file).with_name(_OWNER), self.owner.model_dump_json())

    def succeeded(self) -> None:
        """Stamp that a pass ran to its end, whatever it found."""
        _replace(Path(self.lock_file).with_name(_SUCCESS), now())


def last_settlement(run: Path) -> str:
    """When a settling pass last ran to its end in the run directory `run`, empty if none did.

    The service manager's own last run says only that a pass was started.
    """
    try:
        return (run / _SUCCESS).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _replace(path: Path, text: str) -> None:
    """Write `text` so a reader finds the old file or the new one, never half of it."""
    staged = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    staged.write_text(text, encoding="utf-8")
    staged.replace(path)


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
        owes them an outcome; zero when another process held settlement and nothing was inspected.
    busy: that holder, when there was one; the states read meanwhile are cached.
    failed: jobs newly `failed`/`vanished` this sweep.
    """

    running: int = 0
    busy: SettlementBusy | None = None
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
