# How a live dispatched job is doing right now: how far through its cells, how long since its
# output last grew, and how busy its host's cards are. `wait` blocked in silence and `jobs` said
# only `running`, so every agent wrote its own loop around `logs` and `nvidia-smi`; this is that
# loop, once. The cells come off the runner's beacon in the log (`jobs/beacon.py`). The silence
# is measured: each look remembers the output's length and when it last grew in a file beside
# the dispatch state, so a `jobs` after a `wait` knows what the wait saw, and a job is called
# quiet only once seen twice. Cards are read only where that is one more command over the
# connection already open for the log, a host whose scheduler runs the job there; a cluster's
# login node carries no card of the job's and a rented machine is not asked.

import json
import os
import shlex
from contextlib import suppress
from time import time
from typing import TYPE_CHECKING

from patos import FrozenModel
from plumbum import ProcessExecutionError

from .core.errors import MissionError
from .core.project import Project
from .dispatch.backends.base import route
from .dispatch.schedulers import HostUnreachable, registry
from .dispatch.wrapping import connection
from .jobs.beacon import Progress

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .board import Board
    from .dispatch.state import RunRecord
    from .dispatch.transport import Machine

# The family of schedulers reached over ssh, whose logs are read over one connection per host.
_SSH_FAMILY = "ssh-family"
# The scheduler kinds whose jobs run on the machine the connection lands on, so its cards are
# the job's cards. A PBS or Slurm login node runs nothing of the job's.
_ON_HOST = frozenset({"ssh", "local"})
# The one query that reads every card's busyness, one integer percentage per line.
_UTILIZATION = ("nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits")
# Where the looks remember each live job's output, beside the rest of the dispatch state.
_MEMORY = "pulse.json"
# Seconds a job stays remembered past its last growth, since another process may be watching
# a job this look was not asked about.
_FORGET = 86400.0


class Reading(FrozenModel):
    """What one look got out of a live job's host.

    output: the job's captured output so far, None when its host did not answer.
    gpu_pct: the busiest card on the job's host, None where that is not cheap or not there.
    """

    output: str | None = None
    gpu_pct: int | None = None


# Reads every given run's output and its host's cards in one look.
type Reader = Callable[[Sequence[RunRecord]], dict[RunRecord, Reading]]


class Pulse(FrozenModel):
    """One live job as a look found it.

    progress: what its beacon says so far.
    quiet_s: seconds since its output last grew, None until it has been seen twice.
    gpu_pct: the busiest card on its host, None where unknown.
    """

    handle: str
    target: str
    progress: Progress = Progress()
    quiet_s: int | None = None
    gpu_pct: int | None = None


class Pulses:
    """Looks at live jobs, remembering between looks when each one's output last grew."""

    def __init__(
        self, board: Board, *, read: Reader | None = None, clock: Callable[[], float] = time
    ) -> None:
        """read: the look at the hosts, a `Probe` of `board` when None.

        clock: wall-clock seconds, shared across processes since the memory outlives them.
        """
        self.read = read or Probe(board)
        self.clock = clock
        self.memory = board.root / Project().out_dir / _MEMORY

    def taken(self, records: Sequence[RunRecord]) -> dict[RunRecord, Pulse]:
        """Every running record's pulse, leaving out a run whose host did not answer and one that
        printed nothing yet, which may still be queued for all its log can say. A queued run is
        not a caller's to pass, since its silence is the queue's.
        """
        if not records:
            return {}
        readings = self.read(records)
        now = self.clock()
        held = self.recalled()
        seen = set(held)
        pulses: dict[RunRecord, Pulse] = {}
        for record, reading in readings.items():
            if not reading.output:
                continue
            key = f"{record.target}/{record.handle}"
            size, grew = held.get(key, (-1, now))
            if size != len(reading.output):
                size, grew = len(reading.output), now
            held[key] = (size, grew)
            pulses[record] = Pulse(
                handle=record.handle,
                target=record.target,
                progress=Progress.read(reading.output),
                quiet_s=int(now - grew) if key in seen else None,
                gpu_pct=reading.gpu_pct,
            )
        live = {f"{record.target}/{record.handle}" for record in records}
        self.remembered(
            {key: kept for key, kept in held.items() if key in live or now - kept[1] < _FORGET}
        )
        return pulses

    def recalled(self) -> dict[str, tuple[int, float]]:
        """Each remembered job's output length and when it last grew, empty when none is."""
        with suppress(OSError, ValueError, TypeError, AttributeError):
            held = json.loads(self.memory.read_text(encoding="utf-8"))
            return {key: (int(size), float(grew)) for key, (size, grew) in held.items()}
        return {}

    def remembered(self, held: dict[str, tuple[int, float]]) -> None:
        """Write the memory back whole; a look that cannot still answers, the next knows less."""
        with suppress(OSError):
            self.memory.parent.mkdir(parents=True, exist_ok=True)
            staged = self.memory.with_suffix(f".{os.getpid()}.tmp")
            staged.write_text(json.dumps(held), encoding="utf-8")
            staged.replace(self.memory)


class Probe:
    """The look itself: each ssh host once for all its runs' logs, each rental on its own."""

    def __init__(self, board: Board) -> None:
        self.board = board

    def __call__(self, records: Sequence[RunRecord]) -> dict[RunRecord, Reading]:
        """Every record's reading, a host that will not answer reading as silence."""
        readings: dict[RunRecord, Reading] = {}
        hosts: dict[str, list[RunRecord]] = {}
        for record in records:
            if route(record.kind) == _SSH_FAMILY:
                hosts.setdefault(record.target, []).append(record)
                continue
            output = self.board.job(record.handle, host=record.target).transcript()
            readings[record] = Reading(output=output)
        for target, owned in hosts.items():
            readings.update(self.hosted(target, owned))
        return readings

    def hosted(self, target: str, records: Sequence[RunRecord]) -> dict[RunRecord, Reading]:
        """One ssh host's runs, read over a single connection, its cards read on the way out."""
        try:
            root = self.board.on(target).remote_root()
            with connection(target) as remote:
                outputs = {record: logged(remote, root, record) for record in records}
                on_host = any(record.kind in _ON_HOST for record in records)
                busiest = utilization(remote) if on_host else None
        except HostUnreachable, MissionError, OSError, ProcessExecutionError:
            return {}
        return {
            record: Reading(output=output, gpu_pct=busiest) for record, output in outputs.items()
        }


def logged(remote: Machine, root: str, record: RunRecord) -> str:
    """`record`'s output over an open connection, empty when its scheduler has none.

    The scheduler is the one the run was dispatched under, whatever the host's profile says now.
    """
    scheduler = registry.SCHEDULERS.select(record.kind, default="ssh")
    try:
        return scheduler.logs(remote, root, handle=record.handle)
    except ProcessExecutionError:
        return ""


def utilization(remote: Machine) -> int | None:
    """The busiest card's utilization on `remote`, None where no card or no driver answers.

    Asked through a login shell like every other probe, so the driver's tool is found wherever
    that host's profile puts it.
    """
    status, said, _ = remote["bash"][["-lc", shlex.join(_UTILIZATION)]].run(retcode=None)
    readings = [int(word) for word in str(said).split() if word.isdigit()]
    return max(readings) if status == 0 and readings else None
