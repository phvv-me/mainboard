# `mainboard host serve`: the node side of a kept line, run by the keeper inside each allocation.
#
# It claims the oldest queued job that fits the time left, runs it through the job script's own
# runner (which writes the log and the `.exit` artifact under the job's handle, as for any PBS
# job), and ends before its allocation does, so a job is never killed by the walltime it was
# promised. Nothing here talks to a scheduler: the line, the queue and the results are files.

import os
import signal
import socket
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=runs the job script this dispatch staged, not untrusted input since=2026-10-09
import time
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

from .core.errors import MissionError
from .dispatch.spool import Beat, Claim, Spool, Submission
from .log import logger
from .runtime.job import walltime_seconds

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

# Seconds between looks at the queue, and between beats while nothing changes.
_TICK = 1.0
_BEAT = 5.0
# How long a job gets to end after its SIGTERM before it is killed outright.
_KILL_AFTER = 60.0
# What a terminated runner records, `128 + SIGTERM`.
_TERMINATED = 143


class Server:
    """One allocation's worth of a line: claim, run, repeat, and end in time.

    spool: the line's spool, on the shared filesystem.
    gen: the allocation generation the keeper opened this for.
    walltime: how long the allocation lasts, `HH:MM:SS`, counted from now.
    """

    def __init__(
        self,
        spool: Spool,
        *,
        gen: int,
        walltime: str,
        environ: Mapping[str, str] = os.environ,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.spool, self.gen = spool, gen
        self.environ, self.clock, self.sleep = environ, clock, sleep
        self.line = spool.line()
        self.ends = clock() + walltime_seconds(walltime)
        self.directory = spool.path / "gen" / str(gen)
        self.running = ""
        self.beaten = 0.0

    def run(self) -> int:
        """Serve until the line is released, its deadline or this allocation's end is near."""
        node = self.verified()
        self.spool.replace(self.directory / "alloc", self.environ["PBS_JOBID"])
        self.spool.replace(self.directory / "node", node)
        logger.info("serving generation {} on {}", self.gen, node)
        with ExitStack() as ending:
            ending.callback(self.end)
            while not self.finished():
                if (entry := self.next()) is not None:
                    self.work(entry, node)
                else:
                    self.beat("ready")
                    self.sleep(_TICK)
        return 0

    def end(self) -> None:
        """Say this allocation is over, however it came to be."""
        self.running = ""
        self.beat("ended", force=True)
        self.spool.replace(self.directory / "ended", str(self.clock()))

    def verified(self) -> str:
        """This node's name, refused unless a scheduler allocated it.

        Nothing a user queued may run on the login node.
        """
        nodefile = Path(self.environ.get("PBS_NODEFILE", ""))
        node = socket.gethostname().split(".")[0]
        if not (self.environ.get("PBS_JOBID") and nodefile.is_file()):
            raise MissionError("host serve runs inside a PBS allocation; this is not one")
        allocated = nodefile.read_text(encoding="utf-8").split()
        if node not in {name.split(".")[0] for name in allocated}:
            raise MissionError(f"{node} is not in this allocation's node file")
        return node

    @property
    def usable(self) -> float:
        """Seconds left before the allocation or the line's deadline, whichever comes first."""
        return min(self.ends, self.line.deadline) - self.clock()

    def finished(self) -> bool:
        """Whether nothing more may start: released, or no job could fit with its grace."""
        return self.spool.stopped() or self.usable <= self.line.grace

    def next(self) -> Submission | None:
        """The oldest queued job that fits the time left."""
        for entry in self.spool.queued():
            if self.spool.fits(entry, remaining=self.usable):
                return entry
        return None

    def work(self, entry: Submission, node: str) -> None:
        """Claim `entry` and run it to an exit artifact, ending it on its tombstone."""
        claim = Claim(
            gen=self.gen,
            alloc=self.environ["PBS_JOBID"],
            node=node,
            pid=os.getpid(),
            at=self.clock(),
        )
        if not self.spool.claim(entry, claim):
            return
        self.running = entry.handle
        self.beat("running", force=True)
        cancelled = self.spool.tombstoned(entry.handle)
        self.record(entry, _TERMINATED if cancelled else self.execute(entry))
        self.running = ""

    def execute(self, entry: Submission) -> int:
        """Run the job script as PBS would, answering the process's own status.

        Its handle is `PBS_JOBID`, so the runner writes `logs/<handle>.log` and `.exit`.
        """
        self.spool.logs.mkdir(parents=True, exist_ok=True)
        with (self.spool.logs / f"{entry.handle}.log").open("ab") as log:
            process = subprocess.Popen(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=the job script this dispatch staged since=2026-10-09
                ["sh", entry.script],
                cwd=entry.cwd,
                env={**self.environ, "PBS_JOBID": entry.handle},
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            terminated = 0.0
            while process.poll() is None:
                self.beat("running")
                if not terminated and (self.spool.tombstoned(entry.handle) or self.usable <= 0):
                    terminated = self.clock()
                    os.killpg(process.pid, signal.SIGTERM)
                elif terminated and self.clock() - terminated > _KILL_AFTER:
                    os.killpg(process.pid, signal.SIGKILL)
                self.sleep(_TICK)
        return process.returncode

    def record(self, entry: Submission, status: int) -> None:
        """Leave an exit artifact when the runner left none.

        That is a script that could not start, or a claim cancelled before it ran.
        """
        if self.spool.exit_of(entry.handle) is None:
            self.spool.replace(self.spool.logs / f"{entry.handle}.exit", f"exit={status}\n")

    def beat(self, state: str, *, force: bool = False) -> None:
        """Say what this allocation is doing, at most every few seconds unless it changed."""
        now = self.clock()
        if force or now - self.beaten >= _BEAT:
            self.beaten = now
            beat = Beat(at=now, state=state, running=self.running)
            self.spool.replace(self.directory / "beat", beat.model_dump_json())
