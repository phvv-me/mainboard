# What a rendered bash script once did (exit trap, `timeout` re-exec, `trap 'exit 143' TERM`,
# exports, sourced activation) is a step here, so every scheduler runs the same code.
#
# The order is the contract: build the environment, enter it, export what the dispatch decided,
# attest, start the sampler, point the command at its receipts file, and run it from the pinned
# tree under the walltime. Once entered, the receipts are always framed back and the exit status
# is the command's own, a signal's `128 + N`, or the walltime's `124`/`137`.

import os
import signal
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=runs the dispatched job and this tool's own verbs, argv built from the job record since=2026-09-25
import sys
from contextlib import contextmanager
from time import monotonic
from typing import TYPE_CHECKING

from ....constants import RECEIPTS_VAR
from ....core import ProcessTree, Project
from ....core.events.frames import JobFrames
from ....enums import Stop
from ....exceptions import Refusal
from ....runtime.job import walltime_seconds
from ....runtime.tree import FileBudget
from .output import JobOutput
from .receipts import Receipts

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from types import FrameType

    from ....runtime.job import Job, ToolCall

# The signals ending a job: the command's tree and the job end with `128 + N`, the status
# `trap 'exit 143' TERM` gave it.
_ENDINGS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)

# How long a sampler is given to notice the job is over before it is stopped outright.
_SAMPLER_GRACE = 5.0


class Runner:
    """Run one `Job` to its exit status.

    `tool` calls this very tool for the verbs around a command, in its own interpreter rather
    than whichever `mainboard` a PATH names.

    grace: seconds a command ended at its walltime is given before it is killed outright.
    """

    tool: tuple[str, ...] = (sys.executable, "-m", Project().package)

    @contextmanager
    def _endings(self) -> Generator[None]:
        """Hold every ending signal for the run, ending the command's tree when one arrives."""
        previous = {number: signal.signal(number, self._ended) for number in _ENDINGS}
        try:
            yield
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)

    @contextmanager
    def _sampling(self, environment: Mapping[str, str]) -> Generator[None]:
        """Run the sampler beside the body, stopping it however the body ends."""
        sampler = self._sample(environment)
        try:
            yield
        finally:
            if sampler is not None:
                ProcessTree(sampler, grace=_SAMPLER_GRACE).stop()

    def __init__(
        self, job: Job, *, environ: Mapping[str, str] | None = None, grace: float = 30.0
    ) -> None:
        self.job = job
        self.environ = {
            name: value
            for name, value in (os.environ if environ is None else environ).items()
            if name not in RECEIPTS_VAR.names
        }
        self.grace = grace
        self.output = JobOutput(job.logs, self.environ)
        self.receipts = Receipts()
        self.frames = JobFrames(job.frames, say=self.output.say)
        self.stop = Stop.exited
        self.tree: ProcessTree | None = None
        self.signalled = 0
        self.deadline = monotonic() + walltime_seconds(job.walltime) if job.walltime else None

    def _argv(self) -> list[str]:
        """The command's argv: its container, else `bash -c`."""
        return list(self.job.container) if self.job.container else ["bash", "-c", self.job.command]

    def _call(
        self, call: ToolCall | None, environment: Mapping[str, str], *, quiet: bool = True
    ) -> int:
        """Run one of this tool's own verbs to its exit status, 0 when there is none to run.

        Its output is discarded, since it belongs in the job's receipts rather than its log; a
        call that is not `quiet` keeps its errors in the log.
        """
        if call is None:
            return 0
        self.output.flush()
        with self._verb(call, environment, quiet=quiet) as process:
            return process.wait()

    def _command(self, environment: dict[str, str]) -> int:
        """Run the command from the pinned tree under the walltime and answer its status."""
        self.output.flush()
        process = subprocess.Popen(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=the job's own command, run the way its dispatch spelled it since=2026-09-25
            self._argv(), cwd=self.job.root, env=environment
        )
        self.tree = ProcessTree(process, grace=self.grace)
        remaining = None if self.deadline is None else max(self.deadline - monotonic(), 0.0)
        status = self.tree.wait(remaining)
        if status is not None:
            return status
        status = 137 if self.tree.stop() else 124
        self.stop = Stop.walltime
        self.output.say(f"mainboard: killed at walltime {self.job.walltime} (exit {status})")
        return status

    def _command_pid(self) -> int | None:
        """The command's process once it has started, what the frames' beats measure."""
        return self.tree.process.pid if self.tree else None

    def _ended(self, number: int, frame: FrameType | None) -> None:
        """Remember the signal the job is ending on and pass it to the command's tree."""
        del frame
        self.signalled = 128 + number
        self.stop = Stop.signaled
        if self.tree is not None:
            self.tree.terminate()

    def _environment(self) -> dict[str, str]:
        """The environment the command starts in: entered, then the dispatch's own variables."""
        entered = self.job.activation.entered(self.environ, cwd=self.job.root)
        if self.job.pythonpath:
            entered["PYTHONPATH"] = self.job.pythonpath
        else:
            entered.pop("PYTHONPATH", None)
        entered.update(self.job.variables)
        return entered

    def _outcome(self) -> int:
        """Every step up to and including the command, answering the job's exit status."""
        self._provide()
        try:
            environment = self._environment()
        except Refusal as refused:
            self.stop = Stop.refused
            self.output.say(refused.message, error=True)
            return refused.status
        self._call(self.job.attestation, environment)
        with self._sampling(environment):
            self.receipts.stage(environment)
            if self.signalled:
                return self.signalled
            status = self._command(environment)
            return self.signalled or status

    def _provide(self) -> None:
        """Build the environment this job was dispatched against, unless the host has it.

        Its failure is only said: entering afterwards refuses in clearer words.
        """
        if self.job.provide is None:
            return
        if self._call(self.job.provide, self.environ, quiet=False):
            self.output.say(
                "mainboard: could not build the environment this job was dispatched with"
            )

    def _sample(self, environment: Mapping[str, str]) -> subprocess.Popen[bytes] | None:
        """Start the sampler beside the command, following this runner so it cannot outlive it."""
        call = self.job.sampler
        return (
            None if call is None else self._verb(call, environment, "--parent", str(os.getpid()))
        )

    def _verb(
        self, call: ToolCall, environment: Mapping[str, str], *extra: str, quiet: bool = True
    ) -> subprocess.Popen[bytes]:
        return subprocess.Popen(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=this tool's own verb in its own interpreter since=2026-09-25
            [*self.tool, *call.args, *extra],
            cwd=call.cwd or self.job.root,
            env=dict(environment),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL if quiet else None,
        )

    def run(self) -> int:
        """Run the job, frame its receipts, record its exit, and answer its exit status."""
        with (
            self.output.redirected(),
            self._endings(),
            FileBudget.permitted(),
            self.frames.running(self._command_pid),
        ):
            status = self._outcome()
            self.frames.settled(status, self.stop)
            self.output.say(self.receipts.frame())
            self.output.record_exit(status)
        return status
