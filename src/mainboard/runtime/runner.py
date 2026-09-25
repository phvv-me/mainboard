# The one way a dispatched job runs, on whichever machine it landed on.
#
# A job used to be a rendered bash script: an exit trap framing receipts, a `timeout` re-exec for
# the walltime, a `trap 'exit 143' TERM`, `export` lines and a sourced activation. Each of those is
# a step here instead, in the order the script ran them, so PBS, pueue, a rented machine and a
# Windows box all run the same code, and what a job does no longer depends on which shell a
# scheduler happened to hand it to.
#
# The order is the contract. Build the environment the job was dispatched against, enter it,
# export what the dispatch decided, attest to the machine, start the sampler, point the command at
# its receipts file, and run it from the pinned tree under the walltime. Whatever happens after
# the environment is entered, the receipts the command wrote are framed back and the exit status
# is the command's own, a signal's `128 + N`, or the walltime's `124`/`137`.

import json
import os
import platform
import shlex
import shutil
import signal
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=runs the dispatched job and this tool's own verbs, argv built from the job record since=2026-09-25
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import gettempdir
from time import monotonic
from typing import TYPE_CHECKING, TextIO

from ..core.project import Project
from ..dispatch.evidence import RECEIPTS_VAR, framed
from .entry import Refusal, entering
from .job import walltime_seconds
from .tree import ProcessTree

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, MutableMapping
    from types import FrameType

    from .entry import Entering
    from .job import Job, ToolCall

# The signals a job is ended by, whichever of them this platform has. Receiving one ends the
# command's whole tree and the job with `128 + N`, the status `trap 'exit 143' TERM` gave it.
_ENDINGS = tuple(
    getattr(signal, name) for name in ("SIGTERM", "SIGINT", "SIGHUP") if hasattr(signal, name)
)

# How long a sampler is given to notice the job is over before it is stopped outright.
_SAMPLER_GRACE = 5.0


class Receipts:
    """The file a run writes its trial receipts to, framed back through the job's output.

    Named for this runner's process, since a cluster node runs several jobs out of one temporary
    directory and two sharing a receipts file would hand each other's trials to whichever
    settled first. See `dispatch.evidence` for why receipts are written to a file at all.
    """

    def __init__(self) -> None:
        self.path = Path(gettempdir()) / f"mainboard-receipts.{os.getpid()}.ndjson"
        self.staged = False

    def stage(self, environment: MutableMapping[str, str]) -> None:
        """Point the command at an empty receipts file, emptied for a restarted container."""
        self.path.write_bytes(b"")
        environment[RECEIPTS_VAR] = str(self.path)
        self.staged = True

    def frame(self) -> str:
        """The framed block of what the command wrote, empty when it wrote nothing."""
        if not self.staged:
            return ""
        try:
            written = self.path.read_bytes()
        except FileNotFoundError:
            return ""
        self.path.unlink()
        return framed(written) if written else ""


class Runner:
    """Run one `Job` to its exit status.

    `tool` is how the runner calls this very tool for the verbs around a command, its own
    interpreter rather than whichever `mainboard` a PATH names.

    job: what the dispatch decided.
    environ: the environment this runner was started with, the process's own by default.
    how: this machine's way of entering an environment, chosen by platform by default.
    grace: seconds a command ended at its walltime is given before it is killed outright.
    """

    tool: tuple[str, ...] = (sys.executable, "-m", Project().name)

    def __init__(
        self,
        job: Job,
        *,
        environ: Mapping[str, str] | None = None,
        how: Entering | None = None,
        grace: float = 30.0,
    ) -> None:
        self.job = job
        self.environ = {
            name: value
            for name, value in (os.environ if environ is None else environ).items()
            if name != RECEIPTS_VAR
        }
        self.how = how or entering()
        self.grace = grace
        self.receipts = Receipts()
        self.tree: ProcessTree | None = None
        self.signalled = 0
        self.log: TextIO | None = None
        self.deadline = monotonic() + walltime_seconds(job.walltime) if job.walltime else None

    def run(self) -> int:
        """Run the job, frame its receipts, record its exit, and answer its exit status."""
        with self.output(), self.endings():
            status = self.outcome()
            self.say(self.receipts.frame())
            if self.job.logs:
                self.say(f"exit={status}")
                self.exit_artifact().write_text(f"exit={status}\n", encoding="utf-8")
        return status

    def outcome(self) -> int:
        """Every step up to and including the command, answering the job's exit status."""
        self.provide()
        try:
            environment = self.environment()
        except Refusal as refused:
            self.say(refused.message, error=True)
            return refused.status
        self.call(self.job.attestation, environment)
        sampler = self.sample(environment)
        try:
            self.receipts.stage(environment)
            if self.signalled:
                return self.signalled
            status = self.command(environment)
            return self.signalled or status
        finally:
            if sampler is not None:
                ProcessTree(sampler, grace=_SAMPLER_GRACE).stop()

    def provide(self) -> None:
        """Build the environment this job was dispatched against, unless the host has it.

        Its failure is not the job's: entering the environment afterwards refuses in words that
        name it, which is the clearer of the two messages.
        """
        if self.job.provide is None:
            return
        if self.call(self.job.provide, self.environ, quiet=False):
            self.say("mainboard: could not build the environment this job was dispatched with")

    def environment(self) -> dict[str, str]:
        """The environment the command starts in: entered, then the dispatch's own variables."""
        entered = self.job.activation.entered(self.environ, cwd=self.job.root, how=self.how)
        if self.job.pythonpath:
            entered["PYTHONPATH"] = self.job.pythonpath
        elif self.job.isolate_pythonpath:
            entered.pop("PYTHONPATH", None)
        entered.update(self.job.variables)
        return entered

    def command(self, environment: dict[str, str]) -> int:
        """Run the command from the pinned tree under the walltime and answer its status."""
        self.flush()
        process = subprocess.Popen(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=the job's own command, run the way its dispatch spelled it since=2026-09-25
            self.argv(environment),
            cwd=self.job.root,
            env=environment,
            stdout=self.log,
            stderr=self.log,
        )
        self.tree = ProcessTree(process, grace=self.grace)
        remaining = None if self.deadline is None else max(self.deadline - monotonic(), 0.0)
        status = self.tree.wait(remaining)
        if status is not None:
            return status
        status = 137 if self.tree.stop() else 124
        self.say(f"mainboard: killed at walltime {self.job.walltime} (exit {status})")
        return status

    def argv(self, environment: Mapping[str, str]) -> list[str]:
        """The command as the process started for it: its container, `bash -c`, or its words.

        Windows has no shell to hand a line to, so the line is split into the argv it spells
        and its program is looked up on the environment's own `PATH`, which is what a shell
        would have searched.
        """
        if self.job.container:
            return list(self.job.container)
        if platform.system() != "Windows":
            return ["bash", "-c", self.job.command]
        words = shlex.split(self.job.command)
        program = shutil.which(words[0], path=environment.get("PATH")) or words[0]
        return [program, *words[1:]]

    def call(
        self, call: ToolCall | None, environment: Mapping[str, str], *, quiet: bool = True
    ) -> int:
        """Run one of this tool's own verbs to its exit status, 0 when there is none to run.

        The verb runs in this very tool, so a job is provisioned and watched by the tool that is
        running it. A quiet call's output
        is discarded, since it belongs in the job's receipts rather than its log; a loud one keeps
        its errors in the log.

        call: the verb and where it runs, None for nothing to run.
        environment: the environment it runs in, before its own credentials.
        quiet: discard its errors too, not only its output.
        """
        if call is None:
            return 0
        self.flush()
        return subprocess.call(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=this tool's own verb in its own interpreter since=2026-09-25
            [*self.tool, *call.args],
            cwd=call.cwd or self.job.root,
            env={**environment, **self.credentials(call)},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL if quiet else self.log,
        )

    def sample(self, environment: Mapping[str, str]) -> subprocess.Popen[bytes] | None:
        """Start the sampler beside the command, following this runner so it cannot outlive it."""
        call = self.job.sampler
        if call is None:
            return None
        return subprocess.Popen(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=this tool's own verb in its own interpreter since=2026-09-25
            [*self.tool, *call.args, "--parent", str(os.getpid())],
            cwd=call.cwd or self.job.root,
            env={**environment, **self.credentials(call)},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    @staticmethod
    def credentials(call: ToolCall) -> dict[str, str]:
        """The variables `call` alone receives, nothing when it names no file or a missing one."""
        if not call.credentials:
            return {}
        try:
            staged: dict[str, str] = json.loads(Path(call.credentials).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        return staged

    def exit_artifact(self) -> Path:
        """Where a PBS job's status is kept for a server that has since forgotten the job."""
        return Path(self.job.logs) / f"{self.job_id}.exit"

    @property
    def job_id(self) -> str:
        """The PBS job's bare number, the stem its log and exit artifact are named by."""
        return self.environ["PBS_JOBID"].split(".", maxsplit=1)[0]

    def say(self, text: str, *, error: bool = False) -> None:
        """Write one line of the runner's own into the job's output, nothing for empty text."""
        if not text:
            return
        stream = self.log or (sys.stderr if error else sys.stdout)
        stream.write(text if text.endswith("\n") else f"{text}\n")
        stream.flush()

    def flush(self) -> None:
        """Flush this runner's own streams, so a child's output lands after what came before."""
        for stream in (sys.stdout, sys.stderr):
            stream.flush()

    @contextmanager
    def output(self) -> Generator[None]:
        """Send a PBS job's merged output to the log a later poll reads, and nothing otherwise.

        A PBS server spools a job's output where no poll looks, and purges the job from its own
        records soon after it ends, so the job appends to its own log under the dispatch state
        directory and writes its exit status beside it.
        """
        if not self.job.logs:
            yield
            return
        Path(self.job.logs).mkdir(parents=True, exist_ok=True)
        with (Path(self.job.logs) / f"{self.job_id}.log").open("a", encoding="utf-8") as log:
            self.log = log
            try:
                yield
            finally:
                self.log = None

    @contextmanager
    def endings(self) -> Generator[None]:
        """Hold every ending signal for the run, ending the command's tree when one arrives."""
        previous = {number: signal.signal(number, self.ended) for number in _ENDINGS}
        try:
            yield
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)

    def ended(self, number: int, frame: FrameType | None) -> None:
        """Remember the signal the job is ending on and pass it to the command's tree."""
        del frame
        self.signalled = 128 + number
        if self.tree is not None:
            self.tree.terminate()
