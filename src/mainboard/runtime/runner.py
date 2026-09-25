# The one way a dispatched job runs, on whichever machine it landed on.
#
# What a rendered bash script once did (exit trap, `timeout` re-exec, `trap 'exit 143' TERM`,
# exports, sourced activation) is a step here, so every scheduler and Windows run the same code.
#
# The order is the contract: build the environment, enter it, export what the dispatch decided,
# attest, start the sampler, point the command at its receipts file, and run it from the pinned
# tree under the walltime. Once entered, the receipts are always framed back and the exit status
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
from typing import TYPE_CHECKING

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

# The signals ending a job, whichever this platform has: the command's tree and the job end with
# `128 + N`, the status `trap 'exit 143' TERM` gave it.
_ENDINGS = tuple(
    getattr(signal, name) for name in ("SIGTERM", "SIGINT", "SIGHUP") if hasattr(signal, name)
)

# How long a sampler is given to notice the job is over before it is stopped outright.
_SAMPLER_GRACE = 5.0


class Receipts:
    """The file a run writes its trial receipts to, framed back through the job's output.

    Named for this runner's process, since jobs sharing a node's temporary directory would
    otherwise hand each other's trials over. `dispatch.evidence` says why receipts are a file.
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

    `tool` calls this very tool for the verbs around a command, in its own interpreter rather
    than whichever `mainboard` a PATH names.

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

        Its failure is only said: entering afterwards refuses in clearer words.
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
            self.argv(environment), cwd=self.job.root, env=environment
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
        """The command's argv: its container, `bash -c`, or on shell-less Windows its own words,
        the program looked up on the environment's `PATH` as a shell would."""
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

        Its output is discarded, since it belongs in the job's receipts rather than its log; a
        call that is not `quiet` keeps its errors in the log.
        """
        if call is None:
            return 0
        self.flush()
        with self._verb(call, environment, quiet=quiet) as process:
            return process.wait()

    def sample(self, environment: Mapping[str, str]) -> subprocess.Popen[bytes] | None:
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
            env={**environment, **self.credentials(call)},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL if quiet else None,
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
        """Write one line of the runner's own into the job's output, nothing for empty text.

        Straight to the descriptor like a shell's `echo`, so it lands wherever the command's
        output goes, a PBS log included, in order.
        """
        if text:
            self.flush()
            os.write(2 if error else 1, (text if text.endswith("\n") else f"{text}\n").encode())

    def flush(self) -> None:
        """Flush this runner's own streams, so a child's output lands after what came before."""
        for stream in (sys.stdout, sys.stderr):
            stream.flush()

    @contextmanager
    def output(self) -> Generator[None]:
        """Send a PBS job's merged output to the log a later poll reads, and nothing otherwise.

        A PBS server spools output where no poll looks and soon purges the job, so the job
        appends to its own log (exit status beside it). The runner's own standard streams move,
        as `exec >> log 2>&1` moved a script's, so everything lands in the one log until the run
        ends.
        """
        if not self.job.logs:
            yield
            return
        Path(self.job.logs).mkdir(parents=True, exist_ok=True)
        self.flush()
        kept = [os.dup(stream) for stream in (1, 2)]
        with (Path(self.job.logs) / f"{self.job_id}.log").open("ab") as log:
            for stream in (1, 2):
                os.dup2(log.fileno(), stream)
        try:
            yield
        finally:
            self.flush()
            for stream, descriptor in zip((1, 2), kept, strict=True):
                os.dup2(descriptor, stream)
                os.close(descriptor)

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
