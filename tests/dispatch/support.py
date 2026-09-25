import base64
import os
import subprocess
import sys
import threading
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pytest

from mainboard import ExecutionPlan
from mainboard.dispatch import now
from mainboard.dispatch.agent import Link
from mainboard.dispatch.agent import program as agent_program
from mainboard.dispatch.allocation import Allocation
from mainboard.dispatch.state import Cache, RunRecord
from mainboard.dispatch.vocabulary import JobState, Resources
from mainboard.manifest import Container, HostProfile


def _links() -> bool:
    """Whether this account can make a symbolic link, which Windows grants only on request."""
    with TemporaryDirectory() as scratch:
        try:
            os.symlink(scratch, os.path.join(scratch, "probe"))
        except OSError:
            return False
    return True


# A pin links the environment, the data and the results back to the mirror, and a mirror
# carries a link as a link, so an account that cannot make one cannot stand in for a host.
links_on_this_host = pytest.mark.skipif(
    not _links(), reason="this account cannot create symbolic links"
)
# A setgid directory passes its bit to what is made under it on Linux; BSD, macOS included,
# inherits the group without the bit, so the bit a Linux host keeps cannot be checked here.
setgid_inherits = pytest.mark.skipif(
    sys.platform != "linux", reason="setgid-bit inheritance is Linux filesystem behaviour"
)

if TYPE_CHECKING:
    from types import TracebackType

    from mainboard.dispatch.transport import Machine

type FieldValue = str | HostProfile | Container | dict[str, str] | None

# One `(marker, retcode, output)` rule: when `marker` appears in the argv, the command answers
# `retcode` with `output` on stdout for a clean exit and on stderr otherwise.
type Rule = tuple[str, int, str]

# One `(marker, error)` pair: when `marker` appears in the argv, the command raises instead of
# answering, the way a client whose daemon refused its control socket does.
type Fault = tuple[str, BaseException]


class InProcess:
    """A `Process` running the agent's own `run` on a thread over real pipes, in this process.

    The framed source is read off the pipe and compiled, which proves it parses, and then the
    already imported agent answers, so every line it runs is measured and a test pays no
    interpreter start.
    """

    def __init__(self) -> None:
        stdin, feed = os.pipe()
        drain, stdout = os.pipe()
        listen, stderr = os.pipe()
        self.stdin = os.fdopen(feed, "wb")
        self.stdout = os.fdopen(drain, "rb")
        self.stderr = os.fdopen(listen, "rb")
        self.pid = os.getpid()
        self.returncode: int | None = None
        self.thread = threading.Thread(
            target=self.__serve, args=(stdin, stdout, stderr), daemon=True
        )
        self.thread.start()

    def wait(self, timeout: float | None = None) -> int:
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise subprocess.TimeoutExpired("agent", timeout or 0.0)
        return self.returncode or 0

    def __serve(self, stdin: int, stdout: int, stderr: int) -> None:
        with (
            open(stdin, "rb") as source,
            open(stdout, "wb") as answer,
            open(stderr, "w", encoding="utf-8") as said,
        ):
            compile(source.read(int(source.readline())), "mainboard-agent", "exec")
            try:
                self.returncode = agent_program.run(source, answer, said)
            except Exception:
                traceback.print_exc(file=said)
                self.returncode = 1


class InProcessLink(Link):
    """A `Link` whose target is this machine's own file system, reached without a process.

    commands: every command line the agent was started with.
    """

    def __init__(self, host: str = "local") -> None:
        super().__init__(host)
        self.commands: list[str] = []

    def spawn(self, command: str) -> InProcess:
        self.commands.append(command)
        return InProcess()

    def end(self, process: InProcess) -> None:
        process.wait()


class RecordingAgent:
    """An `Agent` double that records each request and answers the one record a pin gives back.

    answer: raised instead of answering when it is an exception, the records to give back else.
    """

    def __init__(
        self, answer: list[dict[str, str]] | BaseException | None = None, host: str = "gold"
    ) -> None:
        self.answer = [{"path": "pinned"}] if answer is None else answer
        self.host = host
        self.requests: list[dict[str, dict[str, object]]] = []

    def ask(
        self, request: dict[str, dict[str, object]], *, payload: Callable[..., None] | None = None
    ) -> list[dict[str, str]]:
        del payload
        self.requests.append(request)
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


def created_request() -> Allocation:
    """A reserved provider request backed by the same in-memory cache used by dispatch tests."""
    store = cache()
    record = RunRecord(
        handle="mainboard-test-creation",
        creation="mainboard-test-creation",
        target="provider-host",
        kind="vast",
        script="true",
        args="",
        git_sha="abc",
        dirty=0,
        submitted_at=now(),
        state="prepared",
        verdict="prepared",
        evidence="pending",
    )
    store.reserve(record)
    return Allocation(cache=store, record=record)


class RecordingCommand:
    """A plumbum-command stand-in that records its argv and replays a canned answer.

    `remote["bash"][["-lc", cmd]]` indexes a command then binds args, and calling it (or
    `.run(retcode=None)`) runs it. Every bound argv lands in the machine's shared call log and
    the machine decides the answer, so a scheduler test asserts the exact command string built
    without any real process or ssh.
    """

    def __init__(self, name: str, machine: RecordingMachine) -> None:
        self.name = name
        self.machine = machine
        self.bound: list[str] = []
        self.stdin = ""

    def __call__(self, *_, **__) -> str:
        return self.machine.answer([self.name, *self.bound], stdin=self.stdin)[1]

    def __getitem__(self, args: str | list[str] | tuple[str, ...]) -> RecordingCommand:
        extra = list(args) if isinstance(args, list | tuple) else [args]
        return self.__bound([*self.bound, *(str(a) for a in extra)], stdin=self.stdin)

    def __lshift__(self, data: str) -> RecordingCommand:
        """Bind stdin the way plumbum's `cmd << text` does, so a written file is readable here."""
        return self.__bound(self.bound, stdin=str(data))

    def run(self, *_, **__) -> tuple[int, str, str]:
        retcode, output = self.machine.answer([self.name, *self.bound], stdin=self.stdin)
        return (retcode, output, "") if retcode == 0 else (retcode, "", output)

    def __bound(self, args: list[str], *, stdin: str) -> RecordingCommand:
        child = RecordingCommand(self.name, self.machine)
        child.bound = list(args)
        child.stdin = stdin
        return child


class RecordingMachine:
    """A fake plumbum machine, and the connection a `wrapping.connection()` double hands back.

    Three knobs answer every shape the dispatch subsystem asks for. `outputs` is the queue a
    probe reads, one entry per call with the last entry answering every call after it, so a
    snapshot a backend re-reads inside one operation stays the same. `rules` answer ahead of the
    queue whenever their marker appears in the argv, which is how a host shell says yes to
    `command -v uv` and no to `command -v curl`. `faults` raise instead of answering.
    """

    def __init__(
        self,
        outputs: Sequence[str] = (),
        *,
        rules: Sequence[Rule] = (),
        faults: Sequence[Fault] = (),
    ) -> None:
        self.calls: list[list[str]] = []
        self.inputs: list[str] = []
        self.outputs = list(outputs)
        self.rules = list(rules)
        self.faults = list(faults)

    def __enter__(self) -> RecordingMachine:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return False

    def __getitem__(self, name: str) -> RecordingCommand:
        return RecordingCommand(name, self)

    @property
    def lines(self) -> list[str]:
        """The trailing argument of every recorded call, the shell line each command carried."""
        return [argv[-1] for argv in self.calls if argv]

    def answer(self, argv: list[str], *, stdin: str = "") -> tuple[int, str]:
        """The scripted `(retcode, output)` for `argv`, recording it as run.

        argv: the full command line, its binary first.
        stdin: what the caller piped into it, recorded on `inputs` when there was any.
        """
        self.calls.append(argv)
        if stdin:
            self.inputs.append(stdin)
        joined = " ".join(argv)
        for marker, error in self.faults:
            if marker in joined:
                raise error
        for marker, retcode, output in self.rules:
            if marker in joined:
                return retcode, output
        if len(self.outputs) > 1:
            return 0, self.outputs.pop(0)
        return 0, self.outputs[0] if self.outputs else ""

    def ran(self, marker: str) -> bool:
        """Whether any command run so far carried `marker`."""
        return any(marker in " ".join(argv) for argv in self.calls)

    def close(self) -> None:
        """Nothing to release."""
        return


class RecordingTransport:
    """A bounded-transport double answering every one-shot ssh the way a scripted host would.

    The same three knobs as `RecordingMachine`, matched against the argv with any PowerShell
    `-EncodedCommand` payload decoded back to its script, so a rule written against the words
    of a probe matches whether the host is asked in bash or in PowerShell.
    """

    def __init__(
        self,
        outputs: Sequence[str] = (),
        *,
        rules: Sequence[Rule] = (),
        faults: Sequence[Fault] = (),
    ) -> None:
        self.machine = RecordingMachine(outputs, rules=rules, faults=faults)
        self.options: tuple[str, ...] = ("-o", "BatchMode=yes")
        self.endpoint = None

    @property
    def calls(self) -> list[list[str]]:
        """Every argv invoked, PowerShell scripts decoded."""
        return self.machine.calls

    @property
    def scripts(self) -> list[str]:
        """The decoded script (or bash line) each call carried."""
        return self.machine.lines

    def destination(self, host: str) -> str:
        return host

    def invoke(
        self, command: Sequence[str], host: str, *, operation: str, **_: object
    ) -> tuple[int, str, str]:
        del host, operation
        argv = list(command)
        if "-EncodedCommand" in argv:
            argv[-1] = base64.b64decode(argv[-1]).decode("utf-16-le")
        retcode, output = self.machine.answer(argv)
        return (retcode, output, "") if retcode == 0 else (retcode, "", output)

    def ran(self, marker: str) -> bool:
        """Whether any script run so far carried `marker`."""
        return self.machine.ran(marker)


def machine_with(
    *outputs: str, rules: Sequence[Rule] = (), faults: Sequence[Fault] = ()
) -> RecordingMachine:
    """A recording machine queued with these stdout strings, one per command call.

    The double stands in for the `Machine` union everywhere dispatch runs a command. Nothing
    type checks this suite (pyrefly reads `src/**` alone), so it is handed over as itself rather
    than cast, which keeps its call log readable at every assertion.

    outputs: stdout replayed in order, the last entry answering every later call.
    rules: `(marker, retcode, output)` answers matched against the argv ahead of the queue.
    faults: `(marker, error)` pairs raised instead of answering.
    """
    return RecordingMachine(outputs, rules=rules, faults=faults)


def keypair(home: Path, name: str = "id_ed25519") -> Path:
    """A key pair under `home`'s `.ssh`, the shape a rental's `identity` reads off a machine."""
    ssh = home / ".ssh"
    ssh.mkdir(parents=True, exist_ok=True)
    private = ssh / name
    private.write_text("PRIVATE\n", encoding="utf-8")
    Path(f"{private}.pub").write_text("ssh-ed25519 AAAA me@here\n", encoding="utf-8")
    return private


class Naps:
    """A sleeper that records how long it was asked to wait and never really waits.

    A log poll and a rental's ssh knock both retry on a schedule, so a test has to prove the wait
    was driven without paying for it in wall time.
    """

    def __init__(self) -> None:
        self.waited: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waited.append(seconds)


class RecordingScheduler:
    """A `Scheduler` double recording each call and replaying canned results."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str | tuple[str, ...], ...]]] = []
        self.submit_handle = "H1"
        self.submit_resources = Resources()
        self.state_result = JobState(handle="H1", state="F", exit_code=0, verdict="ok")
        self.queue_list: list[str] = []
        self.revive_cleared: list[str] = []

    def cancel(self, remote: Machine, root: str, *, handle: str) -> None:
        self.calls.append(("cancel", (root, handle)))

    def jobs(self, remote: Machine, root: str) -> list[JobState]:
        self.calls.append(("jobs", (root,)))
        return [self.state_result]

    def logs(self, remote: Machine, root: str, *, handle: str) -> str:
        self.calls.append(("logs", (root, handle)))
        return ""

    def queues(self, remote: Machine, root: str) -> list[str]:
        self.calls.append(("queues", (root,)))
        return self.queue_list

    def revive(self, remote: Machine, root: str) -> list[str]:
        self.calls.append(("revive", (root,)))
        return self.revive_cleared

    def state(self, remote: Machine, root: str, *, handle: str) -> JobState:
        self.calls.append(("state", (root, handle)))
        return self.state_result

    def states(self, remote: Machine, root: str, handles: Sequence[str]) -> dict[str, JobState]:
        self.calls.append(("states", (root, tuple(handles))))
        return {self.state_result.handle: self.state_result}

    def stream(self, remote: Machine, root: str, *, handle: str) -> JobState:
        self.calls.append(("stream", (root, handle)))
        return self.state_result

    def submit(
        self,
        remote: Machine,
        root: str,
        *,
        script: str,
        args: Sequence[str],
        resources: Resources,
    ) -> str:
        self.calls.append(("submit", (root, script, tuple(args))))
        self.submit_resources = resources
        return self.submit_handle

    def wait(self, remote: Machine, root: str, *, handle: str) -> JobState:
        self.calls.append(("wait", (root, handle)))
        return self.state_result


def plan(**overrides: FieldValue) -> ExecutionPlan:
    """An `ExecutionPlan` for gold's default environment, overridden field by field."""
    fields: dict[str, FieldValue] = {
        "host": "gold",
        "profile": HostProfile(kind="ssh", root="/repo", sync={"include": ["src"]}),
        "env": "default",
    }
    fields.update(overrides)
    return ExecutionPlan.model_validate(fields)


def cache() -> Cache:
    """A dispatch state cache in a private in-memory database.

    Every table the file-backed store creates is created here too, so a test reads and writes
    exactly what production does without paying the WAL journal's fsync once per test, which is
    what made this slice the slowest in the suite.
    """
    return Cache(Path(":memory:"))


def run_record(handle: str, *, target: str = "gold", submitted_at: str = "t0") -> RunRecord:
    """One dispatched run's provenance row, the unit the registry stores and reconciles."""
    return RunRecord(
        handle=handle,
        target=target,
        kind="ssh",
        script="job.sh",
        args="",
        git_sha="abc1234",
        dirty=0,
        submitted_at=submitted_at,
    )
