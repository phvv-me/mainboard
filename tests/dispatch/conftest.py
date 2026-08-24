from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mainboard import ExecutionPlan
from mainboard.dispatch.state import Cache, RunRecord
from mainboard.dispatch.vocabulary import JobState, Resources
from mainboard.manifest import Container, HostProfile

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from mainboard.dispatch.transport import Machine

type FieldValue = str | HostProfile | Container | dict[str, str] | None

# One `(marker, retcode, output)` rule: when `marker` appears in the argv, the command answers
# `retcode` with `output` on stdout for a clean exit and on stderr otherwise.
type Rule = tuple[str, int, str]

# One `(marker, error)` pair: when `marker` appears in the argv, the command raises instead of
# answering, the way a client whose daemon refused its control socket does.
type Fault = tuple[str, BaseException]


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run the body in a fresh empty CWD so cache/state files stay hermetic."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


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

    def __call__(self, *_, **__) -> str:
        return self.machine.answer([self.name, *self.bound])[1]

    def __getitem__(self, args: str | list[str] | tuple[str, ...]) -> RecordingCommand:
        extra = list(args) if isinstance(args, list | tuple) else [args]
        child = RecordingCommand(self.name, self.machine)
        child.bound = [*self.bound, *(str(a) for a in extra)]
        return child

    def run(self, *_, **__) -> tuple[int, str, str]:
        retcode, output = self.machine.answer([self.name, *self.bound])
        return (retcode, output, "") if retcode == 0 else (retcode, "", output)


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

    def answer(self, argv: list[str]) -> tuple[int, str]:
        """The scripted `(retcode, output)` for `argv`, recording it as run.

        argv: the full command line, its binary first.
        """
        self.calls.append(argv)
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
