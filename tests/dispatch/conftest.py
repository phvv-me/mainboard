import os
from typing import TYPE_CHECKING

import pytest
from plumbum import local

from mainboard.dispatch.schedulers import JobState, Resources
from mainboard.manifest import HostProfile

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path
    from types import TracebackType

    from mainboard.dispatch.transport import Machine

type FieldValue = str | int | float | bool | None | dict[str, "FieldValue"] | list["FieldValue"]


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run the body in a fresh empty CWD so cache/state files stay hermetic."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def stub_bin(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, str]]:
    """Stub scheduler/transport executables on PATH so plumbum's `local["cmd"]` resolves."""
    bindir = tmp_path_factory.mktemp("bin")
    paths: dict[str, str] = {}
    for tool in (
        "qstat",
        "qsub",
        "qdel",
        "sacct",
        "squeue",
        "sbatch",
        "scancel",
        "pueue",
        "rsync",
    ):
        executable = bindir / tool
        executable.write_text("#!/bin/sh\n")
        executable.chmod(0o755)
        paths[tool] = str(executable)
    with local.env(PATH=f"{bindir}{os.pathsep}{local.env['PATH']}"):
        yield paths


class RecordingCommand:
    """A plumbum-command stand-in that records its argv and replays a canned stdout.

    `remote["bash"][["-lc", cmd]]` indexes a command then binds args; calling it (or
    `.run(retcode=None)`) runs it. This double records every bound argv into a shared list and
    returns the queued stdout for the matching call, so a scheduler test asserts the exact
    command string built without any real process or ssh.
    """

    def __init__(self, name: str, calls: list[list[str]], outputs: list[str]) -> None:
        self.name = name
        self.calls = calls
        self.outputs = outputs
        self.bound: list[str] = []

    def __call__(self, *_, **__) -> str:
        self.calls.append([self.name, *self.bound])
        return self.outputs.pop(0) if self.outputs else ""

    def __getitem__(self, args: str | list[str] | tuple[str, ...]) -> RecordingCommand:
        extra = list(args) if isinstance(args, list | tuple) else [args]
        child = RecordingCommand(self.name, self.calls, self.outputs)
        child.bound = [*self.bound, *(str(a) for a in extra)]
        return child

    def run(self, *_, **__) -> tuple[int, str, str]:
        return (0, self.__call__(), "")


class RecordingMachine:
    """A fake plumbum machine: `machine["cmd"]` yields a `RecordingCommand`."""

    def __init__(self, outputs: list[str] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.outputs = list(outputs or [])

    def __getitem__(self, name: str) -> RecordingCommand:
        return RecordingCommand(name, self.calls, self.outputs)


def machine_with(*outputs: str) -> RecordingMachine:
    """A recording machine queued with these stdout strings, one per command call."""
    return RecordingMachine(list(outputs))


@pytest.fixture
def remote() -> RecordingMachine:
    """A recording fake `SshMachine`/`local` for scheduler command-construction tests."""
    return RecordingMachine()


class FakeRemote:
    """A context-manager stand-in for what `wrapping.connection()` returns.

    `with connection(name) as remote:` only needs `__enter__`/`__exit__`; a scheduler double
    ignores the remote entirely. `remote["bash"][[...]].run(retcode=None)` (the dispatch
    preflight's `verify` check) defaults to a clean exit; a test exercising a broken host passes
    `healthy=False` (and optionally `stderr`) instead.
    """

    def __init__(self, *, healthy: bool = True, stderr: str = "") -> None:
        self.healthy = healthy
        self.stderr = stderr

    def __enter__(self) -> FakeRemote:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return False

    def __getitem__(self, _name: str) -> _RemoteCommand:
        return _RemoteCommand(self.healthy, self.stderr)


class _RemoteCommand:
    """A minimal plumbum-command double for `remote["bash"][[...]].run(retcode=None)`."""

    def __init__(self, healthy: bool, stderr: str) -> None:
        self.healthy = healthy
        self.stderr = stderr

    def __getitem__(self, _args: str | list[str] | tuple[str, ...]) -> _RemoteCommand:
        return self

    def run(self, *_, **__) -> tuple[int, str, str]:
        return (0, "", "") if self.healthy else (1, "", self.stderr)


class RecordingScheduler:
    """A `Scheduler` double recording each call and replaying canned results."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str | tuple[str, ...], ...]]] = []
        self.submit_handle = "H1"
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


def profile(**overrides: FieldValue) -> HostProfile:
    """A `HostProfile` for dispatch tests, defaulting to an ssh host with a sync allowlist."""
    fields: dict[str, FieldValue] = {"kind": "ssh", "root": "/repo", "sync": {"include": ["src"]}}
    fields.update(overrides)
    return HostProfile.model_validate(fields)
