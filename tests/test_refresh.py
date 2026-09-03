import subprocess
from collections.abc import Sequence
from pathlib import Path

import psutil
import pytest

from mainboard import _refresh


def test_the_worker_waits_for_its_parent_before_replacing_the_tool_and_records_the_result(
    tmp_path: Path,
) -> None:
    """The lock holder goes first, uv second, and its diagnostic remains after both exit."""
    events: list[str] = []
    log = tmp_path / "self-update.log"

    def wait(parent: int) -> None:
        events.append(f"wait:{parent}")

    def execute(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        events.append(f"run:{' '.join(command)}")
        return subprocess.CompletedProcess(command, 7, stdout="out\n", stderr="err\n")

    command = ("uv", "tool", "install", "mainboard")
    assert _refresh.after_parent(314, command, log, wait=wait, execute=execute) == 7
    assert events == ["wait:314", "run:uv tool install mainboard"]
    assert log.read_text(encoding="utf-8") == "exit=7\nout\nerr\n"


def test_the_worker_defaults_to_its_real_process_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production calls use the same two injectable boundaries the ordering test observes."""
    waited: list[int] = []
    executed: list[Sequence[str]] = []
    monkeypatch.setattr(_refresh, "_wait", waited.append)

    def execute(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        executed.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="done\n", stderr="")

    monkeypatch.setattr(_refresh, "_execute", execute)
    log = tmp_path / "refresh.log"
    assert _refresh.after_parent(12, ("uv", "tool", "install"), log) == 0
    assert waited == [12]
    assert executed == [("uv", "tool", "install")]


def test_the_parent_wait_absorbs_only_a_process_that_already_vanished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waited: list[float] = []

    class Process:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def wait(self, timeout: float) -> None:
            waited.append(timeout)
            if self.pid == 2:
                raise psutil.NoSuchProcess(self.pid)

    monkeypatch.setattr(_refresh.psutil, "Process", Process)
    _refresh._wait(1)
    _refresh._wait(2)
    assert waited == [60.0, 60.0]


def test_the_executor_runs_the_exact_argv_without_a_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Sequence[str], dict[str, bool]]] = []

    def run(command: Sequence[str], **options: bool) -> subprocess.CompletedProcess[str]:
        calls.append((command, options))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(_refresh.subprocess, "run", run)
    command = ("uv", "tool", "install", "mainboard")
    assert _refresh._execute(command).returncode == 0
    assert calls == [(command, {"capture_output": True, "check": False, "text": True})]


def test_the_cli_entrypoint_delegates_to_the_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, tuple[str, ...], Path]] = []

    def after_parent(parent: int, command: tuple[str, ...], log: Path) -> int:
        calls.append((parent, command, log))
        return 9

    monkeypatch.setattr(_refresh, "after_parent", after_parent)
    log = Path("refresh.log")
    assert _refresh.main(42, log, "uv", "tool", "install") == 9
    assert calls == [(42, ("uv", "tool", "install"), log)]
