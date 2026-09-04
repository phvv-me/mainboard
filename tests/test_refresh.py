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
    assert log.read_text(encoding="utf-8") == "attempt=1 exit=7\nout\nerr\n"


def test_windows_tool_directory_locks_back_off_and_preserve_every_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both transient lock failures reach the log before the successful retry begins."""
    log = tmp_path / "self-update.log"
    pauses: list[float] = []
    before: list[str] = []
    results = iter(
        (
            subprocess.CompletedProcess(
                (),
                1,
                stdout="first out\n",
                stderr="failed to remove directory Scripts: Acesso negado. (os error 5)\n",
            ),
            subprocess.CompletedProcess(
                (),
                1,
                stdout="second out\n",
                stderr="failed to remove directory Scripts: sharing violation (os error 32)\n",
            ),
            subprocess.CompletedProcess((), 0, stdout="installed\n", stderr=""),
        )
    )

    def execute(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        before.append(log.read_text(encoding="utf-8"))
        return next(results)

    monkeypatch.setattr(_refresh.platform, "system", lambda: "Windows")

    assert (
        _refresh.after_parent(
            314,
            ("uv", "tool", "install", "mainboard"),
            log,
            wait=lambda parent: None,
            execute=execute,
            pause=pauses.append,
        )
        == 0
    )
    assert pauses == [0.25, 0.5]
    assert before == [
        "",
        "attempt=1 exit=1\n"
        "first out\n"
        "failed to remove directory Scripts: Acesso negado. (os error 5)\n",
        "attempt=1 exit=1\n"
        "first out\n"
        "failed to remove directory Scripts: Acesso negado. (os error 5)\n"
        "attempt=2 exit=1\n"
        "second out\n"
        "failed to remove directory Scripts: sharing violation (os error 32)\n",
    ]
    assert log.read_text(encoding="utf-8").endswith("attempt=3 exit=0\ninstalled\n")


@pytest.mark.parametrize(
    ("operating_system", "stderr"),
    [
        pytest.param(
            "Windows",
            "authentication failed (os error 5)",
            id="unrelated-windows-access-denied",
        ),
        pytest.param(
            "Linux",
            "failed to remove directory Scripts (os error 5)",
            id="same-signature-on-linux",
        ),
    ],
)
def test_refresh_does_not_retry_unrelated_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operating_system: str,
    stderr: str,
) -> None:
    """Only a Windows uv directory-removal lock enters the retry loop."""
    calls: list[Sequence[str]] = []
    pauses: list[float] = []

    def execute(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, stdout="", stderr=stderr)

    monkeypatch.setattr(_refresh.platform, "system", lambda: operating_system)

    assert (
        _refresh.after_parent(
            314,
            ("uv", "tool", "install", "mainboard"),
            tmp_path / "refresh.log",
            wait=lambda parent: None,
            execute=execute,
            pause=pauses.append,
        )
        == 1
    )
    assert len(calls) == 1
    assert pauses == []


def test_windows_tool_directory_lock_retries_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persistent sharing violation stops after five recorded attempts and four delays."""
    calls: list[Sequence[str]] = []
    pauses: list[float] = []

    def execute(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="failed to remove directory Scripts: access denied (os error 5)\n",
        )

    monkeypatch.setattr(_refresh.platform, "system", lambda: "Windows")
    log = tmp_path / "refresh.log"

    assert (
        _refresh.after_parent(
            314,
            ("uv", "tool", "install", "mainboard"),
            log,
            wait=lambda parent: None,
            execute=execute,
            pause=pauses.append,
        )
        == 1
    )
    assert len(calls) == 5
    assert pauses == [0.25, 0.5, 1.0, 2.0]
    transcript = log.read_text(encoding="utf-8")
    assert transcript.count("failed to remove directory") == 5
    assert "attempt=1 exit=1" in transcript
    assert "attempt=5 exit=1" in transcript


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
