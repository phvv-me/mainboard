import subprocess
from collections.abc import Sequence
from pathlib import Path

import psutil
import pytest

from mainboard import _refresh

COMMAND = ("uv", "tool", "install", "mainboard")


class Worker:
    """The worker's process boundaries, scripted: each install answers the next outcome.

    The last outcome repeats, so one outcome answers every attempt.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, log: Path) -> None:
        self.log = log
        self.fault: type[Exception] | None = None
        self.outcomes: list[tuple[int, str, str]] = []
        self.events: list[str] = []
        self.timeouts: list[float] = []
        self.options: list[dict[str, bool]] = []
        self.pauses: list[float] = []
        self.before: list[str] = []
        monkeypatch.setattr(_refresh.psutil, "Process", self.process)
        monkeypatch.setattr(_refresh.subprocess, "run", self.run)
        monkeypatch.setattr(_refresh, "sleep", self.pauses.append)

    def __call__(self, *outcomes: tuple[int, str, str]) -> int:
        self.outcomes = list(outcomes)
        return _refresh.main(314, self.log, *COMMAND)

    def process(self, pid: int) -> Worker:
        self.events.append(f"wait:{pid}")
        return self

    def wait(self, timeout: float) -> None:
        self.timeouts.append(timeout)
        if self.fault is not None:
            raise self.fault(314)

    def run(self, command: Sequence[str], **options: bool) -> subprocess.CompletedProcess[str]:
        self.events.append(f"run:{' '.join(command)}")
        self.options.append(options)
        self.before.append(self.log.read_text(encoding="utf-8"))
        code, out, err = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        return subprocess.CompletedProcess(command, code, stdout=out, stderr=err)


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Worker:
    return Worker(monkeypatch, tmp_path / "self-update.log")


@pytest.mark.parametrize(
    "fault",
    [
        pytest.param(None, id="a-parent-that-exited-while-we-waited"),
        pytest.param(psutil.NoSuchProcess, id="a-parent-that-had-already-gone"),
        pytest.param(psutil.TimeoutExpired, id="a-parent-still-alive-after-the-minute"),
        pytest.param(psutil.AccessDenied, id="a-pid-this-user-may-not-watch"),
    ],
)
def test_every_way_the_parent_wait_ends_leads_to_the_install_and_its_recorded_result(
    worker: Worker, fault: type[Exception] | None
) -> None:
    """The wait is a courtesy and the install is the job; uv runs second, as the exact argv.

    Only a vanished parent was absorbed, so a parent still holding the launcher after the minute,
    or a recycled pid, threw out of the worker before it had created its log: a deferred update
    that died in silence after `self-update` exited 0. The diagnostic remains after both exit,
    and the marker that kept later commands from scheduling a second worker is gone.
    """
    pending = worker.log.with_suffix(".pending")
    pending.write_text("314", encoding="utf-8")
    worker.fault = fault

    assert worker((7, "out\n", "err\n")) == 7

    assert worker.events == ["wait:314", "run:uv tool install mainboard"]
    assert worker.timeouts == [60.0]
    assert worker.options == [{"capture_output": True, "check": False, "text": True}]
    assert worker.log.read_text(encoding="utf-8") == "attempt=1 exit=7\nout\nerr\n"
    assert not pending.exists()


def test_windows_tool_directory_locks_back_off_and_preserve_every_attempt(
    worker: Worker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both transient lock failures reach the log before the successful retry begins."""
    first = (1, "first out\n", "failed to remove directory Scripts: Acesso negado. (os error 5)\n")
    second = (
        1,
        "second out\n",
        "failed to remove directory Scripts: sharing violation (os error 32)\n",
    )
    monkeypatch.setattr(_refresh.platform, "system", lambda: "Windows")

    assert worker(first, second, (0, "installed\n", "")) == 0

    assert worker.pauses == [0.25, 0.5]
    assert worker.before == [
        "",
        f"attempt=1 exit=1\n{first[1]}{first[2]}",
        f"attempt=1 exit=1\n{first[1]}{first[2]}attempt=2 exit=1\n{second[1]}{second[2]}",
    ]
    assert worker.log.read_text(encoding="utf-8").endswith("attempt=3 exit=0\ninstalled\n")


@pytest.mark.parametrize(
    ("operating_system", "stderr"),
    [
        pytest.param(
            "Windows", "authentication failed (os error 5)", id="unrelated-windows-access-denied"
        ),
        pytest.param(
            "Linux",
            "failed to remove directory Scripts (os error 5)",
            id="same-signature-on-linux",
        ),
    ],
)
def test_refresh_does_not_retry_unrelated_failures(
    worker: Worker, monkeypatch: pytest.MonkeyPatch, operating_system: str, stderr: str
) -> None:
    """Only a Windows uv directory-removal lock enters the retry loop."""
    monkeypatch.setattr(_refresh.platform, "system", lambda: operating_system)
    assert worker((1, "", stderr)) == 1
    assert len(worker.options) == 1
    assert worker.pauses == []


def test_windows_tool_directory_lock_retries_are_bounded(
    worker: Worker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persistent sharing violation stops after five recorded attempts and four delays."""
    monkeypatch.setattr(_refresh.platform, "system", lambda: "Windows")
    locked = "failed to remove directory Scripts: access denied (os error 5)\n"
    assert worker((1, "", locked)) == 1
    assert len(worker.options) == 5
    assert worker.pauses == [0.25, 0.5, 1.0, 2.0]
    transcript = worker.log.read_text(encoding="utf-8")
    assert transcript.count("failed to remove directory") == 5
    assert "attempt=1 exit=1" in transcript
    assert "attempt=5 exit=1" in transcript
