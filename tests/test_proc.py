import signal
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import psutil
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.core.errors import MissionError
from mainboard.proc import TIMED_OUT, Processes

# A Python that starts a sleeping child, says the child's pid, then sleeps itself.
_PARENT = (
    "import subprocess, sys, time; "
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
    "print(child.pid, flush=True); time.sleep(60)"
)

# A Python that would outlive any test unless something stops it.
_SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]


class Clock:
    """A monotonic clock only the sleeps move, running `arrive` once `after` sleeps have passed.

    after: how many sleeps pass before the awaited condition comes true.
    arrive: what makes the condition true.
    """

    def __init__(self, after: int = -1, arrive: Path | None = None) -> None:
        self.now = 0.0
        self.after = after
        self.arrive = arrive
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        """Advance the clock, making the condition true on the `after`th sleep."""
        self.slept.append(seconds)
        self.now += seconds
        if len(self.slept) == self.after and self.arrive is not None:
            self.arrive.touch()


def reaped() -> int:
    """The pid of a process that has already exited and been reaped."""
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def ended(process: psutil.Process) -> bool:
    """Whether `process` is dead, a zombie included: an orphan waits for init to reap it."""
    try:
        return process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


@pytest.fixture
def family() -> Iterator[tuple[subprocess.Popen[str], psutil.Process]]:
    """A running parent and the sleeping child it started, both reaped whatever the test did."""
    parent = subprocess.Popen([sys.executable, "-c", _PARENT], stdout=subprocess.PIPE, text=True)
    assert parent.stdout is not None
    child = psutil.Process(int(parent.stdout.readline()))
    yield parent, child
    if child.is_running():
        child.kill()
    if parent.poll() is None:
        parent.kill()
    parent.wait()
    parent.stdout.close()


@pytest.mark.parametrize("force", [False, True])
def test_killing_a_process_stops_everything_it_started_and_names_who_was_already_gone(
    family: tuple[subprocess.Popen[str], psutil.Process], force: bool
) -> None:
    """A parent's children die with it, so nothing it spawned is left orphaned and running.

    A pid that no longer exists is not an error: it is reported back as already gone, since the
    caller asked for the process to stop and it has.
    """
    parent, child = family
    missing = reaped()

    gone = Processes().kill([parent.pid, missing], force=force)

    assert gone == [missing]
    assert parent.wait(timeout=10) != 0
    _, alive = psutil.wait_procs([child], timeout=10)
    assert all(ended(process) for process in alive)


@pytest.mark.parametrize(
    ("seconds", "command", "status"),
    [
        (30.0, [sys.executable, "-c", "raise SystemExit(3)"], 3),
        (0.2, _SLEEPER, TIMED_OUT),
    ],
    ids=["finished", "stopped"],
)
def test_a_bounded_command_answers_its_own_status_or_gnu_s_timed_out(
    seconds: float, command: list[str], status: int
) -> None:
    """A command that finishes in time keeps its status; one stopped at the limit reads 124."""
    assert Processes().timeout(seconds, command) == status


@pytest.mark.skipif(sys.platform == "win32", reason="Windows cannot ignore a terminate request")
def test_a_command_that_ignores_the_stop_request_is_killed_after_the_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tree that ignores SIGTERM still ends, since the grace period is followed by SIGKILL.

    The ignore is inherited from this process across the exec, so the command ignores the
    request from its first instruction and no startup race decides which branch runs.
    """
    monkeypatch.setattr("mainboard.proc._GRACE", 0.2)
    held = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        status = Processes().timeout(0.2, _SLEEPER)
    finally:
        signal.signal(signal.SIGTERM, held)

    assert status == TIMED_OUT


@pytest.mark.parametrize(
    ("command", "message"),
    [([], "needs a command"), (["/no/such/program-anywhere"], "cannot start")],
    ids=["empty", "unstartable"],
)
def test_a_command_that_cannot_run_is_refused_with_a_reason(
    command: list[str], message: str
) -> None:
    """Nothing to run, or a program that is not there, is a refusal and never a hang."""
    with pytest.raises(MissionError, match=message):
        Processes().timeout(1.0, command)


@given(appears=st.integers(min_value=0, max_value=8), quarters=st.integers(0, 8))
def test_a_wait_holds_exactly_when_its_condition_arrives_before_the_deadline(
    tmp_path: Path, appears: int, quarters: int
) -> None:
    """Waiting succeeds iff the file appears by the last look, and no deadline waits forever.

    Every look is a quarter second apart, so the file arriving after `appears` sleeps is seen in
    time exactly when the look before it still fell inside the deadline.
    """
    target = tmp_path / f"ready-{appears}-{quarters}"
    target.unlink(missing_ok=True)
    if not appears:
        target.touch()
    clock = Clock(after=appears, arrive=target)

    held = Processes(clock=clock, sleep=clock.sleep).wait(file=target, seconds=quarters / 4)

    assert held == (not quarters or appears == 0 or (appears - 1) < quarters)
    assert all(pause == 0.25 for pause in clock.slept)


def test_a_wait_for_a_port_sees_a_listener_and_times_out_on_a_closed_one() -> None:
    """A listening socket answers at once; a port nothing listens on is waited out."""
    with socket.create_server(("127.0.0.1", 0)) as server:
        port = server.getsockname()[1]
        assert Processes().wait(port=f"127.0.0.1:{port}")
    clock = Clock()

    assert not Processes(clock=clock, sleep=clock.sleep).wait(port=f"127.0.0.1:{port}", seconds=1)
    assert clock.slept == [0.25] * 4


def test_a_wait_for_a_pid_holds_once_the_process_is_gone() -> None:
    """An exited pid holds at once, and a live one, this very process, is waited out."""
    clock = Clock()
    waiting = Processes(clock=clock, sleep=clock.sleep)

    assert waiting.wait(pid=reaped())
    assert not waiting.wait(pid=psutil.Process().pid, seconds=0.5)
    assert clock.slept == [0.25, 0.25]


@pytest.mark.parametrize(
    ("port", "message"),
    [("", "needs --file, --port or --pid"), ("localhost:http", "host:port")],
    ids=["nothing", "port-name"],
)
def test_a_wait_with_no_condition_or_a_misspelled_port_is_refused(port: str, message: str) -> None:
    """A wait on nothing would return at once, and a port name is not a number."""
    with pytest.raises(MissionError, match=message):
        Processes().wait(port=port)
