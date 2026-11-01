"""The keeper: one supervisor per state directory, and nothing of its session left behind."""

import fcntl
import signal
import subprocess
import sys
from collections.abc import Callable
from inspect import getfile
from pathlib import Path

import psutil
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mainboard.serve import keeper
from mainboard.serve.keeper import Layout, Leader, Ledger

# A supervisor stand-in that, as process-compose does, starts a child in a process group of its
# own, and one whose child outlives it: a grandchild with no parent left, still in the session.
_STRANDING = (
    "/bin/sh",
    "-c",
    "set -m; /bin/sleep 600 & /bin/sh -c '/bin/sleep 601 & exit 0'; wait",
)

_LEADER = st.builds(
    Leader,
    pid=st.integers(1, 2**31 - 1),
    started=st.floats(0, 4e9, allow_nan=False, allow_infinity=False),
)

type Spawn = Callable[..., subprocess.Popen[bytes]]
type Eventually = Callable[[Callable[[], bool], float], bool]


@pytest.fixture
def layout(tmp_path: Path) -> Layout:
    state = tmp_path / "serve"
    state.mkdir(mode=0o700)
    return Layout(state)


def _keeper(layout: Layout, *command: str) -> list[str]:
    """The keeper's command line as a unit runs it: by path, isolated."""
    return [sys.executable, "-I", getfile(keeper), str(layout.state), "--", *command]


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(kept=_LEADER, supervisor=_LEADER)
def test_the_ledger_reads_back_the_keeper_and_the_supervisor_it_recorded(
    layout: Layout, kept: Leader, supervisor: Leader
) -> None:
    ledger = Ledger(layout.ledger)
    ledger.record(keeper=kept, supervisor=supervisor)
    assert (ledger.keeper(), ledger.supervisor()) == (kept, supervisor)


@pytest.mark.parametrize(
    "written",
    [
        pytest.param("not json", id="a-torn-write"),
        pytest.param('{"keeper": [1, 2.0]}', id="no-supervisor-recorded"),
        pytest.param('{"keeper": [1], "supervisor": [2, 3.0]}', id="a-keeper-without-its-start"),
    ],
)
def test_a_missing_or_torn_ledger_reads_as_nothing_recorded(layout: Layout, written: str) -> None:
    layout.ledger.write_text(written, encoding="utf-8")
    torn, missing = Ledger(layout.ledger), Ledger(layout.state / "absent.json")
    assert (torn.keeper(), torn.supervisor(), torn.members()) == (None, None, [])
    assert (missing.keeper(), missing.supervisor(), missing.sweep()) == (None, None, 0)


def test_a_second_keeper_on_a_held_lock_exits_75_and_starts_nothing(layout: Layout) -> None:
    with layout.lock.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        done = subprocess.run(
            _keeper(layout, "/usr/bin/touch", "started"), capture_output=True, timeout=30
        )
    assert done.returncode == 75
    assert not (layout.state / "started").exists()
    assert "another keeper holds" in layout.log.read_text(encoding="utf-8")


def test_a_stopped_keeper_leaves_nothing_of_the_session_alive_a_stranded_grandchild_included(
    layout: Layout, spawned: Spawn, eventually: Eventually
) -> None:
    running = spawned(_keeper(layout, *_STRANDING))
    ledger = Ledger(layout.ledger)
    assert eventually(lambda: len(ledger.members()) == 3, 20)
    members = ledger.members()
    running.send_signal(signal.SIGTERM)
    assert running.wait(timeout=30) == 128 + signal.SIGTERM
    assert ([member.pid for member in members if member.is_running()], ledger.supervisor()) == (
        [],
        None,
    )


def test_a_new_keeper_sweeps_the_session_a_crash_left_and_rotates_the_supervisor_log(
    layout: Layout, spawned: Spawn
) -> None:
    orphan = spawned(["/bin/sleep", "600"])
    left = Leader.of(orphan.pid) or Leader(pid=orphan.pid, started=0.0)
    Ledger(layout.ledger).record(keeper=Leader(pid=1, started=1.0), supervisor=left)
    layout.supervisor_log.write_bytes(b"x" * ((8 << 20) + 1))
    done = subprocess.run(_keeper(layout, "/usr/bin/true"), capture_output=True, timeout=30)
    assert (done.returncode, orphan.wait(timeout=10)) == (0, -signal.SIGTERM)
    assert (layout.state / "pc.log.1").stat().st_size == (8 << 20) + 1
    assert "swept 1 processes an earlier run left" in layout.log.read_text(encoding="utf-8")


def test_a_session_id_another_process_now_leads_names_none_of_its_processes(
    layout: Layout, spawned: Spawn
) -> None:
    newcomer = spawned(["/bin/sleep", "600"])
    started = psutil.Process(newcomer.pid).create_time()
    Ledger(layout.ledger).record(
        keeper=Leader(pid=1, started=1.0), supervisor=Leader(newcomer.pid, started - 60)
    )
    assert [process.pid for process in Leader(newcomer.pid, started).members()] == [newcomer.pid]
    assert (Ledger(layout.ledger).sweep(), newcomer.poll()) == (0, None)


def test_the_keeper_narrows_the_socket_its_supervisor_binds_to_0600(
    layout: Layout, spawned: Spawn, eventually: Eventually
) -> None:
    binding = "/usr/bin/touch pc.sock && /bin/chmod 700 pc.sock && exec /bin/sleep 600"
    running = spawned(_keeper(layout, "/bin/sh", "-c", binding))
    assert eventually(
        lambda: layout.socket.exists() and layout.socket.stat().st_mode & 0o777 == 0o600, 20
    )
    running.send_signal(signal.SIGTERM)
    assert running.wait(timeout=30) == 128 + signal.SIGTERM


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["/state"], id="no-separator"),
        pytest.param(["/state", "--"], id="no-command"),
    ],
)
def test_a_keeper_called_without_a_command_is_a_usage_error(argv: list[str]) -> None:
    assert keeper.main(argv) == 64
