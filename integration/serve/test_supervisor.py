"""`serve`'s start, stop and status, over a fake unit and a fake process-compose API."""

import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.core import Verdict
from mainboard.core.services import Unit
from mainboard.manifest.render.interpolate import Json
from mainboard.serve.client import Client
from mainboard.serve.constants import Service
from mainboard.serve.keeper import Leader, Ledger
from mainboard.serve.supervisor import Supervisor

from .fakes import answering, programs

type Spawn = Callable[..., subprocess.Popen[bytes]]

_STATE: dict[str, Json] = {
    "/project/state": {"upTime": 75_000_000_000, "version": "v1.124.0"},
    "/live": {},
}
_RUNNING: dict[str, Json] = {
    "name": "resident",
    "status": "Running",
    "is_running": True,
    "pid": 41,
    "restarts": 2,
}
_SCHEDULED: dict[str, Json] = {"name": "pass", "status": "Scheduled", "exit_code": 0}
# A keeper the ledger names that no longer runs: pid 1 started long after the epoch.
_DEAD_KEEPER = Leader(pid=1, started=0.0)


class FakeUnit(Unit):
    """The service manager's resident unit as a double: it records installs and runs nothing.

    loaded: what `loaded` answers, a unit the manager holds or not.
    """

    @property
    def name(self) -> str:
        return f"fake.serve.{self.stamp}"

    def __init__(self, root: Path, *, loaded: bool = False) -> None:
        super().__init__(root)
        self.held = loaded
        self.installs: list[tuple[list[str], int, Path]] = []
        self.removals = 0

    def install(self, command: Sequence[str], *, stop_seconds: int, log: Path) -> None:
        self.installs.append((list(command), stop_seconds, log))

    def installed(self) -> bool:
        return bool(self.installs)

    def loaded(self) -> bool:
        return self.held

    def remove(self) -> None:
        self.removals += 1
        self.held = False


def _supervisor(
    root: Path, services: Sequence[Service], *, loaded: bool = False, supervisor: str = ""
) -> Supervisor:
    unit = FakeUnit(root, loaded=loaded)
    return Supervisor(
        root, programs=programs(root, supervisor=supervisor), unit=unit, services=services
    )


def _answering(
    root: Path, services: Sequence[Service], processes: Sequence[Json]
) -> tuple[Supervisor, AbstractContextManager[None]]:
    """A supervisor whose API is a fake at a socket short enough to bind, its own socket 0600."""
    supervisor = _supervisor(root, services, loaded=True)
    supervisor.layout.state.mkdir(parents=True)
    supervisor.layout.socket.touch(mode=0o600)
    supervisor.client = Client(root / "s")
    routes: dict[str, Json] = {**_STATE, "/processes": {"data": list(processes)}}
    return supervisor, answering(root / "s", routes)


def _recorded(supervisor: Supervisor, session: Leader) -> None:
    """The ledger a dead keeper of `session` left in `supervisor`'s state."""
    supervisor.layout.state.mkdir(parents=True, exist_ok=True)
    Ledger(supervisor.layout.ledger).record(keeper=_DEAD_KEEPER, supervisor=session)


def _rows(supervisor: Supervisor) -> dict[str, tuple[Verdict, str]]:
    return {row.section: (row.verdict, row.detail) for row in supervisor.status()}


def test_with_nothing_running_each_child_says_it_is_unknown_or_why_it_is_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stand_ins: list[Service]
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    rows = _rows(_supervisor(tmp_path, stand_ins))
    assert rows == {
        "supervisor": (
            Verdict.WARN,
            "not running, and process-compose is not installed here; the owner starts it",
        ),
        "resident": (Verdict.WARN, "unknown: the supervisor is not running"),
        "pass": (Verdict.WARN, "unknown: the supervisor is not running"),
        "sink": (Verdict.PASS, "off: v2-events builds it"),
    }


def test_a_loaded_unit_whose_supervisor_does_not_answer_fails_naming_its_log(
    tmp_path: Path, stand_ins: list[Service]
) -> None:
    head = _supervisor(tmp_path, stand_ins, loaded=True).section()
    assert (head.section, head.verdict) == ("serve", Verdict.FAIL)
    assert "is loaded but nothing answers" in head.detail and "keeper.log" in head.detail


@pytest.mark.parametrize(
    ("processes", "row", "verdict", "said"),
    [
        pytest.param(
            [_RUNNING, _SCHEDULED],
            "resident",
            Verdict.PASS,
            "running, pid 41, 2 restarts",
            id="a-running-child",
        ),
        pytest.param(
            [{**_RUNNING, "status": "Completed", "is_running": False, "exit_code": 3}, _SCHEDULED],
            "resident",
            Verdict.FAIL,
            "completed, exit 3 after 2 restarts",
            id="a-resident-child-past-its-restarts",
        ),
        pytest.param(
            [_RUNNING, {**_SCHEDULED, "exit_code": 2}],
            "pass",
            Verdict.WARN,
            "the last pass exited 2",
            id="a-failed-pass",
        ),
        pytest.param(
            [_SCHEDULED],
            "resident",
            Verdict.FAIL,
            "the running YAML is older",
            id="a-child-the-yaml-lacks",
        ),
    ],
)
def test_each_child_reads_what_the_supervisor_reports_of_it(
    tmp_path: Path,
    stand_ins: list[Service],
    processes: Sequence[Json],
    row: str,
    verdict: Verdict,
    said: str,
) -> None:
    supervisor, api = _answering(tmp_path, stand_ins, processes)
    with api:
        rows = _rows(supervisor)
    assert rows[row][0] is verdict and said in rows[row][1]
    assert rows["supervisor"] == (
        Verdict.PASS,
        f"process-compose v1.124.0 up 0:01:15, keeper pid ?, 0 processes, unit "
        f"{supervisor.unit.name} loaded; socket 0600, no .env name in the 0 environments "
        "readable, nothing listening off loopback",
    )


def test_a_dotenv_name_in_any_process_of_the_tree_fails_the_supervisor_row(
    tmp_path: Path, spawned: Spawn, stand_ins: list[Service]
) -> None:
    (tmp_path / ".env").write_text("CANARY_TOKEN=one\nPLAIN=two\n", encoding="utf-8")
    # This interpreter, not an Apple binary: macOS hides an Apple binary's environment.
    asleep = [sys.executable, "-I", "-c", "import time; time.sleep(600)"]
    child = spawned(asleep, env={"PATH": "/usr/bin:/bin", "CANARY_TOKEN": "x"})
    supervisor, api = _answering(tmp_path, stand_ins, [_RUNNING, _SCHEDULED])
    _recorded(supervisor, Leader.of(child.pid) or Leader(pid=child.pid, started=0.0))
    with api:
        head = supervisor.status()[0]
    assert (head.verdict, head.detail) == (
        Verdict.FAIL,
        "CANARY_TOKEN from .env reach a child (process-compose v1.124.0)",
    )


def test_a_stop_after_the_keeper_died_sweeps_what_it_left_at_once(
    tmp_path: Path, spawned: Spawn, stand_ins: list[Service]
) -> None:
    orphan = spawned(["/bin/sleep", "600"])
    supervisor = _supervisor(tmp_path, stand_ins)
    left = Leader.of(orphan.pid) or Leader(pid=orphan.pid, started=0.0)
    _recorded(supervisor, left)
    started = time.monotonic()
    rows = supervisor.stop()
    assert time.monotonic() - started < 5
    assert [(row.verdict, row.detail) for row in rows] == [
        (
            Verdict.PASS,
            f"{supervisor.unit.name} removed, and nothing of it runs; "
            "swept 1 processes that outstayed it",
        )
    ]
    # This test is the orphan's parent, so the sweep reaped it: ask by pid and start time.
    assert not left.alive()


def test_a_start_without_process_compose_is_refused_before_anything_is_installed(
    tmp_path: Path, stand_ins: list[Service]
) -> None:
    supervisor = _supervisor(tmp_path, stand_ins)
    with pytest.raises(MissionError, match="process-compose is not installed"):
        supervisor.start()
    assert (supervisor.unit.installed(), supervisor.layout.state.exists()) == (False, False)


def test_a_start_whose_socket_path_would_not_fit_is_refused_before_anything_is_written(
    tmp_path: Path, stand_ins: list[Service], refusing: str
) -> None:
    deep = tmp_path / ("w" * 90)
    supervisor = _supervisor(deep, stand_ins, supervisor=refusing)
    with pytest.raises(MissionError, match="a unix socket path holds"):
        supervisor.start()
    assert (supervisor.unit.installed(), supervisor.layout.state.exists()) == (False, False)


def test_a_yaml_process_compose_refuses_is_never_handed_to_the_unit(
    tmp_path: Path, stand_ins: list[Service], refusing: str
) -> None:
    supervisor = _supervisor(tmp_path, stand_ins, supervisor=refusing)
    with pytest.raises(MissionError, match="unknown field 'bogus'"):
        supervisor.start()
    assert supervisor.unit.installed() is False
    assert supervisor.layout.compose.exists() and os.listdir(supervisor.layout.logs) == []
