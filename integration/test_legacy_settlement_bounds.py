"""M34's bridge: a held settlement lock names its holder and a pass reads each stream once.

On 2026-10-10 a `job list` that outlived its killed ssh held the lock for 29 minutes while every
listing printed stale states as settled, and one pass over a lake of small files took 164 s
because every settled run replayed its stream again. Temporary lakes and processes only.
"""

import json
import os
import signal
import subprocess
import sys
from collections.abc import Generator
from contextlib import suppress
from pathlib import Path
from time import monotonic, sleep
from unittest.mock import Mock

import pytest
from filelock import FileLock

from mainboard.batch import Event, Topic
from mainboard.batch.receipts import Journal, SettlementReceipts
from mainboard.board import Board, Job
from mainboard.dispatch import Dispatcher, now
from mainboard.dispatch.state import MonitorReport, RunRecord, SettlementBusy
from mainboard.listing import Listing
from mainboard.monitor import Monitor

# A child that takes the settlement lock the way a pass does, says so, then stalls inside the pass
# until the file `go` exists (a minute at most). Arguments: workspace, held, go.
_HOLDER = """
import sys
from pathlib import Path
from time import monotonic, sleep
from mainboard import monitor
from mainboard.board import Board
from mainboard.dispatch.state import MonitorReport

def stalled(self, scope=None):
    Path(sys.argv[2]).write_text("held", encoding="utf-8")
    end = monotonic() + 60
    while not Path(sys.argv[3]).exists() and monotonic() < end:
        sleep(0.02)
    return MonitorReport()

monitor.Monitor._once = stalled
Board(Path(sys.argv[1])).monitor().once()
"""

# A session: starts the command it is given as its child, names it, and ends when its input does.
_SESSION = """
import subprocess, sys
print(subprocess.Popen(sys.argv[1:], stdout=subprocess.DEVNULL).pid, flush=True)
sys.stdin.read()
"""

# A block that outlives a one-second pass deadline.
_OVERDUE = """
from time import sleep
from mainboard import monitor

monitor._PASS_SECONDS = 1.0
with monitor._bounded_pass():
    sleep(60)
"""


@pytest.fixture
def board(workspace: Path, monkeypatch) -> Generator[Board]:
    monkeypatch.chdir(workspace)
    (workspace / "mb.toml").write_text(
        '[workspace]\nname="it"\n[hosts.alpha]\nkind="ssh"\nroot="/fake/alpha"\n',
        encoding="utf-8",
    )
    found = Board(workspace)
    yield found
    found.dispatcher.cache.close()


def run(number: int, **fields: str) -> RunRecord:
    """A finished run of the batch `cohort`, its settlement still owed."""
    return RunRecord(
        handle=str(number),
        target="alpha",
        kind="ssh",
        script="train.py",
        args="",
        submitted_at="2026-10-10T00:00:00+00:00",
        verdict="ok",
        state="finished",
        exit_code=0,
        name=f"batch:cohort/cell-{number}",
    ).model_copy(update=fields)


def holding(board: Board, tmp_path: Path) -> list[str]:
    """The command of a process that holds the settlement lock inside a pass."""
    return [
        sys.executable,
        "-c",
        _HOLDER,
        str(board.root),
        *(str(tmp_path / n) for n in ("held", "go")),
    ]


def await_hold(process: subprocess.Popen[str], tmp_path: Path) -> None:
    """Block until the pass has taken the lock, failing if its process died first."""
    end = monotonic() + 60
    while not (tmp_path / "held").exists():
        assert process.poll() is None and monotonic() < end, "the pass never took the lock"
        sleep(0.02)


def lock_file(board: Board) -> Path:
    return Path(board.dispatcher.cache.settlement.lock_file)


def free(board: Board) -> None:
    """Take the lock from a plain second instance, waiting for whoever holds it to let go."""
    with FileLock(lock_file(board), is_singleton=False).acquire(timeout=30):
        pass


@pytest.fixture
def holder(board: Board, tmp_path: Path) -> Generator[subprocess.Popen[str]]:
    """Another process holding settlement inside a pass, let go when the test is done."""
    with subprocess.Popen(holding(board, tmp_path)) as process:
        await_hold(process, tmp_path)
        yield process
        (tmp_path / "go").write_text("", encoding="utf-8")


@pytest.fixture
def session(board: Board, tmp_path: Path) -> Generator[subprocess.Popen[str]]:
    """A session process whose child is a pass holding settlement, killed when the test is done."""
    command = [sys.executable, "-c", _SESSION, *holding(board, tmp_path)]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as it:
        assert it.stdout is not None
        child = int(it.stdout.readline())
        await_hold(it, tmp_path)
        yield it
        with suppress(ProcessLookupError):
            os.kill(child, signal.SIGKILL)


def test_a_held_lock_is_named_by_the_pass_that_found_it_and_by_the_listing(
    board: Board, holder: subprocess.Popen[str], monkeypatch
) -> None:
    monkeypatch.setattr(Dispatcher, "states", Mock(side_effect=AssertionError("a host was asked")))
    board.dispatcher.cache.record(run(7, verdict="running", state="running"))
    report = board.monitor().once()
    assert report.running == 0 and report.busy is not None and report.busy.owner is not None
    owner = report.busy.owner
    assert (owner.pid, owner.phase) == (holder.pid, "settling")
    assert owner.host and owner.started_at and owner.heartbeat and owner.deadline
    listed = Listing(board, limit=5, busy=report.busy).taken()
    assert [row.state for row in listed.rows] == ["running"]
    assert str(holder.pid) in listed.note and "states are cached" in listed.note


def test_the_command_line_listing_says_who_holds_settlement(
    holder: subprocess.Popen[str], mb
) -> None:
    document = json.loads(mb("job", "list", "--json").out)
    assert document["sweep"]["busy"]["owner"]["pid"] == holder.pid
    assert "states are cached" in document["note"]
    assert str(holder.pid) in mb("job", "list").err
    assert str(holder.pid) in mb("job", "list", "--batch", "cohort").err


def test_a_pass_whose_parent_session_ends_kills_itself_and_frees_the_lock(
    board: Board, session: subprocess.Popen[str]
) -> None:
    session.communicate(timeout=30)
    free(board)


def test_a_pass_past_its_deadline_kills_its_process() -> None:
    overdue = subprocess.run([sys.executable, "-c", _OVERDUE], timeout=30, check=False)
    assert overdue.returncode == -signal.SIGKILL


@pytest.fixture
def replays(monkeypatch) -> list[str]:
    """The stream of every `Journal.replay` made from here on."""
    made: list[str] = []
    replay = Journal.replay

    def counted(self: Journal) -> list[Event]:
        made.append(self.batch)
        return replay(self)

    monkeypatch.setattr(Journal, "replay", counted)
    return made


def test_a_pass_reads_each_stream_once_for_a_hundred_runs(
    board: Board, replays: list[str], monkeypatch
) -> None:
    for number in range(100):
        board.dispatcher.cache.record(run(number))
    monkeypatch.setattr(Job, "transcript", lambda self: "")
    monitor = board.monitor()
    assert len(monitor.once().finished) == 100
    assert replays == ["cohort"] and monitor.streams == {}
    published = Journal(board.dispatcher.cache.session, "cohort").replay()
    assert sum(event.topic == Topic.EVIDENCE for event in published) == 200


def test_a_line_joins_the_stream_index_only_once_the_bus_took_it() -> None:
    bus = Mock(replay=Mock(return_value=[]), publish=Mock(side_effect=[OSError("full"), None]))
    index = SettlementReceipts(bus, "cohort")
    with pytest.raises(OSError, match="full"):
        index.publish(Topic.EVIDENCE, job="cell", data={"status": "copied"})
    assert index.get(Topic.EVIDENCE, "cell") is None
    index.publish(Topic.EVIDENCE, job="cell", data={"status": "copied"})
    taken = index.get(Topic.EVIDENCE, "cell")
    assert taken is not None and taken.data == {"status": "copied"}


def test_the_stream_index_keeps_each_submissions_newest_line_apart_from_the_jobs() -> None:
    lines = [
        Event(
            at=at,
            batch="cohort",
            topic=Topic.EVIDENCE,
            job="cell",
            data={"handle": handle, "target": "alpha", "submitted_at": "then", "status": status},
        )
        for at, handle, status in [("1", "7", "copied"), ("2", "8", "pending")]
    ]
    index = SettlementReceipts(Mock(replay=Mock(return_value=lines)), "cohort")
    newest = index.get(Topic.EVIDENCE, "cell")
    mine = index.get(Topic.EVIDENCE, "cell", submission=lines[0].data)
    assert newest is not None and newest.data["status"] == "pending"
    assert mine is not None and mine.data["status"] == "copied"


def test_a_holder_without_a_record_still_blocks_and_reads_as_unknown(board: Board) -> None:
    lock = lock_file(board)
    lock.with_name("settlement.owner.json").write_text("{torn", encoding="utf-8")
    with FileLock(lock, is_singleton=False):
        busy = board.monitor().once().busy
    assert busy is not None and busy.owner is None and "holder unknown" in busy.note


def test_a_pass_that_ran_to_its_end_stamps_success_and_leaves_no_record(board: Board) -> None:
    lock = lock_file(board)
    assert SettlementBusy.read(lock).last_success == ""
    board.monitor().once()
    assert SettlementBusy.read(lock).last_success
    assert not lock.with_name("settlement.owner.json").exists()


def test_a_pass_that_dies_stamps_nothing_and_lets_go_of_the_lock(
    board: Board, monkeypatch
) -> None:
    monkeypatch.setattr(Monitor, "_once", Mock(side_effect=OSError("interrupted pass")))
    with pytest.raises(OSError, match="interrupted pass"):
        board.monitor().once()
    assert SettlementBusy.read(lock_file(board)).last_success == ""
    assert not lock_file(board).with_name("settlement.owner.json").exists()
    free(board)


def test_a_wait_names_the_holder_and_waits_on_instead_of_heartbeating_cached_states(
    board: Board, monkeypatch
) -> None:
    cache = board.dispatcher.cache
    cache.record(run(7, verdict="running", state="running", name="solo"))
    busy = SettlementBusy(observed_at=now(), last_success="2026-10-10T00:00:00+00:00")
    monkeypatch.setattr(Monitor, "once", lambda self, *, scope=None: MonitorReport(busy=busy))
    said: list[str] = []
    verdict = board.verdicts().wait(
        "7",
        host="alpha",
        say=said.append,
        poll=lambda seconds: cache.report(
            cache.resolve(cache.run("7", "alpha"), "finished", 0, "ok"), "ok"
        ),
    )
    assert said == [busy.note] and verdict.code == 0
