"""A host that stopped answering costs each command one knock, never a traceback or a stuck wait.

gold went unreachable on 2026-09-30 with two runs still owed: `job list` took 34 seconds, every
pass of a wait on another host paid for it again, and `run --on gold` ended as a traceback.
"""

from pathlib import Path

import pytest

from mainboard import monitor as sweeping
from mainboard.board import Board
from mainboard.core.project import Project
from mainboard.core.section import Verdict
from mainboard.dispatch import wrapping
from mainboard.dispatch.onboard import HostSetup
from mainboard.dispatch.shared import now
from mainboard.dispatch.state import RunRecord
from mainboard.dispatch.transport import HostUnreachable, SshTransport
from mainboard.doctor import Doctor
from mainboard.listing import Listing

_HOST = "deadbox"


@pytest.fixture
def declared(workspace: Path) -> Path:
    """A workspace declaring one ssh host no name resolves to."""
    manifest = f'[workspace]\nname = "it"\n[hosts.{_HOST}]\nkind = "ssh"\nroot = "/srv/jobs"\n'
    (workspace / "mb.toml").write_text(manifest, encoding="utf-8", newline="\n")
    return workspace


@pytest.fixture
def board(declared: Path) -> Board:
    """That workspace owing the host two runs."""
    found = Board(declared)
    for handle in ("7", "8"):
        found.dispatcher.cache.record(
            RunRecord(
                handle=handle,
                target=_HOST,
                kind="ssh",
                script="python train.py",
                args="",
                submitted_at=now(),
                name=f"run-{handle}",
            )
        )
    return found


@pytest.fixture
def knocks(monkeypatch) -> list[str]:
    """Every ssh knock made, each refused the way a host that is gone refuses it."""
    made: list[str] = []

    def refused(self: SshTransport, host: str) -> None:
        made.append(host)
        raise HostUnreachable(f"ssh connect to {host!r} failed: Host is unreachable")

    monkeypatch.setattr(SshTransport, "warm", refused)
    monkeypatch.setattr(wrapping, "_CONNECT_BACKOFF", 0.0)
    return made


def test_a_listing_knocks_once_and_says_what_to_do(board: Board, knocks: list[str]) -> None:
    report = board.monitor().once()
    quiet = {down.host: down.reason for down in report.unreachable_hosts}
    listed = Listing(board, limit=5, quiet=quiet).taken()
    assert knocks == [_HOST]
    assert [row.state for row in listed.rows] == ["unknown", "unknown"]
    assert all(_HOST in row.cause for row in listed.rows)
    assert "2 live run(s)" in listed.note and "job cancel <handle>" in listed.note


def test_a_wait_leaves_a_silent_host_alone_between_passes(
    board: Board, knocks: list[str], monkeypatch
) -> None:
    monitor = board.monitor()
    first = monitor.once()
    second = monitor.once()
    assert knocks == [_HOST]
    assert [down.host for down in second.unreachable_hosts] == [_HOST]
    assert first.running == second.running == 0
    monkeypatch.setattr(sweeping, "_SILENT_SECONDS", 0.0)
    monitor.once()
    assert knocks == [_HOST, _HOST]


def test_a_cancel_settles_a_run_its_host_never_heard_about(
    board: Board, knocks: list[str], capfd
) -> None:
    settled = board.verdicts().cancel("7")
    assert settled.code == 1
    recorded = board.dispatcher.cache.run("7", _HOST)
    assert recorded.verdict == recorded.reported == "cancelled"
    assert "was not stopped there" in capfd.readouterr().err


def test_a_dropped_hosts_runs_are_named_with_what_settles_them(
    board: Board, knocks: list[str]
) -> None:
    """Two ended runs on a released rental were announced every sweep, on no listing of live
    ones, with nothing saying which handles to cancel."""
    ended = RunRecord(
        handle="0",
        target="released",
        kind="ssh",
        script="python train.py",
        args="",
        submitted_at=now(),
        verdict="ok",
    )
    board.dispatcher.cache.record(ended)
    report = board.monitor().once()
    said = {down.host: down.reason for down in report.unreachable_hosts}
    assert f"`{Project().name} job cancel 0 --on released`" in said["released"]
    assert "released" not in knocks
    listed = Listing(board, limit=5, quiet=said).taken()
    assert said["released"] in listed.note


def test_doctor_names_no_fix_for_a_host_the_manifest_dropped(board: Board) -> None:
    """`mb host sync crimson-jobs` was the fix printed for an alias nothing declares any more."""
    cache = board.dispatcher.cache
    cache.save_host(HostSetup(host="dropped", root="/srv/jobs", digest="stale"))
    assert Doctor(board).hosts().verdict is Verdict.PASS
    cache.save_host(HostSetup(host=_HOST, root="/srv/jobs", digest="stale"))
    diverged = Doctor(board).hosts()
    assert diverged.verdict is Verdict.WARN and diverged.fix.endswith(f"host sync {_HOST}")
    assert "dropped" not in diverged.detail


def test_an_unreachable_host_is_a_sentence_not_a_traceback(mb, declared: Path) -> None:
    ran = mb("run", "--on", _HOST, "--", "true", timeout=300)
    assert ran.code == 1
    assert f"ssh connect to '{_HOST}' failed" in ran.err
