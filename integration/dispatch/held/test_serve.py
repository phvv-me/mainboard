"""The node's server: claims on the spool, jobs run in order, and the beat it keeps.

Everything runs on this machine against a temporary directory: the node's allocation is faked by
the two environment variables a PBS job has. No ssh, no cluster.
"""

import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.dispatch import Beat, Spool
from mainboard.dispatch.mirroring.spool import Claim
from mainboard.holding.serve import Server

from .builders import entry, halt, script, serve, until


def test_one_claimant_wins_a_contended_entry(spool: Spool, tmp_path: Path) -> None:
    submission = entry(spool, script(tmp_path, "true"))
    with ThreadPoolExecutor(16) as pool:
        won = list(pool.map(lambda n: spool.claim(submission, Claim(gen=n)), range(64)))
    assert won.count(True) == 1
    assert spool.queued() == []
    assert not list(spool.claims.glob(".tmp.*")), "temporary names are removed"


def test_oldest_fitting_entry_is_next_and_claimed_or_cancelled_ones_are_skipped(
    spool: Spool, tmp_path: Path
) -> None:
    first = entry(spool, script(tmp_path, "true"), name="h01")
    second = entry(spool, script(tmp_path, "true"), name="h02")
    third = entry(spool, script(tmp_path, "true"), name="h03")
    assert [s.handle for s in spool.queued()] == ["h01", "h02", "h03"]
    spool.claim(first, Claim())
    spool.publish(spool.cancel, second.handle, text="")
    assert [s.handle for s in spool.queued()] == [third.handle]


def test_the_node_runs_jobs_in_order_and_leaves_an_exit_artifact_each(
    spool: Spool, node: dict[str, str], tmp_path: Path
) -> None:
    passing = entry(spool, script(tmp_path, "echo first\n"), name="h01")
    bad = entry(spool, script(tmp_path, "echo second; exit 3\n"), name="h02")
    thread = serve(spool, node)
    until(lambda: spool.exit_of(bad.handle) is not None)
    halt(spool, thread)
    assert (spool.exit_of(passing.handle), spool.exit_of(bad.handle)) == (0, 3)
    assert "first" in (spool.logs / f"{passing.handle}.log").read_text()
    claim = Claim.model_validate_json((spool.claims / f"{passing.handle}.json").read_text())
    assert (claim.alloc, claim.gen, claim.node) == (
        "123.opbs",
        1,
        socket.gethostname().split(".")[0],
    )
    assert (spool.path / "gen" / "1" / "ended").exists()


def test_a_tombstone_ends_a_running_job_with_a_terminated_exit(
    spool: Spool, node: dict[str, str], tmp_path: Path
) -> None:
    sleeper = entry(spool, script(tmp_path, "sleep 60\n"), name="h01")
    thread = serve(spool, node)
    until(lambda: (spool.claims / "h01.json").exists())
    time.sleep(0.3)
    spool.publish(spool.cancel, sleeper.handle, text="")
    until(lambda: spool.exit_of(sleeper.handle) is not None)
    assert spool.exit_of(sleeper.handle) in {143, -15}
    halt(spool, thread)


def test_a_job_that_cannot_fit_the_allocation_waits_and_the_server_ends_early(
    spool: Spool, node: dict[str, str], tmp_path: Path
) -> None:
    big = entry(spool, script(tmp_path, "true"), walltime="00:00:30", name="h01")
    thread = serve(spool, node, walltime="00:02:02")
    thread.join(20)
    assert not thread.is_alive(), "no job fits, so the allocation is given back"
    assert [s.handle for s in spool.queued()] == [big.handle]


def test_a_server_refuses_to_run_outside_an_allocation(spool: Spool, tmp_path: Path) -> None:
    server = Server(spool, gen=1, walltime="01:00:00", environ={"PATH": os.environ["PATH"]})
    with pytest.raises(MissionError, match="PBS allocation"):
        server.run()


def test_the_beat_names_the_job_in_flight(
    spool: Spool, node: dict[str, str], tmp_path: Path
) -> None:
    entry(spool, script(tmp_path, "sleep 1\n"), name="h01")
    thread = serve(spool, node)
    until(lambda: (spool.path / "gen" / "1" / "beat").exists())
    until(
        lambda: Beat.model_validate_json((spool.path / "gen/1/beat").read_text()).running == "h01"
    )
    halt(spool, thread)
