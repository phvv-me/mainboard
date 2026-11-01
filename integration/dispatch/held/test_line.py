"""The held line from the center: what its spool says, the keeper that renews it, its records.

The center's shell is executed by a local bash (plumbum's `local` is a `Machine`) and `qsub` is a
stub on PATH.
"""

import os
import stat
import subprocess
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from plumbum import local

from mainboard import Board, ExecutionPlan, MissionError
from mainboard.dispatch import Beat, Dispatcher, Order, Resources, Shipment, Source, Spool
from mainboard.dispatch.jobs import vocabulary
from mainboard.dispatch.mirroring.spool import LineSpec, Remote
from mainboard.dispatch.schedulers import Held
from mainboard.dispatch.state import Cache
from mainboard.holding.line import Line, keeper_script
from mainboard.manifest import HostProfile, QueuePolicy
from mainboard.runtime import walltime_seconds
from mainboard.runtime.job import walltime
from mainboard.state import Lake

from .builders import pbs_board

# A stamp this far ahead of any clock reads as fresh, so no row depends on when it runs.
_AHEAD = 2**40
_KEEPING = {"keeper.beat": str(_AHEAD)}
_ALLOCATED = {**_KEEPING, "gen/1/alloc": "7.opbs"}
_READY = {**_ALLOCATED, "gen/1/beat": Beat(at=_AHEAD, state="ready").model_dump_json()}
_BUSY = {
    **_ALLOCATED,
    "gen/1/beat": Beat(at=_AHEAD, state="running", running="h1").model_dump_json(),
}
_ENDED = {**_BUSY, "gen/1/ended": "1"}


def status_for(spool: Spool, center: Path, **files: str) -> Line:
    """A line whose spool holds `files`, read through a local shell."""
    for name, text in files.items():
        spool.replace(spool.path / name, text)
    return Line(pbs_board(center, str(spool.path)))


def fake_qsub(directory: Path, body: str) -> dict[str, str]:
    """A `qsub` on PATH, in the new `directory`, running `body`."""
    directory.mkdir()
    stub = directory / "qsub"
    stub.write_text(f"#!/bin/bash\n{body}\n", encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return {**os.environ, "PATH": f"{directory}:{os.environ['PATH']}"}


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        pytest.param({}, ("unkept", "", 0), id="no-keeper-beating"),
        pytest.param(_KEEPING, ("opening", "", 0), id="a-keeper-and-no-allocation"),
        pytest.param(_READY, ("ready", "7.opbs", 1), id="an-allocation-answering"),
        pytest.param(_BUSY, ("busy", "7.opbs", 1), id="an-allocation-running-a-job"),
        pytest.param(_ENDED, ("renewing", "", 1), id="an-allocation-gone-for-the-next"),
        pytest.param({**_ENDED, "stop": ""}, ("ended", "", 1), id="a-released-line"),
    ],
)
def test_a_line_reads_as_what_its_spool_adds_up_to(
    spool: Spool, tmp_path: Path, files: dict[str, str], expected: tuple[str, str, int]
) -> None:
    line = status_for(spool, tmp_path / "center", **files)
    read = line.read(local, Remote(str(spool.path)))
    assert (read.state, read.alloc, read.generation) == expected


def test_the_keeper_opens_a_generation_per_allocation_until_released(
    spool: Spool, tmp_path: Path
) -> None:
    text = keeper_script(
        spool.line(),
        spool=str(spool.path),
        ownership="abc",
        flags=["-q", "interact-g", "-l", "select=1:mem=100gb"],
        template="serve --gen @GEN@ --walltime @WALL@",
    )
    (spool.path / "keeper.token").write_text("abc")
    env = fake_qsub(
        tmp_path / "bin",
        f'echo "$@" >> {tmp_path}/calls; n=$(ls {spool.path}/gen | sort -n | tail -1); '
        f'echo "j$n" > {spool.path}/gen/$n/alloc; '
        f'[ "$n" -ge 2 ] && touch {spool.path}/stop; exit 0',
    )
    subprocess.run(["bash", "-c", text], env=env, timeout=60, check=True)
    first, second = (tmp_path / "calls").read_text().splitlines()
    assert first.startswith("-I -q interact-g -l select=1:mem=100gb -l walltime=0")
    assert "-N mbhold -- /bin/bash -lc serve --gen 1 --walltime " in first
    assert "--gen 2 --walltime " in second
    assert (spool.path / "keeper.beat").exists() and (spool.path / "keeper.json").exists()
    assert "keeper finished" in (spool.path / "keeper.log").read_text()


def test_an_older_keeper_exits_once_a_newer_one_holds_the_token(
    spool: Spool, tmp_path: Path
) -> None:
    text = keeper_script(
        spool.line(),
        spool=str(spool.path),
        ownership="old",
        flags=[],
        template="serve",
    )
    (spool.path / "keeper.token").write_text("new")
    env = fake_qsub(tmp_path / "bin", "exit 99")
    subprocess.run(["bash", "-c", text], env=env, timeout=30, check=True)
    assert not (spool.path / "gen" / "1").exists(), "a fenced keeper opens nothing"


def test_the_keeper_script_is_valid_bash(spool: Spool) -> None:
    text = keeper_script(
        spool.line(), spool="/a b/c", ownership="t", flags=["-q", "x y"], template="serve 'q'"
    )
    assert subprocess.run(["bash", "-n"], input=text, text=True, check=False).returncode == 0
    assert LineSpec.grace == 120


@pytest.fixture
def dispatcher(tmp_path: Path) -> Dispatcher:
    """A dispatcher whose runs are recorded beneath `tmp_path`."""
    return Dispatcher(cache=Cache(Lake.at(tmp_path)), root=tmp_path)


@pytest.fixture
def order() -> Order:
    """A command for the held queue of `miyabi-g`."""
    profile = HostProfile(kind="pbs", queues={"held": QueuePolicy(scheduler="held")})
    return Order(
        plan=ExecutionPlan(host="miyabi-g", profile=profile, env="default"),
        shipment=Shipment.of_command("echo hi", source=Source(identity="t", key="t"), imports=()),
        resources=Resources(queue="held", walltime="00:05:00"),
    )


def test_a_held_submission_is_recorded_before_it_is_queued_and_closed_when_refused(
    dispatcher: Dispatcher, order: Order, held: Held, spool: Spool, root: str
) -> None:
    handle = dispatcher.handed(order, held, local, pinned=root, script="job.sh")
    queued = dispatcher.cache.run(handle, "miyabi-g")
    assert (queued.verdict, queued.kind) == (vocabulary.Word.QUEUED, "held")
    assert queued.creation.startswith("mainboard-"), "the intent's label is kept"
    (spool.path / "stop").touch()
    with pytest.raises(MissionError, match="no line is held"):
        dispatcher.handed(order, held, local, pinned=root, script="job.sh")
    refused = [run for run in dispatcher.cache.listed(limit=20) if run.handle != handle]
    assert [run.verdict for run in refused] == [vocabulary.Word.FAILED]


def test_only_a_declared_pbs_host_has_a_line(workspace: Path) -> None:
    manifest = '[workspace]\nname = "it"\n[hosts.p]\nkind = "pbs"\n[hosts.g]\nkind = "ssh"\n'
    (workspace / "mb.toml").write_text(manifest, encoding="utf-8")
    board = Board(workspace)
    kept = Line.of(board, "p")
    assert kept is not None and kept.host == "p"
    assert Line.of(board, "g") is None
    assert Line.of(board, "rented-4090") is None


@given(st.integers(min_value=0, max_value=10 * 24 * 3600))
def test_a_walltime_past_a_day_keeps_its_hours(seconds: int) -> None:
    assert walltime_seconds(walltime(seconds)) == seconds
