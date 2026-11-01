"""A job's frames on its host: a start, beats of the command's tree and a settle, in the runner."""

import os
import signal
import sys
import threading
from pathlib import Path
from time import monotonic, sleep

import psutil
import pytest

from mainboard.constants import FRAMES_MARK
from mainboard.core.events import Event, EventSpool, JobFrame, JobReading, Key, Stop
from mainboard.verbs.work.runner import Runner

from .builders import environment, job_in, spawned
from .constants import CANARY, HOST

_COUNTERS = {JobReading.wall_s, JobReading.cpu_s, JobReading.rss_bytes, JobReading.processes}


def _frames_in(directory: Path) -> list[Event]:
    """The events of every spool file in `directory`."""
    return [event for path in directory.glob("*.jsonl") for event in EventSpool.read(path).events]


def _awaited(directory: Path, *, count: int, seconds: float = 20.0) -> None:
    """Wait until a runner still writing into `directory` has written `count` events."""
    deadline = monotonic() + seconds
    while monotonic() < deadline and len(_frames_in(directory)) < count:
        sleep(0.05)
    assert len(_frames_in(directory)) >= count, f"fewer than {count} frames in {seconds:g}s"


def test_a_local_job_frames_its_start_then_beats_then_its_settle(tmp_path: Path) -> None:
    runner = Runner(job_in(tmp_path, "sleep 0.7; exit 3"), environ=environment())
    runner.frames.interval = 0.1
    status = runner.run()
    names = [event.name for event in _frames_in(tmp_path / "frames")]
    assert (status, names[0], names[-1]) == (3, JobFrame.started, JobFrame.settled)
    assert names.count(JobFrame.heartbeat) >= 2
    assert set(names[1:-1]) == {JobFrame.heartbeat}


def test_a_frame_carries_the_host_and_counters_and_nothing_of_the_environment(
    tmp_path: Path,
) -> None:
    runner = Runner(job_in(tmp_path, "sleep 0.4"), environ=environment())
    runner.frames.interval = 0.1
    runner.run()
    (path,) = (tmp_path / "frames").glob("*.jsonl")
    started, beat, *_, settled = _frames_in(tmp_path / "frames")
    assert started.attributes == {Key.host: HOST}
    assert set(beat.attributes) == {Key.host, *_COUNTERS}
    assert settled.attributes[JobReading.exit_code] == 0
    assert (CANARY not in path.read_text(), path.stat().st_mode & 0o777) == (True, 0o600)


def test_the_job_output_names_the_spool_file_the_center_will_read(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    Runner(job_in(tmp_path, "true"), environ=environment()).run()
    (path,) = (tmp_path / "frames").glob("*.jsonl")
    assert f"{FRAMES_MARK}{path.name}\n" in capfd.readouterr().out


@pytest.mark.parametrize(
    ("command", "walltime", "prefix", "stop"),
    [
        pytest.param(
            "sleep 30", "00:00:01", sys.prefix, Stop.walltime, id="killed-at-its-walltime"
        ),
        pytest.param("true", "", "/nonexistent", Stop.refused, id="an-environment-that-refuses"),
        pytest.param("exit 5", "", sys.prefix, Stop.exited, id="a-command-that-fails-on-its-own"),
    ],
)
def test_the_settle_says_why_the_job_ended_and_with_what_status(
    tmp_path: Path, command: str, walltime: str, prefix: str, stop: Stop
) -> None:
    job = job_in(tmp_path, command, walltime=walltime, prefix=prefix)
    status = Runner(job, environ=environment(), grace=0.5).run()
    settled = _frames_in(tmp_path / "frames")[-1]
    assert settled.attributes[JobReading.stop] == stop
    assert settled.attributes[JobReading.exit_code] == status


def test_a_signal_to_the_runner_settles_the_job_as_signaled(tmp_path: Path) -> None:
    runner = Runner(job_in(tmp_path, "sleep 30"), environ=environment(), grace=0.5)
    threading.Timer(0.5, os.kill, args=(os.getpid(), signal.SIGTERM)).start()
    status = runner.run()
    settled = _frames_in(tmp_path / "frames")[-1]
    assert (status, settled.attributes[JobReading.stop]) == (143, Stop.signaled)


def test_a_job_killed_outright_leaves_its_start_and_beats_and_no_settle(tmp_path: Path) -> None:
    process = spawned(job_in(tmp_path, "sleep 30"))
    _awaited(tmp_path / "frames", count=3)
    orphans = psutil.Process(process.pid).children(recursive=True)
    process.kill()
    process.wait()
    for orphan in orphans:
        orphan.kill()
    names = {event.name for event in _frames_in(tmp_path / "frames")}
    assert names == {JobFrame.started, JobFrame.heartbeat}


def test_a_job_dispatched_without_a_frames_folder_runs_and_names_no_spool_file(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    job = job_in(tmp_path, "exit 4").model_copy(update={"frames": None})
    status = Runner(job, environ=environment()).run()
    assert (status, FRAMES_MARK in capfd.readouterr().out, (tmp_path / "frames").exists()) == (
        4,
        False,
        False,
    )


def test_a_frames_folder_that_cannot_be_made_costs_the_job_nothing(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "file").write_text("not a folder", encoding="utf-8")
    job = job_in(tmp_path, "exit 4", directory=tmp_path / "file")
    status = Runner(job, environ=environment()).run()
    assert (status, FRAMES_MARK in capfd.readouterr().out) == (4, False)
