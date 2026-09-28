from collections.abc import Sequence
from time import monotonic, sleep
from types import TracebackType
from typing import TYPE_CHECKING, NamedTuple

import pytest

from mainboard import Board, ExecutionPlan, Job
from mainboard.batch import Topic
from mainboard.batch.receipts import Journal
from mainboard.cli import build
from mainboard.dispatch import Handle
from mainboard.dispatch.state import Cache, RunRecord
from mainboard.dispatch.vocabulary import JobState, Resources
from mainboard.manifest import Tracking
from mainboard.runtime.job import ToolCall

if TYPE_CHECKING:
    from pathlib import Path

_HOST = "miyabi-g"


class Asked(NamedTuple):
    """What one submit asked the dispatcher's run for."""

    name: str
    sampler: ToolCall | None
    root: str


class FakeRemote:
    """An ssh connection that answers every command with nothing."""

    def __init__(self, command: str = "") -> None:
        self.command = command

    def __call__(self) -> str:
        return ""

    def __enter__(self) -> FakeRemote:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        fault: BaseException | None,
        trace: TracebackType | None,
    ) -> bool:
        return False

    def __getitem__(self, argv: str | tuple[str, ...]) -> FakeRemote:
        return FakeRemote(argv if isinstance(argv, str) else " ".join(argv))

    def close(self) -> None:
        return


def tracking(board: Board, **fields: float) -> Board:
    """This workspace with its live lane at whatever the caller declared."""
    board.shared["manifest"] = board.manifest.model_copy(update={"tracking": Tracking(**fields)})
    return board


def test_every_stream_publishes_into_the_workspace_lake(board: Board) -> None:
    """The composition root, so no flow has to know where its receipts are kept."""
    bus = board.receipts("plain")
    assert isinstance(bus, Journal) and bus.batch == "plain"


def test_a_sampler_takes_the_interval_the_manifest_declared(board: Board) -> None:
    tuned = tracking(board, interval=42.0)
    assert tuned.samples("s", job="j").interval == 42.0
    assert tuned.samples("s", job="j", interval=1.5).interval == 1.5


def submitting(board: Board, monkeypatch: pytest.MonkeyPatch) -> list[Asked]:
    """Pin the dispatcher's own run to a stand-in, recording what each submit asked it for."""
    asked: list[Asked] = []

    def fake_run(
        plan: ExecutionPlan,
        cmd: str,
        *,
        root: str,
        name: str = "",
        sampler: ToolCall | None = None,
        **rest: str | int | float | bool,
    ) -> Handle:
        asked.append(Asked(name=name, sampler=sampler, root=root))
        return Handle(id="77", host=plan.host, root=root, kind=plan.profile.kind)

    monkeypatch.setattr(board.dispatcher, "run", fake_run)
    return asked


def test_a_run_that_named_itself_nothing_is_named_here_so_its_stream_has_a_key(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep on another day settles the run, so the key has to outlive this process; and a
    submission is recorded whether or not anything samples it."""
    asked = submitting(tracking(board, interval=0.0), monkeypatch)
    board.on(_HOST).submit("python train.py")
    [seen] = asked
    assert seen.name.startswith(f"{_HOST}-") and seen.sampler is None
    stream = seen.name
    [line] = board.receipts(stream).replay()
    assert (line.topic, line.job, line.data["handle"]) == (Topic.SUBMITTED, stream, "77")
    assert line.data["target"] == _HOST and line.data["command"] == "python train.py"
    # The node field is optional both ways: absent when nothing declared one, on the line and
    # in the run registry when the dispatch did.
    assert "node" not in line.data
    board.on(_HOST).submit("python train.py", name="noded", node="tax-law")
    [noded] = board.receipts("noded").replay()
    assert noded.data["node"] == "tax-law"


def test_a_batch_job_still_watches_itself_though_only_the_batch_publishes_for_it(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The node says the one thing only it can say, and the batch says everything else."""
    asked = submitting(tracking(board, interval=20.0), monkeypatch)
    board.on(_HOST).submit("echo hi", name="batch:smoke-1/gold-1")
    sampler = asked[0].sampler
    assert sampler is not None and sampler.args[:4] == ("sample", "smoke-1", "--job", "gold-1")
    assert board.receipts("smoke-1").replay() == []


def test_a_dispatched_job_is_handed_the_line_that_makes_it_watch_itself(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seam that carries the live lane onto a host."""
    asked = submitting(tracking(board, interval=15.0), monkeypatch)
    monkeypatch.setattr(
        "mainboard.dispatch.shells.connection", lambda host, ssh=None: FakeRemote()
    )
    board.on(_HOST).submit("python train.py", walltime="01:00:00")
    stream = asked[0].name
    assert asked[0].sampler == ToolCall(
        args=("sample", stream, "--job", stream, "--interval", "15", "--seconds", "3600")
    )


def test_nothing_is_sampled_or_attested_when_the_interval_is_zero(board: Board) -> None:
    quiet = tracking(board, interval=0.0)
    assert quiet.sampling(("s", "j"), resources=Resources()) is None
    assert quiet.attesting(("s", "j")) is None


def swept(board: Board, monkeypatch: pytest.MonkeyPatch, verdict: str) -> None:
    """Pin the sweep's batched probe so every tracked handle answers with `verdict`."""

    def states(handles: Sequence[Handle]) -> dict[str, JobState]:
        return {
            handle.id: JobState(handle=handle.id, state="F", exit_code=0, verdict=verdict)
            for handle in handles
        }

    monkeypatch.setattr(board.dispatcher, "states", states)


def recorded(handle: str, name: str) -> None:
    """Record one dispatched run in the shared cache under `name`."""
    Cache().record(
        RunRecord(
            handle=handle,
            target=_HOST,
            kind="pbs",
            script="job.sh",
            args="",
            git_sha="abc1234",
            dirty=0,
            submitted_at="2026-08-22T00:00:00",
            name=name,
        )
    )


def test_the_durable_sweep_publishes_for_every_run_a_batch_does_not_already_own(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What makes a plain submit and a study trial as recorded as a batch job, sampled or not."""
    recorded("81", "nightly")
    recorded("82", "batch:smoke-1")
    quiet = tracking(board, interval=0.0)
    swept(quiet, monkeypatch, "running")
    quiet.monitor().once()
    assert [line.topic for line in board.receipts("nightly").replay()] == [Topic.STATE]
    assert board.receipts("smoke-1").replay() == []


def test_a_quiet_sweep_writes_nothing_and_a_terminal_one_writes_the_last_line(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This runs on a cron, so an unchanged run must cost the stream no line at all."""
    recorded("83", "nightly-two")
    swept(board, monkeypatch, "running")
    board.monitor().once()
    board.monitor().once()
    stream = board.receipts("nightly-two")
    assert [line.topic for line in stream.replay()] == [Topic.STATE]

    fresh = Board(board.root)
    swept(fresh, monkeypatch, "ok")
    monkeypatch.setattr(fresh.dispatcher, "fetch", lambda handle, **kw: None)
    # The settled run left no output to capture, and reading it would dial the real host.
    monkeypatch.setattr(Job, "transcript", lambda job: "")
    fresh.monitor().once()
    published = stream.replay()
    assert [line.topic for line in published] == [
        Topic.STATE,
        Topic.EVIDENCE,
        Topic.EVIDENCE,
        Topic.STATE,
        Topic.SETTLED,
    ]
    assert [line.data["status"] for line in published if line.topic == Topic.EVIDENCE] == [
        "copied",
        "verified",
    ]
    assert published[-1].data["verdict"] == "ok"
    fresh.monitor().once()
    assert stream.replay() == published


def test_the_sample_verb_watches_this_machine_until_it_is_told_to_stop(
    depot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The verb a job script calls, and the one somebody runs by hand beside a long job."""
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["sample", "live-1", "--interval", "0.01", "--seconds", "0.05"])
    published = Board(depot).receipts("live-1").replay()
    assert published and {line.topic for line in published} == {Topic.SAMPLE}
    assert published[0].job == "live-1"
    reading = published[0].data
    assert {"gpu_used_gb", "host_used_gb", "host_cap_gb", "host_frac"} <= set(reading)


def test_the_loop_keeps_reading_until_its_budget_runs_out(depot: Path) -> None:
    """More than the first reading, which is what a series watched live actually needs."""
    sampler = Board(depot).samples("live-2", job="j", interval=0.005, seconds=10.0)
    with sampler:
        deadline = monotonic() + 10.0
        while len(sampler.bus.replay()) < 3 and monotonic() < deadline:
            sleep(0.01)
    assert len(sampler.bus.replay()) >= 3
