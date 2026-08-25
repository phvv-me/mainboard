import subprocess
import sys
from collections.abc import Sequence
from time import monotonic, sleep
from types import TracebackType
from typing import TYPE_CHECKING

import pytest

from mainboard import Board, ExecutionPlan
from mainboard.batch import Mirrored, Receipts, Topic
from mainboard.batch.runner import directory
from mainboard.cli import build
from mainboard.dispatch import Handle
from mainboard.dispatch.state import Cache, RunRecord
from mainboard.dispatch.vocabulary import JobState, Resources
from mainboard.manifest import Tracking

from .support import FakeWandb, keyed

if TYPE_CHECKING:
    from pathlib import Path

_HOST = "miyabi-g"
_ROOT = "/work/p"


class FakeRemote:
    """The one ssh connection a credential staging opens, keeping what it was piped."""

    piped: list[tuple[str, str]] = []

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

    def __lshift__(self, text: str) -> FakeRemote:
        FakeRemote.piped.append((self.command, text))
        return self


def tracking(board: Board, **fields: str | float) -> Board:
    """Turn this workspace's tracking lane on, at whatever the caller declared."""
    board.shared["manifest"] = board.manifest.model_copy(update={"tracking": Tracking(**fields)})
    return board


def test_a_stream_writes_its_own_file_and_mirrors_it_only_when_the_workspace_asked(
    board: Board, service: FakeWandb
) -> None:
    """The composition root, so no flow has to know that a reporting service exists."""
    assert isinstance(board.receipts("plain"), Receipts)
    composed = tracking(board).receipts("mirrored")
    assert isinstance(composed, Mirrored)
    assert isinstance(composed.canonical, Receipts)
    assert composed.canonical.path == directory(board, "mirrored") / "events.ndjson"


def test_a_sampler_takes_the_interval_the_manifest_declared(board: Board) -> None:
    tuned = tracking(board, interval=42.0)
    assert tuned.samples("s", job="j").interval == 42.0
    assert tuned.samples("s", job="j", interval=1.5).interval == 1.5


def submitting(board: Board, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str]]:
    """Pin the dispatcher's own run to a stand-in, recording what each submit asked it for."""
    asked: list[dict[str, str]] = []

    def fake_run(
        plan: ExecutionPlan,
        cmd: str,
        *,
        root: str,
        name: str = "",
        sampler: str = "",
        **rest: str | int | float | bool,
    ) -> Handle:
        asked.append({"name": name, "sampler": sampler, "root": root})
        return Handle(id="77", host=plan.host, root=root, kind=plan.profile.kind)

    monkeypatch.setattr(board.dispatcher, "run", fake_run)
    return asked


def test_a_run_that_named_itself_nothing_is_named_here_so_its_stream_has_a_key(
    board: Board, monkeypatch: pytest.MonkeyPatch, service: FakeWandb
) -> None:
    """A sweep on another day settles the run, so the key has to outlive this process."""
    keyed(monkeypatch)
    asked = submitting(tracking(board, interval=0.0), monkeypatch)
    monkeypatch.setattr("mainboard.board.connection", FakeRemote)
    board.on(_HOST).submit("python train.py")
    [seen] = asked
    assert seen["name"].startswith(f"{_HOST}-") and seen["sampler"] == ""
    stream = seen["name"]
    [line] = Receipts(directory(board, stream) / "events.ndjson").replay()
    assert (line.topic, line.job, line.data["handle"]) == (Topic.SUBMITTED, stream, "77")
    assert line.data["target"] == _HOST and line.data["command"] == "python train.py"
    # The node field is optional both ways: absent when nothing declared one, on the line and
    # in the run registry when the dispatch did.
    assert "node" not in line.data
    board.on(_HOST).submit("python train.py", name="noded", node="tax-law")
    [noded] = Receipts(directory(board, "noded") / "events.ndjson").replay()
    assert noded.data["node"] == "tax-law"


def test_a_batch_job_still_watches_itself_though_only_the_batch_publishes_for_it(
    board: Board, monkeypatch: pytest.MonkeyPatch, service: FakeWandb
) -> None:
    """The node says the one thing only it can say, and the batch says everything else."""
    keyed(monkeypatch)
    asked = submitting(tracking(board, interval=20.0), monkeypatch)
    board.on(_HOST).submit("echo hi", name="batch:smoke-1/gold-1")
    assert "mainboard sample smoke-1 --job gold-1" in asked[0]["sampler"]
    assert not (directory(board, "smoke-1") / "events.ndjson").exists()


def test_a_dispatched_job_is_handed_the_line_that_makes_it_watch_itself(
    board: Board, monkeypatch: pytest.MonkeyPatch, service: FakeWandb
) -> None:
    """The seam that carries the live lane onto a host, plus the one credential it needs there."""
    keyed(monkeypatch, key="secret")
    FakeRemote.piped = []
    asked = submitting(tracking(board, interval=15.0), monkeypatch)
    monkeypatch.setattr("mainboard.board.connection", FakeRemote)
    board.on(_HOST).submit("python train.py", walltime="01:00:00")
    line = asked[0]["sampler"]
    assert "mainboard sample" in line and "--interval 15 --seconds 3600" in line
    [(command, text)] = FakeRemote.piped
    assert "umask 077" in command and "tracking.env" in command
    assert text.startswith("WANDB_API_KEY=") and "secret" in text


def test_a_machine_holding_no_credential_stages_nothing_and_still_samples(
    board: Board, monkeypatch: pytest.MonkeyPatch, service: FakeWandb
) -> None:
    keyed(monkeypatch)
    FakeRemote.piped = []
    asked = submitting(tracking(board, interval=5.0), monkeypatch)
    monkeypatch.setattr("mainboard.board.connection", FakeRemote)
    board.on(_HOST).submit("python train.py")
    assert "mainboard sample" in asked[0]["sampler"]
    assert FakeRemote.piped == []


def test_staging_is_for_a_machine_that_is_not_this_one(
    board: Board, monkeypatch: pytest.MonkeyPatch, service: FakeWandb
) -> None:
    """This machine reads the workspace `.env` directly, so nothing is ever written for it."""
    keyed(monkeypatch, key="secret")
    FakeRemote.piped = []
    monkeypatch.setattr("mainboard.board.connection", FakeRemote)
    tracking(board).stage(_ROOT)
    assert FakeRemote.piped == []


def test_nothing_is_sampled_and_nothing_is_staged_when_the_lane_is_off(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeRemote.piped = []
    monkeypatch.setattr("mainboard.board.connection", FakeRemote)
    assert board.sampling(("s", "j"), root=_ROOT, resources=Resources()) == ""
    board.stage(_ROOT)
    assert FakeRemote.piped == []


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
    board: Board, monkeypatch: pytest.MonkeyPatch, service: FakeWandb
) -> None:
    """What makes a plain submit and a study trial as tracked as a batch job."""
    keyed(monkeypatch)
    recorded("81", "nightly")
    recorded("82", "batch:smoke-1")
    tracked = tracking(board)
    swept(tracked, monkeypatch, "running")
    tracked.monitor().once()
    stream = Receipts(directory(board, "nightly") / "events.ndjson")
    assert [line.topic for line in stream.replay()] == [Topic.STATE]
    assert not (directory(board, "smoke-1") / "events.ndjson").exists()


def test_a_quiet_sweep_writes_nothing_and_a_terminal_one_writes_the_last_line(
    board: Board, monkeypatch: pytest.MonkeyPatch, service: FakeWandb
) -> None:
    """This runs on a cron, so an unchanged run must cost the stream no line at all."""
    keyed(monkeypatch)
    recorded("83", "nightly-two")
    tracked = tracking(board)
    swept(tracked, monkeypatch, "running")
    tracked.monitor().once()
    tracked.monitor().once()
    stream = Receipts(directory(board, "nightly-two") / "events.ndjson")
    assert [line.topic for line in stream.replay()] == [Topic.STATE]

    fresh = tracking(Board(board.root))
    swept(fresh, monkeypatch, "ok")
    monkeypatch.setattr(fresh.dispatcher, "fetch", lambda handle, **kw: None)
    fresh.monitor().once()
    published = stream.replay()
    assert [line.topic for line in published] == [Topic.STATE, Topic.STATE, Topic.SETTLED]
    assert published[-1].data["verdict"] == "ok"


def test_a_sweep_on_a_workspace_that_tracks_nothing_publishes_nothing(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded("84", "nightly-three")
    swept(board, monkeypatch, "running")
    board.monitor().once()
    assert not (directory(board, "nightly-three") / "events.ndjson").exists()


def test_the_sample_verb_watches_this_machine_until_it_is_told_to_stop(
    depot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The verb a job script calls, and the one somebody runs by hand beside a long job."""
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["sample", "live-1", "--interval", "0.01", "--seconds", "0.05"])
    published = Receipts(depot / ".mainboard" / "batches" / "live-1" / "events.ndjson").replay()
    assert published and {line.topic for line in published} == {Topic.SAMPLE}
    assert published[0].job == "live-1"
    reading = published[0].data
    assert {"gpu_used_gb", "host_used_gb", "host_cap_gb", "host_frac"} <= set(reading)


def test_the_loop_keeps_reading_until_its_budget_runs_out(depot: Path) -> None:
    """More than the first reading, which is what a series watched live actually needs."""
    sampler = Board(depot).samples("live-2", job="j", interval=0.005, seconds=0.4)
    with sampler:
        deadline = monotonic() + 3.0
        while len(sampler.bus.replay()) < 3 and monotonic() < deadline:
            sleep(0.01)
    assert len(sampler.bus.replay()) >= 3


def test_a_fresh_process_minting_by_name_finds_the_wandb_sink() -> None:
    """Importing the tracking package alone registers every sink, which is the CLI's own path.

    Guards the registration seam: a sink joins the registry when its module loads, and the
    package initializer is what loads it, so `mainboard submit` in a fresh interpreter can mint
    the declared service by name. The suite's own imports register sinks as a side effect, so
    only a child interpreter proves the production path.
    """
    probing = "from mainboard.tracking import Tracker\nTracker.find('wandb')\n"
    proof = subprocess.run(
        [sys.executable, "-c", probing], capture_output=True, text=True, timeout=60, check=False
    )
    assert proof.returncode == 0, proof.stderr
