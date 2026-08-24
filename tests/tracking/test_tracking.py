import sys
from typing import TYPE_CHECKING

import pytest
from pydantic import JsonValue

from mainboard import MissionError
from mainboard.batch import Event, Mirrored, Receipts, Topic
from mainboard.manifest import Tracking, TrackingMode
from mainboard.tracking import Tracker, credential, is_batched, mirrored, sink, streamed
from mainboard.tracking.wandb import WandbSink, exit_code, module

from ..batch.support import Recorder
from .support import FakeWandb, keyed

if TYPE_CHECKING:
    from pathlib import Path

_STREAM = "smoke-1"
_JOB = "trial-a"


def opened(tmp_path: Path, declared: Tracking | None = None, *, workspace: str = "") -> WandbSink:
    """A sink over one stream, at whatever the caller declared."""
    return WandbSink(
        _STREAM,
        declared=declared or Tracking(),
        directory=tmp_path,
        workspace=workspace,
    )


def event(topic: Topic, **data: JsonValue) -> Event:
    """One receipt of `topic` about this module's job, carrying `data`."""
    return Event(at="2026-08-22T00:00:00Z", batch=_STREAM, topic=topic, job=_JOB, data=data)


def test_a_workspace_that_declares_nothing_is_tracked_and_one_word_turns_it_off() -> None:
    """The table exists to tune the lane or stop it, never to switch it on."""
    default = Tracking()
    assert (default.provider, default.mode, default.on) == ("wandb", TrackingMode.ONLINE, True)
    assert default.interval > 0
    assert Tracking(mode=TrackingMode.OFF).on is False
    assert Tracking(provider="").on is False


def test_a_declared_sink_mirrors_the_file_and_an_undeclared_one_leaves_it_alone(
    tmp_path: Path, service: FakeWandb, declared: Tracking
) -> None:
    canonical = Receipts(tmp_path / "events.ndjson")
    assert (
        mirrored(canonical, Tracking(mode=TrackingMode.OFF), stream=_STREAM, directory=tmp_path)
        is canonical
    )
    composed = mirrored(canonical, declared, stream=_STREAM, directory=tmp_path)
    assert isinstance(composed, Mirrored) and composed.canonical is canonical
    assert isinstance(composed.mirrors[0], WandbSink)
    assert service.runs == []


def test_a_provider_nothing_registered_is_refused_with_the_roster(tmp_path: Path) -> None:
    with pytest.raises(MissionError, match="no tracking provider 'graphs'"):
        sink("s", declared=Tracking(provider="graphs"), directory=tmp_path)
    assert credential(Tracking()) == "WANDB_API_KEY"
    assert credential(Tracking(mode=TrackingMode.OFF)) == ""


@pytest.mark.parametrize(
    ("name", "handle", "expected"),
    [
        ("batch:smoke-1/gold-1", "7", ("smoke-1", "gold-1")),
        ("batch:smoke-1", "7", ("smoke-1", "smoke-1")),
        ("study:abc/trial-3", "7", ("abc", "trial-3")),
        ("study:abc", "7", ("abc", "abc")),
        ("nightly", "7", ("nightly", "nightly")),
        ("", "7", ("run-7", "7")),
    ],
    ids=[
        "a batch job",
        "a whole batch",
        "a trial",
        "a whole study",
        "a named run",
        "an unnamed run",
    ],
)
def test_every_dispatch_label_routes_to_the_stream_and_job_it_names(
    name: str, handle: str, expected: tuple[str, str]
) -> None:
    """One router, so a plain submit and a study trial reach the run a batch job reaches."""
    assert streamed(name, handle=handle) == expected


def test_only_a_batch_publishes_for_a_batch_job_though_the_job_still_samples_itself() -> None:
    """Its own flow writes every receipt about it, so a second publisher would double each row."""
    assert is_batched("batch:smoke-1/gold-1") is True
    assert is_batched("study:abc/trial-3") is False
    assert is_batched("nightly") is False


def test_a_sink_is_where_events_go_and_never_where_they_come_from(tmp_path: Path) -> None:
    assert opened(tmp_path).replay() == []
    assert issubclass(WandbSink, Tracker)


def test_the_mapping_turns_one_stream_of_receipts_into_one_run_per_job(
    tmp_path: Path, service: FakeWandb, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identity as config, scalars as history, cost on the summary, terminal closing the run."""
    keyed(monkeypatch, key="secret")
    tracked = opened(tmp_path, workspace="lab")
    tracked.publish(
        Event(at="t", batch=_STREAM, topic=Topic.OPENED, data={"name": "smoke", "root": "/repo"})
    )
    tracked.publish(event(Topic.PREPARED, paths=["packages"], files=2, raw_bytes=9))
    tracked.publish(event(Topic.SUBMITTED, handle="77", target="gold", kind="ssh", command="go"))
    tracked.publish(event(Topic.SAMPLE, gpu_used_gb=1.5, host_frac=0.25))
    tracked.publish(event(Topic.COST, platform="gold", setup_s=3.0, observed=True))
    tracked.publish(event(Topic.SETTLED, handle="77", verdict="ok", detail="results/run"))

    [run] = service.runs
    assert run.options["group"] == _STREAM
    assert run.options["name"] == _JOB
    assert run.options["project"] == "lab"
    assert run.options["mode"] == TrackingMode.ONLINE
    assert run.options["resume"] == "allow"
    assert run.options["id"] == run.options["config"]["run_id"]
    assert run.options["config"]["name"] == "smoke"
    assert run.config.held["handle"] == "77" and run.config.options == {"allow_val_change": True}
    steps = [step for step, _ in run.history]
    assert steps == [0, 1, 2, 3, 4]
    assert run.history[0][1] == {"prepared/files": 2, "prepared/raw_bytes": 9}
    assert run.history[2][1] == {"sample/gpu_used_gb": 1.5, "sample/host_frac": 0.25}
    assert run.summary.held["cost/setup_s"] == 3.0
    assert run.summary.held["settled/verdict"] == "ok"
    assert (run.finished, run.exit_code) == (True, 0)


def test_a_refused_job_closes_its_run_rather_than_leaving_it_open(
    tmp_path: Path, service: FakeWandb, monkeypatch: pytest.MonkeyPatch
) -> None:
    keyed(monkeypatch)
    tracked = opened(tmp_path)
    tracked.publish(event(Topic.REFUSED, target="gold", reason="asleep"))
    [run] = service.runs
    assert (run.finished, run.exit_code) == (True, 1)
    assert run.summary.held["refused/reason"] == "asleep"


def test_a_second_process_continues_the_series_instead_of_rewriting_its_beginning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run id is ours, so the sweep that settles a job resumes the run the dispatch opened."""
    keyed(monkeypatch)
    resumed = FakeWandb(resume_at=12)
    monkeypatch.setitem(sys.modules, "wandb", resumed)
    tracked = opened(tmp_path)
    tracked.publish(event(Topic.STATE, handle="77", state="R", verdict="running"))
    tracked.publish(event(Topic.STATE, handle="77", state="C", verdict="ok"))
    [run] = resumed.runs
    assert [step for step, _ in run.history] == [12, 13]

    again = opened(tmp_path)
    again.publish(event(Topic.STATE, handle="77", state="C", verdict="ok"))
    assert len(resumed.runs) == 2
    assert resumed.runs[0].options["id"] == resumed.runs[1].options["id"]


@pytest.mark.parametrize(
    ("mode", "key", "expected"),
    [
        (TrackingMode.OFFLINE, "secret", TrackingMode.OFFLINE),
        (TrackingMode.ONLINE, "secret", TrackingMode.ONLINE),
        (TrackingMode.ONLINE, "", TrackingMode.OFFLINE),
    ],
    ids=["offline stays offline", "online with a key", "online with no key queues offline"],
)
def test_a_machine_with_no_key_queues_offline_instead_of_blocking_on_a_login(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: TrackingMode,
    key: str,
    expected: TrackingMode,
) -> None:
    """A dispatched job could never answer a login prompt, so it is never shown one."""
    keyed(monkeypatch, key=key)
    assert opened(tmp_path, Tracking(mode=mode)).mode == expected


@pytest.mark.parametrize(
    ("declared_project", "workspace", "expected"),
    [("runs", "lab", "runs"), ("", "lab", "lab"), ("", "", _STREAM)],
    ids=["what the manifest declared", "this workspace", "the stream itself"],
)
def test_the_project_falls_back_from_the_manifest_to_the_workspace_to_the_stream(
    tmp_path: Path, declared_project: str, workspace: str, expected: str
) -> None:
    tracked = opened(tmp_path, Tracking(project=declared_project), workspace=workspace)
    assert tracked.project == expected


@pytest.mark.parametrize(
    ("topic", "data", "expected"),
    [
        (Topic.SETTLED, {"verdict": "ok"}, 0),
        (Topic.SETTLED, {"verdict": "failed"}, 1),
        (Topic.SETTLED, {"verdict": "failed", "exit_code": 137}, 137),
        (Topic.REFUSED, {"reason": "asleep"}, 1),
    ],
    ids=["a clean end", "a failure", "the code the receipt carried", "a refusal"],
)
def test_a_closing_receipt_ends_the_run_at_what_it_actually_says(
    topic: Topic, data: dict[str, object], expected: int
) -> None:
    assert exit_code(event(topic, **data)) == expected


def test_the_sink_refuses_cleanly_when_the_package_is_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seal every other test runs under is also the machine that never installed it."""
    keyed(monkeypatch)
    with pytest.raises(MissionError, match="mainboard add wandb"):
        module()
    bus = Recorder()
    composed = Mirrored(bus, opened(tmp_path))
    composed.publish(event(Topic.STATE, verdict="running"))
    assert [line.topic for line in composed.replay()] == [Topic.STATE]
