import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pytest

from mainboard.cli import build
from mainboard.core.project import Project
from mainboard.dispatch import HostSetup
from mainboard.dispatch.dispatcher import Dispatcher
from mainboard.dispatch.schedulers import HostUnreachable
from mainboard.dispatch.state import Cache, RunRecord
from mainboard.dispatch.vocabulary import JobState
from mainboard.jobs.beacon import Progress
from mainboard.pulse import Pulse, Pulses

if TYPE_CHECKING:
    from pathlib import Path

    from mainboard.dispatch import Handle

    from .support import Relayed

_FIELD_VALUE_HEADER = "field\tvalue"

# The cluster the fixture manifest declares a root for, so a live run on it can be rebuilt and
# asked about without any host being contacted.
_CLUSTER = "miyabi-g"

# One settled run as the jobs table projects it, the row every case below is a variation on. A
# settled row carries no live columns: what it is doing now is that it has ended.
_ROW = {
    "state": "ok",
    "host": "gold",
    "name": "train",
    "handle": "H1",
    "since": "",
    "starts": "",
    "submitted_at": "2026-08-01T00:00:00",
    "cause": "",
    "cells": "",
    "quiet_s": None,
    "gpu_pct": None,
}


def seed_run(
    handle: str = "H1",
    submitted_at: str = "2026-08-01T00:00:00",
    *,
    target: str = "gold",
    kind: str = "ssh",
    verdict: str | None = "ok",
) -> None:
    """Record one dispatched run in the shared cache, the way a submit would have.

    verdict: the settled word the cache memoized, None for a run still in flight.
    """
    Cache().record(
        RunRecord(
            handle=handle,
            target=target,
            kind=kind,
            script="job.sh",
            args="",
            git_sha="abc1234",
            dirty=0,
            submitted_at=submitted_at,
            name="train",
            state="F" if verdict else "Q",
            verdict=verdict,
        )
    )


def answering(monkeypatch: pytest.MonkeyPatch, answer: JobState | None) -> list[list[str]]:
    """Pin the batched scheduler probe to `answer`, one entry per round trip a host took.

    The seam is the batched probe, which is what says both that the live runs were resolved
    against their host and that the host was asked once rather than once per run. `None` is the
    host that will not answer at all.
    """
    trips: list[list[str]] = []

    def states(self: Dispatcher, handles: Sequence[Handle]) -> dict[str, JobState]:
        trips.append([handle.id for handle in handles])
        if answer is None:
            raise HostUnreachable("ssh: connect to host miyabi-g port 22: no route")
        return {handle.id: answer.model_copy(update={"handle": handle.id}) for handle in handles}

    monkeypatch.setattr(Dispatcher, "states", states)
    return trips


def onboarded(host: str = "gold") -> HostSetup:
    """What onboarding recorded for a host, the row the hosts table reads back."""
    return HostSetup(
        host=host,
        root="/repo",
        env="default",
        activate="/repo/.mainboard/activate.sh",
        installer="uv",
        rejected=(("pip", "reported unavailable"),),
        tool="0.1.0",
        onboarded_at="2026-08-17T00:00:00+00:00",
    )


@pytest.mark.parametrize(
    ("flags", "fragments"),
    [
        (["--json", "--fields", "hostname,schema_version"], ()),
        ([], ("hostname", "facts")),
        (["--agent"], (_FIELD_VALUE_HEADER, "hostname")),
    ],
    ids=["a projection over the probed fields", "the default rich table", "the compact record"],
)
def test_the_facts_verb_prints_this_machines_own_probe(
    depot: Path, capsys: pytest.CaptureFixture[str], flags: list[str], fragments: tuple[str, ...]
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["facts", *flags])
    out = capsys.readouterr().out
    if not fragments:
        payload = json.loads(out)
        assert set(payload) == {"hostname", "schema_version"}
        assert payload["schema_version"] >= 1
        return
    assert all(fragment in out for fragment in fragments)


@pytest.mark.parametrize(
    ("flags", "fragments"),
    [
        (["--json"], ()),
        ([], ("setup", "gold")),
    ],
    ids=["the record as json", "the default rich table"],
)
def test_the_setup_verb_shows_what_the_host_became(
    depot: Path,
    relayed: Sequence[Relayed],
    capsys: pytest.CaptureFixture[str],
    flags: list[str],
    fragments: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["setup", "gold", *flags])
    out = capsys.readouterr().out
    if not fragments:
        assert json.loads(out)["installer"] == "uv"
        return
    assert all(fragment in out for fragment in fragments)


def test_a_settled_failure_carries_what_it_said_on_the_way_out(
    depot: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A wave of identical `failed` rows says nothing, and the log that says it is already home.

    So the listing reads it: one line per failed row, off the tail the sweep pulled back beside
    that run's receipts, which is what thirty two GH200 jobs were missing on 2026-09-05.
    """
    seed_run("H9", verdict="failed")
    stored = depot / Project().out_dir / "batches" / "train" / "H9.log"
    stored.parent.mkdir(parents=True, exist_ok=True)
    stored.write_text("Traceback:\n  frame\nRuntimeError: the gate failed\nexit=1\n")

    with pytest.raises(SystemExit, match="0"):
        build(depot)(["jobs", "--json"])

    [row] = json.loads(capsys.readouterr().out)
    assert row["state"] == "failed"
    assert row["cause"] == "RuntimeError: the gate failed"


@pytest.mark.parametrize(
    ("seeded", "flags", "expected"),
    [
        (["H1"], [], [_ROW]),
        (["H1"], ["--fields", "handle,state"], [{"handle": "H1", "state": "ok"}]),
        (
            ["H1", "H2"],
            ["--limit", "1"],
            [{**_ROW, "handle": "H2", "submitted_at": "2026-08-02T00:00:00"}],
        ),
        ([], [], []),
    ],
    ids=[
        "every projected field of one run",
        "a projection over two of them",
        "the newest settled run only, under the limit",
        "a cache nobody has dispatched from yet",
    ],
)
def test_the_jobs_verb_lists_settled_runs_newest_first(
    depot: Path,
    capsys: pytest.CaptureFixture[str],
    seeded: Sequence[str],
    flags: list[str],
    expected: list[dict[str, str]],
) -> None:
    for index, handle in enumerate(seeded):
        seed_run(handle, submitted_at=f"2026-08-0{index + 1}T00:00:00")
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["jobs", "--json", *flags])
    assert json.loads(capsys.readouterr().out) == expected


def test_a_live_wave_is_shown_whole_and_its_host_asked_once_for_all_of_it(
    depot: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fault this verb was fixed for: twenty rows of a thirty five job wave, state blank.

    The live runs are shown whatever the limit says, they carry what the scheduler says right
    now rather than what the cache last memoized, and the whole host is asked in one query.
    """
    for index in range(3):
        seed_run(
            f"L{index}",
            submitted_at=f"2026-09-0{index + 1}T00:00:00",
            target=_CLUSTER,
            kind="pbs",
            verdict=None,
        )
    trips = answering(
        monkeypatch,
        JobState(
            handle="",
            state="Q",
            verdict="running",
            stage="queued",
            since="2026-09-04T05:00:00+00:00",
            estimated_start="2026-09-04T06:12:00+00:00",
        ),
    )
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["jobs", "--json", "--limit", "1"])
    listed = json.loads(capsys.readouterr().out)
    assert [row["handle"] for row in listed] == ["L2", "L1", "L0"]
    assert {row["state"] for row in listed} == {"queued"}
    assert listed[0]["since"] == "2026-09-04T05:00:00+00:00"
    assert listed[0]["starts"] == "2026-09-04T06:12:00+00:00"
    assert trips == [["L2", "L1", "L0"]]


def test_a_running_job_shows_its_cells_its_silence_and_its_busiest_card(
    depot: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The columns agents wrote their own loops for, and only a running job is looked at.

    A queued job has printed nothing its log could tell from a hang, so it is not read at all.
    """
    for handle in ("R1", "Q1"):
        seed_run(handle, target=_CLUSTER, kind="pbs", verdict=None)
    running = JobState(handle="", state="R", verdict="running", stage="running")
    queued = JobState(handle="", state="Q", verdict="running", stage="queued")

    def states(self: Dispatcher, handles: Sequence[Handle]) -> dict[str, JobState]:
        answers = {"R1": running, "Q1": queued}
        return {
            handle.id: answers[handle.id].model_copy(update={"handle": handle.id})
            for handle in handles
        }

    looked: list[list[str]] = []

    def taken(self: Pulses, records: Sequence[RunRecord]) -> dict[RunRecord, Pulse]:
        looked.append([record.handle for record in records])
        progress = Progress(total=4, cells=(("a", "passed"), ("b", "failed")))
        return {
            records[0]: Pulse(
                handle="R1", target=_CLUSTER, progress=progress, quiet_s=42, gpu_pct=97
            )
        }

    monkeypatch.setattr(Dispatcher, "states", states)
    monkeypatch.setattr(Pulses, "taken", taken)
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["jobs", "--json", "--fields", "handle,cells,quiet_s,gpu_pct"])
    listed = {row["handle"]: row for row in json.loads(capsys.readouterr().out)}
    assert looked == [["R1"]]
    assert listed["R1"] == {
        "handle": "R1",
        "cells": "2/4 (1 failed)",
        "quiet_s": 42,
        "gpu_pct": 97,
    }
    assert listed["Q1"] == {"handle": "Q1", "cells": "", "quiet_s": None, "gpu_pct": None}


def test_a_live_job_its_queue_has_finished_is_named_rather_than_spelled_with_a_letter(
    depot: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PBS job that finished clean showed as `F` beside `queued` and was read as failed.

    The sweep has not settled it yet, which is a real and nameable moment, so the column says so
    instead of handing the reader a backend's own letter to interpret (handle 3294174).
    """
    seed_run(
        "3294174", submitted_at="2026-09-04T00:00:00", target=_CLUSTER, kind="pbs", verdict=None
    )
    answering(monkeypatch, JobState(handle="", state="F", exit_code=0, verdict="ok"))
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["jobs", "--json"])
    [listed] = json.loads(capsys.readouterr().out)
    assert listed["state"] == "finished"


def test_a_listing_that_leaves_runs_out_says_so_instead_of_stopping_quietly(
    depot: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for index in range(3):
        seed_run(f"H{index}", submitted_at=f"2026-08-0{index + 1}T00:00:00")
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["jobs", "--json", "--limit", "2"])
    printed = capsys.readouterr()
    assert len(json.loads(printed.out)) == 2
    assert "showing 2 of 3 runs" in printed.err


def test_a_host_that_will_not_answer_costs_its_runs_their_live_state_and_nothing_else(
    depot: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run whose host went quiet still has a row, and the silence is named rather than shown.

    The alternative reads as a job that stopped moving, which is the one thing this table must
    never imply about a host problem.
    """
    seed_run("L0", target=_CLUSTER, kind="pbs", verdict=None)
    answering(monkeypatch, None)
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["jobs", "--json"])
    printed = capsys.readouterr()
    assert json.loads(printed.out)[0]["state"] == "Q"
    assert "miyabi-g did not answer: ssh: connect" in printed.err


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], ("H1", "gold", "jobs")),
        (
            ["--agent"],
            ("state\thost\tname\thandle\tcells\tquiet_s\tgpu_pct\tsince\tstarts", "H1"),
        ),
    ],
    ids=["the default rich table", "the compact table"],
)
def test_the_jobs_verb_tables_what_it_listed(
    depot: Path, capsys: pytest.CaptureFixture[str], flags: list[str], expected: Sequence[str]
) -> None:
    seed_run()
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["jobs", *flags])
    out = capsys.readouterr().out
    assert all(fragment in out for fragment in expected)
