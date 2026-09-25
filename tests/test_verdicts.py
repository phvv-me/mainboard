import json
from typing import TYPE_CHECKING

import pytest
from filelock import FileLock

from mainboard import Board, Job, MissionError
from mainboard.batch.receipts import Event, Receipts, Topic, publish
from mainboard.batch.runner import directory
from mainboard.dispatch.state import Cache, RunRecord
from mainboard.monitor import Monitor
from mainboard.verdicts import (
    STALLED,
    StreamVerdict,
    TrialVerdict,
    Verdicts,
    gated,
    lined,
    qualified,
    stopped,
)
from mainboard.vigil import Linger, Look, Vigil

if TYPE_CHECKING:
    from pathlib import Path

_STREAM = "study-receipts"


def receipt(run: str, case: str, **fields: str) -> str:
    """One printed `trial_receipt` line for `case` of `run`, passed unless `fields` say not."""
    return json.dumps(
        {"trial_receipt": {"run": run, "case_id": case, "outcome": "passed", **fields}}
    )


@pytest.mark.parametrize("status", ["pending", "copied"])
def test_a_cached_success_is_not_delivered_before_its_first_evidence_event(
    board: Board, status: str
) -> None:
    recorded(board, "79", name="crashed-before-event", verdict="ok")
    record = board.dispatcher.cache.run("79")
    board.dispatcher.cache.delivery(record, status)
    assert board.verdicts().handled("79").code == 2
    assert board.verdicts().handled("79").trials[0].verdict == "blocked"
    under = directory(board, "crashed-before-event")
    under.mkdir(parents=True, exist_ok=True)
    (under / "receipts.ndjson").write_text(receipt("r", "case", verdict="validated") + "\n")
    assert board.verdicts().of("crashed-before-event").code == 2


def test_delivery_correction_preserves_claim_but_does_not_claim_verified_evidence(
    board: Board,
) -> None:
    recorded(board, "77", name="lost-transfer", verdict="ok")
    under = directory(board, "lost-transfer")
    under.mkdir(parents=True, exist_ok=True)
    path = under / "receipts.ndjson"
    original = "".join(
        receipt(run, "same-case", verdict="validated") + "\n"
        for run in ("first-run", "second-run")
    )
    path.write_text(original)
    publish(
        Receipts(under / "events.ndjson"),
        "lost-transfer",
        Topic.EVIDENCE,
        job="lost-transfer",
        data={
            "handle": "77",
            "target": "gold",
            "status": "unverified",
            "detail": "raw artifacts missing",
            "trials": [["first-run", "same-case"]],
        },
    )
    result = board.verdicts().handled("77")
    assert result.code == 3
    first = next(trial for trial in result.trials if trial.run)
    assert (first.verdict, first.settled) == ("unverified", "validated")
    assert {trial.run for trial in result.trials if trial.run} == {"first-run"}
    second = next(trial for trial in lined(path) if trial.run == "second-run")
    assert second.verdict == "passed"
    assert path.read_text() == original
    assert board.dispatcher.cache.run("77").verdict == "ok"


def recorded(
    board: Board,
    handle: str,
    *,
    name: str = "",
    verdict: str | None = None,
    target: str = "gold",
    commit: str = "",
    digest: str = "",
) -> None:
    """One dispatched run in the registry, the durable floor a verdict reads from."""
    board.dispatcher.cache.record(
        RunRecord(
            handle=handle,
            target=target,
            kind="ssh",
            script="job.sh",
            args="",
            git_sha="abc",
            dirty=0,
            submitted_at=f"2026-08-25T00:00:0{handle[-1]}+00:00",
            name=name,
            node="tax-law" if name else "",
            state="finished" if verdict else None,
            exit_code=0 if verdict == "ok" else None,
            verdict=verdict,
            commit=commit,
            digest=digest,
        )
    )


def published(board: Board, stream: str) -> None:
    """A stream holding one job of every outcome a batch can leave behind."""
    bus = Receipts(directory(board, stream) / "events.ndjson")
    lines: list[tuple[Topic, str, dict[str, str | int]]] = [
        (
            Topic.SUBMITTED,
            "a",
            {"handle": "1", "target": "gold", "command": "true", "node": "law"},
        ),
        (Topic.STATE, "a", {"handle": "1", "state": "F", "verdict": "ok"}),
        (
            Topic.SETTLED,
            "a",
            {"handle": "1", "verdict": "ok", "exit_code": 0, "detail": "results/run"},
        ),
        (Topic.SUBMITTED, "b", {"handle": "2", "target": "gold", "command": "false"}),
        (Topic.STATE, "b", {"handle": "2", "state": "R", "verdict": "running"}),
        # Settled under an older handle and dispatched again: the stale settlement must not
        # silence the run of it that is still going.
        (
            Topic.SETTLED,
            "b",
            {"handle": "0", "verdict": "failed", "exit_code": 1, "detail": "old"},
        ),
        (Topic.REFUSED, "c", {"target": "vast", "reason": "no key"}),
        (Topic.SUBMITTED, "d", {"handle": "4", "target": "gold"}),
    ]
    for topic, job, data in lines:
        publish(bus, stream, topic, job=job, data=data)


def dispatched(board: Board, stream: str, lines: tuple[tuple[str, Topic, str, str], ...]) -> None:
    """Dispatch lines stamped by the test and appended in its order, bypassing `publish`.

    The file order must be free to disagree with the stamps, the way a broker's redelivery does.
    """
    bus = Receipts(directory(board, stream) / "events.ndjson")
    for at, topic, job, said in lines:
        taken = topic is Topic.SUBMITTED
        bus.publish(
            Event(
                at=f"2026-09-04T{at}+00:00",
                batch=stream,
                topic=topic,
                job=job,
                data={"handle": said, "target": "miyabi-g"}
                if taken
                else {"target": "miyabi-g", "reason": said},
            )
        )


def test_the_newest_dispatch_line_decides_the_row_whatever_its_topic(board: Board) -> None:
    """Taken, turned away and held on a quota answer one request, so the last answer stands.

    Refusals at 13:43 went out at 18:43 under new handles (miyabi-g njobs-g, 2026-09-04). The
    stamps decide, not the file order, which the envelope contract never promises.
    """
    stream = "superseded"
    dispatched(
        board,
        stream,
        (
            ("13:43:14", Topic.REFUSED, "shell", "njobs-g full"),
            ("18:43:08", Topic.SUBMITTED, "shell", "3294907"),
            ("18:43:16", Topic.SUBMITTED, "sql", "3294908"),
            ("18:43:23", Topic.REFUSED, "sql", "the queue was removed"),
            ("18:43:22", Topic.SUBMITTED, "rust", "3294909"),
            ("13:43:31", Topic.REFUSED, "rust", "njobs-g full"),
            ("13:43:39", Topic.HELD, "tex", "njobs-g full"),
            ("18:43:28", Topic.SUBMITTED, "tex", "3294910"),
            ("13:00:00", Topic.SUBMITTED, "math", "3294206"),
            ("18:43:42", Topic.HELD, "math", "njobs-g full"),
        ),
    )
    settled = board.verdicts().of(stream)
    rows = {trial.job: trial for trial in settled.trials}
    # A re-dispatch supersedes the refusal that came before it, whichever line was appended last.
    assert (rows["shell"].verdict, rows["shell"].handle) == ("running", "3294907")
    assert (rows["rust"].verdict, rows["rust"].handle) == ("running", "3294909")
    assert (rows["tex"].verdict, rows["tex"].handle) == ("running", "3294910")
    # A refusal after a submission is terminal, in the target's own words.
    assert (rows["sql"].verdict, rows["sql"].detail) == ("refused", "the queue was removed")
    # And a hold after one says the sweep is offering the job again, which is still in flight.
    assert (rows["math"].verdict, rows["math"].code) == ("held", 2)
    assert settled.code == 1


def test_a_job_a_wave_left_out_gets_a_row_that_is_already_over(board: Board) -> None:
    """A `--only` wave's unselected jobs are a row terminal from the start, neither waited on
    nor counted as a failure, since nothing dispatched cannot move."""
    stream = "left-out"
    dispatched(
        board,
        stream,
        (
            ("18:43:08", Topic.SUBMITTED, "shell", "3294907"),
            ("18:43:30", Topic.SKIPPED, "python", "skipped: not named by --only"),
            ("18:43:31", Topic.SKIPPED, "cpp", "skipped: not named by --only"),
        ),
    )
    settled = board.verdicts().of(stream)
    rows = {trial.job: trial for trial in settled.trials}
    assert rows["python"] == TrialVerdict(
        job="python",
        target="miyabi-g",
        state="skipped",
        verdict="skipped",
        detail="skipped: not named by --only",
    )
    assert rows["python"].exit_code is None
    # Not in flight, so the one job that IS flying is the only thing holding the stream at 2.
    assert (rows["python"].code, rows["cpp"].code) == (0, 0)
    assert settled.code == 2

    recorded(board, "3294907", name=f"batch:{stream}/shell", verdict="ok", target="miyabi-g")

    assert board.verdicts().of(stream).code == 0


def test_a_skip_never_outranks_a_dispatch_however_much_newer_it_is(board: Board) -> None:
    """A skip says no offer was made in this wave, not that an earlier run unhappened (rep92,
    2026-09-04); a wave that skips a job and then dispatches it reports the dispatch."""
    stream = "waves"
    dispatched(
        board,
        stream,
        (
            ("13:41:47", Topic.SUBMITTED, "ran-then-skipped", "3294189"),
            ("18:43:30", Topic.SKIPPED, "ran-then-skipped", "skipped: not named by --only"),
            ("13:41:48", Topic.SKIPPED, "skipped-then-ran", "skipped: not named by --only"),
            ("18:43:08", Topic.SUBMITTED, "skipped-then-ran", "3294907"),
        ),
    )
    recorded(board, "3294189", name=f"batch:{stream}/a", verdict="ok", target="miyabi-g")
    recorded(board, "3294907", name=f"batch:{stream}/b", verdict="ok", target="miyabi-g")
    rows = {trial.job: trial for trial in board.verdicts().of(stream).trials}

    assert (rows["ran-then-skipped"].handle, rows["ran-then-skipped"].verdict) == ("3294189", "ok")
    assert (rows["skipped-then-ran"].handle, rows["skipped-then-ran"].verdict) == ("3294907", "ok")


def test_a_stream_reads_the_outcome_the_durable_sweep_already_settled(board: Board) -> None:
    """A batch whose watching session died is settled by the cron pass in the registry only, so
    the registry row is joined onto the rows the receipts left in flight (2026-09-04)."""
    stream = "swept-batch"
    bus = Receipts(directory(board, stream) / "events.ndjson")
    publish(bus, stream, Topic.SUBMITTED, job="tex", data={"handle": "3294910", "target": "gold"})
    # Nothing dispatched under that handle yet, so the row stands exactly as the stream left it.
    assert (board.verdicts().of(stream).trials[0].verdict, board.verdicts().of(stream).code) == (
        "running",
        2,
    )
    recorded(board, "3294910", name=f"batch:{stream}/tex", verdict="ok")
    settled = board.verdicts().of(stream)
    assert settled.trials == (
        TrialVerdict(
            job="tex",
            handle="3294910",
            target="gold",
            state="finished",
            verdict="ok",
            exit_code=0,
        ),
    )
    assert settled.code == 0


@pytest.mark.parametrize(
    ("attested", "flagged"),
    [
        pytest.param(None, "", id="a-run-that-attested-nothing-at-all"),
        pytest.param({"idle": True, "gpu_pct": 0}, "", id="a-run-that-started-on-an-idle-node"),
        pytest.param(
            {"idle": False, "gpu_pct": 47},
            "gpu 47% busy at start",
            id="a-run-that-started-while-another-job-held-the-gpu",
        ),
    ],
)
def test_a_measurement_taken_under_contention_says_so_on_every_row_of_its_run(
    board: Board, attested: dict | None, flagged: str
) -> None:
    """A contended artifact otherwise looks exactly as authoritative as a clean one."""
    stream = f"contention-{flagged.count('%')}-{attested is not None}"
    bus = Receipts(directory(board, stream) / "events.ndjson")
    publish(bus, stream, Topic.SUBMITTED, job="a", data={"handle": "9", "target": "gold"})
    if attested is not None:
        publish(bus, stream, Topic.ATTESTED, job="a", data=attested)
    publish(
        bus,
        stream,
        Topic.SETTLED,
        job="a",
        data={"handle": "9", "verdict": "ok", "exit_code": 0, "detail": ""},
    )
    (settled,) = board.verdicts().of(stream).trials
    assert settled.contended == flagged


def test_a_stream_answers_with_one_settled_row_per_job_and_the_completion_exit(
    board: Board,
) -> None:
    """Rows come from the receipts alone, and a stream that produced rows carries no note.

    A re-dispatched job ignores the old run's settlement, a refusal is terminal in its own
    words, and a job submitted but never probed still has a row.
    """
    published(board, _STREAM)
    settled = board.verdicts().of(_STREAM)
    assert (settled.stream, settled.note) == (_STREAM, "")
    by_job = {trial.job: trial for trial in settled.trials}
    assert by_job["a"] == TrialVerdict(
        job="a",
        handle="1",
        target="gold",
        node="law",
        state="F",
        verdict="ok",
        exit_code=0,
        detail="results/run",
    )
    assert (by_job["b"].verdict, by_job["b"].code) == ("running", 2)
    assert (by_job["c"].verdict, by_job["c"].detail, by_job["c"].code) == (
        "refused",
        "no key",
        1,
    )
    assert (by_job["d"].verdict, by_job["d"].state) == ("running", "")
    assert settled.code == 1


@pytest.mark.parametrize(
    ("verdicts", "code"),
    [
        (("ok", "passed"), 0),
        (("ok", "failed"), 1),
        (("ok", "running"), 2),
        (("ok", "vanished"), 3),
        ((), 3),
    ],
    ids=["all clean", "one failure", "one in flight", "one vanished", "no receipts at all"],
)
def test_the_stream_exit_ranks_failure_over_flight_over_doubt(
    verdicts: tuple[str, ...], code: int
) -> None:
    """0 only when every row settled clean, and receipts that do not exist prove nothing."""
    stream = StreamVerdict(
        stream="s",
        trials=tuple(TrialVerdict(job=str(at), verdict=word) for at, word in enumerate(verdicts)),
    )
    assert stream.code == code


def test_a_receipts_file_reads_both_written_shapes_and_skips_what_is_neither(
    board: Board, tmp_path: Path
) -> None:
    """Both shapes are read and a torn line skipped; a receipt naming no outcome, or whose
    payload is not even a mapping, still settles ok because the harness printed it at all."""
    path = tmp_path / "receipts.jsonl"
    event = {
        "at": "2026-08-25T00:00:00+00:00",
        "batch": "s",
        "topic": "job.submitted",
        "job": "a",
        "data": {"handle": "1", "target": "gold"},
    }
    printed = {
        "trial_receipt": {
            "run_id": "r1",
            "outcome": "passed",
            "producer": "lab",
            "node": "invariance-tax-law",
            "gates": [{"status": "passed", "reason": ""}, {"status": "passed", "reason": ""}],
        }
    }
    bare = {"trial_receipt": {"kind": "gemm", "median_ms": 0.01}}
    garbage = {"trial_receipt": ["not", "a", "mapping"]}
    lines = [json.dumps(event), "  ", json.dumps(printed), json.dumps(bare), json.dumps(garbage)]
    path.write_text("\n".join([*lines, "not json", "[1]"]), encoding="utf-8")
    settled = board.verdicts().of(str(path))
    assert [trial.job for trial in settled.trials] == ["a", "r1", "", ""]
    assert settled.trials[1] == TrialVerdict(
        job="r1",
        node="invariance-tax-law",
        verdict="passed",
        producer="lab",
        gates="2 passed",
    )
    assert (settled.trials[2].verdict, settled.trials[3].verdict) == ("ok", "ok")
    assert lined(path) == settled.trials


@pytest.mark.parametrize(
    ("sweep", "cell"),
    [
        ([{"status": "passed", "reason": ""}], "1 passed"),
        ([{"status": "blocked", "reason": "no gpu"}], "blocked: no gpu"),
        ([{"status": "failed", "reason": "broke"}, {"status": "passed"}], "failed: broke"),
        (["garbage"], "0 passed"),
        ([], ""),
        ("not a list", ""),
    ],
    ids=["all passed", "blocked", "failed first", "entries skipped", "empty", "not a list"],
)
def test_the_gate_sweep_summarizes_to_the_first_non_passing_gate(sweep: object, cell: str) -> None:
    assert gated(sweep) == cell


def test_a_handle_answers_from_its_registry_row_when_the_workspace_tracked_nothing(
    board: Board,
) -> None:
    """The registry is the durable floor, so a run in a workspace tracking nothing settles."""
    recorded(board, "71", name="tax-run", verdict="ok")
    settled = board.verdicts().of("71")
    assert settled.stream == "tax-run"
    assert settled.trials == (
        TrialVerdict(
            job="tax-run",
            handle="71",
            target="gold",
            node="tax-law",
            state="finished",
            verdict="ok",
            exit_code=0,
        ),
    )
    assert settled.code == 0


_IMPORT = "ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.15' not found"


@pytest.mark.parametrize(
    ("verdict", "log", "cause"),
    [
        pytest.param(
            "failed",
            f'Traceback (most recent call last):\n  File "run.py", line 1, in <module>\n'
            f"    import sqlite3\n{_IMPORT}\nmainboard-receipts-begin\n"
            "mainboard-receipt:e30K\nmainboard-receipts-end\nexit=1\n",
            _IMPORT,
            id="a-failed-row-names-its-exception-not-its-frames-receipts-or-exit-stamp",
        ),
        pytest.param("ok", "all good\n", "", id="a-clean-row-is-never-read-for-a-cause"),
    ],
)
def test_a_failed_row_says_what_it_said_on_the_way_out(
    board: Board, verdict: str, log: str, cause: str
) -> None:
    """Thirty two GH200 jobs printed identical `failed` rows while the line that explained each
    sat in the log the sweep had already brought home (2026-09-05)."""
    recorded(board, "9", name="doomed", verdict=verdict)
    stored = directory(board, "doomed") / "9.log"
    stored.parent.mkdir(parents=True, exist_ok=True)
    stored.write_text(log, encoding="utf-8")
    [row] = board.verdicts().of("9").trials
    assert row.cause == cause


def test_every_dispatched_row_carries_the_provenance_its_mirror_could_not_derive(
    board: Board,
) -> None:
    """A row measured on a mirror with no history still names the commit and bytes it ran."""
    recorded(board, "3", name="sealed", verdict="ok", commit="e975499f" * 5, digest="9a" * 32)
    recorded(board, "1", name=_STREAM, verdict="ok", commit="c0ffee" * 6, digest="7b" * 32)
    published(board, _STREAM)

    [alone] = board.verdicts().of("3").trials
    receipted_row = next(row for row in board.verdicts().of(_STREAM).trials if row.handle == "1")

    assert (alone.commit, alone.digest) == ("e975499f" * 5, "9a" * 32)
    assert (receipted_row.commit, receipted_row.digest) == ("c0ffee" * 6, "7b" * 32)


def test_a_handle_prefers_its_own_receipts_rows_over_the_registry_floor(board: Board) -> None:
    """Receipts are the source, so a handle whose stream holds lines answers from them."""
    recorded(board, "1", name=_STREAM, verdict="ok")
    published(board, _STREAM)
    settled = board.verdicts().of("1")
    assert [trial.job for trial in settled.trials] == ["a"]
    assert settled.trials[0].detail == "results/run"


@pytest.mark.parametrize("shared_handle", [False, True])
@pytest.mark.parametrize("missing_submission", [False, True])
def test_same_named_host_jobs_keep_their_own_state_and_child_receipts(
    board: Board, shared_handle: bool, missing_submission: bool
) -> None:
    """A finished Hopper launcher cannot settle a running crimson launcher."""
    stream = "scaling-gpt2"
    remote = "3325153"
    active = remote if shared_handle else "1679"
    recorded(board, remote, name=stream, target="miyabi-g", verdict="ok")
    recorded(board, active, name=stream, target="crimson")
    bus = Receipts(directory(board, stream) / "events.ndjson")
    launchers = [("miyabi-g", remote), ("crimson", active)]
    for target, handle in launchers[:1] if missing_submission else launchers:
        publish(
            bus, stream, Topic.SUBMITTED, job=stream, data={"handle": handle, "target": target}
        )
    flying = {"handle": active, "state": "Running", "verdict": "running"}
    publish(bus, stream, Topic.STATE, job=stream, data=flying)
    ended = {"handle": remote, "state": "F", "verdict": "ok", "exit_code": 0}
    for topic in (Topic.STATE, Topic.SETTLED):
        publish(bus, stream, topic, job=stream, data=ended)
    receipts = []
    for target, handle in launchers:
        cases = [[f"{target}-{index}", f"case-{index}"] for index in range(3)]
        receipts.extend(receipt(run, case) for run, case in cases)
        publish(
            bus,
            stream,
            Topic.EVIDENCE,
            job=stream,
            data={
                "handle": handle,
                "target": target,
                "submitted_at": board.dispatcher.cache.run(handle, target).submitted_at,
                "status": "verified",
                "trials": cases,
            },
        )
    path = directory(board, stream) / "receipts.ndjson"
    original = "\n".join(receipts) + "\n"
    path.write_text(original)

    for target, handle, code in [("crimson", active, 2), ("miyabi-g", remote, 0)]:
        settled = board.verdicts().handled(handle, host=target)
        # Passing children do not prove that the launcher ended.
        assert (settled.code, settled.trials[0].verdict) == (code, "running" if code else "ok")
        assert {row.run for row in settled.trials if row.run} == {
            f"{target}-{i}" for i in range(3)
        }
    assert board.verdicts().of(stream).code == 2
    assert path.read_text() == original


def test_a_target_that_is_nothing_at_all_is_refused_with_the_three_shapes_named(
    board: Board,
) -> None:
    with pytest.raises(MissionError, match="receipts file, a stream, or a recorded handle"):
        board.verdicts().of("never-dispatched")
    with pytest.raises(MissionError, match="nothing to wait on"):
        board.verdicts().wait("never-dispatched", timeout=0.001, poll=lambda _: None)


def test_wait_sweeps_the_monitor_path_until_terminal_and_answers_from_the_receipts(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every poll is the durable pass the cron runs, so waiting settles rather than watches."""
    recorded(board, "9", name="waited")
    passes: list[int] = []

    def sweeping(monitor: Monitor) -> None:
        passes.append(1)
        if len(passes) == 2:
            recorded(board, "9", name="waited", verdict="ok")
            board.dispatcher.cache.report(board.dispatcher.cache.run("9"), "ok")

    monkeypatch.setattr(Monitor, "once", sweeping)
    settled = board.verdicts().wait("9", interval=0.0, poll=lambda seconds: None)
    assert len(passes) == 2
    assert settled.code == 0
    assert settled.trials[0].verdict == "ok"


def test_wait_on_a_batch_id_sweeps_until_every_job_settles_and_answers_the_batch_verdict(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One `wait` serves a handle and a batch alike, and a batch's answer is its whole verdict."""
    stream = "smoke-1"
    bus = Receipts(directory(board, stream) / "events.ndjson")
    publish(bus, stream, Topic.SUBMITTED, job="a", data={"handle": "1", "target": "gold"})
    publish(bus, stream, Topic.SUBMITTED, job="b", data={"handle": "2", "target": "gold"})
    publish(
        bus, stream, Topic.SETTLED, job="a", data={"handle": "1", "verdict": "ok", "exit_code": 0}
    )
    passes: list[int] = []

    def sweeping(monitor: Monitor) -> None:
        passes.append(1)
        if len(passes) == 2:
            publish(
                bus,
                stream,
                Topic.SETTLED,
                job="b",
                data={"handle": "2", "verdict": "failed", "exit_code": 1},
            )

    monkeypatch.setattr(Monitor, "once", sweeping)
    settled = board.verdicts().wait(stream, interval=0.0, poll=lambda seconds: None)
    assert len(passes) == 2
    assert settled.stream == stream
    assert [trial.verdict for trial in settled.trials] == ["ok", "failed"]
    assert settled.code == 1


@pytest.mark.parametrize("batch", [False, True], ids=["a-handle", "a-batch-id"])
def test_wait_gives_up_at_the_deadline_and_reports_the_run_still_in_flight(
    board: Board, monkeypatch: pytest.MonkeyPatch, batch: bool
) -> None:
    """A bounded wait is the contract: exit 2 with the truth, never a hang."""
    if batch:
        bus = Receipts(directory(board, "smoke-2") / "events.ndjson")
        publish(bus, "smoke-2", Topic.SUBMITTED, job="a", data={"handle": "1", "target": "gold"})
    else:
        recorded(board, "8", name="smoke-2")
    monkeypatch.setattr(Monitor, "once", lambda monitor: None)
    waited = "smoke-2" if batch else "8"
    settled = board.verdicts().wait(waited, timeout=1e-6, interval=0.0, poll=lambda s: None)
    assert (settled.stream, settled.code, settled.trials[0].verdict) == ("smoke-2", 2, "running")


def test_cancel_kills_through_the_backend_and_settles_the_record_in_the_same_pass(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killing, settling and releasing are one pass, the reported cursor moving last."""
    recorded(board, "7", name="doomed", target="miyabi-g")
    acted: list[str] = []
    monkeypatch.setattr(Monitor, "once", lambda monitor: acted.append("swept"))
    monkeypatch.setattr(Job, "kill", lambda self: acted.append("killed"))
    monkeypatch.setattr(Job, "release", lambda self: acted.append("released"))
    monkeypatch.setattr(Job, "transcript", lambda self: "")
    settled = board.verdicts().cancel("7")
    assert acted == ["killed", "released"]
    assert settled.trials[0].verdict == "cancelled"
    # Exit 1: the stop was deliberate, and a completion check must still not call it complete.
    assert settled.code == 1
    stored = board.dispatcher.cache.run("7")
    assert (stored.verdict, stored.reported) == ("cancelled", "cancelled")
    # Settled for good, so the durable sweep never owes this run another probe.
    assert stored not in board.dispatcher.cache.tracked()


@pytest.mark.parametrize(
    ("body", "said"),
    [
        pytest.param("", "is empty", id="a-file-nothing-has-been-written-to-yet"),
        pytest.param(
            '{"certificate": {"claim": "x", "status": "verified"}}\n{"certificate": {}}\n',
            "none of which is evidence this verb reads",
            id="a-harness-writing-a-shape-this-verb-was-never-taught",
        ),
    ],
)
def test_an_empty_table_says_why_instead_of_reading_as_a_failure(
    board: Board, tmp_path: Path, body: str, said: str
) -> None:
    """An empty table is the one answer a reader cannot act on, so the note says why."""
    path = tmp_path / "evidence.jsonl"
    path.write_text(body, encoding="utf-8")
    settled = board.verdicts().of(str(path))
    assert settled.trials == ()
    assert said in settled.note
    # Still exit 3: receipts that prove nothing prove nothing, whatever the note explains.
    assert settled.code == 3


def test_the_captured_tail_is_preferred_over_a_backend_that_may_no_longer_exist(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A settled run's host gets cleaned and a rental's disk is already gone, so the copy wins."""
    recorded(board, "5", name="chatty", target="miyabi-g")
    monkeypatch.setattr(Job, "transcript", lambda self: "live output")
    assert board.verdicts().captured("5") == "live output"
    stored = directory(board, "chatty") / "5.log"
    stored.parent.mkdir(parents=True, exist_ok=True)
    stored.write_text("what the sweep brought home\n", encoding="utf-8")
    assert board.verdicts().captured("5") == "what the sweep brought home\n"


def test_a_backend_that_will_not_answer_costs_a_transcript_and_never_a_sweep(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host quiet between the probe and the read must not take every other job's outcome."""
    recorded(board, "4", name="quiet", target="miyabi-g")

    def refuse(self: Job) -> str:
        raise MissionError("host went away")

    monkeypatch.setattr(Job, "logs", refuse)
    assert board.verdicts().captured("4") == ""


def test_cancelling_a_run_that_already_settled_touches_nothing(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A terminal verdict can never change, so a late cancel reports rather than rewrites."""
    recorded(board, "6", name="done", verdict="ok", target="miyabi-g")
    board.dispatcher.cache.report(board.dispatcher.cache.run("6"), "ok")
    monkeypatch.setattr(Job, "kill", lambda self: pytest.fail("a settled run was killed"))
    settled = board.verdicts().cancel("6")
    assert (settled.trials[0].verdict, settled.code) == ("ok", 0)


def test_the_board_hands_out_the_reader_bound_to_itself(board: Board) -> None:
    reader = board.verdicts()
    assert isinstance(reader, Verdicts)
    assert reader.board is board


def test_cancelling_a_prepared_creation_claims_it_so_no_create_can_follow(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelling a handle-less prepared row is winning its claim; one a creation won first is
    left alone, since settling it would hide whatever it rented."""
    monkeypatch.setattr(Job, "kill", lambda self: pytest.fail("a creation with no handle killed"))
    recorded(board, "2", name="beaten", verdict="prepared")
    recorded(board, "3", name="abandoned", verdict="prepared")
    claim = Cache.leave_prepared

    def beaten(cache: Cache, run: RunRecord, verdict: str) -> RunRecord:
        claim(cache, run, "submitting")
        return claim(cache, run, verdict)

    with monkeypatch.context() as racing:
        racing.setattr(Cache, "leave_prepared", beaten)
        with pytest.raises(MissionError, match="changed during cancellation"):
            board.verdicts().cancel("2")
    assert board.dispatcher.cache.run("2").verdict == "submitting"

    board.verdicts().cancel("3")
    stored = board.dispatcher.cache.run("3")
    assert (stored.verdict, stored.reported) == ("cancelled", "cancelled")


@pytest.mark.parametrize(
    ("transcript", "release", "evidence", "reported"),
    [
        ('{"trial_receipt": "torn"}\n', None, "unverified", "cancelled"),
        ("", MissionError("the provider refused the cancel"), "copied", None),
    ],
    ids=[
        "evidence that fails verification is recorded as unverified",
        "a release that fails leaves the cursor for the next pass",
    ],
)
def test_a_cancel_says_what_it_could_not_verify_and_retries_what_it_could_not_release(
    board: Board,
    monkeypatch: pytest.MonkeyPatch,
    transcript: str,
    release: MissionError | None,
    evidence: str,
    reported: str | None,
) -> None:
    """A stop never calls lost evidence verified, and a failed release keeps the cursor."""

    def released(job: Job) -> None:
        if release is not None:
            raise release

    recorded(board, "4", name="stopped", target="miyabi-g")
    monkeypatch.setattr(Job, "kill", lambda self: None)
    monkeypatch.setattr(Job, "transcript", lambda self: transcript)
    monkeypatch.setattr(Job, "release", released)
    board.verdicts().cancel("4")
    stored = board.dispatcher.cache.run("4")
    assert (stored.verdict, stored.evidence, stored.reported) == ("cancelled", evidence, reported)


def test_a_stream_row_the_sweep_settled_as_failed_says_why(board: Board) -> None:
    """A joined outcome carries the cause from the log the sweep brought home, as a floor does."""
    stream = "swept-failure"
    bus = Receipts(directory(board, stream) / "events.ndjson")
    publish(bus, stream, Topic.SUBMITTED, job="tex", data={"handle": "3294911", "target": "gold"})
    recorded(board, "3294911", name=f"batch:{stream}/tex", verdict="failed")
    (directory(board, stream) / "3294911.log").write_text(
        "Traceback (most recent call last):\nMemoryError: CUDA out of memory\n", encoding="utf-8"
    )
    [row] = board.verdicts().of(stream).trials
    assert (row.verdict, row.cause) == ("failed", "MemoryError: CUDA out of memory")


def test_an_evidence_line_with_no_readable_trial_list_still_qualifies_its_own_run() -> None:
    """A correction that lost its case list still names the run it is about by handle."""
    trial = TrialVerdict(job="j", handle="7", target="gold", verdict="ok")
    correction = Event(
        at="2026-09-04T00:00:00+00:00",
        batch="s",
        topic=Topic.EVIDENCE,
        job="j",
        data={"handle": "7", "target": "gold", "status": "unverified", "trials": "torn"},
    )
    [row] = qualified((trial,), [correction])
    assert row.verdict == "unverified"


def test_a_wait_stops_with_its_own_exit_status_on_a_job_silent_on_an_idle_card(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The vigil looks at the run still running after the pass; the answer exits 4, not 2."""
    recorded(board, "8", name="stuck", verdict="running")
    looked: list[list[str]] = []

    def look(vigil: Vigil, records: list[RunRecord]) -> Look:
        looked.append([record.handle for record in records])
        return Look(stalled="8 on gold printed nothing for 1300s")

    monkeypatch.setattr(Monitor, "once", lambda monitor: None)
    monkeypatch.setattr(Vigil, "look", look)
    settled = board.verdicts().wait("8", poll=lambda seconds: pytest.fail("a stall waited on"))
    assert looked == [["8"]]
    assert (settled.code, settled.stalled) == (STALLED, "8 on gold printed nothing for 1300s")
    assert settled.trials[0].verdict == "running"


@pytest.mark.parametrize(("session", "verdict", "code"), [(0, "ok", 0), (1, "failed", 1)])
def test_a_job_whose_session_ended_while_its_process_lingers_settles_on_the_session(
    board: Board, monkeypatch: pytest.MonkeyPatch, session: int, verdict: str, code: int
) -> None:
    """`1 known in 27s`, then half an hour of `running` (2026-09-19): it settles on the session."""
    recorded(board, "7", name="lingered", target="miyabi-g", verdict="running")
    acted: list[str] = []
    monkeypatch.setattr(Monitor, "once", lambda monitor: None)
    monkeypatch.setattr(Job, "kill", lambda self: acted.append("killed"))
    monkeypatch.setattr(Job, "release", lambda self: acted.append("released"))
    monkeypatch.setattr(Job, "transcript", lambda self: "")
    monkeypatch.setattr(
        Vigil,
        "look",
        lambda vigil, records: Look(
            lingering=(Linger(handle="7", target="miyabi-g", session=session),)
        ),
    )
    settled = board.verdicts().wait("7", poll=lambda seconds: pytest.fail("a settle waited on"))
    assert acted == ["killed", "released"]
    assert (settled.trials[0].verdict, settled.trials[0].exit_code, settled.code) == (
        verdict,
        session,
        code,
    )
    assert stopped(verdict, session).startswith(f"pytest session ended with exit {session}")


def test_a_batch_wait_looks_only_at_its_own_running_jobs_and_a_held_claim_defers_a_settle(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A conclusion under another process's settlement claim waits for the next look."""
    stream = "wave-1"
    bus = Receipts(directory(board, stream) / "events.ndjson")
    publish(bus, stream, Topic.SUBMITTED, job="a", data={"handle": "1", "target": "gold"})
    recorded(board, "1", name=stream, verdict="running")
    recorded(board, "2", name="another", verdict="running")
    looked: list[list[str]] = []
    looks = iter(
        [
            Look(lingering=(Linger(handle="1", target="gold", session=0),)),
            Look(stalled="1 on gold printed nothing for 1300s"),
        ]
    )

    def look(vigil: Vigil, records: list[RunRecord]) -> Look:
        looked.append([record.handle for record in records])
        return next(looks)

    monkeypatch.setattr(Monitor, "once", lambda monitor: None)
    monkeypatch.setattr(Vigil, "look", look)
    monkeypatch.setattr("mainboard.verdicts.SETTLEMENT_SECONDS", 0.05)
    monkeypatch.setattr(Job, "kill", lambda self: pytest.fail("killed under a held claim"))
    with FileLock(board.dispatcher.cache.path.with_suffix(".settlement.lock")):
        settled = board.verdicts().wait(stream, poll=lambda seconds: None)
    assert looked == [["1"], ["1"]]
    assert settled.code == STALLED
