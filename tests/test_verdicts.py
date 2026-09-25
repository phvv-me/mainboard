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
    receipted,
    stopped,
)
from mainboard.vigil import Linger, Look, Vigil

if TYPE_CHECKING:
    from pathlib import Path

_STREAM = "study-receipts"


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
    (under / "receipts.ndjson").write_text(
        json.dumps(
            {
                "trial_receipt": {
                    "run": "r",
                    "case_id": "case",
                    "outcome": "passed",
                    "verdict": "validated",
                }
            }
        )
        + "\n"
    )
    assert board.verdicts().of("crashed-before-event").code == 2


def test_delivery_correction_preserves_claim_but_does_not_claim_verified_evidence(
    board: Board,
) -> None:
    recorded(board, "77", name="lost-transfer", verdict="ok")
    under = directory(board, "lost-transfer")
    under.mkdir(parents=True, exist_ok=True)
    path = under / "receipts.ndjson"
    payload = {
        "trial_receipt": {
            "run": "first-run",
            "case_id": "same-case",
            "outcome": "passed",
            "verdict": "validated",
        }
    }
    original = json.dumps(payload) + "\n"
    second = (
        json.dumps(
            {
                "trial_receipt": {
                    "run": "second-run",
                    "case_id": "same-case",
                    "outcome": "passed",
                    "verdict": "validated",
                }
            }
        )
        + "\n"
    )
    path.write_text(original + second)
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
    second_trial = next(trial for trial in lined(path) if trial.run == "second-run")
    assert second_trial.verdict == "passed"
    assert path.read_text() == original + second
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
    publish(
        bus,
        stream,
        Topic.SUBMITTED,
        job="a",
        data={"handle": "1", "target": "gold", "kind": "ssh", "command": "true", "node": "law"},
    )
    publish(bus, stream, Topic.STATE, job="a", data={"handle": "1", "state": "F", "verdict": "ok"})
    publish(
        bus,
        stream,
        Topic.SETTLED,
        job="a",
        data={"handle": "1", "verdict": "ok", "exit_code": 0, "detail": "results/run"},
    )
    publish(
        bus,
        stream,
        Topic.SUBMITTED,
        job="b",
        data={"handle": "2", "target": "gold", "kind": "ssh", "command": "false"},
    )
    publish(
        bus, stream, Topic.STATE, job="b", data={"handle": "2", "state": "R", "verdict": "running"}
    )
    # A job settled under an older handle and dispatched again: the stale settlement must not
    # silence the run of it that is still going.
    publish(
        bus,
        stream,
        Topic.SETTLED,
        job="b",
        data={"handle": "0", "verdict": "failed", "exit_code": 1, "detail": "old run"},
    )
    publish(bus, stream, Topic.REFUSED, job="c", data={"target": "vast", "reason": "no key"})
    publish(bus, stream, Topic.SUBMITTED, job="d", data={"handle": "4", "target": "gold"})


def dispatched(board: Board, stream: str, lines: tuple[tuple[str, Topic, str, str], ...]) -> None:
    """A stream of dispatch lines stamped by the test, in the order it wants them appended.

    Written through the envelope rather than through `publish` because both halves of this are
    about clocks: the stamps have to be chosen, and the file order has to be free to disagree
    with them the way a broker's redelivery does.
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
    """Taken, turned away and held on a quota are three answers to one request.

    So the last answer is the true one. Four jobs miyabi-g's `njobs-g` limit refused at 13:43
    went out at 18:43 under new handles (2026-09-04), and a fold that consulted the refusal
    because it was a refusal buried the run that actually went. The stamps decide it rather than
    the order the lines landed in, since the envelope contract promises no order at all and a
    file is only accidentally in one.
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
    """A `--only` wave's unselected jobs were invisible to the verb that reports the batch.

    They had no row at all, because the row set was built from the three answers a target gives
    and nothing was ever offered to a target for these. Now the skip is the row, terminal from
    the start: nothing dispatched cannot move, so the stream neither waits on it nor counts it
    as a failure.
    """
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
    """A skip says no offer was made in this wave, not that an earlier run unhappened.

    Nine jobs of the rep92 batch ran at 13:43 and were left out of the 18:43 `--only` wave
    (2026-09-04), so ranking the skip by its clock alone would have thrown nine outcomes away.
    A skip decides a row exactly when nothing was ever dispatched for the job, which is also
    what makes the other direction work: a wave that skips a job and then dispatches it reports
    the dispatch.
    """
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
    """A batched job's settled line has exactly one publisher, and it is not the sweep.

    So a batch whose watching session died is probed, pulled and memoized by the cron pass with
    nothing ever reaching its stream, and reading the stream alone left thirteen finished jobs
    saying `running` beside their thirteen pulled logs (2026-09-04). The registry row is that
    outcome, and it is joined onto the rows the receipts left in flight.
    """
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
    """A contended artifact otherwise looks exactly as authoritative as a clean one.

    Only the unwelcome half is rendered, so a column full of the word `idle` never buries the
    one row that matters, and the busy figure rides along so a reader weighs it.
    """
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
    """The stream read is the anti-fabrication read: rows come from the receipts alone.

    A settled job carries its node, exit code and detail; a re-dispatched job ignores the old
    run's settlement and reads as running; a refusal is terminal in its own words; a job that
    was submitted and never probed still has a row. Anything still in flight makes the whole
    stream exit 2, since a completion check must not call a running batch done.
    """
    published(board, _STREAM)
    settled = board.verdicts().of(_STREAM)
    assert settled.stream == _STREAM
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
    """One verb over an events log and a harness's own `trial_receipt` lines.

    The reproducibility evidence files are real streams, so the file mode accepts the printed
    receipt shape beside the envelope shape, tolerates a torn line the way replay does, and a
    receipt that names no outcome still settles as ok because the harness printed it at all.
    """
    path = tmp_path / "receipts.jsonl"
    event = {
        "at": "2026-08-25T00:00:00+00:00",
        "batch": "s",
        "topic": "job.submitted",
        "job": "a",
        "data": {"handle": "1", "target": "gold"},
    }
    receipt = {
        "trial_receipt": {
            "run_id": "r1",
            "outcome": "passed",
            "producer": "lab",
            "node": "invariance-tax-law",
            "gates": [{"status": "passed", "reason": ""}, {"status": "passed", "reason": ""}],
        }
    }
    bare = {"trial_receipt": {"kind": "gemm", "median_ms": 0.01}}
    lines = [json.dumps(event), "  ", json.dumps(receipt), json.dumps(bare), "not json", "[1]"]
    path.write_text("\n".join(lines), encoding="utf-8")
    settled = board.verdicts().of(str(path))
    assert [trial.job for trial in settled.trials] == ["a", "r1", ""]
    assert settled.trials[1] == TrialVerdict(
        job="r1",
        node="invariance-tax-law",
        verdict="passed",
        producer="lab",
        gates="2 passed",
    )
    assert settled.trials[2].verdict == "ok"
    assert lined(path) == settled.trials


def test_a_trial_receipt_payload_that_is_not_a_mapping_still_answers() -> None:
    """A harness that printed garbage under the key gets an empty row, not a refusal."""
    assert receipted(["not", "a", "mapping"]).verdict == "ok"


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
    """The registry is the durable floor, so a run with no receipts still settles.

    The fixture manifest tracks nothing, which is exactly the workspace whose receipts are
    absent, and the row carries the node the dispatch recorded.
    """
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


def test_a_failed_row_says_what_it_said_on_the_way_out(board: Board) -> None:
    """Thirty two GH200 jobs printed thirty two identical `failed` rows (2026-09-05).

    The line that told anyone anything was in the log the sweep had already brought home, one
    directory over from the receipts this verb reads, and finding it meant knowing that. It is a
    column now: the last thing the run said, which for a traceback is its exception and for a
    loader failure is the symbol that was missing.
    """
    recorded(board, "9", name="doomed", verdict="failed")
    stored = directory(board, "doomed") / "9.log"
    stored.parent.mkdir(parents=True, exist_ok=True)
    stored.write_text(
        "Traceback (most recent call last):\n"
        '  File "run.py", line 1, in <module>\n'
        "    import sqlite3\n"
        "ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.15' not found\n"
        "mainboard-receipts-begin\n"
        "mainboard-receipt:e30K\n"
        "mainboard-receipts-end\n"
        "exit=1\n",
        encoding="utf-8",
    )

    [row] = board.verdicts().of("9").trials

    assert row.cause == ("ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.15' not found")
    # The frames above it, the receipts frame below it and the wrapper's own exit stamp are not
    # the cause: one is where, one is the wrapper's channel and the last is a column of its own.
    assert "File" not in row.cause
    assert row.exit_code is None or "exit=" not in row.cause


def test_a_clean_row_carries_no_cause_and_is_never_read_for_one(board: Board) -> None:
    """A run that ended well has nothing to explain, and a live one has printed nothing home."""
    recorded(board, "8", name="fine", verdict="ok")
    stored = directory(board, "fine") / "8.log"
    stored.parent.mkdir(parents=True, exist_ok=True)
    stored.write_text("all good\n", encoding="utf-8")

    assert board.verdicts().of("8").trials[0].cause == ""


def test_every_dispatched_row_carries_the_provenance_its_mirror_could_not_derive(
    board: Board,
) -> None:
    """A row measured on a host with no history still names the commit and the bytes it ran.

    Joined onto settled rows as much as onto live ones, since what a run was measured from does
    not stop being true when the job ends, and the mirror it ran in never knew it.
    """
    recorded(board, "3", name="sealed", verdict="ok", commit="e975499f" * 5, digest="9a" * 32)
    recorded(board, "1", name=_STREAM, verdict="ok", commit="c0ffee" * 6, digest="7b" * 32)
    published(board, _STREAM)

    [alone] = board.verdicts().of("3").trials
    receipted_row = next(row for row in board.verdicts().of(_STREAM).trials if row.handle == "1")

    assert (alone.commit, alone.digest) == ("e975499f" * 5, "9a" * 32)
    # And a row the receipts settled carries it too, joined from the registry, since the stream
    # a batch writes has no column for what the dispatch was taken from.
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
    for handle, target in [(remote, "miyabi-g"), (active, "crimson")]:
        if missing_submission and target == "crimson":
            continue
        publish(
            bus,
            stream,
            Topic.SUBMITTED,
            job=stream,
            data={"handle": handle, "target": target},
        )
    publish(
        bus,
        stream,
        Topic.STATE,
        job=stream,
        data={"handle": active, "state": "Running", "verdict": "running"},
    )
    for topic in (Topic.STATE, Topic.SETTLED):
        publish(
            bus,
            stream,
            topic,
            job=stream,
            data={
                "handle": remote,
                "state": "F",
                "verdict": "ok",
                "exit_code": 0,
            },
        )
    receipts = []
    for target, handle in [("miyabi-g", remote), ("crimson", active)]:
        cases = [[f"{target}-{index}", f"case-{index}"] for index in range(3)]
        for run, case in cases:
            receipts.append(
                json.dumps(
                    {
                        "trial_receipt": {
                            "run": run,
                            "case_id": case,
                            "outcome": "passed",
                        }
                    }
                )
            )
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

    waiting = board.verdicts().handled(active, host="crimson")
    assert waiting.code == 2  # Passing children do not prove that the launcher ended.
    assert waiting.trials[0].verdict == "running"
    assert {trial.run for trial in waiting.trials if trial.run} == {
        f"crimson-{index}" for index in range(3)
    }
    finished = board.verdicts().handled(remote, host="miyabi-g")
    assert finished.code == 0
    assert {trial.run for trial in finished.trials if trial.run} == {
        f"miyabi-g-{index}" for index in range(3)
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
    """Every poll is the durable pass the cron runs, so waiting settles rather than watches.

    The stand-in sweep terminalizes the run on its second pass exactly as a real one would
    write the cache, and the answer is the registry-derived row with the job's own exit code.
    """
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


def test_wait_on_a_batch_id_gives_up_at_the_deadline_with_the_batch_still_in_flight(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = "smoke-2"
    bus = Receipts(directory(board, stream) / "events.ndjson")
    publish(bus, stream, Topic.SUBMITTED, job="a", data={"handle": "1", "target": "gold"})
    monkeypatch.setattr(Monitor, "once", lambda monitor: None)
    settled = board.verdicts().wait(stream, timeout=1e-6, interval=0.0, poll=lambda s: None)
    assert settled.stream == stream
    assert settled.code == 2


def test_cancel_kills_through_the_backend_and_settles_the_record_in_the_same_pass(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancellation with no receipt trail is what killing a job over ssh by hand leaves behind.

    Killing, settling, publishing and releasing are one pass, and the reported cursor moves last
    so a cancel killed halfway repeats rather than loses the outcome.
    """
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
    """An empty table is the one answer a reader cannot act on, since it looks the same either way.

    The two shapes are named rather than the tools that write them, so a harness earns the same
    reading by printing the same line and nothing here has to learn what that harness is called.
    """
    path = tmp_path / "evidence.jsonl"
    path.write_text(body, encoding="utf-8")
    settled = board.verdicts().of(str(path))
    assert settled.trials == ()
    assert said in settled.note
    # Still exit 3: receipts that prove nothing prove nothing, whatever the note explains.
    assert settled.code == 3


def test_a_stream_that_did_produce_rows_carries_no_note_at_all(board: Board) -> None:
    """The note exists for the empty case, so a table that says something says only that."""
    published(board, _STREAM)
    assert board.verdicts().of(_STREAM).note == ""


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


def test_wait_gives_up_at_the_deadline_and_reports_the_run_still_in_flight(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bounded wait is the contract: exit 2 with the truth, never a hang."""
    recorded(board, "8", name="stuck")
    monkeypatch.setattr(Monitor, "once", lambda monitor: None)
    settled = board.verdicts().wait("8", timeout=0.000001, poll=lambda seconds: None)
    assert settled.code == 2
    assert settled.trials[0].verdict == "running"


def test_the_board_hands_out_the_reader_bound_to_itself(board: Board) -> None:
    reader = board.verdicts()
    assert isinstance(reader, Verdicts)
    assert reader.board is board


def test_cancelling_a_prepared_creation_claims_it_so_no_create_can_follow(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prepared row has no provider handle yet, so cancelling it is winning the claim on it.

    A creation that claimed it first owns it now, and settling that row as cancelled would
    leave whatever it rented billing under a row that says nothing is there.
    """
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
    """A deliberate stop may lose work, but it never calls lost evidence verified.

    And a run whose release failed may still be billing, so its cursor stays where the durable
    sweep will find it and release it again.
    """

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
    """A stalled job used to hold its waiter to the whole hour; now the wait says so and ends.

    What the vigil looks at is the run still running after the pass, and the answer is the
    run's own row with the stall beside it, exit 4 rather than the timeout's 2.
    """
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
    """`1 known in 27s` and then half an hour of `running` (2026-09-19): the answer was written.

    The lingering process is stopped the way a cancel stops it, evidence first, and the run
    settles on what its session said rather than as cancelled.
    """
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
