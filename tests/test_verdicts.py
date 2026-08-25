import json
from typing import TYPE_CHECKING

import pytest

from mainboard import Board, MissionError
from mainboard.batch.receipts import Receipts, Topic, publish
from mainboard.batch.runner import directory
from mainboard.dispatch.state import RunRecord
from mainboard.monitor import Monitor
from mainboard.verdicts import StreamVerdict, TrialVerdict, Verdicts, gated, lined, receipted

if TYPE_CHECKING:
    from pathlib import Path

_STREAM = "study-receipts"


def recorded(board: Board, handle: str, *, name: str = "", verdict: str | None = None) -> None:
    """One dispatched run in the registry, the durable floor a verdict reads from."""
    board.dispatcher.cache.record(
        RunRecord(
            handle=handle,
            target="gold",
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


def test_a_handle_prefers_its_own_receipts_rows_over_the_registry_floor(board: Board) -> None:
    """Receipts are the source, so a handle whose stream holds lines answers from them."""
    recorded(board, "1", name=_STREAM, verdict="ok")
    published(board, _STREAM)
    settled = board.verdicts().of("1")
    assert [trial.job for trial in settled.trials] == ["a"]
    assert settled.trials[0].detail == "results/run"


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

    monkeypatch.setattr(Monitor, "once", sweeping)
    settled = board.verdicts().wait("9", interval=0.0, poll=lambda seconds: None)
    assert len(passes) == 2
    assert settled.code == 0
    assert settled.trials[0].verdict == "ok"


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
