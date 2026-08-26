# The anti-fabrication read behind `mainboard verdict` and the block behind `mainboard wait`.
# Everything printed here is derived from on-disk receipts and the durable run registry, never
# from a dashboard, a digest or anything a live session remembered. A notification says a job
# probably ended; this module is where its outcome is actually read.
#
# Three targets resolve to one settled view. A receipts file is read line by line, accepting
# both shapes the workspace writes, the batch `Event` envelope and the printed `trial_receipt`
# line, so a study's events stream and a harness's own receipts file answer through one verb. A
# stream id reads the workspace's own `events.ndjson` for that stream. A handle resolves through
# the run registry to the stream its dispatch was tracked under, and the registry row itself is
# the floor the receipts overlay, so a run whose workspace tracks nothing still answers.

import json
from time import monotonic, sleep
from typing import TYPE_CHECKING

from patos import FrozenModel

from .batch.receipts import Event, Receipts, Topic, latest
from .batch.runner import directory
from .core.errors import MissionError
from .dispatch import vocabulary
from .tracking import streamed

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from pydantic import JsonValue

    from .board import Board
    from .dispatch.state import RunRecord

# The key a printed trial receipt line carries its payload under, the shape any experiment
# harness may write; spelled here rather than imported so reading a receipt never drags the lab
# machinery in.
_RECEIPT = "trial_receipt"

# How a settled word maps to a process exit code, the same table `Verdict.code` answers from:
# 0 ok, 1 failed, 2 still running or legitimately waiting, 3 vanished or unknown.
_EXITS = {
    vocabulary.OK: 0,
    "passed": 0,
    vocabulary.FAILED: 1,
    "refused": 1,
    vocabulary.TIMEOUT: 1,
    vocabulary.RUNNING: 2,
    "blocked": 2,
    "": 2,
    vocabulary.QUEUED: 2,
}


class TrialVerdict(FrozenModel):
    """One trial or job as its receipts left it.

    job: the trial's name inside its stream, or a trial receipt's run id.
    handle: the scheduler or provider handle, empty for an in-process trial.
    target: the alias it ran on, empty when the receipt never named one.
    node: the ledger slug the run serves, empty when none was declared.
    state: the scheduler's own word, empty when nothing reported one.
    verdict: the settled word, `running` while nothing terminal is on file.
    exit_code: the process exit status, when a receipt recorded one.
    detail: where results landed, why it failed, or a trial receipt's own reason.
    gates: the gate sweep summarized, empty when the receipts carry none.
    producer: the harness that stamped a trial receipt, empty for event streams.
    contended: what the machine was already doing when this run started, empty when it attested
        an idle node and empty when nothing attested at all. A cell here is the difference
        between a measurement and a measurement taken while another job held the GPU.
    """

    job: str
    handle: str = ""
    target: str = ""
    node: str = ""
    state: str = ""
    verdict: str = ""
    exit_code: int | None = None
    detail: str = ""
    gates: str = ""
    producer: str = ""
    contended: str = ""

    @property
    def code(self) -> int:
        """This trial's exit code, 3 for a word outside the settled table."""
        return _EXITS.get(self.verdict, 3)


class StreamVerdict(FrozenModel):
    """The settled truth of one stream, every trial's row and what they add up to.

    stream: the stream these rows were read from.
    trials: one row per trial or job, in first-seen order.
    """

    stream: str
    trials: tuple[TrialVerdict, ...]

    @property
    def code(self) -> int:
        """The one exit status a completion check branches on.

        A failure anywhere outranks everything, then anything still in flight, then a trial
        that vanished, and only a stream whose every row settled clean exits zero. An empty
        stream is unknown rather than clean, since receipts that do not exist prove nothing.
        """
        codes = {trial.code for trial in self.trials}
        if 1 in codes:
            return 1
        if 2 in codes:
            return 2
        if 3 in codes or not codes:
            return 3
        return 0


class Verdicts:
    """The receipts-derived outcomes of a workspace's runs, read fresh on every ask."""

    def __init__(self, board: Board) -> None:
        """board: the workspace whose receipts, run registry and sweep this reads."""
        self.board = board

    def handled(self, handle: str, *, host: str = "") -> StreamVerdict:
        """The settled truth of one dispatched run, its registry row under its receipts.

        The registry row is durable dispatch state and always exists for a real handle, so a
        workspace that tracks nothing still gets an answer, and the receipts overlay it with
        whatever richer truth they hold.
        """
        try:
            record = self.board.dispatcher.cache.run(handle, host or None)
        except LookupError as missing:
            raise MissionError(
                f"{handle!r} is not a receipts file, a stream, or a recorded handle: {missing}"
            ) from None
        stream, job = streamed(record.name or "", handle=record.handle)
        stream_file = directory(self.board, stream) / "events.ndjson"
        recorded = eventful(Receipts(stream_file).replay()) if stream_file.is_file() else ()
        mine = [trial for trial in recorded if trial.handle == record.handle]
        return StreamVerdict(stream=stream, trials=tuple(mine) or (registered(record, job=job),))

    def of(self, target: str, *, host: str = "") -> StreamVerdict:
        """The settled truth of `target`, a receipts file, a stream id, or a dispatched handle.

        target: what to read, tried in that order.
        host: the alias narrowing a handle recorded on several hosts.
        """
        path = self.board.dispatcher.local(target)
        if path.is_file():
            return StreamVerdict(stream=target, trials=lined(path))
        stream_file = directory(self.board, target) / "events.ndjson"
        if stream_file.is_file():
            return StreamVerdict(stream=target, trials=eventful(Receipts(stream_file).replay()))
        return self.handled(target, host=host)

    def record(self, handle: str, *, host: str = "") -> RunRecord:
        """The run registry's row for `handle`, refusing a handle nothing ever dispatched."""
        try:
            return self.board.dispatcher.cache.run(handle, host or None)
        except LookupError as missing:
            raise MissionError(f"nothing to wait on: {missing}") from None

    def wait(
        self,
        handle: str,
        *,
        host: str = "",
        timeout: float = 0.0,
        interval: float = vocabulary.POLL_SECONDS,
        poll: Callable[[float], None] = sleep,
    ) -> StreamVerdict:
        """Block until `handle` settles, sweeping the same durable path the monitor cron runs.

        Every pass is one `Monitor.once`, so waiting here pulls results back, releases rentals
        and writes receipts exactly as an unattended sweep would, and a wait killed halfway
        loses nothing. The answer is the receipts-derived outcome, so the exit code a caller
        branches on is the job's own.

        handle: the dispatched run to wait on.
        host: the alias narrowing a handle recorded on several hosts.
        timeout: give up after this many wall seconds, 0 to wait as long as it takes; the
            answer then reports the run still in flight and exits 2.
        interval: seconds between sweeps.
        poll: the sleeper between sweeps, injectable for tests.
        """
        deadline = monotonic() + timeout if timeout else None
        monitor = self.board.monitor()
        while True:
            monitor.once()
            record = self.record(handle, host=host)
            if record.verdict in vocabulary.TERMINAL:
                return self.handled(handle, host=host)
            if deadline is not None and monotonic() >= deadline:
                return self.handled(handle, host=host)
            poll(interval)


def eventful(events: Iterable[Event]) -> tuple[TrialVerdict, ...]:
    """Every job's settled row out of one stream's event envelopes.

    The cursor logic is `latest` per topic per job, the same read every resumed pass uses, so a
    re-dispatched job answers with its newest run and a refused job still has a row.
    """
    recorded = list(events)
    submitted = latest(recorded, Topic.SUBMITTED)
    states = latest(recorded, Topic.STATE)
    settled = latest(recorded, Topic.SETTLED)
    refused = latest(recorded, Topic.REFUSED)
    attested = latest(recorded, Topic.ATTESTED)
    jobs = list(dict.fromkeys([*submitted, *refused]))
    return tuple(
        _joined(
            job,
            submitted=submitted.get(job),
            state=states.get(job),
            ended=settled.get(job),
            refusal=refused.get(job),
            attestation=attested.get(job),
        )
        for job in jobs
    )


def lined(path: Path) -> tuple[TrialVerdict, ...]:
    """Every row a receipts file holds, whichever of the two written shapes each line is.

    An `Event` line joins its stream's per-job cursor; a `trial_receipt` line is one trial,
    whole. A line that is neither readable JSON nor either shape is skipped rather than fatal,
    the same tolerance the receipts replay itself extends to a torn log.
    """
    events: list[Event] = []
    trials: list[TrialVerdict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if _RECEIPT in payload:
            trials.append(receipted(payload[_RECEIPT]))
        else:
            events.append(Event.model_validate(payload))
    return (*eventful(events), *trials)


def receipted(payload: JsonValue) -> TrialVerdict:
    """One printed `trial_receipt` payload as a settled row.

    The contract names `run_id`, `outcome`, `producer`, `node` and `gates` as optional fields
    and any harness may add its own, so everything is read leniently and an absent field is an
    empty cell rather than a refusal.
    """
    data = payload if isinstance(payload, dict) else {}
    return TrialVerdict(
        job=str(data.get("run_id", "")),
        node=str(data.get("node", "")),
        verdict=str(data.get("outcome", "")) or vocabulary.OK,
        detail=str(data.get("reason", "")),
        gates=gated(data.get("gates")),
        producer=str(data.get("producer", "")),
    )


def gated(sweep: JsonValue) -> str:
    """A trial receipt's gate sweep as one cell, the first non-passing gate named.

    sweep: the receipt's `gates` list, entries of `status` and `reason`.
    """
    if not isinstance(sweep, list) or not sweep:
        return ""
    checks = [entry for entry in sweep if isinstance(entry, dict)]
    for check in checks:
        status = str(check.get("status", ""))
        if status and status != "passed":
            return f"{status}: {check.get('reason', '')}"
    return f"{len(checks)} passed"


def registered(record: RunRecord, *, job: str) -> TrialVerdict:
    """The run registry's own row as a settled row, the floor a receiptless run answers from."""
    return TrialVerdict(
        job=job,
        handle=record.handle,
        target=record.target,
        node=record.node,
        state=record.state or "",
        verdict=record.verdict or vocabulary.RUNNING,
        exit_code=record.exit_code,
    )


def contention(attestation: Event | None) -> str:
    """What a job's attestation says it started under, empty for an idle node or no attestation.

    Only the unwelcome half is rendered, since a clean measurement's whole point is that there is
    nothing to say about it, and a column full of the word `idle` would bury the one row that
    matters. The busy figure rides along so a reader can weigh it rather than take the flag's
    word for it.

    attestation: the run's `job.attested` line, None when nothing attested.
    """
    if attestation is None or attestation.data.get("idle"):
        return ""
    return f"gpu {attestation.data.get('gpu_pct', 0)}% busy at start"


def _joined(
    job: str,
    *,
    submitted: Event | None,
    state: Event | None,
    ended: Event | None,
    refusal: Event | None,
    attestation: Event | None = None,
) -> TrialVerdict:
    """One job's row folded from its latest line per topic.

    A settled line wins the verdict, a state line stands in while the job flies, and a refusal
    is terminal in its own words. A job submitted again after settling compares handles, so a
    stale settlement never silences the run of it that is still going. An attestation is carried
    onto every row the run has, since what the machine was doing at the start is as true of the
    finished measurement as it was of the running one.
    """
    handle = str(submitted.data.get("handle", "")) if submitted else ""
    target = str(submitted.data.get("target", "")) if submitted else ""
    node = str(submitted.data.get("node", "")) if submitted else ""
    contended = contention(attestation)
    if submitted is None and refusal is not None:
        return TrialVerdict(
            job=job,
            target=str(refusal.data.get("target", "")),
            verdict="refused",
            detail=str(refusal.data.get("reason", "")),
        )
    current = str(state.data.get("state", "")) if state else ""
    verdict = str(state.data.get("verdict", "")) if state else ""
    if ended is not None and str(ended.data.get("handle", "")) == handle:
        code = ended.data.get("exit_code")
        return TrialVerdict(
            job=job,
            handle=handle,
            target=target,
            node=node,
            state=current,
            verdict=str(ended.data.get("verdict", "")),
            exit_code=code if isinstance(code, int) else None,
            detail=str(ended.data.get("detail", "")),
            contended=contended,
        )
    return TrialVerdict(
        job=job,
        handle=handle,
        target=target,
        node=node,
        state=current,
        verdict=verdict or vocabulary.RUNNING,
        contended=contended,
    )
