"""Settling one run: its output captured, its results pulled home, its evidence recorded, what it
holds released, and every move published into its own receipts stream."""

import json
import shlex
from collections.abc import Iterable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from plumbum.commands.processes import ProcessExecutionError

from ...batch.receipts import SettlementReceipts, Topic
from ...core.errors import MissionError
from ...dispatch.jobs import vocabulary
from ...dispatch.jobs.evidence import covered_in, receipts_in
from ...dispatch.schedulers import HostUnreachable, is_quota_refusal
from ...dispatch.state import Failed, Held, Resumed
from ...dispatch.state.captured import Captured
from ...log import logger
from ...tracking import is_batched, streamed
from ...trials.artifacts import Artifacts

if TYPE_CHECKING:
    from pydantic import JsonValue

    from ...dispatch.jobs.vocabulary import JobState
    from ...dispatch.state import RunRecord
    from ...fleet.board import Board, Run

# What a resubmission may fail with before it becomes this run's row rather than the end of the
# sweep, the set a batch dispatch absorbs: one target refusing says nothing about the next run.
_REFUSALS = (MissionError, HostUnreachable, OSError, LookupError, SystemExit)


class Unverifiable(MissionError):
    """A finished run whose evidence can never be verified: the host holds nothing more to wait
    for, so the run settles with its evidence marked unverified instead of pending."""


class Settlement:
    """What settling one run takes, whichever pass or verb settles it.

    `Monitor` settles every tracked run this way and a cancel settles one. Capture must precede
    release, which destroys a rented instance and its log.

    quiet: the hosts that went quiet in this pass, each with why.
    fetched: what this pass collected, by host, root and path: the runs of one experiment share
        its folder, and a pass collected it once per run (a dozen scans of 190,000 files on
        miyabi's Lustre, 2026-10-09).
    """

    @staticmethod
    def _artifacts(line: str) -> list[JsonValue]:
        """The artifacts one captured receipt declares."""
        payload = _receipt(line)
        if not isinstance(payload, dict) or not isinstance(
            declared := payload.get("artifacts", {}), dict
        ):
            raise ValueError("malformed trial receipt; raw line preserved in captured log")
        return list(declared.values())

    @staticmethod
    def _cases(receipts: Iterable[str]) -> list[JsonValue]:
        """Associate valid envelopes only; raw malformed lines stay in the captured log."""
        payloads = [
            payload
            for line in receipts
            if isinstance(payload := Settlement._parsed(line), Mapping)
        ]
        return [
            [str(payload.get("run", "")), str(payload.get("case_id") or payload.get("run_id", ""))]
            for payload in payloads
        ]

    @staticmethod
    def _parsed(line: str) -> JsonValue:
        """The receipt one captured line carries, None for a line that is no JSON at all."""
        with suppress(json.JSONDecodeError):
            return _receipt(line)
        return None

    def __init__(self, board: Board) -> None:
        self.board = board
        self.cache = board.dispatcher.cache
        self.captured = Captured(self.cache.session)
        self.streams: dict[str, SettlementReceipts] = {}
        self.quiet: dict[str, str] = {}
        self.fetched: dict[tuple[str, str, str], str | None] = {}

    def _asked(self, record: RunRecord) -> Run | Failed | None:
        """Ask `record`'s target for its held dispatch again, None while its quota is full."""
        if record.request is None:
            return None
        try:
            return self.board.dispatch(record.request)
        except _REFUSALS as refusal:
            return self._refused(record, refusal)

    def _checked(self, receipts: tuple[str, ...], pulled: str) -> None:
        """Check the artifacts `receipts` reference against what was pulled to `pulled`.

        A lake the host no longer holds makes the evidence unverifiable rather than pending.
        """
        try:
            Artifacts.verify(
                receipts, directory=self.board.dispatcher.local(pulled), boundary=self.board.root
            )
        except (MissionError, FileNotFoundError) as fault:
            if not str(fault).startswith("no lake around"):
                raise
            raise Unverifiable(
                f"receipts reference artifacts the host no longer holds ({fault})"
            ) from fault

    def _refused(self, record: RunRecord, refusal: BaseException) -> Failed | None:
        """A held dispatch's refusal: None for a full quota, which the next sweep asks again.

        Any other refusal (a queue that does not exist, an account without permission) answers
        the same way forever, so the row settles failed and that failure is returned for the
        only pass that still sees it.
        """
        if is_quota_refusal(str(refusal)):
            return None
        failed = vocabulary.Word.FAILED
        self.cache.report(self.cache.resolve(record, failed, None, failed), failed)
        logger.warning("held dispatch for {} refused: {}", record.target, refusal)
        return failure(record, f"held dispatch refused: {refusal}")

    def _unpulled(self, job: Run, fault: Exception) -> None:
        """Warn about a failed pull, and note a host that went quiet for the rest of the pass."""
        if isinstance(fault, HostUnreachable):
            self.quiet[job.handle.host] = str(fault)
        logger.warning(
            "could not pull {} from {}: {}", job.handle.fetch_path, job.handle.host, fault
        )

    def answering(self, job: Run) -> None:
        """Refuse to contact a host that already went quiet in this pass.

        A dead host costs a full connect timeout per contact, and a settled run is contacted for
        its results and again for its log, so a handful of runs on one dead host held every pass
        for minutes. Later runs on that host are left pending for the next pass unasked.
        """
        if (fault := self.quiet.get(job.handle.host)) is not None:
            raise MissionError(f"{job.handle.host} went quiet earlier in this pass: {fault}")

    def capture(self, record: RunRecord, job: Run, *, required: bool = True) -> tuple[str, ...]:
        """Save the native log and deduplicated receipts before a rental can be destroyed.

        A retry can read the saved log after the provider disappears. Files are flushed before
        the delivery checkpoint is published; receipt references are verified separately.

        required: a log that could not be read keeps the run pending (B03); a cancel, which
            settles the run whatever its host says, takes what was saved instead.
        """
        stream, name = streamed(record.name or "", handle=record.handle)
        transcript = job.transcript()
        if not transcript:
            saved = self.captured.transcript(stream, handle=record.handle)
            if required and transcript is None and saved is None:
                # Settled now, the run would be released with a log nobody read (B03); pending,
                # it is asked again next pass, and its release deadline still bounds the rental.
                raise MissionError("the run's log could not be read; kept for the next pass")
            return receipts_in(saved or "")
        harvested = receipts_in(transcript)
        self.captured.keep_transcript(stream, handle=record.handle, transcript=transcript)
        self.captured.keep_receipts(stream, harvested)
        logger.info(
            "captured {} log lines for {} ({})", transcript.count("\n"), record.handle, name
        )
        return harvested

    def evidence(
        self, record: RunRecord, receipts: tuple[str, ...], *, status: str, detail: str = ""
    ) -> None:
        """Append transfer status without rewriting the computational receipt or exit code."""
        self.cache.delivery(record, status)
        stream, job = streamed(record.name or "", handle=record.handle)
        bus = self.stream(stream)
        data: dict[str, JsonValue] = {
            **identity(record),
            "status": status,
            "detail": detail,
            "trials": Settlement._cases(receipts),
        }
        seen = bus.get(Topic.EVIDENCE, job)
        if seen is None or seen.data != data:
            try:
                bus.publish(Topic.EVIDENCE, job=job, data=data)
            except OSError as fault:
                logger.error(
                    "evidence status is cached but its event could not be saved: {}", fault
                )

    def pull(self, job: Run) -> str | None:
        """Bring a job's recorded results home whatever its verdict, answering where they landed.

        A crashed run keeps everything it wrote, the store staging and renaming each fragment so
        those readings survive the trial that killed it, so a failed job is the one most worth
        carrying home. None when no results path was recorded at dispatch or the pull failed (a
        directory never written, a host dropped mid-transfer, a rented disk that dies with the
        rental): one missing artifact is a warning, never a sweep that dies holding every other
        job's outcome.
        """
        path = job.handle.fetch_path
        if not path:
            return None
        # A run finished before this pass asked, so a collection earlier in the pass holds it.
        key = (job.handle.host, job.handle.root, path)
        if key in self.fetched:
            return self.fetched[key]
        self.fetched[key] = None
        try:
            job.pull()
        except (HostUnreachable, ProcessExecutionError, MissionError, OSError) as fault:
            self._unpulled(job, fault)
            return None
        self.fetched[key] = path
        return path

    def release(self, job: Run) -> bool:
        """Let a settled run go, so nothing keeps billing for work that already ended.

        A scheduler job releases nothing; a provider run is cancelled here, the only thing that
        ends the rental. Asking twice is expected, since a pass may re-run one an earlier pass
        released, and a refused cancel is a warning, never the end of the sweep.
        """
        try:
            job.release()
        except (MissionError, OSError) as fault:
            logger.warning("could not release {} on {}: {}", job.handle.id, job.handle.host, fault)
            return False
        return True

    def resumed(self, record: RunRecord) -> Resumed | Held | Failed:
        """Ask for one quota-held dispatch again: resumed under its new handle, held, or refused.

        A request that goes through replaces its placeholder row under the new handle and tells the
        batch that asked, through its own receipts.
        """
        run = self._asked(record)
        if run is None:
            return Held(handle=record.handle, target=record.target, reason=record.reason)
        if isinstance(run, Failed):
            return run
        self.cache.forget(record)
        self.submitted(record, run)
        logger.info("held dispatch went through as {} on {}", run.handle.id, record.target)
        return Resumed(handle=run.handle.id, target=record.target, name=record.name)

    def stream(self, name: str) -> SettlementReceipts:
        """The receipts stream `name`, replayed on first use and then kept for the whole pass."""
        if name not in self.streams:
            self.streams[name] = SettlementReceipts(self.board.receipts(name), name)
        return self.streams[name]

    def submitted(self, record: RunRecord, run: Run) -> None:
        """Publish the submission a batch would have written for its held job that went out here.

        Without it a watch reading submissions out of the receipts shows the job held forever
        while it runs.
        """
        if not is_batched(record.name):
            return
        stream, job = streamed(record.name, handle=run.handle.id)
        self.stream(stream).publish(
            Topic.SUBMITTED,
            job=job,
            data={
                "handle": run.handle.id,
                "target": record.target,
                "kind": run.handle.kind,
                "command": record.script,
                **({"node": record.node} if record.node else {}),
            },
        )

    def track(self, record: RunRecord, state: JobState, *, detail: str) -> None:
        """Publish what this pass learned about one run into that run's own receipts stream.

        This tracks a plain submit like a batch job. A batched run is skipped,
        since its batch's watch already publishes every line and a second publisher would double
        every row. Only a move is published, so a cron pass that finds nothing new writes nothing.

        detail: where its results landed or why it failed, empty while it is still in flight.
        """
        label = record.name or ""
        if is_batched(label):
            return
        stream, job = streamed(label, handle=record.handle)
        bus = self.stream(stream)
        moved = {**identity(record), "state": state.state or "", "verdict": state.verdict}
        if (seen := bus.get(Topic.STATE, job)) is not None and seen.data == moved:
            return
        bus.publish(Topic.STATE, job=job, data=moved)
        if state.verdict in vocabulary.Word.settled():
            settled = {
                **identity(record),
                "verdict": state.verdict,
                "exit_code": state.exit_code,
                "detail": detail,
            }
            bus.publish(Topic.SETTLED, job=job, data=settled)

    def verify(
        self, job: Run, pulled: str | None, receipts: tuple[str, ...], *, owed: bool = False
    ) -> None:
        """A successful transport is not proof that its declared artifacts arrived.

        owed: the run owes a captured receipt (`owes_receipt`).
        """
        artifacts = [value for line in receipts for value in Settlement._artifacts(line)]
        if job.handle.fetch_path and pulled is None:
            raise MissionError("result transfer failed; remote evidence retained")
        if owed and job.handle.fetch_path and not receipts:
            raise Unverifiable("native trial has no captured receipt; evidence is unverified")
        if pulled is not None and receipts:
            self._checked(receipts, pulled)
        elif any(isinstance(value, dict) for value in artifacts):
            raise MissionError("receipt references artifacts but no fetch was declared")


def failure(record: RunRecord, reason: str) -> Failed:
    """`record`'s row in a report's failed list."""
    return Failed(handle=record.handle, target=record.target, reason=reason)


def identity(record: RunRecord) -> dict[str, JsonValue]:
    """The fields that tie a published line to this one submission of `record`."""
    return {"handle": record.handle, "target": record.target, "submitted_at": record.submitted_at}


def owes_receipt(
    record: RunRecord, *, transcript: str = "", verdict: str = vocabulary.Word.OK
) -> bool:
    """Whether `record`'s run owes a captured receipt: a native trial claiming success.

    A transcript of a trials session whose every cell was already complete and skipped owes none,
    the one native run that legitimately captures no receipt. Only a run claiming success owes
    one: holding a crash pending re-pulled it every sweep for the cache's lifetime, and ten such
    runs cost every `wait` a minute per pass.
    """
    native = any(
        Path(token.partition("::")[0]).name.startswith("test_")
        for token in shlex.split(record.script)
    )
    return native and verdict == vocabulary.Word.OK and not covered_in(transcript)


def _receipt(line: str) -> JsonValue:
    """The `trial_receipt` payload of one captured line, None when it is not a receipt envelope."""
    envelope = json.loads(line)
    return envelope.get("trial_receipt") if isinstance(envelope, dict) else None
