# THE BATCH EVENT CONTRACT, and the one place it is written down.
#
# Everything a batch learns is published here as one line, and every later verb reads its own
# cursor back out of these lines rather than out of memory. That is deliberate: the store behind
# this module is a file today and a broker tomorrow, and the swap has to be a transport swap
# alone, so nothing downstream may depend on the lines being local, ordered by inode, or written
# by the same process that reads them.
#
# THE ENVELOPE. Every line is one `Event`: `at` (ISO-8601), `batch` (the stream it belongs to and
# the key a broker partitions on), `topic` (the routing key), `job` (the job inside the batch,
# empty for a batch-wide line) and `data` (that topic's own payload). Envelope fields never carry
# payload and payload never carries routing, so a subscriber filters on the envelope without
# parsing what it is filtering.
#
# THE TOPICS, and what each one's `data` holds:
#   batch.opened    {"name", "jobs": [job names], "root"}         once, by the first verb to write
#   job.prepared    a `TransferSet`                               what must still reach the target
#   job.estimated   a `JobEstimate`                               what it is expected to cost
#   job.submitted   {"handle", "target", "kind", "command"}       a scheduler or provider took it,
#                   plus "node" (the ledger slug the run serves) only when one was declared
#   job.refused     {"target", "reason"}                          the target would not take it
#   job.held        {"target", "reason"}                          the target's quota was full, so
#                   the request is kept at this workstation and the durable sweep asks again;
#                   a `job.submitted` for the same job later is what says it finally went
#   job.skipped     {"target", "reason"}                          the run's selection left it out,
#                   so nothing is coming for it and no reader should wait on it
#   job.state       {"handle", "state", "verdict"}                published only when it changed
#   job.attested    a machine reading plus "idle", taken in the foreground on the node itself the
#                   instant before the command started, so a measurement can say what conditions
#                   it was taken under rather than leaving a reader to assume clean ones
#   job.sample      a live machine reading, at the declared interval, from the node itself
#   job.cost        {"platform", "gpu", "setup_s", "run_s", "observed", "expected_usd",
#                   "actual_usd", "delta_usd"}, what the run was quoted at beside what it
#                   came to, so the cost model learns from its own misses
#   job.settled     {"handle", "verdict", "exit_code", "detail"}  terminal computation
#   job.evidence    {"handle", "target", "submitted_at", "status", "trials", "detail"}
#                   delivery/release checkpoint or append-only correction, keyed to the run;
#                   status is pending, copied, verified, not_started, or unverified
#   batch.closed    {"jobs", "ok", "failed", "skipped"}           every job settled, once
#
# THE RULES that make the transport swappable. Every line is derived from durable state (the
# dispatch cache and the lines already published), so a pass that dies republishes nothing and a
# pass that never ran loses nothing, it just publishes later. Every line is idempotent in
# meaning: a topic that must happen once is written once because the log itself is the cursor,
# which is exactly what a broker's at-least-once delivery needs from its producers. And ordering
# is per job rather than global, so a partitioned topic reads the same as this file does. What a
# job spends is published before the job settles, so the terminal line is genuinely the last one
# a subscriber sees about that job and a sink may close its own record on reading it.

import json
import logging
import os
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from filelock import FileLock
from patos import FrozenModel
from pydantic import JsonValue, ValidationError

from ..dispatch.shared import now

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

# One logger for the flow, so a caller quieting `mainboard.batch` quiets every module in it.
logger = logging.getLogger("mainboard.batch")


class Topic(StrEnum):
    """Every routing key a batch publishes under, dotted the way a broker subject is."""

    OPENED = "batch.opened"
    PREPARED = "job.prepared"
    ESTIMATED = "job.estimated"
    SUBMITTED = "job.submitted"
    REFUSED = "job.refused"
    HELD = "job.held"
    SKIPPED = "job.skipped"
    STATE = "job.state"
    ATTESTED = "job.attested"
    SAMPLE = "job.sample"
    SETTLED = "job.settled"
    EVIDENCE = "job.evidence"
    COST = "job.cost"
    CLOSED = "batch.closed"


# The three answers a target can give about one offered job. They are named together because no
# reader wants them apart: a row's question is what the last word about this job was, not what
# was submitted and separately what was refused, and `latest` over the three is that question.
OFFERED = (Topic.SUBMITTED, Topic.REFUSED, Topic.HELD)


class Event(FrozenModel):
    """One published line: where it belongs, what it says, and what it says it about.

    at: ISO-8601 publish time.
    batch: the batch id, the stream this line belongs to.
    topic: the routing key, from `Topic`.
    job: the job name inside the batch, empty for a batch-wide line.
    data: the topic's own payload, exactly as this module's contract documents it.
    """

    at: str
    batch: str
    topic: Topic
    job: str = ""
    data: dict[str, JsonValue] = {}


class Bus(Protocol):
    """Where a batch's receipts go: one file now, a broker later, the same two verbs either way."""

    def publish(self, event: Event) -> None:
        """Hand one event to the transport."""

    def replay(self) -> list[Event]:
        """Every event this batch has published, oldest first."""


class Receipts:
    """The file transport: one NDJSON line per event under the batch's own directory.

    Append-only and read whole, since a batch is tens of jobs and a few lines each. A line that
    is not readable JSON is skipped rather than fatal, so a log truncated by a crash still
    replays everything written before the tear.
    """

    def __init__(self, path: Path) -> None:
        """path: the batch's `events.ndjson`, created with its directory on first publish."""
        self.path = path

    def publish(self, event: Event) -> None:
        """Append one event line durably."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(self.path.with_suffix(".lock")), self.path.open("a+b") as opened:
            if opened.tell():
                opened.seek(-1, os.SEEK_END)
                if opened.read(1) != b"\n":
                    opened.write(b"\n")
            opened.write((event.model_dump_json() + "\n").encode())
            opened.flush()
            os.fsync(opened.fileno())

    def replay(self) -> list[Event]:
        """Every recorded event, oldest first, empty when nothing has been published yet."""
        if not self.path.is_file():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        events: list[Event] = []
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                events.append(Event.model_validate_json(line))
            except ValidationError:
                logger.warning("unreadable receipt line retained at %s:%d", self.path, number)
        return events


class Mirrored:
    """One canonical transport with best-effort copies beside it, the shape a reporting sink joins.

    The canonical bus is the record and every mirror is a courtesy, which is the whole contract
    here. The canonical publish happens first and a mirror that raises is logged and dropped
    after it, so an expired token, a rate limit or a node with no route out costs one copy of one
    line and never the line itself. `replay` reads the canonical bus alone for the same reason: a
    cursor a resumed pass reads has to come from the transport guaranteed to hold every line, and
    a sink is somewhere events go rather than somewhere they come from.

    A mirror is caught broadly on purpose. It is a whole vendor SDK behind one call, so the set
    of ways it can fail is not ours to enumerate, and the one outcome this class exists to
    prevent is a batch dying because a dashboard did.
    """

    def __init__(self, canonical: Bus, *mirrors: Bus) -> None:
        """canonical: the transport that must receive every event.

        mirrors: the copies, each published to after the canonical one and never instead of it.
        """
        self.canonical = canonical
        self.mirrors = mirrors

    def publish(self, event: Event) -> None:
        """Record `event` durably, then offer it to each mirror."""
        self.canonical.publish(event)
        for mirror in self.mirrors:
            try:
                mirror.publish(event)
            except Exception:
                logger.warning(
                    "mirror %s dropped %s for %s",
                    type(mirror).__name__,
                    event.topic,
                    event.job or event.batch,
                    exc_info=True,
                )

    def replay(self) -> list[Event]:
        """Every event, from the canonical transport, which is the only one that holds them all."""
        return self.canonical.replay()


def publish(
    bus: Bus, batch: str, topic: Topic, *, job: str = "", data: Mapping[str, JsonValue]
) -> Event:
    """Stamp `data` as one event of `topic` and hand it to `bus`, returning what was published.

    The one place an event is built, so every line carries the same envelope however far from
    here the payload was assembled.
    """
    event = Event(at=now(), batch=batch, topic=topic, job=job, data=dict(data))
    bus.publish(event)
    return event


def payload(record: FrozenModel) -> dict[str, JsonValue]:
    """`record` as an event payload, the JSON round trip a broker would put it through anyway."""
    return json.loads(record.model_dump_json())


def latest(events: Iterable[Event], *topics: Topic) -> dict[str, Event]:
    """The most recent event of `topics` per job, the cursor a resumed pass reads.

    Recent by the envelope's own `at` rather than by where the line happened to land, because
    the transport this contract is written for is a broker: a partition delivers at least once
    and promises order per job at best, so a reader that simply took the last line it saw would
    let a redelivered older line overwrite the newer one it already had. An ISO-8601 stamp sorts
    as the instant it names, and a tie keeps the later arrival, which is what a file transport
    appending twice inside one clock tick means.

    Several topics read as one cursor, which is how a reader asks the question it actually has.
    `job.submitted`, `job.refused` and `job.held` are three answers to a single offer, so what a
    row wants is the newest of the three rather than the newest of each and a rule for ranking
    them afterwards; asked here, the rule is the clock and there is only one copy of it.
    """
    wanted = frozenset(topics)
    newest: dict[str, Event] = {}
    for event in events:
        held = newest.get(event.job)
        if event.topic in wanted and (held is None or event.at >= held.at):
            newest[event.job] = event
    return newest
