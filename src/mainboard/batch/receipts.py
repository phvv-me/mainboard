# THE BATCH EVENT CONTRACT, and the one place it is written down.
#
# Everything a batch learns is published as one line, and every later verb reads its cursor back
# out of these lines rather than memory. The store is the workspace lake today and could be a
# broker tomorrow, so nothing downstream may depend on the lines being local or written by the
# reader.
#
# THE ENVELOPE. Every line is one `Event`. Envelope fields never carry payload and payload never
# carries routing, so a subscriber filters on the envelope without parsing what it filters.
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
# THE RULES that make the transport swappable. Every line derives from durable state (the dispatch
# cache and lines already published), so a dying pass republishes nothing and a pass that never
# ran just publishes later. Every line is idempotent in meaning (the log is the cursor), as a
# broker's at-least-once delivery needs. Ordering is per job, not global. What a job spends is
# published before it settles, so the terminal line is the last one and a sink may close on it.

import json
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from patos import FrozenModel
from pydantic import JsonValue, ValidationError
from sqlalchemy import literal_column, select

from ..dispatch.shared import now
from ..log import logger
from ..state import schema
from ..state.lake import Session

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


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


# The three answers a target gives about one offered job, read together since a row asks only
# for the last word (`latest` over the three).
OFFERED = (Topic.SUBMITTED, Topic.REFUSED, Topic.HELD)


class Event(FrozenModel):
    """One published line: where it belongs, what it says, and what it says it about.

    at: ISO-8601 publish time.
    batch: the batch id, the stream this line belongs to and the key a broker partitions on.
    job: the job name inside the batch, empty for a batch-wide line.
    data: the topic's own payload, as the table above documents it.
    """

    at: str
    batch: str
    topic: Topic
    job: str = ""
    data: dict[str, JsonValue] = {}


class Bus(Protocol):
    """Where a batch's receipts go: the lake now, a broker later, the same two verbs either way."""

    def publish(self, event: Event) -> None:
        """Hand one event to the transport."""

    def replay(self) -> list[Event]:
        """Every event this batch has published, oldest first."""


class Journal:
    """The lake transport: the workspace lake's `events`, one batch's lines selected by its id.

    Replayed in commit order, the order a file would have been appended in. A row whose topic
    this release no longer knows is skipped rather than fatal, the tolerance a torn file gets.
    """

    def __init__(self, session: Session, batch: str) -> None:
        self.session = session
        self.batch = batch

    def publish(self, event: Event) -> None:
        self.session.append(
            schema.events,
            [
                {
                    "ts": event.at,
                    "batch": event.batch,
                    "topic": str(event.topic),
                    "job": event.job,
                    "data": event.data,
                }
            ],
        )

    def replay(self) -> list[Event]:
        kept = schema.events
        rows = self.session.rows(
            select(kept.c.ts, kept.c.batch, kept.c.topic, kept.c.job, kept.c.data)
            .where(kept.c.batch == self.batch)
            .order_by(literal_column("rowid"))
        )
        events: list[Event] = []
        for at, batch, topic, job, data in rows:
            try:
                events.append(
                    Event(
                        at=at.isoformat(),
                        batch=batch,
                        topic=topic,
                        job=job or "",
                        data=json.loads(data) if data else {},
                    )
                )
            except ValidationError:
                logger.warning("unreadable event of {} retained in the lake: {}", batch, topic)
        return events


def publish(
    bus: Bus, batch: str, topic: Topic, *, job: str = "", data: Mapping[str, JsonValue]
) -> Event:
    """Stamp `data` as one event of `topic` and hand it to `bus` (events are built only here)."""
    event = Event(at=now(), batch=batch, topic=topic, job=job, data=dict(data))
    bus.publish(event)
    return event


def payload(record: FrozenModel) -> dict[str, JsonValue]:
    """`record` as an event payload, the JSON round trip a broker would put it through anyway."""
    return json.loads(record.model_dump_json())


def latest(events: Iterable[Event], *topics: Topic) -> dict[str, Event]:
    """The most recent event of any of `topics` per job, the cursor a resumed pass reads.

    Recent by the envelope's `at`, not arrival: a broker redelivers at least once and orders per
    job at best, so taking the last line seen would let an older redelivery win. A tie keeps the
    later arrival (a file appending twice in one clock tick).
    """
    wanted = frozenset(topics)
    newest: dict[str, Event] = {}
    for event in events:
        held = newest.get(event.job)
        if event.topic in wanted and (held is None or event.at >= held.at):
            newest[event.job] = event
    return newest
