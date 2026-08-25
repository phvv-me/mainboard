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
#   job.state       {"handle", "state", "verdict"}                published only when it changed
#   job.sample      a live machine reading, at the declared interval, from the node itself
#   job.cost        {"platform", "gpu", "setup_s", "run_s", "observed"}
#   job.settled     {"handle", "verdict", "exit_code", "detail"}  terminal, once and last
#   batch.closed    {"jobs", "ok", "failed"}                      every job settled, once
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
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from patos import FrozenModel
from pydantic import JsonValue

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
    STATE = "job.state"
    SAMPLE = "job.sample"
    SETTLED = "job.settled"
    COST = "job.cost"
    CLOSED = "batch.closed"


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
        with self.path.open("a", encoding="utf-8") as opened:
            opened.write(event.model_dump_json() + "\n")

    def replay(self) -> list[Event]:
        """Every recorded event, oldest first, empty when nothing has been published yet."""
        if not self.path.is_file():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [Event.model_validate_json(line) for line in lines if line.strip()]


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


def latest(events: Iterable[Event], topic: Topic) -> dict[str, Event]:
    """The most recent event of `topic` per job, the cursor a resumed pass reads."""
    return {event.job: event for event in events if event.topic is topic}
