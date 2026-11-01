"""mb's offline event spool: OTLP log records as JSON lines on the internal disk.

A line is one `ExportLogsServiceRequest` holding one record, so the collect pass POSTs it to the
sink as it stands (the sink rejects a body of two lines), and DuckDB `read_json` reads the file
when no sink runs. A spool file has one writer, which holds its lock for life, and only grows.
Spooled events are telemetry, lossy and untrusted by rule: the sink stores a line twice if it is
sent twice, so the collect pass keeps the offset it has delivered up to and sends no event whose
id the lake already holds.
"""

import fcntl
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Self
from uuid import UUID, uuid7

from patos import FrozenModel
from pydantic import AwareDatetime, Field

from ...constants import EVENT_ID
from ..errors import MissionError
from .identity import Attribute, Identity
from .wire import KeyValue, Record, Request, Value

if TYPE_CHECKING:
    from types import TracebackType

_INFO = 9
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_NANOS_PER_MICRO = 1000
_APPENDING = os.O_WRONLY | os.O_CREAT | os.O_APPEND


class Event(FrozenModel):
    """One operational fact: what happened, when, and the identities and measures around it.

    name: the OTLP `eventName`, dotted and namespaced (`mb.job.settled`).
    severity: the OTLP severity number, 9 for INFO.
    attributes: the keys of `Key` for identities (see `Identity.attributes`), anything else
        beside them.
    id: the record's own id, which makes a delivery single: time-ordered and random unless the
        caller mints it, as a uuid5 makes one request one event however often it is spooled.
    """

    name: str
    at: AwareDatetime
    body: str = ""
    severity: int = _INFO
    attributes: dict[str, Attribute] = {}
    id: UUID = Field(default_factory=uuid7)

    @classmethod
    def parse(cls, line: str) -> Self:
        """The event on one spool line; `ValueError` for a line of another shape or schema."""
        record = Request.model_validate_json(line).record
        attributes = {pair.key: pair.value.read() for pair in record.attributes}
        minted = attributes.pop(EVENT_ID, "")
        return cls(
            name=record.event_name,
            at=_EPOCH + timedelta(microseconds=int(record.time_unix_nano) // _NANOS_PER_MICRO),
            body=str(record.body.read()),
            severity=record.severity_number,
            attributes=attributes,
            id=UUID(str(minted)),
        )

    @property
    def identity(self) -> Identity:
        """The identities this event carries."""
        return Identity.from_attributes(self.attributes)

    def line(self) -> str:
        """The spool line, without its newline."""
        record = Record(
            time_unix_nano=str((self.at - _EPOCH) // timedelta(microseconds=1) * _NANOS_PER_MICRO),
            severity_number=self.severity,
            event_name=self.name,
            body=Value.of(self.body),
            attributes=[
                KeyValue(key=key, value=Value.of(value))
                for key, value in {**self.attributes, EVENT_ID: str(self.id)}.items()
            ],
        )
        return Request.of(record).line()


class Page(FrozenModel):
    """The events a read found and the offset to read from next.

    offset: the byte position after the last complete line, which the collect pass keeps.
    """

    events: tuple[Event, ...]
    offset: int


class EventSpool:
    """The single writer of one spool file, which it creates private and only appends to.

    Opening takes an exclusive lock for the writer's life, so a second writer is refused, and
    cuts a half-written last line a crashed predecessor left, so that line never gets another's
    glued onto it. Name the file for the process (a service and a day), and let the collect
    pass delete a file it has delivered whole once no writer holds it.
    """

    @staticmethod
    def read(path: Path, offset: int = 0) -> Page:
        """The events from byte `offset` to the last complete line of `path`.

        A line another writer is still writing is left for the next read. `ValueError` names the
        offset of a complete line that is not a spool line, since skipping it would hide a bug.
        """
        with path.open("rb") as handle:
            handle.seek(offset)
            complete = handle.read().rpartition(b"\n")[0]
        events: list[Event] = []
        position = offset
        for raw in complete.split(b"\n") if complete else []:
            try:
                events.append(Event.parse(raw.decode()))
            except ValueError as error:
                raise ValueError(f"{path} at byte {position}: {error}") from error
            position += len(raw) + 1
        return Page(events=tuple(events), offset=position)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = os.fdopen(os.open(path, _APPENDING, 0o600), "ab", buffering=0)
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            raise MissionError(f"{path} already has a writer") from None
        self.handle.truncate(path.read_bytes().rfind(b"\n") + 1)

    def append(self, event: Event) -> None:
        """Write `event` as one line, in one `write` unless the disk takes it in parts."""
        pending = memoryview(f"{event.line()}\n".encode())
        while pending:
            pending = pending[self.handle.write(pending) :]

    def close(self) -> None:
        """Release the file and its lock; a second call is harmless."""
        self.handle.close()
