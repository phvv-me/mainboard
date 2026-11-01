"""The drain pass: each spooled event reaches the sink once however often the pass runs, and a sink
that is down, or another pass delivering, loses nothing and fails nothing."""

import fcntl
from collections.abc import Callable, Collection, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mainboard.constants import EVENT_ID
from mainboard.core.events import Event, EventSpool
from mainboard.serve.events.drain import Drain
from mainboard.serve.events.sink import Sink
from mainboard.state import TelemetryLake

from ...state.telemetry.builders import Record, lake_with
from ..fakes import FakeSink

_NOON = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)


def _spooled(lake: TelemetryLake, events: Sequence[Event]) -> None:
    """`events` written to one spool file by a writer that then lets go."""
    with EventSpool(lake.spool / "drained.jsonl") as spool:
        for event in events:
            spool.append(event)


def _drained(sink: Sink, *, stored: Sequence[Event]) -> int:
    """One drain pass over a lake holding `stored`, as the sink stores mb's lines."""
    sink.token.ensure()
    records = [
        Record(
            at=event.at.replace(tzinfo=None),
            attributes={EVENT_ID: str(event.id)},
            observed=event.at.replace(tzinfo=None),
            service="mb",
        )
        for event in stored
    ]
    lake_with(sink.root, records)
    return Drain(sink).drain()


@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(count=st.integers(1, 5), stored=st.sets(st.integers(0, 4)))
def test_draining_twice_sends_each_event_the_lake_lacks_once_and_then_nothing(
    reach: Callable[[Path, Path], Sink], count: int, stored: Collection[int]
) -> None:
    events = [Event(name="mb.test.drained", at=_NOON + timedelta(seconds=n)) for n in range(count)]
    held = [event for at, event in enumerate(events) if at in stored]
    with TemporaryDirectory() as scratch, FakeSink(Path(scratch) / "sink.sock") as fake:
        sink = reach(Path(scratch), fake.path)
        _spooled(sink.lake, events)
        first = _drained(sink, stored=held)
        _spooled(sink.lake, events)
        second = _drained(sink, stored=events)
    assert (first, second) == (count - len(held), 0)
    assert sorted(fake.bodies) == sorted(event.line() for event in events if event not in held)


def test_a_drain_with_the_sink_down_keeps_every_line_and_ends_clean(
    workspace: Path, reach: Callable[[Path, Path], Sink]
) -> None:
    sink = reach(workspace, workspace / "absent.sock")
    _spooled(sink.lake, [Event(name="mb.test.kept", at=_NOON, body=str(n)) for n in range(3)])
    before = sink.lake.pending()
    assert (Drain(sink).drain(), sink.lake.pending()) == (0, before)
    assert [(file.file, file.lines) for file in before] == [("drained.jsonl", 3)]


def test_a_drain_sends_nothing_while_another_pass_delivers_and_all_once_it_ends(
    sink: Sink, fake: FakeSink
) -> None:
    _spooled(sink.lake, [Event(name="mb.test.locked", at=_NOON, body=str(n)) for n in range(2)])
    with (sink.lake.home / "collect.lock").open("wb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        during = Drain(sink).drain(), len(fake.posted)
    assert (during, Drain(sink).drain(), sink.lake.pending()) == ((0, 0), 2, [])
