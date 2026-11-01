"""The collect pass: every spooled event reaches the sink once, a line POSTed alone with the token,
and a file a writer holds, a sink that refuses or a line that is no spool line loses nothing."""

import itertools
import stat
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
    run_state_machine_as_test,
)

from mainboard import MissionError
from mainboard.core.events import Event, EventSpool, JobFrame, Key
from mainboard.serve.events.collect import Collector
from mainboard.serve.events.sink import Sink

from ...state.telemetry.builders import lake_with, stored
from ..fakes import FakeSink

_FILES = ("a.jsonl", "b.jsonl", "c.jsonl")
_NOON = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)


def _spooled(sink: Sink, name: str, bodies: Sequence[str]) -> list[str]:
    """`bodies` written as events to the spool file `name` by a writer that then lets go."""
    with EventSpool(sink.lake.spool / name) as spool:
        events = [Event(name="mb.test.spooled", at=_NOON, body=body) for body in bodies]
        for event in events:
            spool.append(event)
    return [event.line() for event in events]


def _written(sink: Sink, name: str, events: Sequence[Event]) -> None:
    """`events` written to the spool file `name` by a writer that then lets go."""
    with EventSpool(sink.lake.spool / name) as spool:
        for event in events:
            spool.append(event)


class Delivery(RuleBasedStateMachine):
    """Writers appending to spool files, some holding them open, between collect passes."""

    @invariant()
    def nothing_is_sent_twice_or_unwritten(self) -> None:
        assert len(set(self.fake.bodies)) == len(self.fake.bodies)
        assert set(self.fake.bodies) <= set(self.written)

    @precondition(lambda machine: machine.held)
    @rule()
    def release(self) -> None:
        for spool in self.held.values():
            spool.close()
        self.held.clear()

    @rule()
    def collect(self) -> None:
        Collector(self.sink).collect()
        assert sorted(self.fake.bodies) == sorted(self.written)

    @rule(name=st.sampled_from(_FILES), count=st.integers(1, 4), hold=st.booleans())
    def write(self, name: str, count: int, *, hold: bool) -> None:
        spool = self.held.get(name) or EventSpool(self.sink.lake.spool / name)
        for _ in range(count):
            event = Event(name="mb.test.spooled", at=_NOON, body=str(next(self.numbers)))
            spool.append(event)
            self.written.append(event.line())
        if hold:
            self.held[name] = spool
        elif name not in self.held:
            spool.close()

    def __init__(self, reach: Callable[[Path, Path], Sink]) -> None:
        super().__init__()
        self.stack = ExitStack()
        root = Path(self.stack.enter_context(TemporaryDirectory()))
        self.fake = self.stack.enter_context(FakeSink(root / "sink.sock"))
        self.sink = reach(root, self.fake.path)
        self.sink.token.ensure()
        self.held: dict[str, EventSpool] = {}
        self.numbers = itertools.count()
        self.written: list[str] = []

    def teardown(self) -> None:
        for spool in self.held.values():
            spool.close()
        self.stack.close()


def test_every_spooled_line_reaches_the_sink_once_whatever_writers_hold(
    reach: Callable[[Path, Path], Sink],
) -> None:
    class Reaching(Delivery):
        """The machine over the `reach` fixture's sink, built by the runner with no argument."""

        def __init__(self) -> None:
            super().__init__(reach)

    run_state_machine_as_test(
        Reaching, settings=settings(max_examples=20, stateful_step_count=8, deadline=None)
    )


def test_each_line_goes_alone_to_the_logs_route_with_the_bearer_token(
    sink: Sink, fake: FakeSink
) -> None:
    lines = _spooled(sink, "one.jsonl", ["x", "y"])
    Collector(sink).collect()
    assert fake.posted == [("/v1/logs", f"Bearer {sink.token.read()}", line) for line in lines]


def test_the_pass_that_collects_keeps_its_lock_file_private(sink: Sink, fake: FakeSink) -> None:
    _spooled(sink, "one.jsonl", ["x"])
    Collector(sink).collect()
    assert stat.S_IMODE((sink.lake.home / "collect.lock").stat().st_mode) == 0o600


def test_a_file_a_writer_holds_is_sent_but_kept_until_its_writer_lets_go(
    sink: Sink, fake: FakeSink
) -> None:
    with EventSpool(sink.lake.spool / "held.jsonl") as spool:
        spool.append(Event(name="mb.test.spooled", at=_NOON, body="held"))
        sent = Collector(sink).collect()
        kept = (sink.lake.spool / "held.jsonl").exists()
    assert (sent, kept, Collector(sink).collect(), len(fake.posted)) == (1, True, 0, 1)
    assert not (sink.lake.spool / "held.jsonl").exists()


def test_a_file_holding_a_line_that_is_no_spool_line_is_set_aside_unsent(
    sink: Sink, fake: FakeSink
) -> None:
    lines = _spooled(sink, "good.jsonl", ["ok"])
    (sink.lake.spool / "bad.jsonl").write_text('{"resourceLogs": "no"}\n', encoding="utf-8")
    sent = Collector(sink).collect(), Collector(sink).collect()
    assert (sent, fake.bodies) == ((1, 0), lines)
    assert sorted(path.name for path in sink.lake.spool.iterdir()) == ["bad.refused"]


def test_a_sink_refusing_a_line_leaves_the_file_to_be_sent_again(
    sink: Sink, fake: FakeSink
) -> None:
    lines = _spooled(sink, "retry.jsonl", ["first", "second"])
    fake.answer(503)
    with pytest.raises(MissionError, match="answered 503"):
        Collector(sink).collect()
    fake.answer(202)
    assert (Collector(sink).collect(), fake.bodies[1:]) == (2, lines)


def test_an_empty_spool_asks_nothing_of_a_sink_that_is_not_there(
    workspace: Path, reach: Callable[[Path, Path], Sink]
) -> None:
    sink = reach(workspace, workspace / "absent.sock")
    assert (Collector(sink).collect(), (workspace / ".mb").exists()) == (0, False)


def test_a_runs_frames_spooled_again_once_the_lake_holds_them_are_not_sent_again(
    workspace: Path, sink: Sink, fake: FakeSink
) -> None:
    frames = [Event(name=frame, at=_NOON, attributes={Key.job: "gold/42"}) for frame in JobFrame]
    lake_with(workspace, [])
    _written(sink, "first.jsonl", frames)
    first = Collector(sink).collect()
    lake_with(workspace, stored(frames))
    _written(sink, "again.jsonl", frames)
    second = Collector(sink).collect()
    assert (first, second, len(fake.posted)) == (3, 0, 3)
