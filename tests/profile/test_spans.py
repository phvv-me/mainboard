import asyncio
import threading
import time
from collections.abc import AsyncIterator, Iterator

import pytest

from mainboard import Profiler, span
from mainboard.profile import Profile, annotate, spans


class RecordingSession:
    """A `SpanSession` stand-in that records every name opened and every wall time closed."""

    def __init__(self) -> None:
        self.entered: list[str] = []
        self.walls: list[int] = []

    def enter(self, name: str) -> int:
        self.entered.append(name)
        return len(self.entered)

    def exit(self, token: int, *, wall_ns: int) -> None:
        del token
        self.walls.append(wall_ns)


def test_dormant_annotations_call_through_without_reading_a_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no profiler owning the process an annotation performs no collection at all.

    The clock is the cheapest thing the active path touches, so a stubbed-out
    `perf_counter_ns` is what proves the dormant path never enters it.
    """

    def fail() -> int:
        raise AssertionError("clock read")

    monkeypatch.setattr(spans.time, "perf_counter_ns", fail)

    @span
    def bare(value: int) -> int:
        return value + 1

    @span("named")
    def named(value: int) -> int:
        return value * 2

    @span("quiet")
    def produce() -> Iterator[int]:
        yield 9

    with span("context"):
        assert bare(1) == 2
        assert named(2) == 4
        assert list(produce()) == [9]
    assert spans.active() is None


def test_dormant_async_annotations_call_through() -> None:
    """A coroutine and an async generator both pass straight through with no session."""

    @span("worker")
    async def work(value: int) -> int:
        await asyncio.sleep(0)
        return value

    @span("quiet-async")
    async def stream() -> AsyncIterator[int]:
        yield 11

    async def consume() -> list[int]:
        return [await work(3), *[item async for item in stream()]]

    assert asyncio.run(consume()) == [3, 11]
    assert spans.active() is None


def test_one_profiler_collects_nested_context_and_decorator_spans() -> None:
    """All annotation forms feed one profile with dotted nesting paths."""

    @span("child")
    def work() -> None:
        time.sleep(0.0001)

    with Profiler(features=Profiler.Feature.SPANS) as profiler, span("parent"):
        work()

    profile = profiler.result()
    assert isinstance(profile, Profile)
    assert [item.name for item in profile.summaries] == ["parent.child", "parent"]
    assert all(item.wall_ms > 0 for item in profile.summaries)


def test_exception_still_closes_the_span() -> None:
    """A span left by an exception is closed and recorded, not leaked open."""
    profiler = Profiler(features=Profiler.Feature.SPANS)
    with profiler, pytest.raises(ValueError, match="boom"), span("risky"):
        raise ValueError("boom")
    assert profiler.result().summaries[0].name == "risky"


def test_the_active_span_slot_admits_exactly_one_owner() -> None:
    """One process routes spans to one profiler, and a stale owner cannot clear the slot.

    A second profiler is refused rather than silently stealing the spans of the first, and
    the refused instance never comes up active. Closing a token nobody opened, or exiting a
    span that never entered, are both no-ops rather than errors.
    """
    first = Profiler(features=Profiler.Feature.SPANS)
    second = Profiler(features=Profiler.Feature.SPANS)
    with first, pytest.raises(RuntimeError, match="only one"):
        second.__enter__()
    assert second.active is False
    with first, pytest.raises(RuntimeError, match="entered twice"):
        first.__enter__()

    spans.activate(first)
    spans.deactivate(second)  # second never owned the slot, so this changes nothing
    assert spans.active() is first
    spans.deactivate(first)
    spans.finish(None)
    span("unused").__exit__(None, None, None)


def test_span_buffer_is_bounded_and_reports_drops() -> None:
    """Past `max_spans` the oldest measurements are dropped and counted, never grown into."""
    with Profiler(features=Profiler.Feature.SPANS, max_spans=2) as profiler:
        for name in ("one", "two", "three"):
            with span(name):
                pass
    profile = profiler.result()
    assert [item.name for item in profile.summaries] == ["two", "three"]
    assert profile.dropped_spans == 1
    assert "oldest spans dropped" in profile.report()


def test_concurrent_threads_close_their_exact_tokens() -> None:
    """Eight threads opening the same name each close their own token, not a neighbour's."""
    ready = threading.Barrier(8)

    @span("worker")
    def work() -> None:
        ready.wait(timeout=5)

    with Profiler(features=Profiler.Feature.SPANS) as profiler:
        threads = [threading.Thread(target=work) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

    summaries = profiler.result().summaries
    assert len(summaries) == 8
    assert {item.name for item in summaries} == {"worker"}


def test_async_tasks_keep_independent_nesting_paths() -> None:
    """Twenty concurrent tasks each nest under their own parent, never under a sibling's.

    A bare decorator names its span from the function's qualified name, so an unnamed
    coroutine still lands in the profile under something readable.
    """

    @span("worker")
    async def work(value: int) -> int:
        await asyncio.sleep(0)
        return value

    @span
    async def bare() -> int:
        return 1

    async def pipeline(value: int) -> int:
        with span("pipeline"):
            return await work(value)

    async def run() -> list[int]:
        return [*await asyncio.gather(*(pipeline(value) for value in range(20))), await bare()]

    with Profiler(features=Profiler.Feature.SPANS) as profiler:
        assert asyncio.run(run()) == [*range(20), 1]

    names = [item.name for item in profiler.result().summaries]
    assert names.count("pipeline.worker") == 20
    assert names.count("pipeline") == 20
    assert any(name.endswith("bare") for name in names)


def test_automatic_async_spans_keep_independent_task_stacks() -> None:
    """Auto-annotated coroutines never nest inside whichever task happened to run first."""

    async def automatic(value: int) -> int:
        await asyncio.sleep(0)
        return value

    async def run() -> list[int]:
        return await asyncio.gather(*(automatic(value) for value in range(20)))

    annotate.frames().clear()
    with Profiler(features=Profiler.Feature.SPANS) as profiler:
        annotate.enable_auto((automatic.__code__,))
        try:
            assert asyncio.run(run()) == list(range(20))
        finally:
            annotate.disable_auto()
    assert all("automatic.automatic" not in item.name for item in profiler.result().summaries)


def test_generator_spans_cover_consumption_not_creation() -> None:
    """A generator's span opens at the first item, not when the generator object is built.

    Timing only the creation is the silent zero-length-span bug CPython fixed for its own
    decorators, so the proof is that nothing is entered until consumption starts. A bare
    decorator on a generator takes the same path under its qualified name.
    """
    session = RecordingSession()

    @span("gen")
    def produce() -> Iterator[int]:
        yield 1
        yield 2

    @span
    def walk() -> Iterator[int]:
        yield 5

    @span("agen")
    async def stream() -> AsyncIterator[int]:
        yield 7

    async def consume() -> list[int]:
        return [item async for item in stream()]

    spans.activate(session)
    try:
        iterator = produce()
        assert session.entered == []  # creation alone opens nothing
        assert list(iterator) == [1, 2]
        assert list(walk()) == [5]
        assert asyncio.run(consume()) == [7]
    finally:
        spans.deactivate(session)
    assert session.entered[0] == "gen"
    assert session.entered[1].endswith("walk")
    assert session.entered[2] == "agen"
    assert len(session.walls) == 3
