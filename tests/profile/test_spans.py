import asyncio
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import ExitStack

import pytest

from mainboard import Profiler, span
from mainboard.profile import Profile, annotate, spans

from .support import RecordingSession


def test_dormant_annotations_call_through_without_reading_a_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clock is the cheapest thing the active path touches, so a failing clock proves the
    dormant path never enters it."""

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
    profiler = Profiler(features=Profiler.Feature.SPANS)
    with profiler, pytest.raises(ValueError, match="boom"), span("risky"):
        raise ValueError("boom")
    assert profiler.result().summaries[0].name == "risky"


def test_the_active_span_slot_admits_exactly_one_owner() -> None:
    """A stale owner cannot clear the slot; finishing nothing or exiting an unentered span is a
    no-op."""
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
    with Profiler(features=Profiler.Feature.SPANS, max_spans=2) as profiler:
        for name in ("one", "two", "three"):
            with span(name):
                pass
    profile = profiler.result()
    assert [item.name for item in profile.summaries] == ["two", "three"]
    assert profile.dropped_spans == 1
    assert "oldest spans dropped" in profile.report()


def test_concurrent_threads_close_their_exact_tokens() -> None:
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
    """A bare decorator names its span from the function's qualified name."""

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
    """Timing only the creation is the zero-length-span bug CPython fixed for its own
    decorators."""
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

    with ExitStack() as active:
        spans.activate(session)
        active.callback(spans.deactivate, session)
        iterator = produce()
        assert session.entered == []  # creation alone opens nothing
        assert list(iterator) == [1, 2]
        assert list(walk()) == [5]
        assert asyncio.run(consume()) == [7]
    assert session.entered[0] == "gen"
    assert session.entered[1].endswith("walk")
    assert session.entered[2] == "agen"
    assert len(session.walls) == 3
