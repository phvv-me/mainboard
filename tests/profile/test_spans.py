import asyncio
import threading
import time
from collections.abc import AsyncIterator, Iterator

import pytest
from mainboard import Profiler, span
from mainboard.profile import Profile, annotate, spans


def test_dormant_context_and_decorators_do_no_collection() -> None:
    """Annotations call through normally when no profiler owns the process."""

    @span
    def bare(value: int) -> int:
        return value + 1

    @span("named")
    def named(value: int) -> int:
        return value * 2

    with span("context"):
        assert bare(1) == 2
        assert named(2) == 4
    assert spans.active() is None


def test_dormant_annotations_do_not_read_a_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """The inactive path avoids clocks and every collector side effect."""

    def fail() -> int:
        raise AssertionError("clock read")

    monkeypatch.setattr(spans.time, "perf_counter_ns", fail)

    @span
    def work() -> int:
        return 7

    with span("context"):
        assert work() == 7


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
    profiler = Profiler(features=Profiler.Feature.SPANS)
    with profiler, pytest.raises(ValueError, match="boom"), span("risky"):
        raise ValueError("boom")
    assert profiler.result().summaries[0].name == "risky"


def test_only_one_profiler_can_receive_process_spans() -> None:
    first = Profiler(features=Profiler.Feature.SPANS)
    second = Profiler(features=Profiler.Feature.SPANS)
    with first, pytest.raises(RuntimeError, match="only one"):
        second.__enter__()
    assert second.active is False


def test_profiler_instance_cannot_be_entered_twice() -> None:
    profiler = Profiler(features=Profiler.Feature.SPANS)
    with profiler, pytest.raises(RuntimeError, match="entered twice"):
        profiler.__enter__()


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

    assert len(profiler.result().summaries) == 8
    assert {item.name for item in profiler.result().summaries} == {"worker"}


def test_deactivate_ignores_a_session_that_does_not_own_the_slot() -> None:
    first = Profiler(features=Profiler.Feature.SPANS)
    second = Profiler(features=Profiler.Feature.SPANS)
    spans.activate(first)
    spans.deactivate(second)
    assert spans.active() is first
    spans.deactivate(first)


def test_finish_and_inactive_exit_are_safe_noops() -> None:
    spans.finish(None)
    context = span("unused")
    context.__exit__(None, None, None)


def test_dormant_async_decorator_calls_through() -> None:
    @span("worker")
    async def work(value: int) -> int:
        await asyncio.sleep(0)
        return value

    assert asyncio.run(work(3)) == 3


def test_async_tasks_keep_independent_nesting_paths() -> None:
    @span("worker")
    async def work(value: int) -> int:
        await asyncio.sleep(0)
        return value

    async def pipeline(value: int) -> int:
        with span("pipeline"):
            return await work(value)

    async def run() -> list[int]:
        return await asyncio.gather(*(pipeline(value) for value in range(20)))

    with Profiler(features=Profiler.Feature.SPANS) as profiler:
        assert asyncio.run(run()) == list(range(20))

    names = [item.name for item in profiler.result().summaries]
    assert names.count("pipeline.worker") == 20
    assert names.count("pipeline") == 20


def test_bare_async_decorator_uses_qualname() -> None:
    @span
    async def bare() -> int:
        return 1

    with Profiler(features=Profiler.Feature.SPANS) as profiler:
        assert asyncio.run(bare()) == 1
    assert profiler.result().summaries[0].name.endswith("bare")


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


class _RecordingSession:
    def __init__(self) -> None:
        self.entered: list[str] = []
        self.walls: list[int] = []

    def enter(self, name: str) -> int:
        self.entered.append(name)
        return len(self.entered)

    def exit(self, token: int, *, wall_ns: int) -> None:
        self.walls.append(wall_ns)


def test_generator_spans_cover_consumption_not_creation() -> None:
    session = _RecordingSession()

    @span("gen")
    def produce() -> Iterator[int]:
        time.sleep(0.03)
        yield 1
        yield 2

    spans.activate(session)
    try:
        iterator = produce()
        assert session.entered == []
        assert list(iterator) == [1, 2]
    finally:
        spans.deactivate(session)
    assert session.entered == ["gen"]
    assert session.walls[0] >= 25_000_000


def test_bare_decorator_on_generator_uses_consumption_span() -> None:
    session = _RecordingSession()

    @span
    def walk() -> Iterator[int]:
        yield 5

    spans.activate(session)
    try:
        assert list(walk()) == [5]
    finally:
        spans.deactivate(session)
    assert session.entered and session.entered[0].endswith("walk")


def test_async_generator_spans_cover_consumption() -> None:
    session = _RecordingSession()

    @span("agen")
    async def stream() -> AsyncIterator[int]:
        await asyncio.sleep(0.02)
        yield 7

    async def consume() -> list[int]:
        return [item async for item in stream()]

    spans.activate(session)
    try:
        assert asyncio.run(consume()) == [7]
    finally:
        spans.deactivate(session)
    assert session.entered == ["agen"]
    assert session.walls[0] >= 15_000_000


def test_dormant_generator_spans_pass_through() -> None:
    @span("quiet")
    def produce() -> Iterator[int]:
        yield 9

    assert list(produce()) == [9]
    assert spans.active() is None


def test_dormant_async_generator_spans_pass_through() -> None:
    @span("quiet-async")
    async def stream() -> AsyncIterator[int]:
        yield 11

    async def consume() -> list[int]:
        return [item async for item in stream()]

    assert asyncio.run(consume()) == [11]
    assert spans.active() is None
