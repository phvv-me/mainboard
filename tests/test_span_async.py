"""`span` under asyncio: the concurrency case it exists for. `ContextVar`s copy per
task, so many fanned-out tasks each keep their own nesting stack — this is what a real
ingestion pipeline (gate/extract/consolidate/embed/write, dozens of concurrent chunks)
or a sequential recall call actually needs, without any of aizk wired in here.
"""

import asyncio

from mainboard.profiling.collector import Collector, default_collector
from mainboard.profiling.spans import Span, _decorate_async, enable_spans, span

STAGES = ("gate", "extract", "consolidate", "embed", "write")


async def _pipeline_chunk(chunk_id: int, collector: Collector) -> int:
    """One concurrent unit of work: a `pipeline` span wrapping five sequential stages,
    each yielding control (`asyncio.sleep(0)`) so tasks genuinely interleave."""
    with span("pipeline", collector=collector):
        for stage in STAGES:
            with span(stage, collector=collector):
                await asyncio.sleep(0)
    return chunk_id


def test_concurrent_pipeline_tasks_never_bleed_into_each_other() -> None:
    """40 concurrent chunks, each 5 stages deep: every task keeps its own stack, so
    only the 6 expected dotted paths ever appear, each with the right occurrence count."""
    enable_spans()
    collector = Collector()
    task_count = 40

    async def run() -> list[int]:
        return await asyncio.gather(*(_pipeline_chunk(i, collector) for i in range(task_count)))

    results = asyncio.run(run())
    assert sorted(results) == list(range(task_count))

    records = collector.records()
    assert len(records) == task_count * (1 + len(STAGES))

    expected_paths = {"pipeline"} | {f"pipeline.{stage}" for stage in STAGES}
    assert {r.path for r in records} == expected_paths
    for path in expected_paths:
        assert sum(1 for r in records if r.path == path) == task_count

    # No task's stage ever nested under a sibling task's frame: every stage record's
    # depth is exactly 1, and the outer `pipeline` record is always depth 0.
    assert all(r.depth == (0 if r.path == "pipeline" else 1) for r in records)


def test_async_decorator_bare_uses_qualname_and_awaits_the_result() -> None:
    enable_spans()
    collector = Collector()

    async def double(n: int) -> int:
        await asyncio.sleep(0)
        return n * 2

    decorated = Span("double", collector=collector)(double)
    result = asyncio.run(decorated(21))
    assert result == 42
    assert collector.records()[0].name == "double"


def test_async_decorator_named_runs_concurrently_without_cross_talk() -> None:
    """`@span("label")` on a coroutine function: concurrent calls each open their own
    `Span`, so their wall times and stack entries never collide."""
    enable_spans()
    collector = Collector()

    @span("worker", collector=collector)
    async def work(n: int) -> int:
        await asyncio.sleep(0)
        return n

    async def run() -> list[int]:
        return await asyncio.gather(*(work(i) for i in range(20)))

    results = asyncio.run(run())
    assert results == list(range(20))
    records = collector.records()
    assert len(records) == 20
    assert all(r.name == "worker" and r.depth == 0 for r in records)


async def _compute(n: int) -> int:
    await asyncio.sleep(0)
    return n + 1


def test_bare_span_decorator_on_coroutine_function_uses_qualname() -> None:
    """Bare `@span` (the `span(func)` overload) wraps a coroutine function too,
    preserving `functools.wraps`' identity and using its qualname as the label."""
    enable_spans()
    wrapped = span(_compute)
    assert asyncio.run(wrapped(1)) == 2
    assert default_collector().records()[0].name == _compute.__qualname__


def test_span_call_dispatches_to_decorate_async_for_coroutine_functions() -> None:
    """`Span.__call__` on a coroutine function routes to `_decorate_async`, matching
    what calling `_decorate_async` directly produces."""
    enable_spans()
    collector = Collector()

    labelled = Span("labelled-compute", collector=collector)(_compute)
    assert asyncio.run(labelled(1)) == 2
    assert collector.records()[0].name == "labelled-compute"

    direct = _decorate_async(_compute, "direct-label", collector, False)
    assert asyncio.run(direct(1)) == 2
    assert collector.records()[1].name == "direct-label"
