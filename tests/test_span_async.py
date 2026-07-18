import asyncio

from mainboard.profiling import Profiler, span


@span("worker")
async def work(value: int) -> int:
    await asyncio.sleep(0)
    return value


def test_dormant_async_decorator_calls_through() -> None:
    assert asyncio.run(work(3)) == 3


def test_async_tasks_keep_independent_nesting_paths() -> None:
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
