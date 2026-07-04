"""Time nested stages in production code with `span`. Runs anywhere, no GPU needed."""

import asyncio
import time

from mainboard.profiling import Collector, enable_spans, span

enable_spans()  # off by default; call once at startup


@span
def load() -> None:
    time.sleep(0.02)


async def process(chunk_id: int, collector: Collector) -> None:
    with span("pipeline", collector=collector):
        with span("extract", collector=collector):
            await asyncio.sleep(0.01)
        with span("embed", collector=collector):
            await asyncio.sleep(0.02)


async def main() -> None:
    collector = Collector()  # a scoped window: one per operation, isolated from the default
    await asyncio.gather(*(process(i, collector) for i in range(8)))
    collector.show()  # rich table: path, calls, total/mean/p50/p95/max ms


load()  # sync context manager + bare decorator both write to the default collector
asyncio.run(main())
