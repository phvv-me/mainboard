"""Time nested stages in production code with `span`. Runs anywhere, no GPU needed."""

import asyncio
import time

from mainboard.profiling import Profiler, span


@span
def load() -> None:
    time.sleep(0.02)


async def process(chunk_id: int) -> None:
    with span("pipeline"):
        with span("extract"):
            await asyncio.sleep(0.01)
        with span("embed"):
            await asyncio.sleep(0.02)


async def main() -> None:
    with Profiler(features=Profiler.Feature.SPANS) as profiler:
        load()
        await asyncio.gather(*(process(i) for i in range(8)))
    profiler.show()


asyncio.run(main())
