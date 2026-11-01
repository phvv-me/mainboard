# Work over a dependency graph on threads, each item once everything it waits for is done.

import os
from collections.abc import Callable, Hashable, Mapping, Set
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from graphlib import TopologicalSorter


def in_waves[Item: Hashable](
    waits: Mapping[Item, Set[Item]], work: Callable[[Item], None]
) -> None:
    """Run `work` on every item, each on a thread once the items it waits for are done.

    Items that wait for nothing in common run beside each other, in waves of the graph.

    waits: each item with the items that must be done before it starts.
    """
    order = TopologicalSorter(waits)
    order.prepare()
    with ThreadPoolExecutor(os.process_cpu_count()) as pool:
        running: dict[Future[None], Item] = {}
        while order.is_active():
            for item in order.get_ready():
                running[pool.submit(work, item)] = item
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                future.result()
                order.done(running.pop(future))
