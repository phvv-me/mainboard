"""Bounded aggregation for `span`s: reservoir quantiles collapsed into `SpanStat` rows.

A `Collector` owns one collection window — the caller creates one per operation (one
recall call, one pipeline batch) so concurrent windows never share state. `span(...)`
writes into the process-wide `default_collector()` whenever no `collector=` is given,
which covers the simple, single-window case for free.
"""

import threading
from collections import defaultdict
from functools import cache
from random import Random

from .models import SpanRecord, SpanStat


class Reservoir:
    """A fixed-capacity uniform sample of a stream, for streaming quantiles.

    Algorithm R: the first `capacity` values are always kept, so `quantile` is exact
    while the stream is no longer than `capacity`; beyond that, each new value replaces
    a uniformly random slot, so memory stays bounded no matter how long the stream runs.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.seen = 0
        self.values: list[float] = []
        self.random = Random()

    def add(self, value: float) -> None:
        """Fold one value into the sample."""
        self.seen += 1
        if len(self.values) < self.capacity:
            self.values.append(value)
            return
        slot = self.random.randint(0, self.seen - 1)
        if slot < self.capacity:
            self.values[slot] = value

    def quantile(self, q: float) -> float:
        """The `q`-th quantile (0..1) of the sample so far, or 0.0 when empty."""
        if not self.values:
            return 0.0
        ordered = sorted(self.values)
        index = min(len(ordered) - 1, int(q * len(ordered)))
        return ordered[index]


class PathAggregate:
    """Running count/total/max/reservoir for one dotted path.

    Mutated only under the owning `Collector`'s lock, so a single instance never sees
    concurrent writers.
    """

    __slots__ = ("count", "total_ms", "max_ms", "reservoir")

    def __init__(self, reservoir_size: int) -> None:
        self.count = 0
        self.total_ms = 0.0
        self.max_ms = 0.0
        self.reservoir = Reservoir(reservoir_size)

    def add(self, wall_ms: float) -> None:
        """Fold one occurrence's wall time into the running aggregate."""
        self.count += 1
        self.total_ms += wall_ms
        self.max_ms = max(self.max_ms, wall_ms)
        self.reservoir.add(wall_ms)

    def stat(self, path: str) -> SpanStat:
        """Snapshot this path's running aggregate as an immutable `SpanStat`."""
        return SpanStat(
            path=path,
            count=self.count,
            total_ms=self.total_ms,
            mean_ms=self.total_ms / self.count,
            p50_ms=self.reservoir.quantile(0.50),
            p95_ms=self.reservoir.quantile(0.95),
            max_ms=self.max_ms,
        )


class Collector:
    """One collection window: the raw span log plus each path's running aggregate.

    Thread-safe (a lock guards every mutation), so concurrent asyncio tasks or threads
    writing through the same collector never corrupt its state. `reservoir_size` bounds
    the per-path quantile sample; `count`/`total_ms`/`mean_ms`/`max_ms` stay exact
    regardless of how many spans ran.
    """

    def __init__(self, *, reservoir_size: int = 2048) -> None:
        self.reservoir_size = reservoir_size
        self.lock = threading.Lock()
        self.log: list[SpanRecord] = []
        self.paths: defaultdict[str, PathAggregate] = defaultdict(
            lambda: PathAggregate(self.reservoir_size)
        )

    def add(self, record: SpanRecord) -> None:
        """Fold one completed span into the log and its path's running aggregate."""
        with self.lock:
            self.log.append(record)
            self.paths[record.path].add(record.wall_ms)

    def records(self) -> list[SpanRecord]:
        """Every completed span, in completion order (the raw, un-collapsed log)."""
        with self.lock:
            return list(self.log)

    def stats(self) -> list[SpanStat]:
        """Per-path aggregates, slowest total first."""
        with self.lock:
            rows = [aggregate.stat(path) for path, aggregate in self.paths.items()]
        return sorted(rows, key=lambda s: s.total_ms, reverse=True)

    def reset(self) -> None:
        """Clear the window so the next operation starts from zero."""
        with self.lock:
            self.log.clear()
            self.paths.clear()

    def report(self) -> str:
        """Plain-text per-path table (the no-rich fallback of `show`)."""
        from .render import span_text

        return span_text(self.stats())

    def show(self, *, color: bool = True) -> None:
        """Print the per-path stats as a rich table."""
        from .render import show_spans

        show_spans(self.stats(), color=color)

    def __str__(self) -> str:
        return self.report()


@cache
def default_collector() -> Collector:
    """The process-wide collector `span(...)` writes to when no `collector=` is given."""
    return Collector()
