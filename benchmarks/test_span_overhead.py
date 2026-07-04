"""Load test: how much does `span` cost? Measures the disabled/enabled overhead.

Runs the *same* annotated workload with spans off, on (timing only), and on with
`memory=True`, so the deltas are the cost of `span` itself — the near-zero-overhead
requirement is that "disabled" costs barely more than the bare workload.

Run: ``pytest benchmarks --benchmark-only`` (add ``-q``). Compare the ``Mean`` column
across the ``config`` params. Kept out of the default test run (its own directory).
"""

from collections.abc import Iterator

import pytest

from mainboard.profiling.collector import Collector
from mainboard.profiling.spans import disable_spans, enable_spans, span

_ITERS = 200


def _workload(collector: Collector) -> None:
    """A fixed unit of nested, annotated work: no real payload, just the span machinery."""
    for _ in range(_ITERS):
        with span("outer", collector=collector), span("inner", collector=collector):
            pass


_CONFIGS = {
    "disabled": (False, False),
    "enabled_timing_only": (True, False),
    "enabled_with_memory": (True, True),
}


@pytest.fixture(params=list(_CONFIGS))
def configured(request: pytest.FixtureRequest) -> Iterator[tuple[bool, Collector]]:
    """Toggle the module switch for the benchmarked workload; always a fresh collector."""
    enabled, memory = _CONFIGS[request.param]
    if enabled:
        enable_spans()
    else:
        disable_spans()
    yield memory, Collector()
    disable_spans()


@pytest.mark.benchmark(group="span-overhead")
def test_span_overhead(
    benchmark: pytest.FixtureRequest, configured: tuple[bool, Collector]
) -> None:
    """Benchmark the workload under each span config (compare Mean across params)."""
    memory, collector = configured
    if memory:
        benchmark(lambda: _memory_workload(collector))
    else:
        benchmark(lambda: _workload(collector))


def _memory_workload(collector: Collector) -> None:
    for _ in range(_ITERS):
        outer = span("outer", collector=collector, memory=True)
        inner = span("inner", collector=collector, memory=True)
        with outer, inner:
            pass
