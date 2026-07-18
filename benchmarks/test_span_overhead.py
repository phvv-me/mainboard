from collections.abc import Iterator

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from mainboard.profiling import Profiler, span

_ITERATIONS = 200
_CONFIGS = {
    "dormant": None,
    "timing": Profiler.Feature.SPANS,
    "timing_and_markers": Profiler.Feature.SPANS | Profiler.Feature.MARKERS,
}


@span
def annotated() -> None:
    """A permanently annotated function with no payload."""


def workload() -> None:
    """Run the same dormant annotations under every collection policy."""
    for _ in range(_ITERATIONS):
        with span("outer"):
            annotated()


@pytest.fixture(params=tuple(_CONFIGS))
def profiled(request: pytest.FixtureRequest) -> Iterator[None]:
    """Activate one feature set while leaving the annotations unchanged."""
    features = _CONFIGS[request.param]
    if features is None:
        yield
        return
    with Profiler(features=features, max_spans=_ITERATIONS * 2):
        yield


@pytest.mark.benchmark(group="span-overhead")
def test_span_overhead(benchmark: BenchmarkFixture, profiled: None) -> None:
    """Measure dormant, timed, and native-marker annotation costs."""
    benchmark(workload)
