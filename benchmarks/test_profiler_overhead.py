from collections.abc import Iterator

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture

from mainboard.profiling import Profiler, span

_ITERATIONS = 200
_CUDA = torch.cuda.is_available()
_CONFIGS = {
    "dormant": None,
    "spans": Profiler.Feature.SPANS,
    "device": Profiler.Feature.SPANS | Profiler.Feature.DEVICE,
    "activity": Profiler.Feature.SPANS | Profiler.Feature.ACTIVITY,
    "all_local": Profiler.Feature.DEFAULT & ~Profiler.Feature.PYTHON,
}


def workload() -> None:
    """Issue a small matrix multiplication in each annotated span when CUDA exists."""
    value = torch.randn(512, 512, device="cuda") if _CUDA else None
    for _ in range(_ITERATIONS):
        with span("op"):
            if value is not None:
                value = value @ value
    if value is not None:
        torch.cuda.synchronize()


@pytest.fixture(params=tuple(_CONFIGS))
def profiled(request: pytest.FixtureRequest) -> Iterator[None]:
    """Activate one collector set for the same workload."""
    features = _CONFIGS[request.param]
    if features is None:
        yield
        return
    with Profiler(features=features, activities=Profiler.Activity.DEFAULT):
        yield


@pytest.mark.benchmark(group="profiler-overhead")
def test_profiler_overhead(benchmark: BenchmarkFixture, profiled: None) -> None:
    """Measure the incremental cost of each collector set."""
    benchmark(workload)
