"""Load test: how much does profiling cost? Measures the profiler's own overhead.

Runs the *same* annotated workload under each profiler configuration so the deltas are
the cost of profiling itself — disabled vs. snapshot-only vs. deep trace vs. all CUPTI
activity kinds. On CUDA the workload also issues GPU kernels, so the deep-trace numbers
reflect real CUPTI overhead; on CPU it isolates the Python annotation cost.

Run: ``pytest benchmarks --benchmark-only`` (add ``-q``). Compare the ``Mean`` column
across the ``config`` params. Kept out of the default test run (its own directory).
"""

from __future__ import annotations

import pytest

from mainboard.profiling import Activity, Profiler, region

try:
    import torch

    _CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    _CUDA = False

_ITERS = 200


def _workload() -> None:
    """A fixed unit of annotated work — a small GPU matmul per region when on CUDA."""
    x = torch.randn(512, 512, device="cuda") if _CUDA else None
    for _ in range(_ITERS):
        with region("op"):
            if x is not None:
                x = x @ x
    if x is not None:
        torch.cuda.synchronize()


_CONFIGS = {
    "disabled": None,
    "snapshot": False,
    "trace_kernel_memcpy": True,
    "trace_all_kinds": Activity.ALL,
}


@pytest.fixture(params=list(_CONFIGS))
def profiled(request: pytest.FixtureRequest):
    """Activate the profiler config for the benchmarked workload (None = no profiler)."""
    trace = _CONFIGS[request.param]
    if trace is None:
        yield
        return
    with Profiler(sample_interval_ms=10, trace=trace):
        yield


@pytest.mark.benchmark(group="profiler-overhead")
def test_profiler_overhead(benchmark: pytest.FixtureRequest, profiled: None) -> None:
    """Benchmark the workload under each profiler config (compare Mean across params)."""
    benchmark(_workload)
