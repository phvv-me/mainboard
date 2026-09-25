import pytest

from mainboard.probe import GPU
from mainboard.profile import annotate
from mainboard.profile import spans as span_module

from .support import FakeGPU, clock_tracer, one_process_gpu


@pytest.fixture(autouse=True)
def reset_profiling_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """No leftover active profiler, cached tracer, or real device.

    The host probe itself is pinned to an empty fleet, so every session still runs the real
    discovery; a test that cares puts its own fake devices behind the same seam.
    """
    monkeypatch.setattr(span_module, "_active", None, raising=False)
    monkeypatch.setattr(annotate, "_tracer", None, raising=False)
    monkeypatch.setattr(GPU, "all", staticmethod(tuple))


@pytest.fixture
def one_gpu(monkeypatch: pytest.MonkeyPatch) -> FakeGPU:
    """A one-GPU host whose snapshot carries fixed telemetry, with a clock tracer installed."""
    gpu = one_process_gpu()
    monkeypatch.setattr(annotate, "_tracer", clock_tracer())
    return gpu
