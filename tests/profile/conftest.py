import pytest

from mainboard.probe import GPU
from mainboard.profile import annotate
from mainboard.profile import spans as span_module

from .support import FakeGPU, clock_tracer, one_process_gpu


@pytest.fixture(autouse=True)
def reset_profiling_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests hermetic: no leftover active profiler, cached tracer, or real device.

    `monkeypatch.setattr` (rather than a direct `module._private = ...` assignment) both
    resets the module-private state up front and restores it automatically at teardown,
    covering a prior test that failed before clearing its own active session.

    The host probe is pinned to an empty fleet so this suite reads the same on a GPU
    workstation as on CI. The seam is the probe itself rather than the profiler's discovery
    call, so every session still runs the real discovery, and a test that cares what
    discovery finds says so by putting its own fake devices behind the same seam.
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
