import pytest

from mainboard.profile import annotate
from mainboard.profile import spans as span_module

from .support import FakeGPU, clock_tracer, one_process_gpu


@pytest.fixture(autouse=True)
def reset_profiling_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests hermetic: no leftover active profiler or cached tracer between tests.

    `monkeypatch.setattr` (rather than a direct `module._private = ...` assignment) both
    resets the module-private state up front and restores it automatically at teardown,
    covering a prior test that failed before clearing its own active session.
    """
    monkeypatch.setattr(span_module, "_active", None, raising=False)
    monkeypatch.setattr(annotate, "_tracer", None, raising=False)


@pytest.fixture
def one_gpu(monkeypatch: pytest.MonkeyPatch) -> FakeGPU:
    """A one-GPU host whose snapshot carries fixed telemetry, with a clock tracer installed."""
    gpu = one_process_gpu()
    monkeypatch.setattr(annotate, "_tracer", clock_tracer())
    return gpu
