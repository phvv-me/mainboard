"""The one-call bottleneck report and the GPU-contention gating helpers.

Everything here uses a fake `DeviceProbe` and a fake tracer, so the suite runs
identically on a host with no GPU.
"""

from typing import TYPE_CHECKING

import pytest

from mainboard import Profiler as RealProfiler
from mainboard.profile import Activity, KernelTrace, Profile, ProfileReport, annotate, bottleneck

from .conftest import FakeGPU, FakeMemory, FakeUtilization

if TYPE_CHECKING:
    from types import TracebackType


class _FakeTracer:
    """A no-op tracer whose deep-trace support is fixed to `KERNEL | MEMCPY`."""

    def pop(self) -> None:
        return None

    def push(self, name: str) -> None:
        return None

    def supported(self) -> Activity:
        return Activity.KERNEL | Activity.MEMCPY


class _NoTracer:
    """A tracer offering no deep-trace support, standing in for a backend-less host."""

    def supported(self) -> Activity:
        return Activity(0)


class _RecordingProfiler:
    """A `Profiler` stand-in that ignores tracing and returns a fixed kernel `Profile`."""

    Feature = None

    def __init__(self, **_) -> None:
        pass

    def __enter__(self) -> _RecordingProfiler:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def result(self) -> Profile:
        return Profile(device="fake", kernels=(KernelTrace(name="gemm", start_ns=0, end_ns=1000),))


@pytest.fixture
def traceable_gpu(monkeypatch: pytest.MonkeyPatch) -> FakeGPU:
    """A device with a fake tracer and a recording profiler standing in for CUPTI."""
    gpu = FakeGPU(peak_bandwidth_gbs=900.0)
    _RecordingProfiler.Feature = RealProfiler.Feature
    tracer = _FakeTracer()
    monkeypatch.setattr(bottleneck, "tracer", lambda: tracer)
    monkeypatch.setattr(annotate, "tracer", lambda **_: tracer)
    monkeypatch.setattr(bottleneck, "Profiler", _RecordingProfiler)
    return gpu


def test_profile_runs_callable_and_reports(traceable_gpu: FakeGPU) -> None:
    """`profile` warms up, runs the callable `iters` times, and returns its report."""
    ran: list[int] = []
    report = bottleneck.profile(lambda: ran.append(1), gpu=traceable_gpu, iters=3, warmup=2)
    assert isinstance(report, ProfileReport)
    assert report.iterations == 3
    assert len(ran) == 5  # warmup 2 + iters 3
    assert report.dominant_kernel == "gemm"
    assert report.peak_bandwidth_gbps == 900.0


def test_profile_adapts_requested_kinds_to_device_support(traceable_gpu: FakeGPU) -> None:
    """Requested kinds the device lacks are dropped from the trace and noted."""
    report = bottleneck.profile(lambda: None, gpu=traceable_gpu, kinds=Activity.ALL)
    assert "memory" in report.unavailable  # the fake tracer supports only KERNEL|MEMCPY


def test_profile_syncs_between_runs(traceable_gpu: FakeGPU) -> None:
    """A provided `sync` barrier is called after warmup and each timed run."""
    synced: list[int] = []
    bottleneck.profile(
        lambda: None, gpu=traceable_gpu, iters=2, warmup=1, sync=lambda: synced.append(1)
    )
    assert len(synced) >= 3  # one post-warmup + one per timed iter


def test_profile_on_cpu_only_host_is_graceful(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no GPU, `profile` still runs and returns an empty, ungraded report."""
    monkeypatch.setattr(bottleneck, "tracer", lambda: _NoTracer())
    ran: list[int] = []
    report = bottleneck.profile(lambda: ran.append(1), gpu=None, iters=2)
    assert ran == [1, 1]
    assert report.kernels == ()
    assert report.peak_bandwidth_gbps == 0.0
    assert report.unavailable == ()  # no backend -> nothing is flagged unavailable


def test_gpu_busy_true_under_compute_load() -> None:
    """A GPU above the utilization threshold reads as busy."""
    gpu = FakeGPU(utilization=FakeUtilization(gpu_pct=85))
    assert bottleneck.gpu_busy(gpu) is True


def test_gpu_busy_true_under_memory_pressure() -> None:
    """A GPU near its memory capacity reads as busy even when compute is idle."""
    gpu = FakeGPU(utilization=FakeUtilization(gpu_pct=1), memory=FakeMemory(percent_used=95.0))
    assert bottleneck.gpu_busy(gpu) is True


def test_gpu_busy_false_when_idle() -> None:
    """An idle GPU below both thresholds reads as not busy."""
    gpu = FakeGPU(utilization=FakeUtilization(gpu_pct=2), memory=FakeMemory(percent_used=10.0))
    assert bottleneck.gpu_busy(gpu) is False


def test_gpu_busy_false_when_no_gpu() -> None:
    """No device reads as not busy rather than raising."""
    assert bottleneck.gpu_busy(None) is False


def test_wait_for_idle_returns_true_when_already_idle() -> None:
    """An already-idle GPU returns immediately without sleeping."""
    gpu = FakeGPU(utilization=FakeUtilization(gpu_pct=2), memory=FakeMemory(percent_used=10.0))
    slept: list[float] = []
    assert bottleneck.wait_for_idle(gpu, sleep=slept.append) is True
    assert slept == []


def test_wait_for_idle_polls_until_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    """`wait_for_idle` polls until the device clears, then returns True."""
    states = iter([True, True, False])
    monkeypatch.setattr(bottleneck, "gpu_busy", lambda *a, **k: next(states))
    slept: list[float] = []
    assert bottleneck.wait_for_idle(None, timeout=10.0, sleep=slept.append) is True
    assert len(slept) == 2  # two busy polls, then idle


def test_wait_for_idle_times_out_when_stuck_busy() -> None:
    """A GPU that never clears makes `wait_for_idle` return False at the deadline."""
    gpu = FakeGPU(utilization=FakeUtilization(gpu_pct=85))
    slept: list[float] = []
    assert bottleneck.wait_for_idle(gpu, timeout=0.0, sleep=slept.append) is False
