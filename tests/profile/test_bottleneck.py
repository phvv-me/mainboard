# The one-call bottleneck report and the GPU-contention gating helpers. Everything here uses a
# fake `DeviceProbe` and a fake tracer, so the suite runs identically on a host with no GPU.

from typing import TYPE_CHECKING

import pytest

from mainboard import Profiler as RealProfiler
from mainboard.probe import GPU
from mainboard.profile import Activity, Profile, ProfileReport, annotate, bottleneck

from .support import FakeGPU, FakeMemory, FakeUtilization, kernel

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
    """A `Profiler` stand-in that ignores tracing and returns a fixed kernel `Profile`.

    It settles on a device the way the real session does, since the report is scored against
    whichever device the session ended up on rather than against the argument it was handed.
    """

    Feature = None

    def __init__(self, *, gpus: tuple[FakeGPU, ...] = (), **_) -> None:
        self.gpu = gpus[0] if gpus else None

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
        return Profile(device="fake", kernels=(kernel("gemm", 1000),))


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


def test_profile_runs_the_callable_and_reports_where_its_time_went(
    traceable_gpu: FakeGPU,
) -> None:
    """`profile` warms up, runs the callable `iters` times, and returns its report.

    The `sync` barrier fires after the warmup and after each timed run so async device
    work is captured, and any requested kind the device lacks is dropped from the trace
    and named in `unavailable` rather than silently omitted.
    """
    ran: list[int] = []
    synced: list[int] = []
    report = bottleneck.profile(
        lambda: ran.append(1),
        gpu=traceable_gpu,
        iters=3,
        warmup=2,
        sync=lambda: synced.append(1),
        kinds=Activity.ALL,
    )
    assert isinstance(report, ProfileReport)
    assert report.iterations == 3
    assert len(ran) == 5  # warmup 2 + iters 3
    assert len(synced) == 4  # one post-warmup call + one per timed iter
    assert report.dominant_kernel == "gemm"
    assert report.peak_bandwidth_gbps == 900.0
    assert "memory" in report.unavailable  # the fake tracer supports only KERNEL|MEMCPY


def test_profile_on_a_cpu_only_host_is_graceful(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no GPU, `profile` still runs and returns an empty, ungraded report."""
    monkeypatch.setattr(bottleneck, "tracer", lambda: _NoTracer())
    ran: list[int] = []
    report = bottleneck.profile(lambda: ran.append(1), gpu=None, iters=2)
    assert ran == [1, 1]
    assert report.kernels == ()
    assert report.peak_bandwidth_gbps == 0.0
    assert report.unavailable == ()  # no backend -> nothing is flagged unavailable


def test_a_discovered_device_still_scores_the_bandwidth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Naming no device costs the trace nothing, so it must not cost the report its peak either.

    The session finds the host's own card, and the copy bandwidth is graded against that card's
    peak rather than against the zero a caller who passed no argument would otherwise get.
    """
    monkeypatch.setattr(bottleneck, "tracer", lambda: _NoTracer())
    monkeypatch.setattr(GPU, "all", staticmethod(lambda: (FakeGPU(peak_bandwidth_gbs=900.0),)))
    report = bottleneck.profile(lambda: None, iters=1)
    assert report.peak_bandwidth_gbps == 900.0


@pytest.mark.parametrize(
    ("gpu", "busy"),
    [
        (FakeGPU(utilization=FakeUtilization(gpu_pct=85)), True),
        (
            FakeGPU(utilization=FakeUtilization(gpu_pct=1), memory=FakeMemory(percent_used=95.0)),
            True,
        ),
        (
            FakeGPU(utilization=FakeUtilization(gpu_pct=2), memory=FakeMemory(percent_used=10.0)),
            False,
        ),
        (None, False),
    ],
    ids=["compute_load", "memory_pressure", "idle", "cpu_only_host"],
)
def test_gpu_busy_reads_load_off_compute_and_memory(gpu: FakeGPU | None, busy: bool) -> None:
    """Someone else is using the device when either compute or memory is over its threshold."""
    assert bottleneck.gpu_busy(gpu) is busy


def test_wait_for_idle_returns_the_moment_the_device_clears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An already-idle device never sleeps, and a busy one is polled until it clears."""
    idle = FakeGPU(utilization=FakeUtilization(gpu_pct=2), memory=FakeMemory(percent_used=10.0))
    slept: list[float] = []
    assert bottleneck.wait_for_idle(idle, sleep=slept.append) is True
    assert slept == []

    states = iter([True, True, False])
    monkeypatch.setattr(bottleneck, "gpu_busy", lambda *a, **k: next(states))
    assert bottleneck.wait_for_idle(None, timeout=10.0, sleep=slept.append) is True
    assert len(slept) == 2  # two busy polls, then idle


def test_wait_for_idle_times_out_when_stuck_busy() -> None:
    """A GPU that never clears makes `wait_for_idle` return False at the deadline."""
    gpu = FakeGPU(utilization=FakeUtilization(gpu_pct=85))
    slept: list[float] = []
    assert bottleneck.wait_for_idle(gpu, timeout=0.0, sleep=slept.append) is False
