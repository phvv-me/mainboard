"""The one-call bottleneck report and the GPU-contention gating helpers.

Everything here mocks the CUPTI/NVML seam — the `Profiler`, the active `Tracer`, and
`GPU.all` — so the suite runs identically on a host with no GPU. The fakes live in
`conftest` (`FakeTracer`, `RecordingProfiler`) so other profiling tests can reuse them.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

import mainboard
from mainboard.profiling import bottleneck
from mainboard.profiling.models import RegionSummary
from mainboard.profiling.report import Bound, KernelStat, ProfileReport
from mainboard.profiling.result import Profile
from mainboard.profiling.trace import Activity, KernelTrace, MemcpyTrace


def _kernel(name: str, ns: int, **shape: object) -> KernelTrace:
    """A `KernelTrace` of ``name`` lasting ``ns`` nanoseconds from t=0."""
    return KernelTrace(name=name, start_ns=0, end_ns=ns, **shape)  # pyrefly: ignore


class _NoTracer:
    """A tracer offering no deep-trace support, standing in for a backend-less host."""

    def supported(self) -> Activity:
        return Activity(0)


def test_classifies_compute_bound_when_kernels_dominate() -> None:
    """More kernel time than copy time reads as compute-bound."""
    profile = Profile(kernels=(_kernel("gemm", 1000),), memcpys=(MemcpyTrace(end_ns=100),))
    report = ProfileReport.from_profile(profile, iterations=1, peak_bandwidth_gbps=0.0)
    assert report.bound is Bound.COMPUTE


def test_classifies_memory_bound_when_copies_dominate() -> None:
    """More copy time than kernel time reads as memory-bound."""
    profile = Profile(
        kernels=(_kernel("k", 100),),
        memcpys=(MemcpyTrace(end_ns=1000, bytes_moved=4096),),
    )
    report = ProfileReport.from_profile(profile, iterations=1, peak_bandwidth_gbps=10.0)
    assert report.bound is Bound.MEMORY


def test_falls_back_to_utilization_when_no_traces() -> None:
    """With no kernel/copy time, the sampled util decides the verdict."""
    busy_memory = Profile(
        summaries=(
            RegionSummary(name="r", wall_ms=1.0, avg_util_pct=5.0, avg_memory_util_pct=80.0),
        )
    )
    busy_compute = Profile(
        summaries=(
            RegionSummary(name="r", wall_ms=1.0, avg_util_pct=90.0, avg_memory_util_pct=3.0),
        )
    )
    assert ProfileReport.from_profile(busy_memory, iterations=1, peak_bandwidth_gbps=0).bound is (
        Bound.MEMORY
    )
    assert ProfileReport.from_profile(busy_compute, iterations=1, peak_bandwidth_gbps=0).bound is (
        Bound.COMPUTE
    )


def test_unknown_when_nothing_to_classify() -> None:
    """An empty profile cannot be classified, so the verdict is UNKNOWN."""
    report = ProfileReport.from_profile(Profile(), iterations=1, peak_bandwidth_gbps=0.0)
    assert report.bound is Bound.UNKNOWN
    assert report.dominant_kernel == ""
    assert "No kernels traced" in report.report()


def test_dominant_kernel_is_the_hottest_by_total_time() -> None:
    """The breakdown ranks kernels by total time and names the hottest dominant."""
    profile = Profile(
        kernels=(_kernel("a", 100), _kernel("a", 100), _kernel("b", 500)),
    )
    report = ProfileReport.from_profile(profile, iterations=1, peak_bandwidth_gbps=0.0)
    assert report.dominant_kernel == "b"
    assert [k.name for k in report.kernels] == ["b", "a"]
    a = next(k for k in report.kernels if k.name == "a")
    assert a.calls == 2
    assert a.total_ns == 200


def test_kernel_stat_carries_launch_shape() -> None:
    """Occupancy proxy, registers, and the static/dynamic shared split are reported."""
    profile = Profile(
        kernels=(
            _kernel(
                "k",
                1000,
                block="512x1x1",
                registers=48,
                static_shared_mem=2048,
                dynamic_shared_mem=1024,
            ),
        )
    )
    stat = ProfileReport.from_profile(profile, iterations=1, peak_bandwidth_gbps=0.0).kernels[0]
    assert stat.threads_per_block == 512
    assert stat.occupancy_pct == 50.0  # 512 / 1024
    assert stat.registers == 48
    assert stat.static_shared_mem == 2048
    assert stat.dynamic_shared_mem == 1024


def test_copy_bandwidth_scored_against_peak() -> None:
    """Achieved copy bandwidth is bytes over copy-time, shown against the device peak."""
    profile = Profile(memcpys=(MemcpyTrace(end_ns=1000, bytes_moved=2000),))
    report = ProfileReport.from_profile(profile, iterations=1, peak_bandwidth_gbps=10.0)
    assert report.achieved_bandwidth_gbps == 2.0  # 2000 bytes / 1000 ns
    assert "of peak" in report.report()


def test_zero_duration_copy_has_zero_bandwidth() -> None:
    """A copy with no measured time yields no bandwidth rather than dividing by zero."""
    profile = Profile(memcpys=(MemcpyTrace(end_ns=0, bytes_moved=2000),))
    report = ProfileReport.from_profile(profile, iterations=1, peak_bandwidth_gbps=10.0)
    assert report.achieved_bandwidth_gbps == 0.0


def test_unavailable_lists_dropped_activity_kinds() -> None:
    """Kinds requested but unsupported by the device surface in `unavailable`."""
    report = ProfileReport.from_profile(
        Profile(),
        iterations=1,
        peak_bandwidth_gbps=0.0,
        supported=Activity.KERNEL.value,
        requested=Activity.ALL.value,
    )
    assert "memcpy" in report.unavailable
    assert "kernel" not in report.unavailable
    assert "all" not in report.unavailable
    assert "unavailable on this device" in report.report()


def test_unavailable_empty_when_support_unknown() -> None:
    """Without a support/request pair there is nothing to mark unavailable."""
    report = ProfileReport.from_profile(Profile(), iterations=1, peak_bandwidth_gbps=0.0)
    assert report.unavailable == ()


def test_str_renders_the_report() -> None:
    """`str(report)` is the plain-text verdict table."""
    profile = Profile(device="dev", kernels=(_kernel("k", 100),))
    report = ProfileReport.from_profile(profile, iterations=1, peak_bandwidth_gbps=0.0)
    assert str(report) == report.report()
    assert "dev" in str(report)


def test_report_names_cpu_when_no_device() -> None:
    """An empty device name renders as `cpu` in the header."""
    report = ProfileReport.from_profile(Profile(), iterations=1, peak_bandwidth_gbps=0.0)
    assert "device cpu" in report.report()


@given(
    durations=st.lists(st.integers(min_value=1, max_value=10_000), min_size=1, max_size=8),
)
def test_kernel_shares_sum_to_one_hundred(durations: list[int]) -> None:
    """Per-kernel shares always partition 100% of kernel time (no rounding drift away)."""
    kernels = tuple(_kernel(f"k{i}", ns) for i, ns in enumerate(durations))
    report = ProfileReport.from_profile(
        Profile(kernels=kernels), iterations=1, peak_bandwidth_gbps=0.0
    )
    assert abs(sum(k.share_pct for k in report.kernels) - 100.0) < 1e-6


def test_kernel_trace_threads_per_block_ignores_nonnumeric() -> None:
    """A malformed block string degrades to the numeric dims it can parse."""
    assert KernelTrace(block="").threads_per_block == 1
    assert KernelTrace(block="16xNx2").threads_per_block == 32  # N -> 1


def test_kernel_trace_shared_mem_is_the_sum() -> None:
    """`shared_mem` stays the static+dynamic total for backward-compatible reads."""
    kernel = KernelTrace(static_shared_mem=100, dynamic_shared_mem=40)
    assert kernel.shared_mem == 140


def test_profile_runs_callable_and_reports(gpu_profiling_host: object) -> None:
    """`profile` warms up, runs the callable `iters` times, and returns its report."""
    ran: list[int] = []
    report = mainboard.profile(lambda: ran.append(1), iters=3, warmup=2)
    assert isinstance(report, ProfileReport)
    assert report.iterations == 3
    assert len(ran) == 5  # warmup 2 + iters 3
    assert report.dominant_kernel == "gemm"
    assert report.peak_bandwidth_gbps == 900.0  # from the fake GPU


def test_profile_adapts_requested_kinds_to_device_support(gpu_profiling_host: object) -> None:
    """Requested kinds the device lacks are dropped from the trace and noted."""
    report = mainboard.profile(lambda: None, kinds=Activity.ALL)
    assert "memory" in report.unavailable  # the fake tracer supports only KERNEL|MEMCPY


def test_profile_syncs_between_runs(gpu_profiling_host: object) -> None:
    """A provided `sync` barrier is called after warmup and each timed run."""
    synced: list[int] = []
    mainboard.profile(lambda: None, iters=2, warmup=1, sync=lambda: synced.append(1))
    assert len(synced) >= 3  # one post-warmup + one per timed iter


def test_profile_on_cpu_only_host_is_graceful(
    cpu_only_host: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no GPU (no tracing backend), `profile` still runs and returns an empty report."""
    from mainboard.profiling import annotate
    from mainboard.profiling.tracer import Tracer

    monkeypatch.setattr(bottleneck, "tracer", lambda: _NoTracer())
    monkeypatch.setattr(annotate, "tracer", Tracer)  # region() reads this no-op tracer
    ran: list[int] = []
    report = mainboard.profile(lambda: ran.append(1), iters=2)
    assert ran == [1, 1]
    assert report.kernels == ()
    assert report.bound is Bound.UNKNOWN
    assert report.unavailable == ()  # no backend -> nothing is flagged unavailable


def test_gpu_busy_true_under_compute_load(busy_gpu_host: object) -> None:
    """A GPU above the utilization threshold reads as busy."""
    assert mainboard.gpu_busy() is True


def test_gpu_busy_true_under_memory_pressure(memory_pressure_gpu_host: object) -> None:
    """A GPU near its memory capacity reads as busy even when compute is idle."""
    assert mainboard.gpu_busy() is True


def test_gpu_busy_false_when_idle(idle_gpu_host: object) -> None:
    """An idle GPU below both thresholds reads as not busy."""
    assert mainboard.gpu_busy() is False


def test_gpu_busy_false_for_missing_index(idle_gpu_host: object) -> None:
    """An out-of-range device index reads as not busy rather than raising."""
    assert mainboard.gpu_busy(index=9) is False


def test_wait_for_idle_returns_true_when_already_idle(idle_gpu_host: object) -> None:
    """An already-idle GPU returns immediately without sleeping."""
    slept: list[float] = []
    assert mainboard.wait_for_idle(sleep=slept.append) is True
    assert slept == []


def test_wait_for_idle_polls_until_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    """`wait_for_idle` polls until the device clears, then returns True."""
    states = iter([True, True, False])
    monkeypatch.setattr(bottleneck, "gpu_busy", lambda *a, **k: next(states))
    slept: list[float] = []
    assert bottleneck.wait_for_idle(timeout=10.0, sleep=slept.append) is True
    assert len(slept) == 2  # two busy polls, then idle


def test_wait_for_idle_times_out_when_stuck_busy(busy_gpu_host: object) -> None:
    """A GPU that never clears makes `wait_for_idle` return False at the deadline."""
    slept: list[float] = []
    assert mainboard.wait_for_idle(timeout=0.0, sleep=slept.append) is False


def test_kernel_stat_is_frozen() -> None:
    """The report models are immutable like the rest of the profiling schemas."""
    stat = KernelStat(
        name="k",
        calls=1,
        total_ns=1,
        avg_ns=1.0,
        share_pct=100.0,
        grid="",
        block="",
        threads_per_block=1,
        occupancy_pct=0.0,
        registers=0,
        static_shared_mem=0,
        dynamic_shared_mem=0,
    )
    with pytest.raises(ValueError, match="frozen"):
        stat.calls = 2  # pyrefly: ignore
